/**
 * NanoClaw Agent Runner
 * Runs inside a container, receives config via stdin, outputs result to stdout
 *
 * Input protocol:
 *   Stdin: Full ContainerInput JSON (read until EOF, like before)
 *   IPC:   Follow-up messages written as JSON files to /workspace/ipc/input/
 *          Files: {type:"message", text:"..."}.json — polled and consumed
 *          Sentinel: /workspace/ipc/input/_close — signals session end
 *
 * Stdout protocol:
 *   Each result is wrapped in OUTPUT_START_MARKER / OUTPUT_END_MARKER pairs.
 *   Multiple results may be emitted (one per agent teams result).
 *   Final marker after loop ends signals completion.
 */

import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { spawn } from 'child_process';
import { query, HookCallback, PreCompactHookInput, PreToolUseHookInput } from '@anthropic-ai/claude-agent-sdk';
import { fileURLToPath } from 'url';

interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  secrets?: Record<string, string>;
}

interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
}

interface SessionEntry {
  sessionId: string;
  fullPath: string;
  summary: string;
  firstPrompt: string;
}

interface SessionsIndex {
  entries: SessionEntry[];
}

interface SDKUserMessage {
  type: 'user';
  message: { role: 'user'; content: string };
  parent_tool_use_id: null;
  session_id: string;
}

type AgentProvider = 'claude' | 'openai';

interface OpenAiChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface OpenAiSessionState {
  messages: OpenAiChatMessage[];
}

interface BashExecResult {
  command: string;
  exitCode: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
}

type OpenAiDirectCommand =
  | { type: 'bash'; command: string }
  | { type: 'skill-list' }
  | { type: 'skill-show'; name: string }
  | { type: 'skill-use'; name: string; task: string };

interface CodexSseEvent {
  event?: string;
  data?: unknown;
}

const IPC_INPUT_DIR = '/workspace/ipc/input';
const IPC_INPUT_CLOSE_SENTINEL = path.join(IPC_INPUT_DIR, '_close');
const IPC_POLL_MS = 500;
const OPENAI_BASH_MAX_STEPS = 6;
const OPENAI_BASH_TIMEOUT_MS = 120_000;
const OPENAI_BASH_MAX_OUTPUT = 16_000;
const OPENAI_SKILL_MAX_CONTENT = 24_000;

/**
 * Push-based async iterable for streaming user messages to the SDK.
 * Keeps the iterable alive until end() is called, preventing isSingleUserTurn.
 */
class MessageStream {
  private queue: SDKUserMessage[] = [];
  private waiting: (() => void) | null = null;
  private done = false;

  push(text: string): void {
    this.queue.push({
      type: 'user',
      message: { role: 'user', content: text },
      parent_tool_use_id: null,
      session_id: '',
    });
    this.waiting?.();
  }

  end(): void {
    this.done = true;
    this.waiting?.();
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<SDKUserMessage> {
    while (true) {
      while (this.queue.length > 0) {
        yield this.queue.shift()!;
      }
      if (this.done) return;
      await new Promise<void>(r => { this.waiting = r; });
      this.waiting = null;
    }
  }
}

async function readStdin(): Promise<string> {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { data += chunk; });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

function writeOutput(output: ContainerOutput): void {
  console.log(OUTPUT_START_MARKER);
  console.log(JSON.stringify(output));
  console.log(OUTPUT_END_MARKER);
}

function log(message: string): void {
  console.error(`[agent-runner] ${message}`);
}

function getSessionSummary(sessionId: string, transcriptPath: string): string | null {
  const projectDir = path.dirname(transcriptPath);
  const indexPath = path.join(projectDir, 'sessions-index.json');

  if (!fs.existsSync(indexPath)) {
    log(`Sessions index not found at ${indexPath}`);
    return null;
  }

  try {
    const index: SessionsIndex = JSON.parse(fs.readFileSync(indexPath, 'utf-8'));
    const entry = index.entries.find(e => e.sessionId === sessionId);
    if (entry?.summary) {
      return entry.summary;
    }
  } catch (err) {
    log(`Failed to read sessions index: ${err instanceof Error ? err.message : String(err)}`);
  }

  return null;
}

/**
 * Archive the full transcript to conversations/ before compaction.
 */
function createPreCompactHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preCompact = input as PreCompactHookInput;
    const transcriptPath = preCompact.transcript_path;
    const sessionId = preCompact.session_id;

    if (!transcriptPath || !fs.existsSync(transcriptPath)) {
      log('No transcript found for archiving');
      return {};
    }

    try {
      const content = fs.readFileSync(transcriptPath, 'utf-8');
      const messages = parseTranscript(content);

      if (messages.length === 0) {
        log('No messages to archive');
        return {};
      }

      const summary = getSessionSummary(sessionId, transcriptPath);
      const name = summary ? sanitizeFilename(summary) : generateFallbackName();

      const conversationsDir = '/workspace/group/conversations';
      fs.mkdirSync(conversationsDir, { recursive: true });

      const date = new Date().toISOString().split('T')[0];
      const filename = `${date}-${name}.md`;
      const filePath = path.join(conversationsDir, filename);

      const markdown = formatTranscriptMarkdown(messages, summary);
      fs.writeFileSync(filePath, markdown);

      log(`Archived conversation to ${filePath}`);
    } catch (err) {
      log(`Failed to archive transcript: ${err instanceof Error ? err.message : String(err)}`);
    }

    return {};
  };
}

// Secrets to strip from Bash tool subprocess environments.
// These are needed by claude-code for API auth but should never
// be visible to commands Kit runs.
const SECRET_ENV_VARS = ['ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN', 'OPENAI_API_KEY', 'OPENAI_OAUTH_ACCESS_TOKEN'];

const OPENAI_SESSIONS_DIR = '/home/node/.claude/nanoclaw-openai-sessions';

function configureGithubGitAuthEnv(env: Record<string, string | undefined>): void {
  const token = env.GH_TOKEN || env.GITHUB_TOKEN;
  if (!token) return;

  // Normalize for tools that prefer one name or the other.
  if (!env.GH_TOKEN) env.GH_TOKEN = token;
  if (!env.GITHUB_TOKEN) env.GITHUB_TOKEN = token;

  // Configure git via env-only config so pushes work without mutating mounted repos.
  // This rewrites GitHub SSH remotes to HTTPS and serves credentials from GH_TOKEN.
  const entries: Array<[string, string]> = [
    ['url.https://github.com/.insteadof', 'git@github.com:'],
    ['url.https://github.com/.insteadof', 'ssh://git@github.com/'],
    ['credential.https://github.com.helper', '!f() { test "$1" = get || exit 0; echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'],
  ];

  env.GIT_CONFIG_COUNT = String(entries.length);
  for (let i = 0; i < entries.length; i++) {
    env[`GIT_CONFIG_KEY_${i}`] = entries[i][0];
    env[`GIT_CONFIG_VALUE_${i}`] = entries[i][1];
  }
}

function exposeGithubAuthToToolSubprocesses(env: Record<string, string | undefined>): void {
  // Bash/gh/git commands run as subprocesses from this process and need these vars
  // in process.env. Keep API/provider creds isolated in sdkEnv only.
  const passthroughPrefixes = ['GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_'];
  const passthroughExact = new Set([
    'GH_TOKEN',
    'GITHUB_TOKEN',
    'GIT_CONFIG_COUNT',
  ]);

  for (const [key, value] of Object.entries(env)) {
    if (!value) continue;
    if (passthroughExact.has(key) || passthroughPrefixes.some((p) => key.startsWith(p))) {
      process.env[key] = value;
    }
  }
}

function getProvider(env: Record<string, string | undefined>): AgentProvider {
  const configured = env.NANOCLAW_AGENT_PROVIDER?.toLowerCase();
  if (configured === 'openai') return 'openai';
  if (configured === 'claude') return 'claude';
  if (env.OPENAI_API_KEY && !env.ANTHROPIC_API_KEY && !env.CLAUDE_CODE_OAUTH_TOKEN) {
    return 'openai';
  }
  return 'claude';
}

function ensureOpenAiSessionsDir(): void {
  fs.mkdirSync(OPENAI_SESSIONS_DIR, { recursive: true });
}

function openAiSessionPath(sessionId: string): string {
  return path.join(OPENAI_SESSIONS_DIR, `${sessionId}.json`);
}

function loadOpenAiSession(sessionId: string): OpenAiSessionState {
  ensureOpenAiSessionsDir();
  const file = openAiSessionPath(sessionId);
  if (!fs.existsSync(file)) return { messages: [] };
  try {
    const parsed = JSON.parse(fs.readFileSync(file, 'utf-8')) as OpenAiSessionState;
    if (Array.isArray(parsed.messages)) return parsed;
  } catch (err) {
    log(`Failed to load OpenAI session ${sessionId}: ${err instanceof Error ? err.message : String(err)}`);
  }
  return { messages: [] };
}

function saveOpenAiSession(sessionId: string, state: OpenAiSessionState): void {
  ensureOpenAiSessionsDir();
  fs.writeFileSync(openAiSessionPath(sessionId), JSON.stringify(state, null, 2));
}

function readIfExists(filePath: string): string | null {
  try {
    if (fs.existsSync(filePath)) return fs.readFileSync(filePath, 'utf-8');
  } catch {
  }
  return null;
}

function buildOpenAiSystemPrompt(containerInput: ContainerInput): string | undefined {
  const parts: string[] = [];

  const groupMemory = readIfExists('/workspace/group/CLAUDE.md');
  if (groupMemory) {
    parts.push('Group memory (CLAUDE.md):\n' + groupMemory);
  }

  const globalMemory = !containerInput.isMain ? readIfExists('/workspace/global/CLAUDE.md') : null;
  if (globalMemory) {
    parts.push('Global memory (CLAUDE.md):\n' + globalMemory);
  }

  const projectMemory = containerInput.isMain ? readIfExists('/workspace/project/CLAUDE.md') : null;
  if (projectMemory) {
    parts.push('Project memory (CLAUDE.md):\n' + projectMemory);
  }

  if (parts.length === 0) return undefined;
  return [
    'You are the assistant running inside NanoClaw.',
    'Follow the memory files below as operating context.',
    parts.join('\n\n---\n\n'),
  ].join('\n\n');
}

function buildOpenAiBashToolPrompt(): string {
  return [
    'Tooling available: Bash shell commands can be executed by this runtime.',
    'When you need to run a shell command, respond with ONLY a single XML block:',
    '<bash>',
    'your command here',
    '</bash>',
    'Do not include any other text in that message.',
    'After you receive command output, continue reasoning and either emit another <bash> block or provide the final user-facing answer.',
    'Run commands in /workspace/group unless you need another directory.',
  ].join('\n');
}

function extractBashCommand(text: string): string | null {
  const trimmed = text.trim();
  const match = trimmed.match(/^<bash>\s*([\s\S]*?)\s*<\/bash>$/);
  if (!match) return null;
  const command = match[1].trim();
  return command || null;
}

function truncateText(input: string, maxChars: number): string {
  if (input.length <= maxChars) return input;
  return input.slice(0, maxChars) + `\n...[truncated ${input.length - maxChars} chars]`;
}

function parseOpenAiDirectCommand(prompt: string): OpenAiDirectCommand | null {
  const trimmed = extractDirectCommandCandidate(prompt);
  if (!trimmed.startsWith('/')) return null;

  if (trimmed.startsWith('/bash')) {
    const rest = trimmed.slice('/bash'.length).trim();
    if (!rest) return null;
    return { type: 'bash', command: rest };
  }

  if (!trimmed.startsWith('/skill')) return null;
  const after = trimmed.slice('/skill'.length).trim();
  if (!after || after === 'list') return { type: 'skill-list' };

  const lines = trimmed.split('\n');
  const firstLine = lines[0].trim();
  const firstParts = firstLine.split(/\s+/);

  if (firstParts[1] === 'show' && firstParts[2]) {
    return { type: 'skill-show', name: firstParts[2] };
  }

  if (firstParts[1] === 'use' && firstParts[2]) {
    const task = lines.slice(1).join('\n').trim();
    return { type: 'skill-use', name: firstParts[2], task };
  }

  // Shorthand:
  // /skill <name>
  // <task...>
  if (firstParts[1]) {
    const task = lines.slice(1).join('\n').trim();
    return { type: 'skill-use', name: firstParts[1], task };
  }

  return null;
}

function extractDirectCommandCandidate(prompt: string): string {
  const raw = prompt.trim();
  if (raw.startsWith('/')) return raw;

  // NanoClaw often wraps prompts as:
  // <messages><message ...>@Andy /bash ...</message></messages>
  // Extract the last message body and strip a leading @mention.
  const matches = [...raw.matchAll(/<message\b[^>]*>([\s\S]*?)<\/message>/g)];
  if (matches.length === 0) return raw;

  const lastBody = (matches[matches.length - 1][1] || '').trim();
  const withoutMention = lastBody.replace(/^@\S+\s+/, '').trim();
  return withoutMention || lastBody;
}

function buildBashExecEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const key of SECRET_ENV_VARS) {
    delete env[key];
  }
  return env;
}

function formatBashExecForUser(result: BashExecResult): string {
  return [
    `Command: ${result.command}`,
    `Exit code: ${result.exitCode === null ? 'null' : result.exitCode}`,
    `Timed out: ${result.timedOut ? 'yes' : 'no'}`,
    '',
    'STDOUT:',
    '```',
    result.stdout || '(empty)',
    '```',
    '',
    'STDERR:',
    '```',
    result.stderr || '(empty)',
    '```',
  ].join('\n');
}

async function execBashCommand(command: string): Promise<BashExecResult> {
  return new Promise((resolve) => {
    const child = spawn('bash', ['-lc', command], {
      cwd: '/workspace/group',
      env: buildBashExecEnv(),
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let stdout = '';
    let stderr = '';
    let stdoutTruncated = false;
    let stderrTruncated = false;
    let timedOut = false;

    const append = (
      chunk: string,
      current: string,
      truncated: boolean,
    ): { text: string; truncated: boolean } => {
      if (truncated) return { text: current, truncated };
      const remaining = OPENAI_BASH_MAX_OUTPUT - current.length;
      if (remaining <= 0) return { text: current, truncated: true };
      if (chunk.length > remaining) {
        return { text: current + chunk.slice(0, remaining), truncated: true };
      }
      return { text: current + chunk, truncated: false };
    };

    child.stdout.on('data', (data) => {
      const out = append(String(data), stdout, stdoutTruncated);
      stdout = out.text;
      stdoutTruncated = out.truncated;
    });
    child.stderr.on('data', (data) => {
      const out = append(String(data), stderr, stderrTruncated);
      stderr = out.text;
      stderrTruncated = out.truncated;
    });

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, OPENAI_BASH_TIMEOUT_MS);

    child.on('close', (code) => {
      clearTimeout(timeout);
      resolve({
        command,
        exitCode: code,
        stdout: truncateText(stdout, OPENAI_BASH_MAX_OUTPUT),
        stderr: truncateText(stderr, OPENAI_BASH_MAX_OUTPUT),
        timedOut,
      });
    });
  });
}

function formatBashResultForModel(result: BashExecResult): string {
  return [
    'Bash command result:',
    `Command: ${result.command}`,
    `Exit code: ${result.exitCode === null ? 'null' : result.exitCode}`,
    `Timed out: ${result.timedOut ? 'yes' : 'no'}`,
    'STDOUT:',
    result.stdout || '(empty)',
    'STDERR:',
    result.stderr || '(empty)',
    'If you need another command, respond with ONLY a <bash>...</bash> block. Otherwise provide the final answer for the user.',
  ].join('\n');
}

interface SkillInfo {
  name: string;
  path: string;
}

function getSkillBaseDir(): string {
  // Skills are synced by host into each group's ~/.claude/skills/
  return '/home/node/.claude/skills';
}

function listAvailableSkills(): SkillInfo[] {
  const base = getSkillBaseDir();
  if (!fs.existsSync(base)) return [];
  const results: SkillInfo[] = [];
  for (const entry of fs.readdirSync(base)) {
    const skillPath = path.join(base, entry, 'SKILL.md');
    if (fs.existsSync(skillPath)) {
      results.push({ name: entry, path: skillPath });
    }
  }
  return results.sort((a, b) => a.name.localeCompare(b.name));
}

function loadSkillContent(name: string): { name: string; content: string; path: string } | null {
  const skills = listAvailableSkills();
  const normalized = name.trim().toLowerCase();
  const exact = skills.find((s) => s.name.toLowerCase() === normalized);
  if (!exact) return null;
  const raw = fs.readFileSync(exact.path, 'utf-8');
  return {
    name: exact.name,
    path: exact.path,
    content: truncateText(raw, OPENAI_SKILL_MAX_CONTENT),
  };
}

async function handleOpenAiDirectCommand(
  direct: OpenAiDirectCommand,
): Promise<{ handled: true; result?: string; transformedPrompt?: string }> {
  if (direct.type === 'bash') {
    const result = await execBashCommand(direct.command);
    return { handled: true, result: formatBashExecForUser(result) };
  }

  if (direct.type === 'skill-list') {
    const skills = listAvailableSkills();
    if (skills.length === 0) {
      return { handled: true, result: 'No skills found in /home/node/.claude/skills.' };
    }
    return {
      handled: true,
      result: [
        'Available skills:',
        ...skills.map((s) => `- ${s.name}`),
        '',
        'Use `/skill show <name>` to inspect one, or `/skill use <name>` followed by a task on the next lines.',
      ].join('\n'),
    };
  }

  if (direct.type === 'skill-show') {
    const skill = loadSkillContent(direct.name);
    if (!skill) {
      return { handled: true, result: `Skill not found: ${direct.name}` };
    }
    return {
      handled: true,
      result: [
        `Skill: ${skill.name}`,
        `Path: ${skill.path}`,
        '',
        '```md',
        skill.content,
        '```',
      ].join('\n'),
    };
  }

  const skill = loadSkillContent(direct.name);
  if (!skill) {
    return { handled: true, result: `Skill not found: ${direct.name}` };
  }
  if (!direct.task.trim()) {
    return {
      handled: true,
      result: `No task provided. Usage:\n/skill use ${skill.name}\n<your task here>`,
    };
  }

  const transformedPrompt = [
    `Use the skill "${skill.name}" for this task.`,
    '',
    'Skill instructions:',
    skill.content,
    '',
    'Task:',
    direct.task,
  ].join('\n');
  return { handled: true, transformedPrompt };
}

async function callOpenAiChat(
  messages: OpenAiChatMessage[],
  env: Record<string, string | undefined>,
): Promise<string> {
  if (env.OPENAI_API_KEY) {
    return callOpenAiApiChat(messages, env);
  }
  if (env.OPENAI_OAUTH_ACCESS_TOKEN) {
    return callCodexOAuthChat(messages, env);
  }

  throw new Error('OPENAI_API_KEY or OPENAI_OAUTH_ACCESS_TOKEN is required when using NANOCLAW_AGENT_PROVIDER=openai');
}

async function callOpenAiApiChat(
  messages: OpenAiChatMessage[],
  env: Record<string, string | undefined>,
): Promise<string> {
  const bearerToken = env.OPENAI_API_KEY;
  if (!bearerToken) {
    throw new Error('OPENAI_API_KEY or OPENAI_OAUTH_ACCESS_TOKEN is required when using NANOCLAW_AGENT_PROVIDER=openai');
  }

  const baseUrl = (env.OPENAI_BASE_URL || 'https://api.openai.com/v1').replace(/\/+$/, '');
  const model = env.OPENAI_MODEL || 'gpt-5-mini';

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${bearerToken}`,
    },
    body: JSON.stringify({
      model,
      messages,
    }),
  });

  const raw = await response.text();
  let data: any;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new Error(`OpenAI API returned non-JSON (${response.status}): ${raw.slice(0, 300)}`);
  }

  if (!response.ok) {
    const apiMessage =
      (data?.error?.message && String(data.error.message)) ||
      raw.slice(0, 300);
    throw new Error(`OpenAI API error ${response.status}: ${apiMessage}`);
  }

  const content = data?.choices?.[0]?.message?.content;
  if (typeof content === 'string' && content.trim()) return content;
  if (Array.isArray(content)) {
    const text = content
      .map((part: any) => (typeof part?.text === 'string' ? part.text : ''))
      .join('')
      .trim();
    if (text) return text;
  }

  throw new Error('OpenAI API returned no assistant text content');
}

function parseSseBlocks(buffer: string): { events: CodexSseEvent[]; rest: string } {
  const events: CodexSseEvent[] = [];
  let rest = buffer;
  let idx: number;

  while ((idx = rest.indexOf('\n\n')) !== -1) {
    const block = rest.slice(0, idx);
    rest = rest.slice(idx + 2);
    if (!block.trim()) continue;

    const lines = block.split('\n');
    const eventLine = lines.find(l => l.startsWith('event:'));
    const dataLines = lines.filter(l => l.startsWith('data:')).map(l => l.slice(5).trim());
    const dataRaw = dataLines.join('\n');

    if (!dataRaw || dataRaw === '[DONE]') {
      events.push({ event: eventLine?.slice(6).trim(), data: '[DONE]' });
      continue;
    }

    try {
      events.push({
        event: eventLine?.slice(6).trim(),
        data: JSON.parse(dataRaw),
      });
    } catch {
      events.push({
        event: eventLine?.slice(6).trim(),
        data: dataRaw,
      });
    }
  }

  return { events, rest };
}

function collectCodexCompletedText(data: any): string {
  const output = data?.response?.output;
  if (!Array.isArray(output)) return '';

  const chunks: string[] = [];
  for (const item of output) {
    const content = item?.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (typeof part?.text === 'string') {
        chunks.push(part.text);
      } else if (typeof part?.output_text === 'string') {
        chunks.push(part.output_text);
      }
    }
  }
  return chunks.join('').trim();
}

async function callCodexOAuthChat(
  messages: OpenAiChatMessage[],
  env: Record<string, string | undefined>,
): Promise<string> {
  const bearerToken = env.OPENAI_OAUTH_ACCESS_TOKEN;
  if (!bearerToken) {
    throw new Error('OPENAI_OAUTH_ACCESS_TOKEN is required for Codex OAuth backend');
  }

  const accountId = env.OPENAI_CODEX_ACCOUNT_ID;
  const model = env.OPENAI_MODEL || 'gpt-5.3-codex';
  const instructions = messages
    .filter(m => m.role === 'system')
    .map(m => m.content.trim())
    .filter(Boolean)
    .join('\n\n');

  const input = messages
    .filter(m => m.role !== 'system')
    .map(m => ({
      role: m.role,
      content: [{
        type: m.role === 'assistant' ? 'output_text' : 'input_text',
        text: m.content,
      }],
    }));

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
    Authorization: `Bearer ${bearerToken}`,
  };
  if (accountId) {
    headers['ChatGPT-Account-Id'] = accountId;
  }

  const response = await fetch('https://chatgpt.com/backend-api/codex/responses', {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model,
      instructions: instructions || 'You are a helpful assistant.',
      input,
      stream: true,
      store: false,
    }),
  });

  if (!response.ok || !response.body) {
    const raw = await response.text();
    throw new Error(`Codex OAuth error ${response.status}: ${raw.slice(0, 300)}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let text = '';
  let completedText = '';

  while (true) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });

    const parsed = parseSseBlocks(buffer);
    buffer = parsed.rest;

    for (const ev of parsed.events) {
      const data = ev.data;
      if (!data || typeof data !== 'object') continue;
      const typed = data as { type?: string; delta?: string };
      if (typed.type === 'response.output_text.delta' && typeof typed.delta === 'string') {
        text += typed.delta;
      } else if (typed.type === 'response.completed') {
        completedText = collectCodexCompletedText(data);
      }
    }
  }

  const finalText = (completedText || text).trim();
  if (!finalText) {
    throw new Error('Codex OAuth backend returned no assistant text content');
  }
  return finalText;
}

function createSanitizeBashHook(): HookCallback {
  return async (input, _toolUseId, _context) => {
    const preInput = input as PreToolUseHookInput;
    const command = (preInput.tool_input as { command?: string })?.command;
    if (!command) return {};

    const unsetPrefix = `unset ${SECRET_ENV_VARS.join(' ')} 2>/dev/null; `;
    return {
      hookSpecificOutput: {
        hookEventName: 'PreToolUse',
        updatedInput: {
          ...(preInput.tool_input as Record<string, unknown>),
          command: unsetPrefix + command,
        },
      },
    };
  };
}

function sanitizeFilename(summary: string): string {
  return summary
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 50);
}

function generateFallbackName(): string {
  const time = new Date();
  return `conversation-${time.getHours().toString().padStart(2, '0')}${time.getMinutes().toString().padStart(2, '0')}`;
}

interface ParsedMessage {
  role: 'user' | 'assistant';
  content: string;
}

function parseTranscript(content: string): ParsedMessage[] {
  const messages: ParsedMessage[] = [];

  for (const line of content.split('\n')) {
    if (!line.trim()) continue;
    try {
      const entry = JSON.parse(line);
      if (entry.type === 'user' && entry.message?.content) {
        const text = typeof entry.message.content === 'string'
          ? entry.message.content
          : entry.message.content.map((c: { text?: string }) => c.text || '').join('');
        if (text) messages.push({ role: 'user', content: text });
      } else if (entry.type === 'assistant' && entry.message?.content) {
        const textParts = entry.message.content
          .filter((c: { type: string }) => c.type === 'text')
          .map((c: { text: string }) => c.text);
        const text = textParts.join('');
        if (text) messages.push({ role: 'assistant', content: text });
      }
    } catch {
    }
  }

  return messages;
}

function formatTranscriptMarkdown(messages: ParsedMessage[], title?: string | null): string {
  const now = new Date();
  const formatDateTime = (d: Date) => d.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    hour12: true
  });

  const lines: string[] = [];
  lines.push(`# ${title || 'Conversation'}`);
  lines.push('');
  lines.push(`Archived: ${formatDateTime(now)}`);
  lines.push('');
  lines.push('---');
  lines.push('');

  for (const msg of messages) {
    const sender = msg.role === 'user' ? 'User' : 'Andy';
    const content = msg.content.length > 2000
      ? msg.content.slice(0, 2000) + '...'
      : msg.content;
    lines.push(`**${sender}**: ${content}`);
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Check for _close sentinel.
 */
function shouldClose(): boolean {
  if (fs.existsSync(IPC_INPUT_CLOSE_SENTINEL)) {
    try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }
    return true;
  }
  return false;
}

/**
 * Drain all pending IPC input messages.
 * Returns messages found, or empty array.
 */
function drainIpcInput(): string[] {
  try {
    fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });
    const files = fs.readdirSync(IPC_INPUT_DIR)
      .filter(f => f.endsWith('.json'))
      .sort();

    const messages: string[] = [];
    for (const file of files) {
      const filePath = path.join(IPC_INPUT_DIR, file);
      try {
        const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        fs.unlinkSync(filePath);
        if (data.type === 'message' && data.text) {
          messages.push(data.text);
        }
      } catch (err) {
        log(`Failed to process input file ${file}: ${err instanceof Error ? err.message : String(err)}`);
        try { fs.unlinkSync(filePath); } catch { /* ignore */ }
      }
    }
    return messages;
  } catch (err) {
    log(`IPC drain error: ${err instanceof Error ? err.message : String(err)}`);
    return [];
  }
}

/**
 * Wait for a new IPC message or _close sentinel.
 * Returns the messages as a single string, or null if _close.
 */
function waitForIpcMessage(): Promise<string | null> {
  return new Promise((resolve) => {
    const poll = () => {
      if (shouldClose()) {
        resolve(null);
        return;
      }
      const messages = drainIpcInput();
      if (messages.length > 0) {
        resolve(messages.join('\n'));
        return;
      }
      setTimeout(poll, IPC_POLL_MS);
    };
    poll();
  });
}

/**
 * Run a single query and stream results via writeOutput.
 * Uses MessageStream (AsyncIterable) to keep isSingleUserTurn=false,
 * allowing agent teams subagents to run to completion.
 * Also pipes IPC messages into the stream during the query.
 */
async function runQuery(
  prompt: string,
  sessionId: string | undefined,
  mcpServerPath: string,
  containerInput: ContainerInput,
  sdkEnv: Record<string, string | undefined>,
  resumeAt?: string,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const stream = new MessageStream();
  stream.push(prompt);

  // Poll IPC for follow-up messages and _close sentinel during the query
  let ipcPolling = true;
  let closedDuringQuery = false;
  const pollIpcDuringQuery = () => {
    if (!ipcPolling) return;
    if (shouldClose()) {
      log('Close sentinel detected during query, ending stream');
      closedDuringQuery = true;
      stream.end();
      ipcPolling = false;
      return;
    }
    const messages = drainIpcInput();
    for (const text of messages) {
      log(`Piping IPC message into active query (${text.length} chars)`);
      stream.push(text);
    }
    setTimeout(pollIpcDuringQuery, IPC_POLL_MS);
  };
  setTimeout(pollIpcDuringQuery, IPC_POLL_MS);

  let newSessionId: string | undefined;
  let lastAssistantUuid: string | undefined;
  let messageCount = 0;
  let resultCount = 0;

  // Load global CLAUDE.md as additional system context (shared across all groups)
  const globalClaudeMdPath = '/workspace/global/CLAUDE.md';
  let globalClaudeMd: string | undefined;
  if (!containerInput.isMain && fs.existsSync(globalClaudeMdPath)) {
    globalClaudeMd = fs.readFileSync(globalClaudeMdPath, 'utf-8');
  }

  // Discover additional directories mounted at /workspace/extra/*
  // These are passed to the SDK so their CLAUDE.md files are loaded automatically
  const extraDirs: string[] = [];
  const extraBase = '/workspace/extra';
  if (fs.existsSync(extraBase)) {
    for (const entry of fs.readdirSync(extraBase)) {
      const fullPath = path.join(extraBase, entry);
      if (fs.statSync(fullPath).isDirectory()) {
        extraDirs.push(fullPath);
      }
    }
  }
  if (extraDirs.length > 0) {
    log(`Additional directories: ${extraDirs.join(', ')}`);
  }

  for await (const message of query({
    prompt: stream,
    options: {
      cwd: '/workspace/group',
      additionalDirectories: extraDirs.length > 0 ? extraDirs : undefined,
      resume: sessionId,
      resumeSessionAt: resumeAt,
      systemPrompt: globalClaudeMd
        ? { type: 'preset' as const, preset: 'claude_code' as const, append: globalClaudeMd }
        : undefined,
      allowedTools: [
        'Bash',
        'Read', 'Write', 'Edit', 'Glob', 'Grep',
        'WebSearch', 'WebFetch',
        'Task', 'TaskOutput', 'TaskStop',
        'TeamCreate', 'TeamDelete', 'SendMessage',
        'TodoWrite', 'ToolSearch', 'Skill',
        'NotebookEdit',
        'mcp__nanoclaw__*'
      ],
      env: sdkEnv,
      permissionMode: 'bypassPermissions',
      allowDangerouslySkipPermissions: true,
      settingSources: ['project', 'user'],
      mcpServers: {
        nanoclaw: {
          command: 'node',
          args: [mcpServerPath],
          env: {
            NANOCLAW_CHAT_JID: containerInput.chatJid,
            NANOCLAW_GROUP_FOLDER: containerInput.groupFolder,
            NANOCLAW_IS_MAIN: containerInput.isMain ? '1' : '0',
          },
        },
      },
      hooks: {
        PreCompact: [{ hooks: [createPreCompactHook()] }],
        PreToolUse: [{ matcher: 'Bash', hooks: [createSanitizeBashHook()] }],
      },
    }
  })) {
    messageCount++;
    const msgType = message.type === 'system' ? `system/${(message as { subtype?: string }).subtype}` : message.type;
    log(`[msg #${messageCount}] type=${msgType}`);

    if (message.type === 'assistant' && 'uuid' in message) {
      lastAssistantUuid = (message as { uuid: string }).uuid;
    }

    if (message.type === 'system' && message.subtype === 'init') {
      newSessionId = message.session_id;
      log(`Session initialized: ${newSessionId}`);
    }

    if (message.type === 'system' && (message as { subtype?: string }).subtype === 'task_notification') {
      const tn = message as { task_id: string; status: string; summary: string };
      log(`Task notification: task=${tn.task_id} status=${tn.status} summary=${tn.summary}`);
    }

    if (message.type === 'result') {
      resultCount++;
      const textResult = 'result' in message ? (message as { result?: string }).result : null;
      log(`Result #${resultCount}: subtype=${message.subtype}${textResult ? ` text=${textResult.slice(0, 200)}` : ''}`);
      writeOutput({
        status: 'success',
        result: textResult || null,
        newSessionId
      });
    }
  }

  ipcPolling = false;
  log(`Query done. Messages: ${messageCount}, results: ${resultCount}, lastAssistantUuid: ${lastAssistantUuid || 'none'}, closedDuringQuery: ${closedDuringQuery}`);
  return { newSessionId, lastAssistantUuid, closedDuringQuery };
}

async function runOpenAiQuery(
  prompt: string,
  sessionId: string | undefined,
  containerInput: ContainerInput,
  env: Record<string, string | undefined>,
): Promise<{ newSessionId?: string; lastAssistantUuid?: string; closedDuringQuery: boolean }> {
  const direct = parseOpenAiDirectCommand(prompt);
  if (direct) {
    const handled = await handleOpenAiDirectCommand(direct);
    if (handled.result != null) {
      writeOutput({
        status: 'success',
        result: handled.result,
        newSessionId: sessionId,
      });
      return { newSessionId: sessionId, closedDuringQuery: false };
    }
    if (handled.transformedPrompt) {
      prompt = handled.transformedPrompt;
    }
  }

  const newSessionId = sessionId || crypto.randomUUID();
  const session = loadOpenAiSession(newSessionId);

  if (session.messages.length === 0) {
    const systemPrompt = buildOpenAiSystemPrompt(containerInput);
    const parts = [systemPrompt, buildOpenAiBashToolPrompt()].filter(Boolean);
    if (parts.length > 0) {
      session.messages.push({ role: 'system', content: parts.join('\n\n') });
    }
  }

  session.messages.push({ role: 'user', content: prompt });

  let finalAssistantText = '';
  for (let step = 0; step < OPENAI_BASH_MAX_STEPS; step++) {
    log(`OpenAI backend: sending chat completion (session=${newSessionId}, messages=${session.messages.length}, step=${step + 1})`);
    const assistantText = await callOpenAiChat(session.messages, env);
    session.messages.push({ role: 'assistant', content: assistantText });

    const bashCommand = extractBashCommand(assistantText);
    if (!bashCommand) {
      finalAssistantText = assistantText;
      break;
    }

    log(`OpenAI bash tool request: ${bashCommand.slice(0, 200)}`);
    const result = await execBashCommand(bashCommand);
    session.messages.push({
      role: 'user',
      content: formatBashResultForModel(result),
    });
  }

  if (!finalAssistantText) {
    finalAssistantText = 'I hit the maximum number of bash tool steps before finishing. Please narrow the request and try again.';
    session.messages.push({ role: 'assistant', content: finalAssistantText });
  }

  saveOpenAiSession(newSessionId, session);

  writeOutput({
    status: 'success',
    result: finalAssistantText,
    newSessionId,
  });

  return { newSessionId, closedDuringQuery: false };
}

async function main(): Promise<void> {
  let containerInput: ContainerInput;

  try {
    const stdinData = await readStdin();
    containerInput = JSON.parse(stdinData);
    // Delete the temp file the entrypoint wrote — it contains secrets
    try { fs.unlinkSync('/tmp/input.json'); } catch { /* may not exist */ }
    log(`Received input for group: ${containerInput.groupFolder}`);
  } catch (err) {
    writeOutput({
      status: 'error',
      result: null,
      error: `Failed to parse input: ${err instanceof Error ? err.message : String(err)}`
    });
    process.exit(1);
  }

  // Build SDK env: merge secrets into process.env for the SDK only.
  // Secrets never touch process.env itself, so Bash subprocesses can't see them.
  const sdkEnv: Record<string, string | undefined> = { ...process.env };
  for (const [key, value] of Object.entries(containerInput.secrets || {})) {
    sdkEnv[key] = value;
  }
  configureGithubGitAuthEnv(sdkEnv);
  exposeGithubAuthToToolSubprocesses(sdkEnv);
  const provider = getProvider(sdkEnv);
  log(`Provider selected: ${provider}`);

  const __dirname = path.dirname(fileURLToPath(import.meta.url));
  const mcpServerPath = path.join(__dirname, 'ipc-mcp-stdio.js');

  let sessionId = containerInput.sessionId;
  fs.mkdirSync(IPC_INPUT_DIR, { recursive: true });

  // Clean up stale _close sentinel from previous container runs
  try { fs.unlinkSync(IPC_INPUT_CLOSE_SENTINEL); } catch { /* ignore */ }

  // Build initial prompt (drain any pending IPC messages too)
  let prompt = containerInput.prompt;
  if (containerInput.isScheduledTask) {
    prompt = `[SCHEDULED TASK - The following message was sent automatically and is not coming directly from the user or group.]\n\n${prompt}`;
  }
  const pending = drainIpcInput();
  if (pending.length > 0) {
    log(`Draining ${pending.length} pending IPC messages into initial prompt`);
    prompt += '\n' + pending.join('\n');
  }

  // Query loop: run query → wait for IPC message → run new query → repeat
  let resumeAt: string | undefined;
  try {
    while (true) {
      log(`Starting query (session: ${sessionId || 'new'}, resumeAt: ${resumeAt || 'latest'})...`);

      const queryResult = provider === 'openai'
        ? await runOpenAiQuery(prompt, sessionId, containerInput, sdkEnv)
        : await runQuery(prompt, sessionId, mcpServerPath, containerInput, sdkEnv, resumeAt);
      if (queryResult.newSessionId) {
        sessionId = queryResult.newSessionId;
      }
      if (queryResult.lastAssistantUuid) {
        resumeAt = queryResult.lastAssistantUuid;
      }

      // If _close was consumed during the query, exit immediately.
      // Don't emit a session-update marker (it would reset the host's
      // idle timer and cause a 30-min delay before the next _close).
      if (queryResult.closedDuringQuery) {
        log('Close sentinel consumed during query, exiting');
        break;
      }

      // Emit session update so host can track it
      writeOutput({ status: 'success', result: null, newSessionId: sessionId });

      log('Query ended, waiting for next IPC message...');

      // Wait for the next message or _close sentinel
      const nextMessage = await waitForIpcMessage();
      if (nextMessage === null) {
        log('Close sentinel received, exiting');
        break;
      }

      log(`Got new message (${nextMessage.length} chars), starting new query`);
      prompt = nextMessage;
    }
  } catch (err) {
    const errorMessage = err instanceof Error ? err.message : String(err);
    log(`Agent error: ${errorMessage}`);
    writeOutput({
      status: 'error',
      result: null,
      newSessionId: sessionId,
      error: errorMessage
    });
    process.exit(1);
  }
}

main();
