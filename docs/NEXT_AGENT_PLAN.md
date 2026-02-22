# Next Agent Plan (Codex OAuth + Repo Access)

## Current State

- Branch: `codex-oauth-nanoclaw`
- Repo remote `origin`: `git@github.com:akosel/nanoclaw.git`
- NanoClaw daemon runs via `systemd --user` and Docker access works after reboot
- Main/self chat is registered (`groupCount: 1`)

## What Was Implemented

### Codex/OpenAI support in NanoClaw

- Added `openai` provider path to NanoClaw agent runner.
- Added host-side support for:
  - `OPENAI_API_KEY`
  - `CODEX_AUTH_JSON_PATH` (reads `~/.codex/auth.json`)
  - Codex OAuth token extraction (`tokens.access_token`)
  - Codex account id extraction (`tokens.account_id`)

### Codex OAuth backend (working)

- Implemented Codex OAuth request path using ChatGPT Codex backend:
  - `https://chatgpt.com/backend-api/codex/responses`
- Uses SSE streaming parsing (`response.output_text.delta`)
- Supports multi-turn sessions
- Fixed assistant-history encoding for Codex endpoint:
  - assistant history uses `output_text`
  - user history uses `input_text`

### Dev/runtime fixes

- `npm run dev` changed to use `node --import tsx src/index.ts` (avoids `tsx` IPC socket issue in constrained environments)
- Setup verification/tests updated to recognize OpenAI/Codex auth config

## Important Behavior Notes

- `NANOCLAW_AGENT_PROVIDER=openai` now supports:
  - API key path (`OPENAI_API_KEY`) -> OpenAI API (`/v1/chat/completions`)
  - Codex OAuth path (`CODEX_AUTH_JSON_PATH`) -> ChatGPT Codex backend (`/backend-api/codex/responses`)
- API key takes precedence if both are present.

## Logging Notes (Daemon vs Dev)

- Daemon is quieter because:
  - default `LOG_LEVEL=info`
  - systemd unit writes app logs to files instead of journal
- Use:
  - `tail -f logs/nanoclaw.log logs/nanoclaw.error.log`
  - `ls -lt groups/main/logs`
  - `tail -f groups/main/logs/container-*.log`

## Operational Commands

### Daemon

```bash
systemctl --user status nanoclaw --no-pager
systemctl --user restart nanoclaw
journalctl --user -u nanoclaw -f
```

### Verify Docker from user systemd session

```bash
systemd-run --user --pipe --wait docker info
```

## Next Task: Give NanoClaw Agent Access to Git/GitHub Repos

Goal: allow the NanoClaw container agent to work on selected repos using `git` and `gh`, without over-mounting sensitive host paths.

### Recommended Approach

1. Configure mount allowlist (`~/.config/nanoclaw/mount-allowlist.json`)
2. Add specific repo parent dirs as allowed roots (for example `~/src`)
3. Mount selected repos into NanoClaw group container via group config additional mounts
4. Mount only minimal GitHub auth material needed for `gh` (prefer token/env over whole `~/.config/gh`)
5. Validate in-container `git` + `gh auth status`

### Relevant Code

- Mount allowlist validation: `src/mount-security.ts`
- Mount setup step: `setup/mounts.ts`
- Additional mounts are injected in container runner: `src/container-runner.ts`
- Container image already includes `git`; `gh` is not currently installed in `container/Dockerfile`

### Likely Subtasks

1. Add `gh` CLI to `container/Dockerfile`
2. Rebuild `nanoclaw-agent:latest`
3. Decide auth strategy for `gh` inside container:
   - env token (preferred)
   - read-only mount of a minimal `gh` config dir
4. Define safe additional mounts for repos (read-write only where needed)
5. Add docs/example for per-group repo mounts

## Suggested First Checks for Next Session

```bash
git branch --show-current
git status
systemctl --user status nanoclaw --no-pager
systemd-run --user --pipe --wait docker info
tail -n 100 logs/nanoclaw.log
```

