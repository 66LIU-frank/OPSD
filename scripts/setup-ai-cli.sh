#!/usr/bin/env bash
# One-shot configurator for Claude Code + Codex on a fresh machine.
# Usage:  bash setup-ai-cli.sh
# Idempotent: re-running overwrites config files and updates the bashrc block.

set -euo pipefail

CLAUDE_DIR="$HOME/.claude"
CODEX_DIR="$HOME/.codex"
mkdir -p "$CLAUDE_DIR" "$CODEX_DIR"

########################################
# Claude Code — ~/.claude/settings.json
########################################
cat > "$CLAUDE_DIR/settings.json" <<'JSON'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://154.17.19.146:3000",
    "ANTHROPIC_AUTH_TOKEN": "sk-jjSJKRegkpcxPGQ7KNsJWtjZMfiGVxJbhzK6DBpOabcTozYI",
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1"
  },
  "model": "opus[1m]"
}
JSON
chmod 600 "$CLAUDE_DIR/settings.json"

########################################
# Codex — ~/.codex/config.toml
########################################
cat > "$CODEX_DIR/config.toml" <<'TOML'
model_provider = "hicode"
model = "gpt-5.4"
review_model = "gpt-5.4"
disable_response_storage = true
network_access = "enabled"
model_reasoning_effort = "xhigh"
service_tier = "fast"

[model_providers.OpenAI]
name = "OpenAI"
base_url = "http://localhost:8080"
wire_api = "responses"
requires_openai_auth = true

[model_providers.packycode]
name = "packycode"
base_url = "https://www.packyapi.com/v1"
wire_api = "responses"
requires_openai_auth = true
env_key = "PACKYCODE_API_KEY"

[model_providers.sub2api]
name = "sub2api"
base_url = "https://proxyapi.cc.cd/v1"
wire_api = "responses"
requires_openai_auth = true
env_key = "SUB2API_API_KEY"

[model_providers.bailian]
name = "bailian"
base_url = "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
wire_api = "responses"
requires_openai_auth = true
env_key = "DASHSCOPE_API_KEY"

[model_providers.hicode]
name = "hicode"
base_url = "https://www.hi-code.cc"
wire_api = "responses"
requires_openai_auth = true
env_key = "HICODE_API_KEY"
TOML
chmod 600 "$CODEX_DIR/config.toml"

########################################
# Codex — ~/.codex/auth.json
########################################
cat > "$CODEX_DIR/auth.json" <<'JSON'
{
  "OPENAI_API_KEY": "sk-8a590723d58bfad3081e85c228a2d156b14300e7916c0ea34d7abcf4f863e242",
  "HICODE_API_KEY": "sk-a6ceccbc64218ac4bf31af402ead91de81231cb0b44a4640ed55da48b682b7c8"
}
JSON
chmod 600 "$CODEX_DIR/auth.json"

########################################
# ~/.bashrc — managed env block
########################################
BASHRC="$HOME/.bashrc"
BEGIN="# >>> ai-cli env >>>"
END="# <<< ai-cli env <<<"
touch "$BASHRC"
# Strip any existing managed block, then append fresh one.
if grep -qF "$BEGIN" "$BASHRC"; then
  sed -i "/$BEGIN/,/$END/d" "$BASHRC"
fi
cat >> "$BASHRC" <<'BASH'
# >>> ai-cli env >>>
export PACKYCODE_API_KEY=sk-1BTosnILZbUvxQb8c2UZK4gmY6Fu3HsjespnNMcFe6oaJKUX
export SUB2API_API_KEY=sk-d4aad584a2b1abd30667326b32088f93ee4e28c90e808db468a095d537a052bf
export DASHSCOPE_API_KEY=sk-5i4GWL0YQ42YUYw5tlHy7bL46UAh5ut7TXbxZPrVTAiGO0wy
export BAILIAN_API_KEY="$DASHSCOPE_API_KEY"
export HICODE_API_KEY=sk-a6ceccbc64218ac4bf31af402ead91de81231cb0b44a4640ed55da48b682b7c8
export ANTHROPIC_BASE_URL=http://154.17.19.146:3000
export ANTHROPIC_AUTH_TOKEN=sk-jjSJKRegkpcxPGQ7KNsJWtjZMfiGVxJbhzK6DBpOabcTozYI
# <<< ai-cli env <<<
BASH

echo "[ok] wrote $CLAUDE_DIR/settings.json"
echo "[ok] wrote $CODEX_DIR/config.toml"
echo "[ok] wrote $CODEX_DIR/auth.json"
echo "[ok] updated $BASHRC (reload with:  source ~/.bashrc)"
