# Claude2GPT-Speed-Dial

A structured dual-expert reasoning tool for [Claude Code](https://claude.com/claude-code). Lets Claude consult GPT as a **critical evaluator** during builds — not a yes-man. Forces structured input, structured critique, and hard turn limits so the loop can't run away with your budget.

## Why

LLMs tend to agree with each other. This server makes that harder: GPT is prompted as a constraint-enforcer with a required critique format (ASSESSMENT / RISKS / IMPROVEMENTS / VERDICT), context gets summarized each turn to keep tokens bounded, and sessions have a hard ceiling of 10 rounds.

## Hard rules

1. Max 3 turns per session by default (configurable, hard cap 10)
2. Structured I/O only — 13-field input template
3. Force critique — no agreement bias
4. Context is summarized before each send (token control)
5. Don't restate agreement — only new insight

## Tools exposed

| Tool | Purpose |
|------|---------|
| `gpt_start_session` | Start a session tied to a build (model, max_rounds, phase) |
| `gpt_send` | Send a 13-field structured consultation, get structured critique back |
| `gpt_end_session` | Close session, save transcript to build log, return summary |
| `gpt_usage` | Lifetime token spend, per-build totals, recent sessions |
| `gpt_build_log` | View full history for a specific build across all phases |

## Setup

1. **Clone and install dependencies** (recommend a virtualenv):
   ```bash
   git clone https://github.com/dalton25harris/Claude2GPT-Speed-Dial.git
   cd Claude2GPT-Speed-Dial
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Provide an OpenAI API key.** The server looks for `OPENAI_API_KEY` in this order:
   1. Already in your shell environment
   2. `.env` file next to `server.py` (copy `.env.example` → `.env`)
   3. `.env` in the current working directory
   4. Path set via the `GPT_LOOP_ENV_FILE` environment variable

   ```bash
   cp .env.example .env
   # edit .env and paste your sk-... key
   ```

3. **Register with Claude Code.** Add to your `~/.claude/settings.json` (or equivalent `claude_desktop_config.json`):
   ```json
   {
     "mcpServers": {
       "Claude2GPT-Speed-Dial": {
         "command": "/absolute/path/to/Claude2GPT-Speed-Dial/venv/bin/python",
         "args": ["/absolute/path/to/Claude2GPT-Speed-Dial/server.py"]
       }
     }
   }
   ```
   On Windows adjust the `command` to your venv's `python.exe`.

4. **Restart Claude Code.** The five tools above should appear.

## Runtime data

The server writes persistent build logs and a usage log under `~/.claude2gpt-speed-dial/`:

- `~/.claude2gpt-speed-dial/builds/<build-name>.json` — per-build session history, injected into future sessions with the same build name so GPT has memory across phases.
- `~/.claude2gpt-speed-dial/usage_log.json` — lifetime token and cost tracking.

Nothing is written inside the repo directory by default.

## Default model and cost

Default model is `gpt-5.4`. Cost tracking uses `$2.50 / 1M` input and `$15.00 / 1M` output. If you change the default model, update the pricing constants in `gpt_end_session` and `gpt_usage` accordingly (grep for `2.50` / `15.00`).

## Project structure

```
Claude2GPT-Speed-Dial/
├── server.py          # MCP server — the whole tool lives here
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE            # MIT
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
