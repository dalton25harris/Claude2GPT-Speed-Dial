"""
GPT Communication Loop — MCP Server
Structured dual-expert reasoning tool for Claude Code.

Hard rules:
  1. Max 3 turns per session (configurable, never unlimited)
  2. Always structured I/O
  3. Force critique — no agreement bias
  4. Summarize context before sending (token control)
  5. Don't restate agreement — only add new insight
"""

import json
import os
import uuid
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# ── Load API key ────────────────────────────────────────────
# Priority:
#   1. OPENAI_API_KEY already in environment
#   2. .env file next to this server.py
#   3. .env file in current working directory
#   4. Path set via GPT_LOOP_ENV_FILE env var

_candidate_env_files = []
if os.environ.get("GPT_LOOP_ENV_FILE"):
    _candidate_env_files.append(Path(os.environ["GPT_LOOP_ENV_FILE"]))
_candidate_env_files.append(Path(__file__).parent / ".env")
_candidate_env_files.append(Path.cwd() / ".env")

for _p in _candidate_env_files:
    if _p.exists():
        load_dotenv(_p)
        break

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not found. Set it in your environment, or place a .env "
                "file next to server.py with OPENAI_API_KEY=sk-... "
                "(or set GPT_LOOP_ENV_FILE to a custom .env path)."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ── Default system prompt ───────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """You are an expert systems architect participating in a structured multi-model reasoning loop alongside Claude (another AI model).

YOUR ROLE: Critical evaluator and constraint enforcer.
- Identify failure modes and hidden risks
- Challenge weak assumptions
- Reduce unnecessary complexity
- Prioritize correctness over elegance
- Prefer simple, deterministic solutions over clever ones

BEHAVIOR RULES:
- Do NOT agree with the other model unless you have genuinely nothing to add
- If you agree with a point, do NOT restate it. Only respond with NEW information, NEW risks, or NEW alternatives. Restating agreement wastes tokens.
- Challenge assumptions — say what will break
- Be direct, precise, and technical
- No fluff, filler, or motivational language

SPECIFICITY RULES (strict):
- Do NOT give general advice. Every improvement must reference:
  - A specific part of the system (file, function, line, config key)
  - A concrete change (exact code, rule, or config to add/modify/remove)
- If your feedback is generic, refine it until it is actionable.
- Every improvement must be implementable without further clarification.
- When code is provided, reference exact lines or patterns. Suggest concrete replacements when possible.
- Avoid generic phrases like "add tests", "consider improvements", "best practice". Replace them with exact implementations.

ARTIFACT GROUNDING (strict):
- If code with file paths, functions, or line numbers is provided, reference those directly.
- If NO specific file/function is provided in context (e.g., early design or abstract discussion), you MUST:
  - Create a placeholder target (e.g., "server.py::get_db()" or "config.yaml::pool_size")
  - Define its expected role before proposing changes
  - This prevents vague feedback like "update the validation logic" with no anchor point.

CRITIQUE RULES (strict):
- You MUST identify at least one potential flaw or risk. Minimum 1 critical risk required unless explicitly impossible.
- If the design appears correct, challenge assumptions or identify edge cases.
- Prioritize high-impact issues over completeness.
- Focus on failure modes, not improvements. Avoid listing minor suggestions if major risks exist.

CONCISENESS RULES:
- Be concise by default.
- Expand ONLY when: identifying a failure mode, or proposing a fix.
- Do not pad responses with context summaries or restatements.

OUTPUT FORMAT (follow this structure):

ASSESSMENT
- What is correct (brief — don't echo back what was said)
- What is flawed or missing

RISKS (minimum 1 required)
- What will break or degrade — reference specific components
- Hidden failure modes — name the trigger condition
- For each risk: state the component, the failure mode, and the consequence

IMPROVEMENTS
- Each must reference a specific part of the system
- Each must include a concrete change (code snippet, config change, or exact rule)
- If code was provided, point to exact lines/functions

VERDICT
- proceed / needs revision / rethink approach
- One-line rationale

GOAL: Make the system more reliable, more deterministic, and closer to production quality. Optimize for truth and robustness, not agreement."""

# ── Structured input template ──────────────────────────────

INPUT_TEMPLATE = """BUILDING: {building}
APPROACH: {approach}
CONFIDENCE GAPS: {confidence_gaps}
PROJECT SCOPE: {project_scope}
ENDPOINTS / INTEGRATIONS: {endpoints}
CONSTRAINTS: {constraints}
DECISIONS MADE: {decisions_made}
ENVIRONMENT: {environment}
REASON FOR CONSULT: {reason}
STUCK ON: {stuck_on}
CODE FOR REVIEW:
{code}
TESTING IDEAS: {testing_ideas}
FEEDBACK SCOPE: {feedback_scope}"""

# ── Build log persistence ──────────────────────────────────

_BUILDS_DIR = Path.home() / ".claude2gpt-speed-dial" / "builds"


def _load_build_log(build_name: str) -> dict:
    path = _BUILDS_DIR / f"{build_name}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "build_name": build_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_tokens": 0,
        "sessions": [],
    }


def _save_build_log(build_name: str, log: dict) -> None:
    _BUILDS_DIR.mkdir(parents=True, exist_ok=True)
    path = _BUILDS_DIR / f"{build_name}.json"
    log["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(log, indent=2))


def _get_build_history_summary(build_name: str) -> str:
    """Produce a condensed summary of prior GPT sessions for this build."""
    log = _load_build_log(build_name)
    sessions = log.get("sessions", [])
    if not sessions:
        return ""

    parts = [f"BUILD HISTORY for '{build_name}' ({len(sessions)} prior session(s)):"]
    for i, s in enumerate(sessions, 1):
        parts.append(f"\n--- Session {i} (phase: {s.get('phase', '?')}, {s['rounds']} rounds, {s['tokens']} tokens) ---")
        for exchange in s.get("exchanges", []):
            claude_msg = exchange.get("claude", "")
            gpt_msg = exchange.get("gpt", "")
            # Truncate for token control
            if len(claude_msg) > 400:
                claude_msg = claude_msg[:400] + "... [truncated]"
            if len(gpt_msg) > 400:
                gpt_msg = gpt_msg[:400] + "... [truncated]"
            parts.append(f"CLAUDE: {claude_msg}")
            parts.append(f"GPT: {gpt_msg}")

    return "\n".join(parts)


# ── Persistent token tracker ────────────────────────────────

_USAGE_LOG = Path.home() / ".claude2gpt-speed-dial" / "usage_log.json"


def _load_usage_log() -> dict:
    if _USAGE_LOG.exists():
        try:
            return json.loads(_USAGE_LOG.read_text())
        except (json.JSONDecodeError, OSError):
            return {"sessions": [], "lifetime_tokens": 0}
    return {"sessions": [], "lifetime_tokens": 0}


def _save_session_usage(session_id: str, session: dict) -> None:
    log = _load_usage_log()
    entry = {
        "session_id": session_id,
        "build_name": session.get("build_name", ""),
        "phase": session.get("phase", ""),
        "model": session["model"],
        "rounds": session["current_round"],
        "tokens": session["total_tokens"],
        "prompt_tokens": session.get("prompt_tokens", 0),
        "completion_tokens": session.get("completion_tokens", 0),
        "started_at": datetime.fromtimestamp(
            session["started_at"], tz=timezone.utc
        ).isoformat(),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    }
    log["sessions"].append(entry)
    log["lifetime_tokens"] += session["total_tokens"]
    _USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    _USAGE_LOG.write_text(json.dumps(log, indent=2))


# ── Session store ───────────────────────────────────────────

_sessions: dict[str, dict] = {}

HARD_MAX_ROUNDS = 10  # absolute ceiling, even if user asks for more


def _summarize_history(messages: list[dict]) -> str:
    """Compress conversation history into a concise summary for token control."""
    if len(messages) <= 2:
        return ""

    turns = [m for m in messages if m["role"] != "system"]
    if not turns:
        return ""

    summary_parts = []
    for i, turn in enumerate(turns):
        role_label = "CLAUDE" if turn["role"] == "user" else "GPT"
        content = turn["content"]
        if len(content) > 500:
            content = content[:500] + "... [truncated]"
        summary_parts.append(f"Turn {i+1} ({role_label}): {content}")

    return "PRIOR CONTEXT (summarized):\n" + "\n---\n".join(summary_parts)


# ── MCP Server ──────────────────────────────────────────────

mcp = FastMCP("claude2gpt-speed-dial", instructions="Structured Claude-GPT reasoning loop")


@mcp.tool()
def gpt_start_session(
    build_name: str,
    phase: str = "",
    system_prompt: str = "",
    model: str = "gpt-5.4",
    max_rounds: int = 3,
) -> str:
    """Start a structured reasoning session with GPT, tied to a build.

    Args:
        build_name: Name of the build (e.g. 'gpu-deal-finder', 'auth-system'). Sessions with the same build_name share a persistent log so GPT has history across phases.
        phase: Current build phase (e.g. 'phase-1-scraper', 'phase-3-api'). Labels this session in the build log.
        system_prompt: Custom system prompt for GPT. Leave empty for the default critic/architect prompt.
        model: OpenAI model to use (default: gpt-5.4).
        max_rounds: Max back-and-forth rounds (default: 3, hard cap: 10).

    Returns:
        Session ID, build history status, and confirmation.
    """
    _get_client()

    session_id = uuid.uuid4().hex[:12]
    effective_prompt = system_prompt.strip() if system_prompt.strip() else DEFAULT_SYSTEM_PROMPT

    if max_rounds < 1:
        max_rounds = 1
    if max_rounds > HARD_MAX_ROUNDS:
        max_rounds = HARD_MAX_ROUNDS

    # Check for existing build history
    build_log = _load_build_log(build_name)
    prior_sessions = len(build_log.get("sessions", []))
    build_history = _get_build_history_summary(build_name)

    _sessions[session_id] = {
        "build_name": build_name,
        "phase": phase,
        "model": model,
        "max_rounds": max_rounds,
        "current_round": 0,
        "messages": [{"role": "system", "content": effective_prompt}],
        "exchanges": [],  # stored for build log
        "started_at": time.time(),
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "build_history": build_history,
    }

    history_status = (
        f"  build_history: {prior_sessions} prior session(s) loaded — GPT will see them"
        if prior_sessions > 0
        else "  build_history: new build — no prior sessions"
    )

    return (
        f"Session started.\n"
        f"  session_id: {session_id}\n"
        f"  build: {build_name}\n"
        f"  phase: {phase or '(unspecified)'}\n"
        f"  model: {model}\n"
        f"  max_rounds: {max_rounds}\n"
        f"  system_prompt: {'custom' if system_prompt.strip() else 'default (critic/architect)'}\n"
        f"{history_status}\n"
        f"\nStructured input template for gpt_send:\n"
        f"  BUILDING | APPROACH | CONFIDENCE GAPS | PROJECT SCOPE | ENDPOINTS/INTEGRATIONS\n"
        f"  CONSTRAINTS | DECISIONS MADE | ENVIRONMENT | REASON FOR CONSULT\n"
        f"  STUCK ON | CODE FOR REVIEW | TESTING IDEAS | FEEDBACK SCOPE\n"
        f"\nUse gpt_send with this session_id to begin."
    )


@mcp.tool()
def gpt_send(
    session_id: str,
    building: str,
    approach: str,
    confidence_gaps: str,
    project_scope: str,
    endpoints: str = "N/A",
    constraints: str = "None specified",
    decisions_made: str = "None yet",
    environment: str = "Not specified",
    reason: str = "",
    stuck_on: str = "Nothing specific — general review",
    code: str = "N/A",
    testing_ideas: str = "N/A",
    feedback_scope: str = "All areas — no restrictions",
) -> str:
    """Send a structured consultation to GPT using the full input template.

    All 13 fields map to the structured input format. GPT receives build
    history (if any prior sessions exist for this build) plus your input,
    and responds with ASSESSMENT / RISKS / IMPROVEMENTS / VERDICT.

    Args:
        session_id: The session ID from gpt_start_session.
        building: What you are building.
        approach: How you plan to build it.
        confidence_gaps: Where you are NOT confident in the build.
        project_scope: Full scope of the project / where this fits.
        endpoints: Endpoints or integrations you're attaching to, with details.
        constraints: Non-negotiables — things GPT must not suggest changing.
        decisions_made: Trade-offs already accepted and why.
        environment: Hardware, OS, runtime details.
        reason: Why you're calling GPT right now.
        stuck_on: Specific blockers or areas of difficulty.
        code: Code block for review.
        testing_ideas: Ideas for validating / testing the feature.
        feedback_scope: What you want feedback on / what GPT should skip.

    Returns:
        GPT's structured response with round info and token usage.
    """
    if session_id not in _sessions:
        return f"Error: session '{session_id}' not found. Start one with gpt_start_session."

    session = _sessions[session_id]

    if session["current_round"] >= session["max_rounds"]:
        return (
            f"Session '{session_id}' has reached its max of {session['max_rounds']} rounds.\n"
            f"Use gpt_end_session to get the transcript, or start a new session."
        )

    # Build the structured message
    message = INPUT_TEMPLATE.format(
        building=building,
        approach=approach,
        confidence_gaps=confidence_gaps,
        project_scope=project_scope,
        endpoints=endpoints,
        constraints=constraints,
        decisions_made=decisions_made,
        environment=environment,
        reason=reason,
        stuck_on=stuck_on,
        code=code,
        testing_ideas=testing_ideas,
        feedback_scope=feedback_scope,
    )

    # Compose what we send to OpenAI
    system_msg = session["messages"][0]
    send_messages = [system_msg]

    # Inject build history on first round
    if session["current_round"] == 0 and session.get("build_history"):
        send_messages.append({
            "role": "user",
            "content": session["build_history"],
        })
        send_messages.append({
            "role": "assistant",
            "content": "I have the build history. Ready for the current consultation.",
        })

    # Token control: summarize prior rounds within this session
    if session["current_round"] > 0:
        summary = _summarize_history(session["messages"])
        if summary:
            send_messages.append({"role": "user", "content": summary})
            send_messages.append({
                "role": "assistant",
                "content": "Understood. I have the prior context. Proceeding with the new input.",
            })

    send_messages.append({"role": "user", "content": message})

    # Call OpenAI
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=session["model"],
            messages=send_messages,
            temperature=0.3,
            max_completion_tokens=15000,
        )
    except Exception as e:
        return f"OpenAI API error: {e}"

    gpt_reply = response.choices[0].message.content
    usage = response.usage

    # Update session state
    session["messages"].append({"role": "user", "content": message})
    session["messages"].append({"role": "assistant", "content": gpt_reply})
    session["exchanges"].append({"claude": message, "gpt": gpt_reply})
    session["current_round"] += 1
    if usage:
        session["total_tokens"] += usage.total_tokens
        session["prompt_tokens"] += usage.prompt_tokens
        session["completion_tokens"] += usage.completion_tokens

    round_info = f"[Round {session['current_round']}/{session['max_rounds']}]"
    if usage:
        tokens_info = (
            f"[This call: {usage.prompt_tokens:,} in / {usage.completion_tokens:,} out | "
            f"Session: {session['prompt_tokens']:,} in / {session['completion_tokens']:,} out]"
        )
    else:
        tokens_info = "[Tokens: unknown]"

    remaining = session["max_rounds"] - session["current_round"]
    status = f"[{remaining} round(s) remaining]" if remaining > 0 else "[Final round — use gpt_end_session to close]"

    return f"{round_info} {tokens_info} {status}\n\n{gpt_reply}"


@mcp.tool()
def gpt_end_session(session_id: str) -> str:
    """End a GPT session, save to build log, and return transcript.

    Args:
        session_id: The session ID to close.

    Returns:
        Full conversation transcript and usage summary.
    """
    if session_id not in _sessions:
        return f"Error: session '{session_id}' not found."

    session = _sessions.pop(session_id)

    # Save to persistent build log
    build_name = session.get("build_name", "")
    if build_name:
        build_log = _load_build_log(build_name)
        build_log["sessions"].append({
            "session_id": session_id,
            "phase": session.get("phase", ""),
            "model": session["model"],
            "rounds": session["current_round"],
            "tokens": session["total_tokens"],
            "prompt_tokens": session["prompt_tokens"],
            "completion_tokens": session["completion_tokens"],
            "started_at": datetime.fromtimestamp(
                session["started_at"], tz=timezone.utc
            ).isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "exchanges": session.get("exchanges", []),
        })
        build_log["total_tokens"] += session["total_tokens"]
        _save_build_log(build_name, build_log)

    # Save to global usage log
    _save_session_usage(session_id, session)
    elapsed = int(time.time() - session["started_at"])

    # Cost calculation (gpt-5.4 pricing: $2.50/1M input, $15.00/1M output)
    input_cost = (session["prompt_tokens"] / 1_000_000) * 2.50
    output_cost = (session["completion_tokens"] / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    # Build transcript
    lines = [
        f"SESSION TRANSCRIPT — {session_id}",
        f"Build: {build_name or '(none)'}",
        f"Phase: {session.get('phase', '(none)')}",
        f"Model: {session['model']}",
        f"Rounds: {session['current_round']}/{session['max_rounds']}",
        f"Input tokens: {session['prompt_tokens']:,}",
        f"Output tokens: {session['completion_tokens']:,}",
        f"Cost: ${input_cost:.4f} in + ${output_cost:.4f} out = ${total_cost:.4f} total",
        f"Duration: {elapsed}s",
        "=" * 60,
    ]

    for msg in session["messages"]:
        if msg["role"] == "system":
            lines.append(f"\n[SYSTEM PROMPT]\n{msg['content'][:200]}...")
        elif msg["role"] == "user":
            lines.append(f"\n--- CLAUDE ---\n{msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"\n--- GPT ---\n{msg['content']}")

    lines.append("\n" + "=" * 60)
    lines.append("Session closed.")

    return "\n".join(lines)


@mcp.tool()
def gpt_usage() -> str:
    """Show token usage history across all GPT loop sessions.

    Returns:
        Lifetime token count, per-session breakdown, active sessions, and per-build totals.
    """
    log = _load_usage_log()
    lines = ["GPT LOOP — TOKEN USAGE", "=" * 50]

    # Lifetime totals
    lifetime_in = sum(e.get("prompt_tokens", 0) for e in log["sessions"])
    lifetime_out = sum(e.get("completion_tokens", 0) for e in log["sessions"])
    lifetime_cost_in = (lifetime_in / 1_000_000) * 2.50
    lifetime_cost_out = (lifetime_out / 1_000_000) * 15.00
    lifetime_cost = lifetime_cost_in + lifetime_cost_out

    lines.append(f"Lifetime: {lifetime_in:,} in / {lifetime_out:,} out")
    lines.append(f"Lifetime cost: ${lifetime_cost:.4f} (${lifetime_cost_in:.4f} in + ${lifetime_cost_out:.4f} out)")
    lines.append(f"Total sessions: {len(log['sessions'])}")
    lines.append("")

    # Active sessions
    if _sessions:
        lines.append(f"Active sessions: {len(_sessions)}")
        for sid, s in _sessions.items():
            lines.append(
                f"  {sid} [{s.get('build_name', '?')}] round {s['current_round']}/{s['max_rounds']}, "
                f"{s['prompt_tokens']:,} in / {s['completion_tokens']:,} out"
            )
        lines.append("")

    # Per-build totals with cost
    build_data: dict[str, dict] = {}
    for entry in log["sessions"]:
        bname = entry.get("build_name", "(unnamed)")
        if bname not in build_data:
            build_data[bname] = {"in": 0, "out": 0}
        build_data[bname]["in"] += entry.get("prompt_tokens", 0)
        build_data[bname]["out"] += entry.get("completion_tokens", 0)
    if build_data:
        lines.append("Per-build totals:")
        for bname, d in sorted(build_data.items(), key=lambda x: -(x[1]["in"] + x[1]["out"])):
            cost = (d["in"] / 1_000_000) * 2.50 + (d["out"] / 1_000_000) * 15.00
            lines.append(f"  {bname}: {d['in']:,} in / {d['out']:,} out (${cost:.4f})")
        lines.append("")

    # Last 10 completed sessions
    recent = log["sessions"][-10:]
    if recent:
        lines.append("Recent sessions:")
        for entry in reversed(recent):
            lines.append(
                f"  {entry['session_id']} | {entry.get('build_name', '?')} | "
                f"{entry.get('phase', '?')} | {entry['model']} | "
                f"{entry['rounds']} rounds | {entry['tokens']:,} tokens | "
                f"{entry['started_at']}"
            )
    else:
        lines.append("No completed sessions yet.")

    return "\n".join(lines)


@mcp.tool()
def gpt_build_log(build_name: str) -> str:
    """View the full log for a specific build — all sessions, phases, and exchanges.

    Args:
        build_name: The build name to look up.

    Returns:
        Full build history with all session summaries.
    """
    log = _load_build_log(build_name)
    sessions = log.get("sessions", [])

    if not sessions:
        return f"No sessions found for build '{build_name}'."

    lines = [
        f"BUILD LOG — {build_name}",
        f"Created: {log.get('created_at', '?')}",
        f"Last updated: {log.get('updated_at', '?')}",
        f"Total sessions: {len(sessions)}",
        f"Total tokens: {log.get('total_tokens', 0):,}",
        "=" * 60,
    ]

    for i, s in enumerate(sessions, 1):
        lines.append(
            f"\n--- Session {i}: {s.get('phase', '(no phase)')} ---\n"
            f"  Model: {s['model']} | Rounds: {s['rounds']} | "
            f"Tokens: {s['tokens']:,} | {s.get('started_at', '?')}"
        )
        for j, ex in enumerate(s.get("exchanges", []), 1):
            claude_msg = ex.get("claude", "")[:300]
            gpt_msg = ex.get("gpt", "")[:300]
            lines.append(f"\n  Exchange {j}:")
            lines.append(f"    CLAUDE: {claude_msg}{'...' if len(ex.get('claude', '')) > 300 else ''}")
            lines.append(f"    GPT: {gpt_msg}{'...' if len(ex.get('gpt', '')) > 300 else ''}")

    return "\n".join(lines)


# ── Entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
