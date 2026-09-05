from __future__ import annotations

import argparse
import asyncio
import json

from .app import build_runtime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo-user")
    parser.add_argument("--session", default="window-1")
    parser.add_argument("--db", default="agent.db")
    parser.add_argument("--trace", default="trace.jsonl")
    return parser.parse_args()


def _friendly_runtime_error(exc: Exception) -> str:
    """Return a bounded reviewer-facing error without leaking credentials/internal blobs."""
    name = type(exc).__name__
    text = str(exc).replace("\n", " ").strip()
    if len(text) > 700:
        text = text[:697] + "..."
    return f"{name}: {text}" if text else name


def _read_multiline() -> str:
    """Read an explicit multi-line user turn until /end.

    Plain ``input()`` is line-oriented and cannot reliably distinguish an intentional
    multi-paragraph paste from several separate turns across every terminal emulator.
    Echo therefore exposes a deterministic no-dependency multi-line mode. This keeps the
    user-turn boundary explicit and reviewer-reproducible instead of guessing from timing.
    """
    print("... multiline mode; finish with /end")
    lines: list[str] = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            if lines:
                return "\n".join(lines).strip()
            raise
        if line.strip() == "/end":
            return "\n".join(lines).strip()
        lines.append(line)


async def repl() -> int:
    args = parse_args()
    try:
        runtime = build_runtime(args.db, args.trace)
    except RuntimeError as exc:
        print(f"Configuration error: {exc}")
        print("See README.md -> Quick start. No API key should be committed to the repository.")
        return 2

    print("Echo — search by structure, not by words")
    print(f"user={args.user} session={args.session}")
    print("Commands: /multi, /context, /trace, /help, exit")

    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            print("\nInput stream closed. Session state is durable; reopen the same user/session to resume.")
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted. Session state is durable; reopen the same user/session to resume.")
            return 130

        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return 0
        if text == "/help":
            print(
                "/multi = enter one multi-line user turn; finish with /end; "
                "/context = inspect active context budget; /trace = recent tool/guard events; "
                "exit = close the CLI"
            )
            continue
        if text == "/multi":
            try:
                text = _read_multiline()
            except EOFError:
                print("\nInput stream closed. Session state is durable; reopen the same user/session to resume.")
                return 0
            except KeyboardInterrupt:
                print("\nMultiline entry cancelled; no user turn was submitted.")
                continue
            if not text:
                print("echo> Empty multiline message ignored.")
                continue
        elif text == "/context":
            info = runtime.inspect_session(user_id=args.user, session_id=args.session)
            print(json.dumps(info, ensure_ascii=False, indent=2))
            continue
        elif text == "/trace":
            print(json.dumps(runtime.tracer.tail(), ensure_ascii=False, indent=2))
            continue

        print("echo> Thinking... (Ctrl-C interrupts this process; completed session state is durable)")
        try:
            answer = await runtime.run(
                user_id=args.user,
                session_id=args.session,
                user_input=text,
            )
        except KeyboardInterrupt:
            print("\necho> Current turn interrupted. Reopen the same user/session to inspect durable state.")
            continue
        except Exception as exc:
            # One provider/network failure must not kill an interactive session. The runtime
            # persists the user turn before contacting the provider, so the same session remains
            # inspectable/resumable even after a failed request.
            print(f"echo> Request failed: {_friendly_runtime_error(exc)}")
            print("echo> The session is still alive. You can retry, inspect /trace, or exit.")
            continue

        print(f"echo> {answer}")


def main():
    raise SystemExit(asyncio.run(repl()))


if __name__ == "__main__":
    main()
