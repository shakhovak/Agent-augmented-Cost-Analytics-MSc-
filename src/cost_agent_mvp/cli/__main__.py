from __future__ import annotations

import sys

from cost_agent_mvp.cli.doctor import main as doctor_main


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python -m cost_agent_mvp.cli <command>\n")
        print("Commands:")
        print("  doctor   Validate environment and write a run_record.json\n")
        return 0

    cmd = argv[0]
    if cmd == "doctor":
        return doctor_main(argv[1:])

    print(f"Unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
