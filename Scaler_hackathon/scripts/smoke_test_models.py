#!/usr/bin/env python3
"""Check that every role's model works end to end before you run an experiment.

Examples:
    python scripts/smoke_test_models.py --dry-run                 # show config only, no network
    python scripts/smoke_test_models.py --list-models nemotron    # which Nemotron IDs does my key see?
    python scripts/smoke_test_models.py                           # one real call per role
    python scripts/smoke_test_models.py --config config/models.example.yaml --roles planning

Exit code: 0 all passed, 1 a role failed, 2 configuration error.
Keys come from environment variables (or a .env file) and are never printed.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from server.model_config import ROLES, ConfigError, describe_team, load_team_config  # noqa: E402
from server.model_smoke import run_smoke  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON/YAML model config (default: SCALER_MODELS_FILE or env vars)")
    ap.add_argument("--roles", nargs="+", choices=ROLES, help="test only these roles")
    ap.add_argument("--list-models", nargs="?", const="", metavar="SUBSTR",
                    help="list model IDs the key can see (optionally only those containing SUBSTR) and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved config and exit; no network")
    ap.add_argument("--no-probe", action="store_true", help="skip diagnostic retries when a role fails")
    args = ap.parse_args()

    try:
        team = load_team_config(path=args.config)
    except ConfigError as e:
        print(f"[CONFIG ERROR] {e}")
        return 2

    if args.dry_run:
        print("\n".join(describe_team(team)))
        return 0

    return run_smoke(
        team, roles=args.roles, list_filter=args.list_models,
        list_only=args.list_models is not None, probe=not args.no_probe,
    )


if __name__ == "__main__":
    sys.exit(main())
