import argparse
import shutil
from pathlib import Path

from devteam.runner import (
    DevTeamRunner,
)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MARL-coordinated local "
            "LLM software development team"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run = sub.add_parser(
        "run"
    )

    run.add_argument(
        "request",
    )

    run.add_argument(
        "--workspace",
        default=(
            "workspaces/"
            "devteam_project"
        ),
    )

    run.add_argument(
        "--resources",
        type=int,
        default=2,
    )

    run.add_argument(
        "--max-steps",
        type=int,
        default=40,
    )

    run.add_argument(
        "--clean",
        action="store_true",
    )

    args = parser.parse_args()

    workspace = Path(
        args.workspace
    )

    if (
        args.clean
        and workspace.exists()
    ):
        shutil.rmtree(
            workspace
        )

    runner = DevTeamRunner(
        request=args.request,
        workspace=workspace,
        resources=args.resources,
        max_steps=args.max_steps,
    )

    success = (
        runner.run()
    )

    raise SystemExit(
        0 if success else 1
    )


if __name__ == "__main__":
    main()
