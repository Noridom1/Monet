#!/usr/bin/env python3
"""Refresh the locally cached Colab runtime-proxy credentials for one session."""

import argparse

from colab_cli.common import state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, help="Existing Colab session name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session = state.store.get(args.session)
    if session is None:
        raise SystemExit(f"Local Colab session not found: {args.session}")

    assignment = next(
        (item for item in state.client.list_assignments() if item.endpoint == session.endpoint),
        None,
    )
    if assignment is None:
        raise SystemExit(f"Live Colab assignment not found: {session.endpoint}")

    session.token = assignment.runtime_proxy_info.token
    session.url = assignment.runtime_proxy_info.url
    state.store.add(session)
    print(f"[colab] refreshed runtime-proxy credentials for '{args.session}'")


if __name__ == "__main__":
    main()
