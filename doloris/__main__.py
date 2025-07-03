import os
import sys
import argparse

VERSION = "0.0.1"
DOLORIS = R"""  ____          _               _      
 |  _ \   ___  | |  ___   _ __ (_) ___ 
 | | | | / _ \ | | / _ \ | '__|| |/ __|
 | |_| || (_) || || (_) || |   | |\__ \
 |____/  \___/ |_| \___/ |_|   |_||___/
"""

def panel(cache_path: str):
    raise NotImplementedError("别急，还没实现。")

def main():
    parser = argparse.ArgumentParser(
        description="Doloris: Detection Of Learning Obstacles via Risk-aware Interaction Signals"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # version 命令
    subparsers.add_parser("version", help="Print the version of Doloris")

    # panel 命令
    panel_parser = subparsers.add_parser("panel", help="start the Doloris panel")
    panel_parser.add_argument(
        "--cache-path",
        type=str,
        default=os.path.expanduser("~/.doloris/"),
        help="Path to the cached data directory"
    )

    args = parser.parse_args()

    print(DOLORIS)

    if args.command == "version":
        print(f"Doloris version {VERSION}")
    elif args.command == "panel":
        panel(args.cache_path)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
