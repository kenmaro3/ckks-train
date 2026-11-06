#!/usr/bin/env python3
"""Fixed offline StudyTrain build requests; never executes a scientific binary."""
import argparse
import sys
import study_reference_build as reference
import study_reference_build_support as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    for recipe in s._TRAIN:
        reference.build_options(subcommands.add_parser(recipe.subcommand))
    args = parser.parse_args()
    recipe = next(r for r in s._TRAIN if r.subcommand == args.mode)
    reference.build(args, recipe)


if __name__ == "__main__":
    try:
        main()
    except (s.Invalid, OSError, ValueError, KeyError, TypeError) as e:
        print("study-train build rejected: " + str(e), file=sys.stderr)
        sys.exit(1)
