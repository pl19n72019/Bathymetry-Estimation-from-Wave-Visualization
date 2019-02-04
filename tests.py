#!/usr/bin/env python3
from argparse import ArgumentParser
from unittest import TestSuite, TextTestRunner, defaultTestLoader
from glob import glob

import os
import logging


DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "tests")


def get_parser():
    parser = ArgumentParser(
        description="Interface to test the prediction bathymetry scripts"
    )

    parser.add_argument(
        'target',
        nargs='?',
        help="select the target test to run (by default, all tests are run)",
        default=None
    )

    parser.add_argument(
        '-v',
        '--verbose',
        help="enable verbose output",
        action="store_true"
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    verbosity = args.verbose + 1

    if args.verbose:
        logging.basicConfig(
                format="[LOG:%(levelname)s %(message)s]",
                level=logging.DEBUG
                )
    else:
        logging.disable(logging.CRITICAL)

    # Create test suite
    test_suite = TestSuite()

    if args.target is not None:
        targets = [target]
    else:
        targets = [os.path.splitext(os.path.basename(f))[0]
                for f in glob(os.path.join(DIRECTORY, "tests_*.py"))]

    for target in targets:
        target_name = "tests." + target
        test_suite.addTests(
                defaultTestLoader.loadTestsFromName(target_name)
                )

    # Run the unit tests
    print("Running tests...")
    runner = TextTestRunner(verbosity=verbosity)
    exit(not runner.run(test_suite).wasSuccessful())
