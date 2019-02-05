#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: tests.py |
Created by Benjamin on the 2019-02-05 |
Github: https://github.com/pl19n72019/bathy-vagues

Add a description.
"""


import os
import context
import logging

from argparse import ArgumentParser
from unittest import TestSuite, TextTestRunner, defaultTestLoader
from glob import glob


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
                for f in glob("tests/tests_*.py")]

    for target in targets:
        target_name = "tests." + target
        test_suite.addTests(
                defaultTestLoader.loadTestsFromName(target_name))

    # Run the unit tests
    print("Running tests...")
    runner = TextTestRunner(verbosity=verbosity)
    exit(not runner.run(test_suite).wasSuccessful())
