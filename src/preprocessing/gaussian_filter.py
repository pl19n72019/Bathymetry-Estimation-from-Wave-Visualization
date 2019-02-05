#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: gaussian_filter.py |
Created by Benjamin on the 2019-02-05 |
Github: https://github.com/pl19n72019/bathy-vagues

Add a description.
"""


from scipy.ndimage import gaussian_filter
from argparse import ArgumentParser

import numpy as np
import os


RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "results")


def get_parser():
    parser = ArgumentParser(
        description="Interface to gaussian filter"
    )

    parser.add_argument(
        'path_input',
        nargs='?',
        help="path of the input (npy)",
        default=None
    )

    parser.add_argument(
        '-o',
        '--output',
        help="path of the output",
        required=True
    )

    parser.add_argument(
        '-s',
        '--sigma',
        help="standard deviation",
        default=1
    )

    parser.add_argument(
        '-m',
        '--mode',
        help="mode fo the convolution (see doc)",
        default='reflect'
    )

    return parser


def apply_gaussian_filter(path_input, sigma, mode):
    # load the image input
    image_input = np.load(path_input)

    # compute the filtered image
    image_output = gaussian_filter(image_input, sigma, mode=mode)

    return image_output


if __name__ == "__main__":
    args = get_parser().parse_args()
    image_output = apply_gaussian_filter(args.path_input, args.sigma, args.mode)

    # save the image output
    np.save(path_output, image_output)
