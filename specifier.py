"""This script creates the necessary jsons for the data directory.
"""
import argparse
import itertools
import functools
import logging
import json
import os

import config as c

import coloredlogs
coloredlogs.install(level=logging.DEBUG)

from os.path import exists as E
from os.path import join as J

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='The specifier script which creates a set of experiment jsons.'
        ' The hyperparameters specified are in the config.py file.')

    parser.add_argument('specification_dir', type=str, 
        help='The path of the directory to which the experiment jsons will be written.')
    
    parser.add_argument('--start', type=int, default=0, 
        help='The start index of the number of specifications to create')
    
    parser.add_argument('--end', type=int, default=-1,
        help='The end index of the number of specifications to create.')

    parser.add_argument('--info', action='store_true', 
        help='Generates a printout the number of the specifications to generate.')

    return parser.parse_args()


def print_info():
    print("Specifier Info:")

    num_specifications = functools.reduce(lambda x,y: x*y, 
        [len(x) for x in c.HYPERPARAMETERS.values()])
    print("Total number of specifications {}".format(num_specifications))


def main(specification_dir, start, end, info):
    # First construct the cartesian product 
    if info:
        print_info()
        return 

    vals = c.HYPERPARAMETERS.values()
    product = itertools.product(*vals)
    product_to_dict = [{
        k: v[i] for i, k in enumerate(c.HYPERPARAMETERS)
    } for v in product]

    # Create the specification directory
    if not E(specification_dir):
        os.makedirs(specification_dir)
    else:
        if os.listdir(specification_dir):
            logger.warning(
                "Specification directory is not empty, "
                "are you sure you want to create it.")
    
    logger.info("Making specifications.")

    for i, spec in enumerate(product_to_dict):
        # Set the name
        file_name = "{}_{}".format(spec['algorithm'], i)
        spec["name"] = file_name
        alg = spec["algorithm"]
        for k in c.ALGORITHM_SPECIFIC_PARAMETERS[alg]:
            spec[k] = c.ALGORITHM_SPECIFIC_PARAMETERS[alg][k]

        spec["embedding_dim"] = spec["width"]

        for key, value in c.DEFAULT_VALUES_SPEC.items():
            if key not in spec:
                spec[key] = value
        

        with open(J(specification_dir, file_name + ".json"), "w") as f:
            f.write(json.dumps(spec))

    logger.info("Specifications complete.")
        





if __name__ == '__main__':
    args = parse_args()

    main(args.specification_dir, args.start, args.end, args.info)