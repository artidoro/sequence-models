"""Orchestrates experiments
"""
import json
import argparse
import logging
import coloredlogs
import multiprocess
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'The experiment orchestrator. This takes as argument a'
            ' directory containing experiment specifications and '
            'the desired parallelism with which to run the experiment.'))

    parser.add_argument('specification_dir', type=str,
        help='A directory containing experiment specifications '
             'generated by specifier.py')

    parser.add_argument('out_dir', type=str, help='The output directory')

    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')    

    parser.add_argument('--exps_per_gpu', type=int, default=1, help='Number of examples per GPU')

    return parser.parse_args()




def main(specification_dir, out_dir, num_gpus, exps_per_gpu):
    """Run the experiment orchestrator
    """

    # 1. Load the specifications
    
    # 2. Create the output directory

    # 3. Create the workers with specific environment variables

    # 4. Distribute the workload

    # 5. Launch the workers.



if __name__ == '__main__':
    args = parse_args()
    main(args.specification_dir, args.out_dir, args.num_gpus, args.exps_per_gpu)