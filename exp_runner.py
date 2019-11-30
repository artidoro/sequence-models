"""The main class for running experiments.
"""
import logging
import os
import argparse
import json
import config as c

from os.path import exists as E
from os.path import join as J

logger = logging.getLogger(__name__)


def run_experiment(args):
    """Runs an experiment based on the desired experiment specification.
    This process will record the desired response variables and write them to the experiment directory.
    
    Args:
        spec (dict): The JSON object specifying the experiment to run.
        experiment_directory (str):  The directory path to which to write the response variables.
    """
    spec, experiment_directory = args
    
    # Unpack some of the specification information
    try:
        name = spec["name"]
        # Unpack additional arguments <here>

    except KeyError:
        logger.error("Invalid experiment specification: {}".format(spec))
        raise


    # Create the directory
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    else:
        assert c.EXPERIMENT_RUNNER_SHOULD_OVERWRITE, "Experiment directory {} already exists".format(experiment_directory)

    # Output a copy of the experiment specification
    with open(J(experiment_directory, 'params.json'), 'w') as f:
        json.dump(spec, f)
    
    # Todo Run the actual experiment here <> @Ini
    # For now let's just print out the specification
    print(spec)
    
    


    

if __name__ == '__main__':
    # One can also run the experiment directly:
    parser = argparse.ArgumentParser(description="Runs an experiment directly from its JSON.")
    parser.add_argument('specification_json', type=str, help='The JSON which specifies the experiment.')
    parser.add_argument('output_dir', type=str, help='The directory where the output will be written.')
    args = parser.parse_args()

    with open(args.specification_json, 'r') as f:
        spec = json.load(f)

    run_experiment((spec, os.path.join(args.output_dir, spec["name"])))