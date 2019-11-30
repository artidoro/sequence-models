"""The main class for running experiments.
"""
import logging
import os

import config as c

logger = logging.getLogger(__name__)


def run_experiment(args):
    """Runs an experiment based on the desired experiment specification.
    This process will record the desired response variables and write them to the experiment directory.
    
    Args:
        experiment_specification (dict): The JSON object specifying the experiment to run.
        experiment_directory (str):  The directory path to which to write the response variables.
    """
    experiment_specification, experiment_directory = args
    
    # Unpack some of the specification information
    try:
        name = args["name"]
        # Unpack additional arguments <here>

    except KeyError:
        logger.error("Invalid experiment specification: {}".format(experiment_specification))
        raise


    # Create the directory
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    else:
        assert c.EXPERIMENT_RUNNER_SHOULD_OVERWRITE, "Experiment directory {} already exists".format(experiment_directory)

    
    


    

if __name__ == '__main__':
    run_experiment