import argparse

from .constants import DATASET

def parse_job_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASET,
                    required=True)
    
    parser.add_argument('-model',
                    help='name of model to perform;',
                    type=str,
                    required=True)
    
    parser.add_argument('-configuration',
                    help='file that maintains job settings',
                    type=str,
                    default='job.yaml',
                    required=False)
    
    return parser.parse_args()
