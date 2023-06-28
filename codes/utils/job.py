import os
import sys
import yaml

def get_job_config(file):
    with open(file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            sys.exit()