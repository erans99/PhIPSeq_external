import os

if os.path.exists(__file__.replace('.py', '_local.py')):
    # noinspection PyUnresolvedReferences
    from config_local.py import *


##################################################################
# PATHS
##################################################################
PATH_START = globals().get('PATH_START', os.path.join("C:", "sigall", "PhD")) #TODO: your path here
PATH_PHAGE = globals().get('PATH_PHAGE', os.path.join(PATH_START, "final_coding_44_75"))
PATH_DICTS = globals().get('PATH_DICTS', os.path.join(PATH_PHAGE, "dicts"))
BASE_PATH = globals().get('BASE_PATH', os.path.join(PATH_START, "try_PH"))
ANALYSIS_PATH = globals().get('ANALYSIS_PATH', os.path.join(PATH_START, "Analysis"))
PYTHON_PATH = "C:\Users\M\anaconda3\envs\python_env\python.exe" #TODO: you python path here (e.g. /usr/python3.5.3/bin/python


##################################################################
# CONSTANTS
##################################################################
NUM_WELLS_IN_PLATE = 96
MIN_READS_IN_MOCK = 200
NUM_PLATES = 3
NUM_SAMPLES_PER_PLATE = 80
NUM_NC_PER_PLATE = 4
NUM_MOCK_PER_PLATE = 8
NUM_ANCHOR_PER_PLATE = 4
