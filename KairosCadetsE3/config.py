import os
import logging

def validate_and_create_dir(path, description):
    """
    Validates if the given path exists. If not, attempts to create it.
    Args:
        path (str): Directory path to validate/create.
        description (str): Description for logging purposes.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"{description} created at: {path}")
        else:
            print(f"{description} already exists at: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create {description} at '{path}': {e}")

########################################################
#
#                   Artifacts path
#
########################################################

# The directory of the raw logs
RAW_DIR = "/Users/dd/XAI-Project/XAI-CyberSec/json"
assert os.path.exists(RAW_DIR), f"Raw log directory '{RAW_DIR}' does not exist. Please check the path."

# The directory to save all artifacts
ARTIFACT_DIR = "artifact/"
validate_and_create_dir(ARTIFACT_DIR, "Artifact directory")

# The directory to save the vectorized graphs
GRAPHS_DIR = os.path.join(ARTIFACT_DIR, "graphs/")
validate_and_create_dir(GRAPHS_DIR, "Graphs directory")

# The directory to save the models
MODELS_DIR = os.path.join(ARTIFACT_DIR, "models/")
validate_and_create_dir(MODELS_DIR, "Models directory")

# The directory to save the results after testing
TEST_RE = os.path.join(ARTIFACT_DIR, "test_re/")
validate_and_create_dir(TEST_RE, "Test results directory")

# The directory to save all visualized results
VIS_RE = os.path.join(ARTIFACT_DIR, "vis_re/")
validate_and_create_dir(VIS_RE, "Visualization results directory")

# Attack List with malicious nodes
ATTACK_LIST = [
    '2018-04-06 11_18_26.126177915_2018-04-06 11_33_35.116170745.txt',
    '2018-04-06 11_33_35.116170745_2018-04-06 11_48_42.606135188.txt',
    '2018-04-06 11_48_42.606135188_2018-04-06 12_03_50.186115455.txt',
    '2018-04-06 12_03_50.186115455_2018-04-06 14_01_32.489584227.txt',
]

########################################################
#
#               Database settings
#
########################################################

# Database name
DATABASE = 'tc_cadet_dataset_db'

# Host settings for the database
HOST = None  # Set to '/var/run/postgresql/' if needed, otherwise None

# Database user
USER = 'postgres'

# The password to the database user (retrieve from environment variable for security)
password = '#OnePlus8Pro'

# The port number for Postgres
PORT = '5432'
assert PORT.isdigit(), "The port number must be numeric."

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
EDGE_REVERSED = [
    "EVENT_ACCEPT",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edges are the types only considered to construct the temporal graph for experiments.
INCLUDE_EDGE_TYPE = [
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# The map between edge type and edge ID
REL2ID = {
    1: 'EVENT_WRITE',
    'EVENT_WRITE': 1,
    2: 'EVENT_READ',
    'EVENT_READ': 2,
    3: 'EVENT_CLOSE',
    'EVENT_CLOSE': 3,
    4: 'EVENT_OPEN',
    'EVENT_OPEN': 4,
    5: 'EVENT_EXECUTE',
    'EVENT_EXECUTE': 5,
    6: 'EVENT_SENDTO',
    'EVENT_SENDTO': 6,
    7: 'EVENT_RECVFROM',
    'EVENT_RECVFROM': 7
}

########################################################
#
#                   Model dimensionality
#
########################################################

# Node Embedding Dimension
NODE_EMBEDDING_DIM = 16
assert isinstance(NODE_EMBEDDING_DIM, int) and NODE_EMBEDDING_DIM > 0, "Node embedding dimension must be a positive integer."

# Node State Dimension
NODE_STATE_DIM = 100
assert isinstance(NODE_STATE_DIM, int) and NODE_STATE_DIM > 0, "Node state dimension must be a positive integer."

# Neighborhood Sampling Size
NEIGHBOR_SIZE = 20
assert isinstance(NEIGHBOR_SIZE, int) and NEIGHBOR_SIZE > 0, "Neighbor sampling size must be a positive integer."

# Edge Embedding Dimension
EDGE_DIM = 100
assert isinstance(EDGE_DIM, int) and EDGE_DIM > 0, "Edge embedding dimension must be a positive integer."

# The time encoding Dimension
TIME_DIM = 100
assert isinstance(TIME_DIM, int) and TIME_DIM > 0, "Time encoding dimension must be a positive integer."

########################################################
#
#                   Train & Test
#
########################################################

# Batch size for training and testing
BATCH = 1024
assert isinstance(BATCH, int) and BATCH > 0, "Batch size must be a positive integer."

# Parameters for optimizer
LR = 0.00005
assert isinstance(LR, float) and LR > 0, "Learning rate must be a positive float."

EPS = 1e-08
assert isinstance(EPS, float) and EPS > 0, "Epsilon must be a positive float."

WEIGHT_DECAY = 0.01
assert isinstance(WEIGHT_DECAY, float) and WEIGHT_DECAY >= 0, "Weight decay must be a non-negative float."

EPOCH_NUM = 50
assert isinstance(EPOCH_NUM, int) and EPOCH_NUM > 0, "Epoch number must be a positive integer."

# The size of time window, 60000000000 represents 1 min in nanoseconds.
# The default setting is 15 minutes.
TIME_WINDOW_SIZE = 60000000000 * 15
assert isinstance(TIME_WINDOW_SIZE, int) and TIME_WINDOW_SIZE > 0, "Time window size must be a positive integer."

########################################################
#
#                   Threshold
#
########################################################

BETA_DAY_6 = 100
assert isinstance(BETA_DAY_6, int) and BETA_DAY_6 > 0, "Beta for day 6 must be a positive integer."

BETA_DAY_7 = 100
assert isinstance(BETA_DAY_7, int) and BETA_DAY_7 > 0, "Beta for day 7 must be a positive integer."

print("Configuration settings loaded successfully and validated.")
