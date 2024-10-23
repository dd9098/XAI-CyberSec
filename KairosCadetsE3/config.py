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
raw_dir = "/Users/dd/XAI-Project/CADETS_E3/json/"
assert os.path.exists(raw_dir), f"Raw log directory '{raw_dir}' does not exist. Please check the path."

# The directory to save all artifacts
artifact_dir = "artifact/"
validate_and_create_dir(artifact_dir, "Artifact directory")

# The directory to save the vectorized graphs
graphs_dir = os.path.join(artifact_dir, "graphs/")
validate_and_create_dir(graphs_dir, "Graphs directory")

# The directory to save the models
models_dir = os.path.join(artifact_dir, "models/")
validate_and_create_dir(models_dir, "Models directory")

# The directory to save the results after testing
test_re = os.path.join(artifact_dir, "test_re/")
validate_and_create_dir(test_re, "Test results directory")

# The directory to save all visualized results
vis_re = os.path.join(artifact_dir, "vis_re/")
validate_and_create_dir(vis_re, "Visualization results directory")

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
database = 'tc_cadet_dataset_db'

# Host settings for the database
host = None  # Set to '/var/run/postgresql/' if needed, otherwise None

# Database user
user = 'postgres'

# The password to the database user (retrieve from environment variable for security)
password = '#OnePlus8Pro'

# The port number for Postgres
port = '5432'
assert port.isdigit(), "The port number must be numeric."

########################################################
#
#               Graph semantics
#
########################################################

# The directions of the following edge types need to be reversed
edge_reversed = [
    "EVENT_ACCEPT",
    "EVENT_RECVFROM",
    "EVENT_RECVMSG"
]

# The following edges are the types only considered to construct the temporal graph for experiments.
include_edge_type = [
    "EVENT_WRITE",
    "EVENT_READ",
    "EVENT_CLOSE",
    "EVENT_OPEN",
    "EVENT_EXECUTE",
    "EVENT_SENDTO",
    "EVENT_RECVFROM",
]

# The map between edge type and edge ID
rel2id = {
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
node_embedding_dim = 16
assert isinstance(node_embedding_dim, int) and node_embedding_dim > 0, "Node embedding dimension must be a positive integer."

# Node State Dimension
node_state_dim = 100
assert isinstance(node_state_dim, int) and node_state_dim > 0, "Node state dimension must be a positive integer."

# Neighborhood Sampling Size
neighbor_size = 20
assert isinstance(neighbor_size, int) and neighbor_size > 0, "Neighbor sampling size must be a positive integer."

# Edge Embedding Dimension
edge_dim = 100
assert isinstance(edge_dim, int) and edge_dim > 0, "Edge embedding dimension must be a positive integer."

# The time encoding Dimension
time_dim = 100
assert isinstance(time_dim, int) and time_dim > 0, "Time encoding dimension must be a positive integer."

########################################################
#
#                   Train&Test
#
########################################################

# Batch size for training and testing
BATCH = 1024
assert isinstance(BATCH, int) and BATCH > 0, "Batch size must be a positive integer."

# Parameters for optimizer
lr = 0.00005
assert isinstance(lr, float) and lr > 0, "Learning rate must be a positive float."

eps = 1e-08
assert isinstance(eps, float) and eps > 0, "Epsilon must be a positive float."

weight_decay = 0.01
assert isinstance(weight_decay, float) and weight_decay >= 0, "Weight decay must be a non-negative float."

epoch_num = 50
assert isinstance(epoch_num, int) and epoch_num > 0, "Epoch number must be a positive integer."

# The size of time window, 60000000000 represents 1 min in nanoseconds.
# The default setting is 15 minutes.
time_window_size = 60000000000 * 15
assert isinstance(time_window_size, int) and time_window_size > 0, "Time window size must be a positive integer."

########################################################
#
#                   Threshold
#
########################################################

beta_day6 = 100
assert isinstance(beta_day6, int) and beta_day6 > 0, "Beta for day 6 must be a positive integer."

beta_day7 = 100
assert isinstance(beta_day7, int) and beta_day7 > 0, "Beta for day 7 must be a positive integer."

print("Configuration settings loaded successfully and validated.")
