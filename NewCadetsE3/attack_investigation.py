import os
from graphviz import Digraph
import networkx as nx
import datetime
import community.community_louvain as community_louvain
from tqdm import tqdm
from config import *
from kairos_utils import *

# Some common path abstraction for visualization
replace_dic = {
    '/run/shm/': '/run/shm/*',
    '/home/admin/.cache/mozilla/firefox/': '/home/admin/.cache/mozilla/firefox/*',
    '/home/admin/.mozilla/firefox': '/home/admin/.mozilla/firefox*',
    '/data/replay_logdb/': '/data/replay_logdb/*',
    '/home/admin/.local/share/applications/': '/home/admin/.local/share/applications/*',
    '/usr/share/applications/': '/usr/share/applications/*',
    '/lib/x86_64-linux-gnu/': '/lib/x86_64-linux-gnu/*',
    '/proc/': '/proc/*',
    '/stat': '*/stat',
    '/etc/bash_completion.d/': '/etc/bash_completion.d/*',
    '/usr/bin/python2.7': '/usr/bin/python2.7/*',
    '/usr/lib/python2.7': '/usr/lib/python2.7/*',
}

def replace_path_name(path_name):
    for i in replace_dic:
        if i in path_name:
            return replace_dic[i]
    return path_name

# Users should manually put the detected anomalous time windows here
# ARTIFACT_DIR = r'artifact/graph_4_7'  # Use a raw string for Windows paths

# Load the detected anomalous time windows
attack_list = []
for file in ATTACK_LIST:
    file = os.path.join(MALICIOUS_DIR, file)
    attack_list.append(file)


original_edges_count = 0
graphs = []
gg = nx.DiGraph()
count = 0

for path in tqdm(attack_list):
    if os.path.isfile(path):  # Check if the file exists
        line_count = 0
        node_set = set()
        tempg = nx.DiGraph()
        with open(path, "r") as f:  # Use 'with' to handle file opening
            edge_list = []
            for line in f:
                count += 1
                l = line.strip()
                jdata = eval(l)
                edge_list.append(jdata)

        edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)
        original_edges_count += len(edge_list)

        loss_list = []
        for i in edge_list:
            loss_list.append(i['loss'])
        loss_mean = mean(loss_list)
        loss_std = std(loss_list)
        print(loss_mean)
        print(loss_std)
        thr = loss_mean + 1.5 * loss_std
        print("thr:", thr)
        for e in edge_list:
            if e['loss'] > thr:
                tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                               str(hashgen(replace_path_name(e['dstmsg']))))
                gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))), str(hashgen(replace_path_name(e['dstmsg']))),
                            loss=e['loss'], srcmsg=e['srcmsg'], dstmsg=e['dstmsg'], edge_type=e['edge_type'],
                            time=e['time'])

partition = community_louvain.best_partition(gg.to_undirected())

# Generate the candidate subgraphs based on community discovery results
communities = {}
max_partition = 0
for i in partition:
    if partition[i] > max_partition:
        max_partition = partition[i]
for i in range(max_partition + 1):
    communities[i] = nx.DiGraph()
for e in gg.edges:
    communities[partition[e[0]]].add_edge(e[0], e[1])
    communities[partition[e[1]]].add_edge(e[0], e[1])

# Define the attack nodes. They are only used to plot the colors of attack nodes and edges.
def attack_edge_flag(msg):
    attack_nodes = ATTACK_NODES
    return any(i in msg for i in attack_nodes)

# Plot and render candidate subgraph
os.makedirs(f'{ARTIFACT_DIR}/graph_visual/', exist_ok=True)  # Create directory if it doesn't exist
graph_index = 0
for c in communities:
    dot = Digraph(name="MyPicture", comment="the test", format="pdf")
    dot.graph_attr['rankdir'] = 'LR'

    for e in communities[c].edges:
        try:
            temp_edge = gg.edges[e]
            srcnode = e[0]
            dstnode = e[1]

            # source node
            src_shape = 'box' if "'subject': '" in temp_edge['srcmsg'] else 'oval' if "'file': '" in temp_edge['srcmsg'] else 'diamond'
            src_node_color = 'red' if attack_edge_flag(temp_edge['srcmsg']) else 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['srcmsg']))), label=str(
                replace_path_name(temp_edge['srcmsg']) + str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])),
                color=src_node_color, shape=src_shape)

            # destination node
            dst_shape = 'box' if "'subject': '" in temp_edge['dstmsg'] else 'oval' if "'file': '" in temp_edge['dstmsg'] else 'diamond'
            dst_node_color = 'red' if attack_edge_flag(temp_edge['dstmsg']) else 'blue'
            dot.node(name=str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=str(
                replace_path_name(temp_edge['dstmsg']) + str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])),
                color=dst_node_color, shape=dst_shape)

            edge_color = 'red' if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']) else 'blue'
            dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                     str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                     color=edge_color)

        except KeyError:
            continue  # Skip if edge does not exist

    dot.render(f'{ARTIFACT_DIR}/graph_visual/subgraph_{graph_index}', view=False)
    graph_index += 1
