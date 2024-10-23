import os
import json
from graphviz import Digraph
import networkx as nx
import datetime
import community.community_louvain as community_louvain
from tqdm import tqdm
from config import *
from kairos_utils import *
import shutil
import ast

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

artifact_dir = 'artifact/malacious/'
attack_list = [os.path.join(artifact_dir, filename) for filename in ATTACK_LIST]

original_edges_count = 0
graphs = []
gg = nx.DiGraph()
count = 0

for path in tqdm(attack_list):
    if os.path.isfile(path):
        try:
            line_count = 0
            node_set = set()
            tempg = nx.DiGraph()
            edge_list = []

            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    try:
                        # Use ast.literal_eval to safely evaluate Python-like dict strings
                        jdata = ast.literal_eval(line)
                        edge_list.append(jdata)
                    except (ValueError, SyntaxError) as e:
                        print(f"Skipping malformed line in {path}: {e}")
                        continue

            edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)
            original_edges_count += len(edge_list)

            loss_list = [i['loss'] for i in edge_list]
            loss_mean = mean(loss_list)
            loss_std = std(loss_list)
            print(f"Mean: {loss_mean}, Std: {loss_std}")
            thr = loss_mean + 1.5 * loss_std
            print(f"Threshold: {thr}")

            for e in edge_list:
                if e['loss'] > thr:
                    tempg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                                   str(hashgen(replace_path_name(e['dstmsg']))))
                    gg.add_edge(str(hashgen(replace_path_name(e['srcmsg']))),
                                str(hashgen(replace_path_name(e['dstmsg']))),
                                loss=e['loss'], srcmsg=e['srcmsg'], dstmsg=e['dstmsg'],
                                edge_type=e['edge_type'], time=e['time'])
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    else:
        print(f"File {path} does not exist, skipping...")

# Check if the graph is not empty before community detection
if len(gg) > 0:
    partition = community_louvain.best_partition(gg.to_undirected())
else:
    print("Graph is empty, skipping community detection.")
    partition = {}

# Define communities as an empty dictionary in case partition is empty
communities = {}

# Check if partition is empty before proceeding
if partition:
    communities = {i: nx.DiGraph() for i in range(max(partition.values()) + 1)}
    for e in gg.edges:
        communities[partition[e[0]]].add_edge(e[0], e[1])
        communities[partition[e[1]]].add_edge(e[0], e[1])
else:
    print("Partition is empty, no communities to create.")

def attack_edge_flag(msg):
    attack_nodes = [
        '/tmp/vUgefal',
        'vUgefal',
        '/var/log/devc',
        '/etc/passwd',
        '81.49.200.166',
        '61.167.39.128',
        '78.205.235.65',
        '139.123.0.113',
        "'nginx'",
    ]
    return any(i in msg for i in attack_nodes)

os.makedirs(f'{artifact_dir}graph_visual/', exist_ok=True)

# Only process communities if there are any
if communities:
    graph_index = 0
    for c in communities:
        # Check if community graph has any edges before rendering
        if len(communities[c].edges) == 0:
            print(f"Community {c} is empty, skipping rendering.")
            continue

        dot = Digraph(name="MyPicture", comment="the test", format="pdf")
        dot.graph_attr['rankdir'] = 'LR'

        # Debugging: Print graph content before rendering
        print(f"Processing community {c} with {len(communities[c].edges)} edges")

        for e in communities[c].edges:
            try:
                temp_edge = gg.edges[e]
                srcnode = e[0]
                dstnode = e[1]

                src_shape = 'box' if "'subject': '" in temp_edge['srcmsg'] else 'oval' if "'file': '" in temp_edge['srcmsg'] else 'diamond'
                src_node_color = 'red' if attack_edge_flag(temp_edge['srcmsg']) else 'blue'
                dot.node(name=str(hashgen(replace_path_name(temp_edge['srcmsg']))), 
                         label=str(replace_path_name(temp_edge['srcmsg']) + str(partition[str(hashgen(replace_path_name(temp_edge['srcmsg'])))])),
                         color=src_node_color, shape=src_shape)

                dst_shape = 'box' if "'subject': '" in temp_edge['dstmsg'] else 'oval' if "'file': '" in temp_edge['dstmsg'] else 'diamond'
                dst_node_color = 'red' if attack_edge_flag(temp_edge['dstmsg']) else 'blue'
                dot.node(name=str(hashgen(replace_path_name(temp_edge['dstmsg']))), 
                         label=str(replace_path_name(temp_edge['dstmsg']) + str(partition[str(hashgen(replace_path_name(temp_edge['dstmsg'])))])),
                         color=dst_node_color, shape=dst_shape)

                edge_color = 'red' if attack_edge_flag(temp_edge['srcmsg']) and attack_edge_flag(temp_edge['dstmsg']) else 'blue'
                dot.edge(str(hashgen(replace_path_name(temp_edge['srcmsg']))),
                         str(hashgen(replace_path_name(temp_edge['dstmsg']))), label=temp_edge['edge_type'],
                         color=edge_color)

            except KeyError:
                continue  # Skip if edge does not exist

        # Check if Graphviz is installed before rendering
        if shutil.which("dot") is None:
            print("Graphviz is not installed or 'dot' is not in your PATH. Please install Graphviz and try again.")
        else:
            # print(dot.source)
            # Render the graph and save it as PDF
            dot.render(f'{artifact_dir}graph_visual/subgraph_{graph_index}', view=False)
            graph_index += 1
else:
    print("No communities to process.")
