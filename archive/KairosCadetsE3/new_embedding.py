from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import TemporalData
from tqdm import tqdm

import numpy as np
import logging
import torch
import os

from config import ARTIFACT_DIR, NODE_EMBEDDING_DIM, REL2ID, GRAPHS_DIR, INCLUDE_EDGE_TYPE
from kairos_utils import gen_nodeid2msg, datetime_to_ns_time_US, init_database_connection

# Setting for logging
logger = logging.getLogger("embedding_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(ARTIFACT_DIR, 'embedding.log'))
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def path2higlist(p):
    if not isinstance(p, str):
        logger.error(f"path2higlist expects a string, got {type(p)}")
        return []
    return ["/".join(p.strip().split('/')[:i+1]) for i in range(len(p.strip().split('/')))]


def ip2higlist(p):
    if not isinstance(p, str):
        logger.error(f"ip2higlist expects a string, got {type(p)}")
        return []
    return [".".join(p.strip().split('.')[:i+1]) for i in range(len(p.strip().split('.')))]


def list2str(l):
    if not isinstance(l, list):
        logger.error(f"list2str expects a list, got {type(l)}")
        return ''
    return ''.join(str(i) for i in l)


def gen_feature(cur):
    # Obtain all node labels
    nodeid2msg = gen_nodeid2msg(cur=cur)

    # Construct the hierarchical representation for each node label
    node_msg_dic_list = []
    for node_id, node_msg in tqdm(nodeid2msg.items(), desc="Constructing hierarchical labels"):
        if not isinstance(node_id, int):
            continue

        higlist = []
        if 'netflow' in node_msg:
            higlist = ['netflow'] + ip2higlist(node_msg['netflow'])
        elif 'file' in node_msg:
            higlist = ['file'] + path2higlist(node_msg['file'])
        elif 'subject' in node_msg:
            higlist = ['subject'] + path2higlist(node_msg['subject'])
        else:
            continue

        node_msg_dic_list.append([list2str(higlist)])  # Wrap in list for FeatureHasher compatibility

    # Featurize the hierarchical node labels
    if not node_msg_dic_list:
        raise ValueError("No valid hierarchical labels were found in the dataset.")

    FH_string = FeatureHasher(n_features=NODE_EMBEDDING_DIM, input_type="string")
    node2higvec = FH_string.transform(node_msg_dic_list).toarray()

    # Save the node embeddings
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    torch.save(node2higvec, os.path.join(ARTIFACT_DIR, "node2higvec"))

    return node2higvec


def gen_relation_onehot():
    num_classes = len(REL2ID.keys()) // 2
    relvec = torch.nn.functional.one_hot(torch.arange(0, num_classes), num_classes=num_classes)
    rel2vec = {}
    for rel, idx in REL2ID.items():
        if not isinstance(rel, int):
            rel2vec[rel] = relvec[idx - 1]
    torch.save(rel2vec, os.path.join(ARTIFACT_DIR, "rel2vec"))
    return rel2vec


def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger):
    for day in tqdm(range(2, 14), desc="Generating vectorized graphs"):
        try:
            start_timestamp = datetime_to_ns_time_US(f'2018-04-{day:02d} 00:00:00')
            end_timestamp = datetime_to_ns_time_US(f'2018-04-{day + 1:02d} 00:00:00')
            sql = f"""
            select * from event_table
            where
                  timestamp_rec > '{start_timestamp}' and timestamp_rec < '{end_timestamp}'
                   ORDER BY timestamp_rec;
            """
            cur.execute(sql)
            events = cur.fetchall()
            logger.info(f'2018-04-{day}, events count: {len(events)}')
        except Exception as e:
            logger.error(f"Error fetching events for day 2018-04-{day}: {e}")
            continue

        edge_list = []
        for e in events:
            try:
                edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
                if e[2] in INCLUDE_EDGE_TYPE:
                    edge_list.append(edge_temp)
            except (IndexError, ValueError) as e_idx:
                logger.error(f"Error processing event {e}: {e_idx}")
            except Exception as e_other:
                logger.error(f"Unexpected error processing event {e}: {e_other}")

        logger.info(f'2018-04-{day}, edge list len: {len(edge_list)}')

        dataset = TemporalData()
        src, dst, msg, t = [], [], [], []

        for i in edge_list:
            src_id, dst_id, rel_type, timestamp = i
            if src_id >= len(node2higvec) or dst_id >= len(node2higvec):
                logger.warning(f"Node IDs {src_id} or {dst_id} out of bounds")
                continue
            if rel_type not in rel2vec:
                logger.warning(f"Relation type {rel_type} not in rel2vec")
                continue
            try:
                message = torch.cat([torch.from_numpy(node2higvec[src_id]),
                                     rel2vec[rel_type],
                                     torch.from_numpy(node2higvec[dst_id])])
                msg.append(message)
                src.append(src_id)
                dst.append(dst_id)
                t.append(int(timestamp))
            except Exception as e:
                logger.error(f"Error constructing message for edge {i}: {e}")

        if not msg:
            logger.warning(f'No valid messages for day 2018-04-{day}. Skipping save.')
            continue

        try:
            dataset.src = torch.tensor(src, dtype=torch.long)
            dataset.dst = torch.tensor(dst, dtype=torch.long)
            dataset.t = torch.tensor(t, dtype=torch.long)
            dataset.msg = torch.vstack(msg).to(torch.float)
            torch.save(dataset, os.path.join(GRAPHS_DIR, f"graph_4_{day}.TemporalData.simple"))
        except Exception as e:
            logger.error(f"Error saving dataset for day 2018-04-{day}: {e}")


if __name__ == "__main__":
    logger.info("Start logging.")

    # Create the graphs directory if it doesn't exist
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    try:
        cur, _ = init_database_connection()
    except Exception as e:
        logger.error(f"Error initializing database connection: {e}")
        exit(1)

    try:
        node2higvec = gen_feature(cur=cur)
    except Exception as e:
        logger.error(f"Failed to generate node2higvec: {e}")
        exit(1)

    try:
        rel2vec = gen_relation_onehot()
    except Exception as e:
        logger.error(f"Failed to generate rel2vec: {e}")
        exit(1)

    gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger)
