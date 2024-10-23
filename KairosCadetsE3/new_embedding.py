from sklearn.feature_extraction import FeatureHasher
from torch_geometric.data import *
from tqdm import tqdm

import numpy as np
import logging
import torch
import os

from config import *
from kairos_utils import *

# Ensure artifact directory exists
try:
    if not os.path.exists(ARTIFACT_DIR):
        os.makedirs(ARTIFACT_DIR)
except Exception as e:
    print(f"Error creating artifact directory '{ARTIFACT_DIR}': {e}")
    exit(1)

# Setting up logging
try:
    logger = logging.getLogger("embedding_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(ARTIFACT_DIR, 'embedding.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up logging: {e}")
    exit(1)

def path2higlist(p):
    l = []
    try:
        spl = p.strip().split('/')
        for i in spl:
            if len(l) != 0:
                l.append(l[-1] + '/' + i)
            else:
                l.append(i)
        return l
    except Exception as e:
        logger.error(f"Error in path2higlist with input '{p}': {e}")
        return []

def ip2higlist(p):
    l = []
    try:
        spl = p.strip().split('.')
        for i in spl:
            if len(l) != 0:
                l.append(l[-1] + '.' + i)
            else:
                l.append(i)
        return l
    except Exception as e:
        logger.error(f"Error in ip2higlist with input '{p}': {e}")
        return []

def list2str(l):
    s = ''
    try:
        for i in l:
            s += i
        return s
    except Exception as e:
        logger.error(f"Error in list2str with input '{l}': {e}")
        return ''

def gen_feature(cur):
    try:
        # Firstly obtain all node labels
        nodeid2msg = gen_nodeid2msg(cur=cur)

        if not nodeid2msg:
            logger.warning("No node labels found in the database.")
            return None

        # Construct the hierarchical representation for each node label
        node_msg_dic_list = []
        for i in tqdm(nodeid2msg.keys(), desc="Processing node labels"):
            try:
                if isinstance(i, int):
                    higlist = []
                    if 'netflow' in nodeid2msg[i]:
                        higlist = ['netflow']
                        higlist += ip2higlist(nodeid2msg[i]['netflow'])
                    elif 'file' in nodeid2msg[i]:
                        higlist = ['file']
                        higlist += path2higlist(nodeid2msg[i]['file'])
                    elif 'subject' in nodeid2msg[i]:
                        higlist = ['subject']
                        higlist += path2higlist(nodeid2msg[i]['subject'])
                    else:
                        logger.warning(f"No valid keys found for node id {i}.")
                        continue

                    if higlist:
                        node_msg_dic_list.append(list2str(higlist))
            except Exception as e:
                logger.error(f"Error processing node id {i}: {e}")
                continue

        if not node_msg_dic_list:
            logger.warning("No node messages to featurize.")
            return None

        # Featurize the hierarchical node labels
        FH_string = FeatureHasher(n_features=NODE_EMBEDDING_DIM, input_type="string")
        node2higvec = []
        for i in tqdm(node_msg_dic_list, desc="Featurizing node labels"):
            try:
                vec = FH_string.transform([[i]]).toarray()  # Note the double brackets
                node2higvec.append(vec)
            except Exception as e:
                logger.error(f"Error featurizing node message '{i}': {e}")
                continue

        if not node2higvec:
            logger.error("No node features generated.")
            return None

        node2higvec = np.array(node2higvec).reshape([-1, NODE_EMBEDDING_DIM])
        torch.save(node2higvec, os.path.join(ARTIFACT_DIR, "node2higvec"))
        logger.info("Node features generated and saved successfully.")
        return node2higvec
    except Exception as e:
        logger.error(f"Error in gen_feature: {e}")
        return None

def gen_relation_onehot():
    try:
        num_classes = len(REL2ID.keys()) // 2
        if num_classes == 0:
            logger.error("Number of relation classes is zero.")
            return None

        relvec = torch.nn.functional.one_hot(torch.arange(0, num_classes), num_classes=num_classes)
        rel2vec = {}
        for i in REL2ID.keys():
            try:
                if not isinstance(i, int):
                    idx = REL2ID[i] - 1
                    if 0 <= idx < num_classes:
                        rel2vec[i] = relvec[idx]
                        rel2vec[idx] = i  # Map index to relation name if needed
                    else:
                        logger.error(f"Index out of bounds for relation '{i}': idx={idx}")
            except Exception as e:
                logger.error(f"Error processing relation '{i}': {e}")
                continue
        if not rel2vec:
            logger.error("No relations found in rel2id.")
            return None

        torch.save(rel2vec, os.path.join(ARTIFACT_DIR, "rel2vec"))
        logger.info("Relation vectors generated and saved successfully.")
        return rel2vec
    except Exception as e:
        logger.error(f"Error in gen_relation_onehot: {e}")
        return None

def gen_vectorized_graphs(cur, node2higvec, rel2vec, logger):
    if node2higvec is None or rel2vec is None:
        logger.error("Node features or relation vectors are None. Cannot generate vectorized graphs.")
        return

    for day in tqdm(range(2, 14), desc="Generating vectorized graphs"):
        try:
            start_timestamp = datetime_to_ns_time_US(f'2018-04-{day} 00:00:00')
            end_timestamp = datetime_to_ns_time_US(f'2018-04-{day + 1} 00:00:00')
            sql = """
            SELECT * FROM event_table
            WHERE timestamp_rec > %s AND timestamp_rec < %s
            ORDER BY timestamp_rec;
            """
            cur.execute(sql, (start_timestamp, end_timestamp))
            events = cur.fetchall()
            logger.info(f'2018-04-{day}, events count: {len(events)}')
            
            edge_list = []
            for e in events:
                try:
                    edge_temp = [int(e[1]), int(e[4]), e[2], e[5]]
                    if e[2] in INCLUDE_EDGE_TYPE:
                        edge_list.append(edge_temp)
                except Exception as ex:
                    logger.error(f"Error processing event {e}: {ex}")
                    continue
            
            logger.info(f'2018-04-{day}, edge list length: {len(edge_list)}')
            dataset = TemporalData()
            src = []
            dst = []
            msg = []  # Initialize msg list
            t = []

            for i in edge_list:
                try:
                    src_node = int(i[0])
                    dst_node = int(i[1])
                    relation = i[2]
                    timestamp = int(i[3])

                    src.append(src_node)
                    dst.append(dst_node)
                    t.append(timestamp)

                    # Ensure node2higvec and rel2vec have valid data
                    if src_node < len(node2higvec) and dst_node < len(node2higvec) and relation in rel2vec:
                        src_vec = torch.from_numpy(node2higvec[src_node])
                        dst_vec = torch.from_numpy(node2higvec[dst_node])
                        rel_vec = rel2vec[relation].float()
                        message = torch.cat([src_vec, rel_vec, dst_vec])
                        msg.append(message)
                    else:
                        logger.warning(f"Invalid indices or relation for edge: {i}")
                except Exception as e:
                    logger.error(f"Error processing edge {i}: {e}")
                    continue

            if not src or not dst or not t:
                logger.warning(f"No valid edges for day 2018-04-{day}. Skipping day.")
                continue

            dataset.src = torch.tensor(src, dtype=torch.long)
            dataset.dst = torch.tensor(dst, dtype=torch.long)
            dataset.t = torch.tensor(t, dtype=torch.long)

            # Check if msg is not empty before vstack
            if msg:
                try:
                    dataset.msg = torch.vstack(msg)
                    dataset.msg = dataset.msg.to(torch.float)
                except Exception as e:
                    logger.error(f"Error stacking messages for day 2018-04-{day}: {e}")
                    continue
            else:
                logger.warning(f'No valid messages for day 2018-04-{day}. Skipping vstack.')
                continue  # Since we have no messages, skip saving

            # Save the dataset
            try:
                save_path = os.path.join(GRAPHS_DIR, f"graph_4_{day}.TemporalData.simple")
                torch.save(dataset, save_path)
                logger.info(f"Saved dataset for day 2018-04-{day} to {save_path}")
            except Exception as e:
                logger.error(f"Error saving dataset for day 2018-04-{day}: {e}")

        except Exception as e:
            logger.error(f"Error processing day 2018-04-{day}: {e}")
            continue

if __name__ == "__main__":
    logger.info("Start logging.")

    # Create the graphs directory if it doesn't exist
    try:
        if not os.path.exists(GRAPHS_DIR):
            os.makedirs(GRAPHS_DIR)
    except Exception as e:
        logger.error(f"Error creating graphs directory '{GRAPHS_DIR}': {e}")
        exit(1)

    # Initialize database connection
    try:
        cur, _ = init_database_connection()
    except Exception as e:
        logger.error(f"Error initializing database connection: {e}")
        exit(1)

    # Generate node features
    try:
        node2higvec = gen_feature(cur=cur)
        if node2higvec is None:
            logger.error("Failed to generate node features. Exiting.")
            exit(1)
    except Exception as e:
        logger.error(f"Error generating node features: {e}")
        exit(1)

    # Generate relation vectors
    try:
        rel2vec = gen_relation_onehot()
        if rel2vec is None:
            logger.error("Failed to generate relation vectors. Exiting.")
            exit(1)
    except Exception as e:
        logger.error(f"Error generating relation vectors: {e}")
        exit(1)

    # Generate vectorized graphs
    try:
        gen_vectorized_graphs(cur=cur, node2higvec=node2higvec, rel2vec=rel2vec, logger=logger)
    except Exception as e:
        logger.error(f"Error generating vectorized graphs: {e}")
        exit(1)
