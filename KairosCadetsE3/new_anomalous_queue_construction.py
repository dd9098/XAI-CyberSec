import logging
import os
import torch
import math
import copy
import ast
from tqdm import tqdm
from kairos_utils import *
from config import *

# Setting for logging
try:
    logger = logging.getLogger("anomalous_queue_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(artifact_dir, 'anomalous_queue.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up logging: {e}")
    exit(1)

def cal_anomaly_loss(loss_list, edge_list):
    try:
        if len(loss_list) != len(edge_list):
            logger.error("Mismatch between lengths of loss_list and edge_list.")
            return 0, 0, set(), set()
        
        count = 0
        loss_sum = 0
        loss_std = std(loss_list)
        loss_mean = mean(loss_list)
        edge_set = set()
        node_set = set()

        thr = loss_mean + 1.5 * loss_std
        logger.info(f"thr: {thr}")

        for i in range(len(loss_list)):
            if loss_list[i] > thr:
                count += 1
                src_node = edge_list[i][0]
                dst_node = edge_list[i][1]
                loss_sum += loss_list[i]

                node_set.add(src_node)
                node_set.add(dst_node)
                edge_set.add(edge_list[i][0] + edge_list[i][1])
        
        avg_loss = loss_sum / count if count != 0 else 0
        return count, avg_loss, node_set, edge_set
    except Exception as e:
        logger.error(f"Error in cal_anomaly_loss: {e}")
        return 0, 0, set(), set()

def compute_IDF():
    try:
        node_IDF = {}
        file_list = []

        for graph_dir in ["graph_4_3", "graph_4_4", "graph_4_5"]:
            file_path = os.path.join(artifact_dir, graph_dir)
            if os.path.exists(file_path):
                file_l = os.listdir(file_path)
                for i in file_l:
                    file_list.append(os.path.join(file_path, i))
            else:
                logger.warning(f"Directory '{file_path}' does not exist.")

        if not file_list:
            logger.warning("No files found to process in compute_IDF.")
            return {}, []

        node_set = {}
        for f_path in tqdm(file_list):
            if os.path.exists(f_path):
                with open(f_path) as f:
                    for line in f:
                        try:
                            l = line.strip()
                            jdata = ast.literal_eval(l)
                            if jdata['loss'] > 0:
                                if 'netflow' not in str(jdata['srcmsg']):
                                    node_set.setdefault(str(jdata['srcmsg']), set()).add(f_path)
                                if 'netflow' not in str(jdata['dstmsg']):
                                    node_set.setdefault(str(jdata['dstmsg']), set()).add(f_path)
                        except Exception as e:
                            logger.warning(f"Skipping malformed line in file '{f_path}': {e}")
            else:
                logger.warning(f"File '{f_path}' does not exist.")

        for n in node_set:
            include_count = len(node_set[n])
            IDF = math.log(len(file_list) / (include_count + 1))
            node_IDF[n] = IDF

        torch.save(node_IDF, os.path.join(artifact_dir, "node_IDF"))
        logger.info("IDF weight calculation complete!")
        return node_IDF, file_list
    except Exception as e:
        logger.error(f"Error in compute_IDF: {e}")
        return {}, []

def cal_set_rel(s1, s2, node_IDF, tw_list):
    try:
        def is_include_key_word(s):
            keywords = [
                'netflow',
                '/home/george/Drafts',
                'usr',
                'proc',
                'var',
                'cadet',
                '/var/log/debug.log',
                '/var/log/cron',
                '/home/charles/Drafts',
                '/etc/ssl/cert.pem',
                '/tmp/.31.3022e',
            ]
            return any(i in s for i in keywords)

        new_s = s1 & s2
        count = 0
        for i in new_s:
            if is_include_key_word(i):
                node_IDF[i] = math.log(len(tw_list) / (1 + len(tw_list)))

            IDF = node_IDF.get(i, math.log(len(tw_list)))

            if IDF > (math.log(len(tw_list) * 0.9)):
                logger.info(f"node: {i}, IDF: {IDF}")
                count += 1
        return count
    except Exception as e:
        logger.error(f"Error in cal_set_rel: {e}")
        return 0

def anomalous_queue_construction(node_IDF, tw_list, graph_dir_path):
    try:
        history_list = []
        current_tw = {}

        if not os.path.exists(graph_dir_path):
            logger.error(f"Graph directory '{graph_dir_path}' does not exist.")
            return history_list

        file_l = sorted(os.listdir(graph_dir_path))
        index_count = 0
        for f_path in file_l:
            try:
                logger.info("**************************************************")
                logger.info(f"Time window: {f_path}")

                edge_loss_list = []
                edge_list = []
                logger.info(f'Time window index: {index_count}')

                if f_path.startswith('.'):
                    continue

                with open(os.path.join(graph_dir_path, f_path), "r", encoding="ISO-8859-1") as f:
                    for line in f:
                        l = ''.join([c for c in line.strip().replace('\x00', '') if c.isprintable()])
                        if l:
                            jdata = ast.literal_eval(l)
                            edge_loss_list.append(jdata['loss'])
                            edge_list.append([str(jdata['srcmsg']), str(jdata['dstmsg'])])

                count, loss_avg, node_set, edge_set = cal_anomaly_loss(edge_loss_list, edge_list)
                current_tw = {
                    'name': f_path,
                    'loss': loss_avg,
                    'index': index_count,
                    'nodeset': node_set
                }

                added_que_flag = False
                for hq in history_list:
                    if added_que_flag:
                        break
                    for his_tw in hq:
                        if cal_set_rel(current_tw['nodeset'], his_tw['nodeset'], node_IDF, tw_list) != 0 and current_tw['name'] != his_tw['name']:
                            hq.append(copy.deepcopy(current_tw))
                            added_que_flag = True
                            break

                if not added_que_flag:
                    history_list.append([copy.deepcopy(current_tw)])

                index_count += 1

                logger.info(f"Average loss: {loss_avg}")
                logger.info(f"Num of anomalous edges within the time window: {count}")
                logger.info(f"Percentage of anomalous edges: {count / len(edge_list) if edge_list else 0}")
                logger.info(f"Anomalous node count: {len(node_set)}")
                logger.info(f"Anomalous edge count: {len(edge_set)}")
                logger.info("**************************************************")
            except Exception as e:
                logger.error(f"Error processing file '{f_path}': {e}")

        return history_list
    except Exception as e:
        logger.error(f"Error in anomalous_queue_construction: {e}")
        return []

if __name__ == "__main__":
    try:
        logger.info("Start logging.")

        node_IDF, tw_list = compute_IDF()

        # Validation data
        history_list = anomalous_queue_construction(
            node_IDF=node_IDF,
            tw_list=tw_list,
            graph_dir_path=os.path.join(artifact_dir, "graph_4_5/")
        )
        torch.save(history_list, os.path.join(artifact_dir, "graph_4_5_history_list"))

        # Testing data
        for day in [6, 7]:
            history_list = anomalous_queue_construction(
                node_IDF=node_IDF,
                tw_list=tw_list,
                graph_dir_path=os.path.join(artifact_dir, f"graph_4_{day}/")
            )
            torch.save(history_list, os.path.join(artifact_dir, f"graph_4_{day}_history_list"))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
