import logging
import os
import torch
from sklearn import metrics

from kairos_utils import *
from config import *
from model import *

# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'evaluation.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def classifier_evaluation(y_test, y_test_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_test_pred).ravel()
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    fscore = 2 * (precision * recall) / (precision + recall)
    auc_val = metrics.roc_auc_score(y_test, y_test_pred)
    
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")
    
    return precision, recall, fscore, accuracy, auc_val

def ground_truth_label():
    labels = {}
    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_11")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_12")
    for f in filelist:
        labels[f] = 0

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_13")
    for f in filelist:
        labels[f] = 0

    attack_list = ATTACK_LIST

    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = ATTACK_NODES
        return any(node in line for node in attack_nodes)

    files = []
    attack_list = ATTACK_LIST

    for f in attack_list:
        files.append(f"{ARTIFACT_DIR}/graph_4_11/{f}")

    attack_edge_count = 0
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                if keyword_hit(line):
                    attack_edge_count += 1
    logger.info(f"Num of attack edges: {attack_edge_count}")

if __name__ == "__main__":
    logger.info("Start logging.")

    # Validation data
    anomalous_queue_scores = []
    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_8_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        anomalous_queue_scores.append(anomaly_score)

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_9_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        anomalous_queue_scores.append(anomaly_score)

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_10_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        anomalous_queue_scores.append(anomaly_score)

    logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")

    # Evaluating the testing set
    pred_label = {}

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_11/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_12/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_13/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_11_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        if anomaly_score > beta_day6:
            name_list = [i['name'] for i in hl]
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_12_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        if anomaly_score > beta_day6:
            name_list = [i['name'] for i in hl]
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_13_history_list")
    for hl in history_list:
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = anomaly_score * (hq['loss'] + 1)
        if anomaly_score > beta_day7:
            name_list = [i['name'] for i in hl]
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Calculate the metrics
    labels = ground_truth_label()
    y = []
    y_pred = []

    missing_files = [i for i in labels if i not in pred_label]
    if missing_files:
        logger.warning(f"Files missing from predictions: {missing_files}")

    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label.get(i, 0))  # Default to 0 if file missing in pred_label

    classifier_evaluation(y, y_pred)
