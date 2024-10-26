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
    filelist = os.listdir(f"{ARTIFACT_DIR}graph_4_4")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{ARTIFACT_DIR}graph_4_7")
    for f in filelist:
        labels[f] = 0

    attack_list = [
        '2018-04-04 11_06_12.700056291_2018-04-04 11_21_16.210034949.txt',
        '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
        '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
        '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
    ]
    for i in attack_list:
        labels[i] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            'vUgefal', '/var/log/devc', 'nginx', '81.49.200.166',
            '78.205.235.65', '200.36.109.214', '139.123.0.113',
            '152.111.159.139', '61.167.39.128',
        ]
        return any(node in line for node in attack_nodes)

    files = []
    attack_list = [
        '2018-04-06 11:18:26.126177915~2018-04-06 11:33:35.116170745.txt',
        '2018-04-06 11:33:35.116170745~2018-04-06 11:48:42.606135188.txt',
        '2018-04-06 11:48:42.606135188~2018-04-06 12:03:50.186115455.txt',
        '2018-04-06 12:03:50.186115455~2018-04-06 14:01:32.489584227.txt',
    ]
    for f in attack_list:
        files.append(f"{ARTIFACT_DIR}graph_4_6/{f}")

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
    history_list = torch.load(f"{ARTIFACT_DIR}graph_4_5_history_list")
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

    filelist = os.listdir(f"{ARTIFACT_DIR}graph_4_6/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{ARTIFACT_DIR}graph_4_7/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{ARTIFACT_DIR}graph_4_4_history_list")
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

    history_list = torch.load(f"{ARTIFACT_DIR}graph_4_7_history_list")
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
