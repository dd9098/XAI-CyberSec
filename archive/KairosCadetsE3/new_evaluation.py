from sklearn.metrics import confusion_matrix, roc_auc_score
import logging
import os
import torch
from kairos_utils import *
from config import *
from model import *

# Setting for logging
try:
    logger = logging.getLogger("evaluation_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(ARTIFACT_DIR, 'evaluation.log'), maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up logging: {e}")
    exit(1)

def classifier_evaluation(y_test, y_test_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        logger.info(f'tn: {tn}')
        logger.info(f'fp: {fp}')
        logger.info(f'fn: {fn}')
        logger.info(f'tp: {tp}')

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fscore = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        auc_val = roc_auc_score(y_test, y_test_pred)

        logger.info(f"precision: {precision}")
        logger.info(f"recall: {recall}")
        logger.info(f"fscore: {fscore}")
        logger.info(f"accuracy: {accuracy}")
        logger.info(f"auc_val: {auc_val}")

        return precision, recall, fscore, accuracy, auc_val
    except Exception as e:
        logger.error(f"Error in classifier_evaluation: {e}")
        return 0, 0, 0, 0, 0

def ground_truth_label():
    try:
        labels = {}
        for folder in ["graph_4_6", "graph_4_7"]:
            path = os.path.join(ARTIFACT_DIR, folder)
            if os.path.exists(path):
                filelist = os.listdir(path)
                for f in filelist:
                    labels[f] = 0
            else:
                logger.warning(f"Directory '{path}' does not exist.")

        attack_list = ATTACK_LIST
        for i in attack_list:
            labels[i] = 1

        return labels
    except Exception as e:
        logger.error(f"Error in ground_truth_label: {e}")
        return {}

def calc_attack_edges():
    try:
        def keyword_hit(line):
            attack_nodes = [
                'vUgefal', '/var/log/devc', 'nginx', '81.49.200.166',
                '78.205.235.65', '200.36.109.214', '139.123.0.113',
                '152.111.159.139', '61.167.39.128',
            ]
            return any(node in line for node in attack_nodes)

        files = []
        attack_list = ATTACK_LIST
        for f in attack_list:
            files.append(os.path.join(ARTIFACT_DIR, "graph_4_6", f))

        attack_edge_count = 0
        for fpath in files:
            if os.path.exists(fpath):
                with open(fpath) as f:
                    for line in f:
                        if keyword_hit(line):
                            attack_edge_count += 1
            else:
                logger.warning(f"File '{fpath}' does not exist.")
        logger.info(f"Num of attack edges: {attack_edge_count}")
    except Exception as e:
        logger.error(f"Error in calc_attack_edges: {e}")

if __name__ == "__main__":
    logger.info("Start logging.")

    try:
        # Validation data
        anomalous_queue_scores = []
        history_file = os.path.join(ARTIFACT_DIR, "graph_4_5_history_list")
        if os.path.exists(history_file):
            history_list = torch.load(history_file, weights_only=False)
            for hl in history_list:
                anomaly_score = torch.prod(torch.tensor([hq['loss'] + 1 for hq in hl], dtype=torch.float)).item()
                anomalous_queue_scores.append(anomaly_score)

            logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")
        else:
            logger.error(f"History file '{history_file}' does not exist.")

        # Evaluating the testing set
        pred_label = {}

        for folder in ["graph_4_6", "graph_4_7"]:
            path = os.path.join(ARTIFACT_DIR, folder)
            if os.path.exists(path):
                filelist = os.listdir(path)
                for f in filelist:
                    pred_label[f] = 0
            else:
                logger.warning(f"Directory '{path}' does not exist.")

        for day in [6, 7]:
            history_file = os.path.join(ARTIFACT_DIR, f"graph_4_{day}_history_list")
            if os.path.exists(history_file):
                history_list = torch.load(history_file, weights_only=False)
                for hl in history_list:
                    anomaly_score = torch.prod(torch.tensor([hq['loss'] + 1 for hq in hl], dtype=torch.float)).item()
                    if anomaly_score > (BETA_DAY_6 if day == 6 else BETA_DAY_7):
                        name_list = [i['name'] for i in hl]
                        logger.info(f"Anomalous queue: {name_list}")
                        for i in name_list:
                            pred_label[i] = 1
                        logger.info(f"Anomaly score: {anomaly_score}")
            else:
                logger.error(f"History file '{history_file}' does not exist.")

        # Calculate the metrics
        labels = ground_truth_label()
        y = []
        y_pred = []

        missing_files = [i for i in labels if i not in pred_label]
        if missing_files:
            logger.warning(f"Files missing from predictions: {missing_files}")

        for i in labels:
            y.append(labels[i])
            if i not in pred_label:
                logger.warning(f"Prediction missing for file: {i}")
            y_pred.append(pred_label.get(i, 0))  # Default to 0 if file missing in pred_label

        classifier_evaluation(y, y_pred)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
