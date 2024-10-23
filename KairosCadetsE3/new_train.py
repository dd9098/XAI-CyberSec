##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from kairos_utils import *
from config import *
from new_model import *

# Setting for logging with rotating file handler
try:
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(ARTIFACT_DIR, 'training.log'))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
except Exception as e:
    print(f"Error setting up logging: {e}")
    exit(1)

# Device compatibility check
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Loss criterion
criterion = nn.CrossEntropyLoss()
# Gradient scaler for mixed precision
scaler = GradScaler()

def train(train_data, memory, gnn, link_pred, optimizer, neighbor_loader):
    try:
        memory.train()
        gnn.train()
        link_pred.train()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        batch_size = 1024

        data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)

        for batch in data_loader:
            optimizer.zero_grad()

            src, pos_dst, t, msg = batch.src.to(device), batch.dst.to(device), batch.t.to(device), batch.msg.to(device)

            n_id = torch.cat([src, pos_dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = {n_id[j]: j for j in range(n_id.size(0))}

            # Mixed precision training
            with autocast():
                # Get updated memory of all nodes involved in the computation.
                z, last_update = memory(n_id)
                z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
                pos_out = link_pred(z[torch.tensor([assoc[s] for s in src])], z[torch.tensor([assoc[d] for d in pos_dst])])

                y_pred = torch.cat([pos_out], dim=0)
                y_true = torch.tensor([tensor_find(m[NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1) - 1 for m in msg], device=device)

                loss = criterion(y_pred, y_true)

            # Scale the loss for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            memory.detach()
            total_loss += float(loss) * batch.num_events
        return total_loss / train_data.num_events
    except Exception as e:
        logger.error(f"Error in train function: {e}")
        return None

def load_train_data():
    try:
        graph_4_2 = torch.load(os.path.join(GRAPHS_DIR, "graph_4_2.TemporalData.simple"), map_location=device)
        graph_4_3 = torch.load(os.path.join(GRAPHS_DIR, "graph_4_3.TemporalData.simple"), map_location=device)
        graph_4_4 = torch.load(os.path.join(GRAPHS_DIR, "graph_4_4.TemporalData.simple"), map_location=device)
        return [graph_4_2, graph_4_3, graph_4_4]
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return []

def init_models(node_feat_size):
    try:
        memory = TGNMemory(
            max_node_num,
            node_feat_size,
            NODE_STATE_DIM,
            TIME_DIM,
            message_module=IdentityMessage(node_feat_size, NODE_STATE_DIM, TIME_DIM),
            aggregator_module=LastAggregator(),
        ).to(device)

        gnn = GraphAttentionEmbedding(
            in_channels=NODE_STATE_DIM,
            out_channels=EDGE_DIM,
            msg_dim=node_feat_size,
            time_enc=memory.time_enc,
        ).to(device)

        out_channels = len(INCLUDE_EDGE_TYPE)
        link_pred = LinkPredictor(in_channels=EDGE_DIM, out_channels=out_channels).to(device)

        optimizer = torch.optim.Adam(
            set(memory.parameters()) | set(gnn.parameters())
            | set(link_pred.parameters()), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)

        neighbor_loader = LastNeighborLoader(max_node_num, size=NEIGHBOR_SIZE, device=device)

        return memory, gnn, link_pred, optimizer, neighbor_loader
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return None, None, None, None, None

if __name__ == "__main__":
    try:
        logger.info("Start logging.")

        # Load data for training
        train_data = load_train_data()
        if not train_data:
            logger.error("Training data loading failed. Exiting.")
            exit(1)

        # Initialize the models and the optimizer
        node_feat_size = train_data[0].msg.size(-1)
        memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)
        if memory is None:
            logger.error("Model initialization failed. Exiting.")
            exit(1)

        # Train the model
        for epoch in tqdm(range(1, EPOCH_NUM + 1)):
            for g in train_data:
                loss = train(
                    train_data=g,
                    memory=memory,
                    gnn=gnn,
                    link_pred=link_pred,
                    optimizer=optimizer,
                    neighbor_loader=neighbor_loader
                )
                if loss is not None:
                    logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

        # Save the trained model
        model = [memory, gnn, link_pred, neighbor_loader]
        os.makedirs(MODELS_DIR, exist_ok=True)
        torch.save(model, os.path.join(MODELS_DIR, "models.pt"))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
