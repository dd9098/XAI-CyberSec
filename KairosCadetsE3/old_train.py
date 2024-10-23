##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging

from kairos_utils import *
from config import *
from old_model import *

# Setting for logging
logger = logging.getLogger("training_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def train(train_data,
          memory,
          gnn,
          link_pred,
          optimizer,
          neighbor_loader
          ):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for i in range(0, len(train_data), 1024):
        batch = train_data[i:i+1024]
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        n_id = torch.cat([src, pos_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, train_data.t[e_id], train_data.msg[e_id])
        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])

        y_pred = torch.cat([pos_out], dim=0)
        y_true = []
        for m in msg:
            l = tensor_find(m[NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1) - 1
            y_true.append(l)

        y_true = torch.tensor(y_true).to(device=device)
        y_true = y_true.reshape(-1).to(torch.long).to(device=device)

        loss = criterion(y_pred, y_true)

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events
    return total_loss / train_data.num_events

def load_train_data():
    graph_4_2 = torch.load(GRAPHS_DIR + "/graph_4_2.TemporalData.simple").to(device=device)
    graph_4_3 = torch.load(GRAPHS_DIR + "/graph_4_3.TemporalData.simple").to(device=device)
    graph_4_4 = torch.load(GRAPHS_DIR + "/graph_4_4.TemporalData.simple").to(device=device)
    return [graph_4_2, graph_4_3, graph_4_4]

def init_models(node_feat_size):
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

if __name__ == "__main__":
    logger.info("Start logging.")

    # Load data for training
    train_data = load_train_data()

    # Initialize the models and the optimizer
    node_feat_size = train_data[0].msg.size(-1)
    memory, gnn, link_pred, optimizer, neighbor_loader = init_models(node_feat_size=node_feat_size)

    # train the model
    for epoch in tqdm(range(1, EPOCH_NUM+1)):
        for g in train_data:
            loss = train(
                train_data=g,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                optimizer=optimizer,
                neighbor_loader=neighbor_loader
            )
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}')

    # Save the trained model
    model = [memory, gnn, link_pred, neighbor_loader]

    os.system(f"mkdir -p {MODELS_DIR}")
    torch.save(model, f"{MODELS_DIR}/models.pt")