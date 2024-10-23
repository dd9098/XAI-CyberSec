##########################################################################################
# Some of the code is adapted from:
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
##########################################################################################

import logging
from logging.handlers import RotatingFileHandler
import os
import torch
import time
import ast
from kairos_utils import *
from config import *
from new_model import *
import warnings

# Setting for logging with RotatingFileHandler and console output
try:
    logger = logging.getLogger("reconstruction_logger")
    logger.setLevel(logging.INFO)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        os.path.join(ARTIFACT_DIR, 'reconstruction.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    # Stream handler to print to console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.ERROR)  # Only show errors in the console
    stream_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
except Exception as e:
    print(f"Error setting up logging: {e}")
    exit(1)

# Device compatibility check
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@torch.no_grad()
def test(inference_data,
         memory,
         gnn,
         link_pred,
         neighbor_loader,
         nodeid2msg,
         path
         ):
    """
    Test the model on the given data.

    :param inference_data: Data to test the model on
    :param memory: Memory component of the model
    :param gnn: GNN component of the model
    :param link_pred: Link prediction component of the model
    :param neighbor_loader: Neighbor loader component of the model
    :param nodeid2msg: Mapping of node IDs to messages
    :param path: Path to save the results
    :return: Dictionary containing the losses for each time window
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        memory.eval()
        gnn.eval()
        link_pred.eval()

        memory.reset_state()  # Start with a fresh memory.
        neighbor_loader.reset_state()  # Start with an empty graph.

        time_with_loss = {}  # key: time, value: the losses
        total_loss = 0
        edge_list = []

        unique_nodes = torch.tensor([]).to(device=device)
        total_edges = 0

        start_time = inference_data.t[0]
        event_count = 0
        pos_o = []

        # Record the running time to evaluate the performance
        start = time.perf_counter()

        for i in range(0, len(inference_data), 1024):
            batch = inference_data[i:i+1024]

            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            unique_nodes = torch.cat([unique_nodes, src, pos_dst]).unique()
            total_edges += len(batch.src)

            n_id = torch.cat([src, pos_dst]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc = {n_id[j]: j for j in range(n_id.size(0))}

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, inference_data.t[e_id], inference_data.msg[e_id])

            pos_out = link_pred(z[torch.tensor([assoc[s] for s in src])], z[torch.tensor([assoc[d] for d in pos_dst])])

            pos_o.append(pos_out)
            y_pred = torch.cat([pos_out], dim=0)
            y_true = []
            for m in msg:
                l = tensor_find(m[NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1) - 1
                y_true.append(l)
            y_true = torch.tensor(y_true).to(device=device)
            y_true = y_true.reshape(-1).to(torch.long).to(device=device)

            loss = criterion(y_pred, y_true)
            total_loss += float(loss) * batch.num_events

            # update the edges in the batch to the memory and neighbor_loader
            memory.update_state(src, pos_dst, t, msg)
            neighbor_loader.insert(src, pos_dst)

            # compute the loss for each edge
            each_edge_loss = cal_pos_edges_loss_multiclass(pos_out, y_true)

            for i in range(len(pos_out)):
                srcnode = int(src[i])
                dstnode = int(pos_dst[i])

                # Check if the node IDs are present in the nodeid2msg mapping
                if srcnode in nodeid2msg and dstnode in nodeid2msg:
                    srcmsg = str(nodeid2msg[srcnode])
                    dstmsg = str(nodeid2msg[dstnode])
                else:
                    logger.warning(f"Node ID {srcnode} or {dstnode} not found in nodeid2msg. Skipping.")
                    print(f"Node ID {srcnode} or {dstnode} not found in nodeid2msg. Skipping.")  # Print warning to console
                    continue

                # Extract the edge type and loss for each edge
                t_var = int(t[i])
                edgeindex = tensor_find(msg[i][NODE_EMBEDDING_DIM:-NODE_EMBEDDING_DIM], 1)
                edge_type = REL2ID[edgeindex]
                loss = each_edge_loss[i]

                temp_dic = {
                    'loss': float(loss),
                    'srcnode': srcnode,
                    'dstnode': dstnode,
                    'srcmsg': srcmsg,
                    'dstmsg': dstmsg,
                    'edge_type': edge_type,
                    'time': t_var
                }

                edge_list.append(temp_dic)

            event_count += len(batch.src)
            if t[-1] > start_time + TIME_WINDOW_SIZE:
                # Here is a checkpoint, which records all edge losses in the current time window
                time_interval = ns_time_to_datetime_US(start_time) + "~" + ns_time_to_datetime_US(t[-1])

                end = time.perf_counter()
                time_with_loss[time_interval] = {
                    'loss': total_loss / event_count if event_count != 0 else 0,
                    'nodes_count': len(unique_nodes),
                    'total_edges': total_edges,
                    'costed_time': (end - start)
                }

                time_interval = time_interval.replace(":", "_").replace("~", "_")

                with open(os.path.join(path, f"{time_interval}.txt"), 'w') as log:
                    edge_list = sorted(edge_list, key=lambda x: x['loss'], reverse=True)  # Rank the results based on edge losses
                    for e in edge_list:
                        log.write(str(e) + "\n")

                logger.info(
                    f'Time: {time_interval}, Loss: {total_loss / event_count if event_count != 0 else 0:.4f}, '
                    f'Nodes_count: {len(unique_nodes)}, Edges_count: {event_count}, Cost Time: {(end - start):.2f}s'
                )
                
                event_count = 0
                total_loss = 0
                start_time = t[-1]
                edge_list.clear()

        return time_with_loss
    except Exception as e:
        logger.error(f"Error in test function: {e}")
        print(f"Error in test function: {e}")  # Print error to console
        return {}

def load_data():
    """
    Load graph data from files and return them as a list.

    :return: List of loaded graphs
    """
    try:
        graph_files = [
            "graph_4_3.TemporalData.simple",
            "graph_4_4.TemporalData.simple",
            "graph_4_5.TemporalData.simple",
            "graph_4_6.TemporalData.simple",
            "graph_4_7.TemporalData.simple"
        ]

        graphs = []
        for graph_file in graph_files:
            graph_path = os.path.join(GRAPHS_DIR, graph_file)
            if os.path.exists(graph_path):
                # Suppress warnings only for torch.load
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only=False.*")
                    graph = torch.load(graph_path, map_location=device)
                graphs.append(graph)
            else:
                logger.warning(f"Graph file {graph_file} does not exist at path {graph_path}")

        if not graphs:
            raise FileNotFoundError("No graph files were successfully loaded. Please check the paths and files.")

        return graphs

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")  # Print the error to the terminal for visibility
        return []

if __name__ == "__main__":
    try:
        logger.info("Start logging.")

        # load the map between nodeID and node labels
        try:
            cur, _ = init_database_connection()
            nodeid2msg = gen_nodeid2msg(cur=cur)
        except Exception as e:
            logger.error(f"Error initializing database connection: {e}")
            print(f"Error initializing database connection: {e}")  # Print error to console
            exit(1)

        # Load data
        data = load_data()
        if not data:
            logger.error("Data loading failed. Exiting.")
            print("Data loading failed. Exiting.")  # Print error to console
            exit(1)
        graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7 = data

        # load trained model
        try:
            memory, gnn, link_pred, neighbor_loader = torch.load(f"{MODELS_DIR}/models.pt", map_location=device)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            print(f"Error loading model: {e}")  # Print error to console
            exit(1)

        # Reconstruct the edges in each day
        for graph, day in zip([graph_4_3, graph_4_4, graph_4_5, graph_4_6, graph_4_7], range(3, 8)):
            test(inference_data=graph,
                 memory=memory,
                 gnn=gnn,
                 link_pred=link_pred,
                 neighbor_loader=neighbor_loader,
                 nodeid2msg=nodeid2msg,
                 path=os.path.join(ARTIFACT_DIR, f"graph_4_{day}"))
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error in main execution: {e}")  # Print error to console
