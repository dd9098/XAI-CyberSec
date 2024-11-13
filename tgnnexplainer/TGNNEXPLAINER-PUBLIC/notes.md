notes

changes to code for running existing implementation.
1- no need of tick
2 - change condatick.yml file
2- change in train.sh 
    - source
    - conda environment
    and gpu param 0 for the command
- change in cpckpt.sh 
    - change directory, and trained model name.

3- mkdir check wherever a location is specified. in tg_dataset, learn_edge, process.py


Key considerations for this hypothetical scenario:
Adaptation of the reward function: The reward function in T-GNNExplainer, which uses cross-entropy loss based on the prediction probability, might need to be adapted to align with the reconstruction error-based anomaly score.●
Interpretation of explanations: The interpretation of the explanations provided by T-GNNExplainer might differ in the context of anomaly detection. Instead of pinpointing events leading to a specific event's occurrence, the focus might shift to identifying events that contribute to the model's perception of an event as anomalous.

changes needed to port this model.
the scripts and directory structure needed.
- is preprocessing required for explainer?
- are the logs used?

preprocessing has events, node features and edge features all 3 separated.

subgraphx_tg_run.py

ckpts not used. trained model is referred to in ckpt code. - there is a ckpts for the explainer model wtf!
ok that is trained when none exists, chill. other_baselines_tg.py line 188

from tgnnexplainer.xgraph.dataset.utils_dataset import construct_tgat_neighbor_finder
construct_tgat_neighbor_finder creates 2 adjacency list (user & item) for reddit_ds. 
user, item, time, e_idx are the feats  -> !!have to edit it for kairos
NeighborFinder uses adj_list to create a neighborhood list, sorted by timestamp and offset indexes for fast access.
offset indexes is determined by adj_list.

 -- need utils_dataset.py for construct_tgat_neighbor_finder

PGExplainerExt._call() from subgraph_tg_run.py line 282 trains the initial explainer model, saves ckpt and calculates the candidate score
for target indexes
candidate score are importance scores and set to explain_results
also need base_explainer_tg.py to find all the candidate events.
 -- need base_explainer_tg.py

eventually EvaluatorMCTSTG is initialized. @ metrics_tg.py
params : explain_results & target_event_idxs. !!have to find the link between all these.
evaluate() which gives these outputs
    event_idxs_results
    sparsity_results
    fid_inv_results
    fid_inv_best_results

 -- need metrics_tg.py

sparsity_tg() in metrics_tg_utils.py is the only function. 
merge to metrics_tg.py


tgnnexplainer\TGNNEXPLAINER-PUBLIC\tgnnexplainer\xgraph\models\ext\tgat\processed
tgnnexplainer\TGNNEXPLAINER-PUBLIC\tgnnexplainer\xgraph\models\ext\tgat\saved_checkpoints
tgnnexplainer\TGNNEXPLAINER-PUBLIC\tgnnexplainer\xgraph\saved_mcts_results
tgnnexplainer\TGNNEXPLAINER-PUBLIC\tgnnexplainer\xgraph\explainer_ckpts
tgnnexplainer\TGNNEXPLAINER-PUBLIC\tgnnexplainer\xgraph\dataset\data