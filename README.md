
# Kairos: XAI in Cybersecurity

Kairos is an Explainable AI (XAI) framework designed for cybersecurity applications. This README outlines the steps for setting up and running the Kairos workflow using the CADETS E3 dataset and provides details on utilizing the pre-trained models for quick evaluation.

---

## Steps in OG Kairos - Demo (DARPA CADETS E3)

This demo reproduces the experimental results reported in our paper using the CADETS E3 dataset to demonstrate Kairos' end-to-end workflow.

### 1. Environment Setup
1. Follow the instructions in the [environment settings](https://github.com/ProvenanceAnalytics/kairos/blob/main/DARPA/settings/environment-settings.md) to configure the required environment for Kairos.

### 2. Database Setup
1. Set up the CADETS E3 database by following the instructions in the [database settings](https://github.com/ProvenanceAnalytics/kairos/blob/main/DARPA/settings/database.md).

### 3. Configuration
1. Edit the CADETS E3 configuration file:
   - Open `NewCadetsE3/config.py`.
   - Set the variable `raw_dir` to the absolute path of the folder containing the raw CADETS E3 data.
   - Update database-related variables (e.g., `username`, `password`, etc.) to match your database configuration.

### 4. Run the Workflow
1. Navigate to the NewCadetsE3 directory:
   ```bash
   cd NewCadetsE3
   ```
2. Execute the Kairos workflow (Recomended to run the pipeline in parts, check makefile):
   ```bash
   make pipeline
   ```

### 5. Generated Artifacts
1. Once the workflow completes, artifacts will be stored in the `NewCadetsE3/artifact/` folder.

#### Folder Structure:
```
- artifact/
    - graphs/
    - graph_4_3/
    - graph_4_4/
    - graph_4_5/
    - graph_4_6/
    - graph_4_7/
    - graph_visual/
    - models/
    - embedding.log
    - training.log
    - reconstruction.log
    - anomalous_queue.log
    - evaluation.log
    - some other artifacts
```

#### Explanation of Artifacts:
- `graphs/`: Contains all vectorized graphs.
- `graph_4_*/`: Reconstruction results of graphs.
- `graph_visual/`: Summary graphs for attack investigation.
- `embedding.log`: Records statistics during graph vectorization.
- `training.log`: Records model training losses.
- `reconstruction.log`: Records reconstruction statistics during testing.
- `anomalous_queue.log`: Logs anomalous time windows flagged by Kairos.
- `evaluation.log`: Contains evaluation results for the CADETS E3 dataset.

---

## Using the Pre-trained Model
### Quick Evaluation with Pre-trained Models
1. To skip training and use the pre-trained models:
   - Download the pre-trained models from [this link](https://drive.google.com/drive/u/0/folders/1YAKoO3G32xlYrCs4BuATt1h_hBvvEB6C), this project is CADETS-E3.
   - Replace the model under `artifacts/models/`

2. Run the following commands to evaluate and detect anomalies:
   ```bash
   make test
   make anomaly_detection
   ```

---

## Follow-up Ideas
### TGNN Explainer
- Explore TGNN (Temporal Graph Neural Network) explainers for better interpretability of Kairos outputs.

### TGIB Explainer
- Use Temporal Graph Information Bottleneck (TGIB) techniques to explain and analyze key patterns detected by Kairos.

For additional details, please refer to the official [Kairos GitHub Repository](https://github.com/ProvenanceAnalytics/kairos).
```
