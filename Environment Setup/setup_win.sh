# install conda and create a new environment
conda create -n kairos python
# activate the environment
conda activate kairos
# install the required packages
conda install psycopg2
conda install tqdm
pip install scikit-learn==1.2.0
pip install networkx==2.8.7
pip install xxhash==3.2.0
pip install graphviz==0.20.1

# install pytorch and torch_geometric
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_geometric==2.0.0
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv
