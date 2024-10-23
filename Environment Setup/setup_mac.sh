# install conda and create a new environment
conda create -n kairos python

# activate the environment
conda activate kairos

# install the required packages
conda install psycopg2
conda install tqdm
pip install scikit-learn
pip install networkx
pip install xxhash
pip install graphviz

# install pytorch and torch_geometric
conda install pytorch torchvision torchaudio -c pytorch
pip install torch_geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

"""
While PyTorch with the MPS backend can leverage Apple's GPU (M1/M2) for acceleration, 
PyTorch Geometric (PyG) operations may not fully utilize GPU acceleration at this time, 
as MPS support is still evolving and PyG's heavy reliance on custom CUDA kernels means 
some operations are not fully accelerated yet on Metal.

You can use MPS on Apple Silicon (M1/M2) for GPU acceleration in PyTorch, 
but full support for PyTorch Geometric on MPS is limited. 
However, it will still work, running on a mix of CPU and GPU where applicable.
"""

