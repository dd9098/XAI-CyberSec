
# simulate_v1, simulate_v2, wikipedia, reddit
source  ~/miniforge3/etc/profile.d/conda.sh
conda activate condatick
for i in 0, 1, 2
do
    echo "${i}-th run\n"
    
    dataset=reddit_hyperlinks
    python learn_edge.py -d ${dataset} --bs 512 --n_degree 10 --n_epoch 10 --agg_method attn --attn_mode prod --gpu 0 --n_head 2 --prefix ${dataset}


done
