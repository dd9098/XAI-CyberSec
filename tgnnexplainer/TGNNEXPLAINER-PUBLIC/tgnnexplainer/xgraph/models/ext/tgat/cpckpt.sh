model=tgat
dataset=reddit_hyperlinks
epoch=9

source_path=./saved_checkpoints/${dataset}-attn-prod-123-${epoch}.pth
target_path=~/github/tgnnexplainer/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/checkpoints/${model}_${dataset}_best.pth
cp ${source_path} ${target_path}

echo ${source_path} ${target_path} 'copied'


ls -l ~/github/tgnnexplainer/TGNNEXPLAINER-PUBLIC/tgnnexplainer/xgraph/models/checkpoints
