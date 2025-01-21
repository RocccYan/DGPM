# dataset=$1
# device=$1

[ -z "${dataset_name}" ] && dataset_name="PROTEINS"
[ -z "${device}" ] && device=0

# python main_graph_pretrain.py \
python pretrain/train_cross_learning.py \
	--dataset_name $dataset_name \
	--model_name_node PROTEINS_graphmae \
	--model_name_motif PROTEINS_D=32_S=2_SE=100_LR=0.001_E=200_B=256_W=10_G=1 \
	--train_mode cat \
	--num_epochs 300 \
	--num_workers 8 \
	--batch_size 32 \
# --run_name cross_PROTEINS_D=512 \


