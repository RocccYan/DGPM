device=$1
[ -z "${device}" ] && device=0
# python main_graph_pretrain.py \

for dataset in BA2Motif #"IMDB-BINARY" "NCI1" "MUTAG" "IMDB-MULTI" #"" PROTEINS COLLAB REDDIT-BINARY 
do
	python main_graph.py \
		--device $device \
		--dataset $dataset \
		--mask_rate 0.5 \
		--encoder "gin" \
		--decoder "gin" \
		--in_drop 0.2 \
		--attn_drop 0.1 \
		--num_layers 2 \
		--num_hidden 512 \
		--num_heads 2 \
		--max_epoch 30 \
		--max_epoch_f 0 \
		--lr 0.00015 \
		--weight_decay 0.0 \
		--activation prelu \
		--optimizer adam \
		--drop_edge_rate 0.0 \
		--loss_fn "sce" \
		--seeds 0 1 2 3 4 \
		--linear_prob \
		--deg4feat \
		--save_model \
		# --use_cfg \

done




