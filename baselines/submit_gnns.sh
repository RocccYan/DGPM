#!/bin/sh

outdir="../logs_classification"

test=true

if $test; then
    #datasets="PROTEINS"
    datasets="PROTEINS IMDB-B COX2"
    gnn_types="gcn"
    num_layers="3"
    embed_dims="64"
    global_pools="sum"
    lrs="0.01"
    wds="0.01"
    folds="1 2 3"
    epochs=300
else
    datasets="PROTEINS ENZYMES COX2"
    gnn_types="gcn gin"
    num_layers="1 2 3"
    embed_dims="64 128 256"
    global_pools="mean sum max"
    lrs="0.1 0.01 0.001"
    wds="0.1 0.01 0.001"
    folds="1 2 3 4 5 6 7 8 9 10"
    epochs=300
fi

for dataset in ${datasets}; do
    for gnn_type in ${gnn_types}; do
        for num_layer in ${num_layers}; do
            for embed_dim in ${embed_dims}; do
                for global_pool in ${global_pools}; do
                    for lr in ${lrs}; do
                        for wd in ${wds}; do
                            for fold_idx in ${folds}; do
                                if [ ! -f ${outdir}/${dataset}/${gnn_type}_${num_layer}_${embed_dim}_${global_pool}_${lr}_${wd}/fold-${fold_idx}/results.csv ]; then
                                    python classification_gnn.py --epochs ${epochs} --outdir ${outdir} --dataset ${dataset} --num-layers ${num_layer} --embed-dim ${embed_dim} --gnn-type ${gnn_type} --global-pool ${global_pool} --lr ${lr} --weight-decay ${wd} --fold-idx ${fold_idx}
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done
