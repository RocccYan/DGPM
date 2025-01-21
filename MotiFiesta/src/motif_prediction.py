import torch
import json

import context
from MotiFiesta.src.decode import HashDecoder
from MotiFiesta.utils.pipeline_utils import set_workspace

data_id = "bbbp" # "IMDB-BINARY" #'MUTAG' #,'IMDB-BINARY', 'COX2', , "COLLAB"
# model_id = 'MUTAG_D=8_S=2_SE=100_LR=0.005_E=200_B=256_W=10_G=1'
# model_id = "PROTEINS_D=32_S=2_SE=100_LR=0.005_E=200_B=256_W=10_G=2"
# model_id = "IMDB-BINARY_D=32_S=4_SE=100_LR=0.001_E=200_B=256_W=10_G=1"
# model_file_name = "IMDB-BINARY_D=32_S=4_SE=100_LR=0.001_E=200_B=256_W=10_G=1_200.pth"
# model_file_name = "PROTEINS_D=32_S=2_SE=100_LR=0.005_E=200_B=256_W=10_G=2_80.pth"
# model_file_name = model_id+".pth"
# PROTEINS_D=32_S=2_SE=100_LR=0.005_E=200_B=256_W=10_G=2_80.pth
# IMDB-BINARY_D=32_S=4_SE=100_LR=0.001_E=200_B=256_W=10_G=1_200.pth

model_id = "bbbp_D=8_S=4_B=256_SE=100_E=100_W=10_G=1"
model_file_name = "bbbp_D=8_S=4_B=256_SE=100_E=100_W=10_G=1_MinLoss.pth"
level = 4

set_workspace('/mnt/workspace/graph_pretrain/MotiFiesta')
decoder = HashDecoder(model_id, data_id, level, model_file_name=model_file_name)

decoded_graphs = decoder.decode()

motif_assign_nodes = {}
for idx, decoded_graph in enumerate(decoded_graphs):
    # print(f"Motif assignment for each node: {g.motif_pred}")
    motif_assign_nodes[f'graph_{idx}'] = decoded_graph.motif_pred.numpy().tolist()

with open(f'./results/motifs_{data_id}@{model_id}.json','w') as fh:
    json.dump(motif_assign_nodes, fh)

print('Moif assignment for each node save at ./results/motif_assignment.json!')
"""

class MotifPredictor(torch.nn.Module):

""" 