from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
sys.path.append("/mnt/workspace/graph_pretrain/MotiFiesta/pretrain")
# print(sys.path)