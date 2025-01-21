r"""check all kinds of performance, mostly classification results
 of pretrained/learned models.
"""

import json
import os
import re

import pandas as pd

import context
from MotiFiesta.utils.pipeline_utils import read_json, save_json

def summarize_performance(results_path, summary_filename):
    # results_path = ''
    # summary_filename = ''
    results_files = os.listdir(results_path)

    all_results = []
    for result in results_files:
        if result.endswith(".json"):
            base_info = parse_baseinfo(result)
            base_info.update(read_json(os.path.join(results_path, result)))
            all_results.append(base_info)

    summary_filename = summary_filename + '.csv' if not summary_filename.endswith('.csv') else ""
    pd.DataFrame(all_results).to_csv(os.path.join(results_path,summary_filename))


def parse_baseinfo(text):
    r"""parse the base info like model, dataset and etc. from file name.
    eg. COX2_GIN_B=512_lr=0.0001_T=0.5_model_E=20.pt
    """
    infos = re.sub(".json|.pt","",text)
    print(infos)
    info_list = infos.split('_')
    dataset_name = info_list[0]
    model_type = info_list[1]
    batch_size = re.findall('B=([\d]*)', infos)[0]
    lr = re.findall('lr=([^_]*)', infos)[0]
    temperature = re.findall('T=([^_]*)', infos)[0]
    epoch = re.findall('E=([^_]*)', infos)[0]
    
    return { 
        "dataset_name": dataset_name,
        "model_type": model_type,
        "batch_size": batch_size,
        "lr": lr,
        "T": temperature,
        "epoch": epoch
    }


if __name__ == '__main__':
    # claim results path and summary file name
    results_path = '../classifications'
    summary_filename = 'pretrain_0228'
    summarize_performance(results_path, summary_filename)


