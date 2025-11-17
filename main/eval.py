import sys
import os
# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd

from utils.dataset import get_dataset
from utils.augmentations import Augmenter

from torch.utils.data import DataLoader
from utils.util_progress_log import get_logger, \
    get_dataset_type

import json

from argparse import ArgumentParser
from collections import namedtuple

from algorithms import get_algorithm


def main(args):
    with open(os.path.join(args.experiments_main_folder, args.experiment_folder,
                           str(args.id_src) + "-" + str(args.id_trg), 'commandline_args.txt'), 'r') as f:
        saved_args_dict_ = json.load(f)

    saved_args = namedtuple("SavedArgs", saved_args_dict_.keys())(*saved_args_dict_.values())

    # configure our logger
    log = get_logger(os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                  str(args.id_src) + "-" + str(saved_args.id_trg), "eval_" + saved_args.log))

    # Some functions and variables for logging
    dataset_type = get_dataset_type(saved_args)

    def log_scores(args, dataset_type, metrics_pred):
        if dataset_type == "smd":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        elif dataset_type == "msl":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        elif dataset_type == "boiler":
            log("AUPRC score is : %.4f " % (metrics_pred["avg_prc"]))
            log("Best F1 score is : %.4f " % (metrics_pred["best_f1"]))
            log("Best Prec score is : %.4f " % (metrics_pred["best_prec"]))
            log("Best Rec score is : %.4f " % (metrics_pred["best_rec"]))
        else:
            log("Accuracy score is : %.4f " % (metrics_pred["acc"]))
            log("Macro F1 score is : %.4f " % (metrics_pred["mac_f1"]))
            log("Weighted F1 score is : %.4f " % (metrics_pred["w_f1"]))

    batch_size = saved_args.batch_size
    eval_batch_size = saved_args.eval_batch_size

    # LOAD SOURCE and TARGET datasets (it is MIMIC-IV vs. AUMC by default)
    dataset_test_src = get_dataset(saved_args, domain_type="source", split_type="test")
    dataset_test_trg = get_dataset(saved_args, domain_type="target", split_type="test")

    augmenter = Augmenter()

    # Calculate input_channels_dim and input_static_dim
    input_channels_dim = dataset_test_src[0]['sequence'].shape[1]
    input_static_dim = dataset_test_src[0]['static'].shape[0] if 'static' in dataset_test_src[0] else 0

    # Get our algorithm
    algorithm = get_algorithm(saved_args, input_channels_dim=input_channels_dim, input_static_dim=input_static_dim)

    experiment_folder_path = os.path.join(args.experiments_main_folder, args.experiment_folder,
                                          str(args.id_src) + "-" + str(args.id_trg))

    algorithm.load_state(experiment_folder_path)

    dataloader_test_trg = DataLoader(dataset_test_trg, batch_size=batch_size,
                                     shuffle=False, num_workers=0, drop_last=True)

    dataloader_test_src = DataLoader(dataset_test_src, batch_size=batch_size,
                                     shuffle=False, num_workers=0, drop_last=True)

    # turn algorithm into eval mode
    algorithm.eval()

    for i_batch, sample_batched in enumerate(dataloader_test_trg):
        algorithm.predict_trg(sample_batched)

    # even though the name is "pred_meter_val_trg", in this script it saves test results
    y_test_trg = np.array(algorithm.pred_meter_val_trg.target_list)
    y_pred_trg = np.array(algorithm.pred_meter_val_trg.output_list)
    id_test_trg = np.array(algorithm.pred_meter_val_trg.id_patient_list)
    stay_hour_trg = np.array(algorithm.pred_meter_val_trg.stay_hours_list)

    if len(id_test_trg) == 0 and len(stay_hour_trg) == 0:
        id_test_trg = [-1] * len(y_test_trg)
        stay_hour_trg = [-1] * len(y_test_trg)

    pred_trg_df = pd.DataFrame(
        {"patient_id": id_test_trg, "stay_hour": stay_hour_trg, "y": y_test_trg, "y_pred": y_pred_trg})
    df_save_path_trg = os.path.join(saved_args.experiments_main_folder, args.experiment_folder,
                                    str(args.id_src) + "-" + str(args.id_trg), "predictions_test_target.csv")
    pred_trg_df.to_csv(df_save_path_trg, index=False)

    log("Target results saved to " + df_save_path_trg)

    log("TARGET RESULTS")
    log("loaded from " + saved_args.path_trg)
    log("")

    metrics_pred_test_trg = algorithm.pred_meter_val_trg.get_metrics()

    log_scores(saved_args, dataset_type, metrics_pred_test_trg)
    df_trg = pd.DataFrame.from_dict(metrics_pred_test_trg, orient='index')
    df_trg = df_trg.T
    df_trg.insert(0, 'src_id', args.id_src)
    df_trg.insert(1, 'trg_id', args.id_trg)

    # 创建结果汇总文件夹
    results_summary_dir = 'experiment_results'
    if not os.path.exists(results_summary_dir):
        os.makedirs(results_summary_dir)
    
    # 根据数据集类型生成汇总文件名，仿照MSL的命名格式
    if dataset_type == "msl":
        fname = 'DACAD_MSL_' + args.id_src + ".csv"
    elif dataset_type == "smd":
        fname = 'DACAD_SMD_' + args.id_src + ".csv"
    elif dataset_type == "boiler":
        fname = 'DACAD_Boiler_' + args.id_src + ".csv"
    else:
        fname = 'DACAD_test_' + args.id_src + ".csv"
    
    # 保存到统一的结果文件夹
    fpath = os.path.join(results_summary_dir, fname)
    if os.path.isfile(fpath):
        df_trg.to_csv(fpath, mode='a', header=False, index=False)
    else:
        df_trg.to_csv(fpath, mode='a', header=True, index=False)
    
    log("Summary results saved to " + fpath)

    for i_batch, sample_batched in enumerate(dataloader_test_src):
        algorithm.predict_src(sample_batched)

    # even though the name is "pred_meter_val_src", in this script it saves test results
    y_test_src = np.array(algorithm.pred_meter_val_src.target_list)
    y_pred_src = np.array(algorithm.pred_meter_val_src.output_list)
    id_test_src = np.array(algorithm.pred_meter_val_src.id_patient_list)
    stay_hour_src = np.array(algorithm.pred_meter_val_src.stay_hours_list)

    if len(id_test_src) == 0 and len(stay_hour_src) == 0:
        id_test_src = [-1] * len(y_test_src)
        stay_hour_src = [-1] * len(y_test_src)

    pred_src_df = pd.DataFrame(
        {"patient_id": id_test_src, "stay_hour": stay_hour_src, "y": y_test_src, "y_pred": y_pred_src})
    df_save_path_src = os.path.join(saved_args.experiments_main_folder, saved_args.experiment_folder,
                                    str(saved_args.id_src) + "-" + str(saved_args.id_trg),
                                    "predictions_test_source.csv")
    pred_src_df.to_csv(df_save_path_src, index=False)

    log("Source results saved to " + df_save_path_src)

    log("SOURCE RESULTS")
    log("loaded from " + saved_args.path_src)
    log("")

    metrics_pred_test_src = algorithm.pred_meter_val_src.get_metrics()

    log_scores(saved_args, dataset_type, metrics_pred_test_src)
    
    # 评估完成后，根据参数决定是否删除模型文件以节省空间
    if args.delete_model:
        model_file_path = os.path.join(experiment_folder_path, "model_best.pth.tar")
        if os.path.exists(model_file_path):
            try:
                os.remove(model_file_path)
                log(f"Model file deleted: {model_file_path}")
            except Exception as e:
                log(f"Warning: Failed to delete model file: {e}")

# parse command-line arguments and execute the main method
if __name__ == '__main__':
    parser = ArgumentParser(description="parse args")

    parser.add_argument('-emf', '--experiments_main_folder', type=str, default='results')
    parser.add_argument('-ef', '--experiment_folder', type=str, default='default')
    parser.add_argument('--id_src', type=str, default='1-1')
    parser.add_argument('--id_trg', type=str, default='1-5')
    parser.add_argument('--keep_model', action='store_false', dest='delete_model',
                       help='Keep model file after evaluation (default: delete model after evaluation)')
    parser.set_defaults(delete_model=True)

    args = parser.parse_args()

    main(args)