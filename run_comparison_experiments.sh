#!/bin/bash

# ============================================================================
# MSPAD vs 原始DACAD对比实验脚本
# ============================================================================
# 功能：在MSL、SMD、Boiler三个数据集上对比MSPAD和原始DACAD的性能
#
# 使用方法：
#   1. 设置源域和目标域：
#      ./run_comparison_experiments.sh --dataset MSL --src F-5 --trg C-1
#
#   2. 只设置源域，其他文件为目标域：
#      ./run_comparison_experiments.sh --dataset MSL --src F-5 --all-targets
#
#   3. 运行所有数据集的所有组合：
#      ./run_comparison_experiments.sh --all-datasets --all-combinations
#
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DATASET=""
SOURCE_DOMAIN=""
TARGET_DOMAIN=""
ALL_TARGETS=false
ALL_DATASETS=false
ALL_COMBINATIONS=false
NUM_EPOCHS=20
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
LEARNING_RATE=1e-4
SEED=1234

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --src)
            SOURCE_DOMAIN="$2"
            shift 2
            ;;
        --trg)
            TARGET_DOMAIN="$2"
            shift 2
            ;;
        --all-targets)
            ALL_TARGETS=true
            shift
            ;;
        --all-datasets)
            ALL_DATASETS=true
            shift
            ;;
        --all-combinations)
            ALL_COMBINATIONS=true
            shift
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET          Dataset name: MSL, SMD, or Boiler"
            echo "  --src SOURCE_DOMAIN        Source domain ID"
            echo "  --trg TARGET_DOMAIN        Target domain ID"
            echo "  --all-targets              Use all other files as targets (only set source)"
            echo "  --all-datasets             Run on all datasets"
            echo "  --all-combinations         Run all source-target combinations"
            echo "  --num_epochs N             Number of epochs (default: 20)"
            echo "  --batch_size N             Batch size (default: 256)"
            echo "  --seed N                   Random seed (default: 1234)"
            echo ""
            echo "Examples:"
            echo "  $0 --dataset MSL --src F-5 --trg C-1"
            echo "  $0 --dataset MSL --src F-5 --all-targets"
            echo "  $0 --all-datasets --all-combinations"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ============================================================================
# 辅助函数：获取数据集的所有文件列表
# ============================================================================

get_msl_files() {
    # MSL数据集：从CSV文件中获取MSL航天器的所有通道
    if [ ! -f "datasets/MSL_SMAP/labeled_anomalies.csv" ]; then
        echo -e "${RED}Error: MSL dataset CSV file not found!${NC}"
        return 1
    fi
    
    python3 << EOF
import pandas as pd
import numpy as np
import os

# 读取CSV文件
csv_reader = pd.read_csv('datasets/MSL_SMAP/labeled_anomalies.csv', delimiter=',')
data_info = csv_reader[csv_reader['spacecraft'] == 'MSL']
space_files = np.asarray(data_info['chan_id'])

# 获取所有.npy文件
all_files = os.listdir('datasets/MSL_SMAP/test')
all_names = [name[:-4] for name in all_files if name.endswith('.npy')]

# 找出既有数据文件又有标注的通道
files = [file for file in all_names if file in space_files]
files = sorted(files)

for f in files:
    print(f)
EOF
}

get_smd_files() {
    # SMD数据集：获取所有machine-*.txt文件
    if [ ! -d "datasets/SMD/test" ]; then
        echo -e "${RED}Error: SMD dataset directory not found!${NC}"
        return 1
    fi
    
    for file in datasets/SMD/test/machine-*.txt; do
        if [ -f "$file" ]; then
            basename=$(basename "$file" .txt)
            echo "${basename#machine-}"  # 去掉machine-前缀
        fi
    done | sort
}

get_boiler_files() {
    # Boiler数据集：获取所有.csv文件
    if [ ! -d "datasets/Boiler" ]; then
        echo -e "${RED}Error: Boiler dataset directory not found!${NC}"
        return 1
    fi
    
    for file in datasets/Boiler/*.csv; do
        if [ -f "$file" ]; then
            basename=$(basename "$file" .csv)
            echo "$basename"
        fi
    done | sort
}

# ============================================================================
# 辅助函数：运行单个实验
# ============================================================================

run_experiment() {
    local dataset=$1
    local algo_name=$2
    local src=$3
    local trg=$4
    local exp_folder=$5
    
    # 根据算法选择脚本路径
    if [ "$algo_name" = "dacad" ]; then
        local train_script="main/train.py"
        local eval_script="main/eval.py"
    elif [ "$algo_name" = "MSPAD" ]; then
        local train_script="main_new/train.py"
        local eval_script="main_new/eval.py"
    else
        echo -e "${RED}Unknown algorithm: $algo_name${NC}"
        return 1
    fi
    
    # 根据数据集设置路径和参数
    case $dataset in
        MSL)
            local path_src="datasets/MSL_SMAP"
            local path_trg="datasets/MSL_SMAP"
            local batch_size=$BATCH_SIZE
            local dropout=0.1
            local num_channels_TCN="128-256-512"
            local hidden_dim_MLP=1024
            ;;
        SMD)
            local path_src="datasets/SMD/test"
            local path_trg="datasets/SMD/test"
            local batch_size=128  # SMD使用较小的batch
            local dropout=0.1
            local num_channels_TCN="128-256-512"
            local hidden_dim_MLP=1024
            ;;
        Boiler)
            local path_src="datasets/Boiler"
            local path_trg="datasets/Boiler"
            local batch_size=$BATCH_SIZE
            local dropout=0.2  # Boiler使用较大的dropout
            local num_channels_TCN="128-128-128"  # Boiler使用较小的TCN
            local hidden_dim_MLP=256  # Boiler使用较小的MLP
            ;;
        *)
            echo -e "${RED}Unknown dataset: $dataset${NC}"
            return 1
            ;;
    esac
    
    # 构建训练命令（基础参数）
    local train_cmd=(
        python "$train_script"
        --algo_name "$algo_name"
        --num_epochs $NUM_EPOCHS
        --batch_size $batch_size
        --eval_batch_size $EVAL_BATCH_SIZE
        --learning_rate $LEARNING_RATE
        --dropout $dropout
        --weight_decay 1e-4
        --num_channels_TCN "$num_channels_TCN"
        --dilation_factor_TCN 3
        --kernel_size_TCN 7
        --hidden_dim_MLP $hidden_dim_MLP
        --queue_size 98304
        --momentum 0.99
        --weight_loss_pred 1.0
        --weight_loss_src_sup 0.1
        --weight_loss_trg_inj 0.1
        --id_src "$src"
        --id_trg "$trg"
        --path_src "$path_src"
        --path_trg "$path_trg"
        --experiment_folder "$exp_folder"
        --seed $SEED
    )
    
    # 根据算法添加特定参数
    if [ "$algo_name" = "MSPAD" ]; then
        train_cmd+=(--weight_loss_disc 0.5)
        train_cmd+=(--weight_loss_ms_disc 0.3)
        train_cmd+=(--prototypical_margin 1.0)
    elif [ "$algo_name" = "dacad" ]; then
        train_cmd+=(--weight_loss_disc 0.5)
        # DACAD不需要weight_loss_ms_disc和prototypical_margin参数
    fi
    
    # 运行训练
    echo -e "${BLUE}Training: $algo_name on $dataset ($src -> $trg)...${NC}"
    echo -e "${BLUE}Using script: $train_script${NC}"
    if ! "${train_cmd[@]}"; then
        echo -e "${RED}❌ Training failed: $algo_name on $dataset ($src -> $trg)${NC}"
        return 1
    fi
    
    # 运行评估
    echo -e "${BLUE}Evaluating: $algo_name on $dataset ($src -> $trg)...${NC}"
    echo -e "${BLUE}Using script: $eval_script${NC}"
    local eval_cmd=(
        python "$eval_script"
        --experiments_main_folder results
        --experiment_folder "$exp_folder"
        --id_src "$src"
        --id_trg "$trg"
    )
    
    if ! "${eval_cmd[@]}"; then
        echo -e "${RED}❌ Evaluation failed: $algo_name on $dataset ($src -> $trg)${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Completed: $algo_name on $dataset ($src -> $trg)${NC}"
    return 0
}

# ============================================================================
# 主函数：运行对比实验
# ============================================================================

run_comparison() {
    local dataset=$1
    local src=$2
    local trg=$3
    
    echo ""
    echo "=========================================="
    echo "Dataset: $dataset"
    echo "Source: $src -> Target: $trg"
    echo "=========================================="
    
    # 运行原始DACAD
    local dacad_folder="${dataset}_Baseline_DACAD"
    if ! run_experiment "$dataset" "dacad" "$src" "$trg" "$dacad_folder"; then
        return 1
    fi
    
    # 运行MSPAD
    local mspad_folder="${dataset}_MSPAD_Full"
    if ! run_experiment "$dataset" "MSPAD" "$src" "$trg" "$mspad_folder"; then
        return 1
    fi
    
    echo ""
    echo -e "${GREEN}✓ Comparison completed for $dataset ($src -> $trg)${NC}"
    echo ""
}

# ============================================================================
# 主逻辑
# ============================================================================

main() {
    echo "=========================================="
    echo "MSPAD vs 原始DACAD 对比实验"
    echo "=========================================="
    echo "Parameters:"
    echo "  Epochs: $NUM_EPOCHS"
    echo "  Batch Size: $BATCH_SIZE"
    echo "  Seed: $SEED"
    echo "=========================================="
    
    # 如果设置了所有数据集和所有组合
    if [ "$ALL_DATASETS" = true ] && [ "$ALL_COMBINATIONS" = true ]; then
        echo -e "${YELLOW}Running all datasets with all combinations...${NC}"
        
        # MSL数据集
        echo -e "\n${BLUE}=== MSL Dataset ===${NC}"
        msl_files=($(get_msl_files))
        for src in "${msl_files[@]}"; do
            for trg in "${msl_files[@]}"; do
                if [ "$src" != "$trg" ]; then
                    run_comparison "MSL" "$src" "$trg"
                fi
            done
        done
        
        # SMD数据集
        echo -e "\n${BLUE}=== SMD Dataset ===${NC}"
        smd_files=($(get_smd_files))
        for src in "${smd_files[@]}"; do
            for trg in "${smd_files[@]}"; do
                if [ "$src" != "$trg" ]; then
                    run_comparison "SMD" "$src" "$trg"
                fi
            done
        done
        
        # Boiler数据集
        echo -e "\n${BLUE}=== Boiler Dataset ===${NC}"
        boiler_files=($(get_boiler_files))
        for src in "${boiler_files[@]}"; do
            for trg in "${boiler_files[@]}"; do
                if [ "$src" != "$trg" ]; then
                    run_comparison "Boiler" "$src" "$trg"
                fi
            done
        done
        
        echo -e "\n${GREEN}All experiments completed!${NC}"
        return 0
    fi
    
    # 如果只设置了所有数据集，但没有设置源域和目标域
    if [ "$ALL_DATASETS" = true ]; then
        echo -e "${YELLOW}Please specify source and target domains or use --all-combinations${NC}"
        return 1
    fi
    
    # 检查数据集是否设置
    if [ -z "$DATASET" ]; then
        echo -e "${RED}Error: Dataset not specified. Use --dataset MSL|SMD|Boiler${NC}"
        return 1
    fi
    
    # 检查源域是否设置
    if [ -z "$SOURCE_DOMAIN" ]; then
        echo -e "${RED}Error: Source domain not specified. Use --src SOURCE_DOMAIN${NC}"
        return 1
    fi
    
    # 如果设置了所有目标域
    if [ "$ALL_TARGETS" = true ]; then
        echo -e "${YELLOW}Running with source=$SOURCE_DOMAIN, all other files as targets...${NC}"
        
        case $DATASET in
            MSL)
                files=($(get_msl_files))
                ;;
            SMD)
                files=($(get_smd_files))
                ;;
            Boiler)
                files=($(get_boiler_files))
                ;;
            *)
                echo -e "${RED}Unknown dataset: $DATASET${NC}"
                return 1
                ;;
        esac
        
        for trg in "${files[@]}"; do
            if [ "$trg" != "$SOURCE_DOMAIN" ]; then
                run_comparison "$DATASET" "$SOURCE_DOMAIN" "$trg"
            fi
        done
        
        echo -e "\n${GREEN}All experiments completed!${NC}"
        return 0
    fi
    
    # 如果设置了目标域
    if [ -n "$TARGET_DOMAIN" ]; then
        run_comparison "$DATASET" "$SOURCE_DOMAIN" "$TARGET_DOMAIN"
        return 0
    fi
    
    # 如果没有设置目标域，也没有设置--all-targets
    echo -e "${RED}Error: Target domain not specified. Use --trg TARGET_DOMAIN or --all-targets${NC}"
    return 1
}

# 运行主函数
main "$@"

