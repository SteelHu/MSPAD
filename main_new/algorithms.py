"""
MSPAD算法封装层
MSPAD: Multi-Scale Domain Adversarial Prototypical Anomaly Detection
====================================================================
功能：封装MSPAD算法，实现多尺度域对抗训练和原型网络分类
"""

import sys
sys.path.append("..")
import os
import numpy as np
import math
import torch
import torch.nn as nn
from utils.dataset import get_output_dim
from utils.mlp import MLP
from utils.tcn_no_norm import TemporalConvNet
from utils.augmentations import Augmenter, concat_mask
from utils.util_progress_log import AverageMeter, PredictionMeter, get_dataset_type
from utils.loss import PredictionLoss, SupervisedContrastiveLoss
from models.MSPAD import MSPAD_NN
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def get_algorithm(args, input_channels_dim, input_static_dim):
    """
    根据参数返回对应的算法实例
    
    参数:
        args: 命令行参数
        input_channels_dim: 输入时间序列的特征维度
        input_static_dim: 静态特征的维度
    
    返回:
        算法实例
    """
    if args.algo_name == "MSPAD":
        return MSPAD(args, input_channels_dim, input_static_dim)
    else:
        # 如果使用其他算法，尝试从原始algorithms导入
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'main'))
            from algorithms import get_algorithm as get_original_algorithm
            return get_original_algorithm(args, input_channels_dim, input_static_dim)
        except:
            raise ValueError(f"Unknown algorithm: {args.algo_name}")


def get_num_channels(args):
    return list(map(int, args.num_channels_TCN.split("-")))


class Base_Algorithm(nn.Module):
    """
    算法基类
    定义所有算法的通用接口和功能
    """
    
    def __init__(self, args):
        super(Base_Algorithm, self).__init__()
        
        self.args = args
        self.algo_name = args.algo_name
        self.dataset_type = get_dataset_type(args)
        
        self.pred_loss = PredictionLoss(self.dataset_type, args.task, args.weight_ratio)
        self.sup_cont_loss = SupervisedContrastiveLoss(self.dataset_type)
        
        self.output_dim = get_output_dim(args)
        self.num_channels = get_num_channels(args)
        
        # 根据数据集类型选择主要评估指标
        self.main_pred_metric = ""
        if self.dataset_type == "smd":
            self.main_pred_metric = "avg_prc"
        elif self.dataset_type == "msl":
            self.main_pred_metric = "avg_prc"
        elif self.dataset_type == "boiler":
            self.main_pred_metric = "avg_prc"
        else:
            self.main_pred_metric = "mac_f1"
        
        self.init_pred_meters_val()
    
    def init_pred_meters_val(self):
        self.pred_meter_val_src = PredictionMeter(self.args)
        self.pred_meter_val_trg = PredictionMeter(self.args)
    
    def init_score_pred(self):
        if self.dataset_type == "smd":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "msl":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "boiler":
            return AverageMeter('ROC AUC', ':6.2f')
        else:
            return AverageMeter('Macro F1', ':6.2f')


# ============================================================================
# MSPAD算法实现
# MSPAD: Multi-Scale Domain Adversarial Prototypical Anomaly Detection
# ============================================================================
class MSPAD(Base_Algorithm):
    """
    MSPAD算法实现
    =============
    
    核心改进：
    1. 多尺度域对抗：在TCN的多个中间层同时进行域判别
    2. 层次化域对齐：从低层到高层逐步对齐域特征
    3. 原型网络分类器：使用原型网络替代Deep SVDD
    4. 加权多尺度损失：不同层使用不同权重
    """
    
    def __init__(self, args, input_channels_dim, input_static_dim):
        """
        初始化MSPAD算法
        
        参数:
            args: 超参数配置
            input_channels_dim: 时间序列的特征维度
            input_static_dim: 静态特征的维度
        """
        super(MSPAD, self).__init__(args)
        
        self.input_channels_dim = input_channels_dim
        self.input_static_dim = input_static_dim
        
        # ========== 创建MSPAD模型 ==========
        # 解析scale_weights参数（如果提供）
        scale_weights = None
        if hasattr(args, 'scale_weights') and args.scale_weights is not None:
            if isinstance(args.scale_weights, str):
                # 从字符串解析，如 "0.1,0.3,0.6"
                scale_weights = [float(x.strip()) for x in args.scale_weights.split(',')]
            elif isinstance(args.scale_weights, list):
                scale_weights = args.scale_weights
        
        # 解析use_layer_mask参数（如果提供）
        use_layer_mask = None
        if hasattr(args, 'use_layer_mask') and args.use_layer_mask is not None:
            if isinstance(args.use_layer_mask, str):
                # 从字符串解析，如 "1,1,0"
                use_layer_mask = [int(x.strip()) for x in args.use_layer_mask.split(',')]
            elif isinstance(args.use_layer_mask, list):
                use_layer_mask = args.use_layer_mask
        
        self.model = MSPAD_NN(
            num_inputs=(1+args.use_mask)*input_channels_dim,
            output_dim=self.output_dim,
            num_channels=self.num_channels,
            num_static=input_static_dim,
            mlp_hidden_dim=args.hidden_dim_MLP,
            use_batch_norm=args.use_batch_norm,
            kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN,
            dilation_factor=args.dilation_factor_TCN,
            dropout=args.dropout,
            K=args.queue_size,
            m=args.momentum,
            scale_weights=scale_weights,
            use_layer_mask=use_layer_mask
        )
        
        self.augmenter = None
        self.concat_mask = concat_mask
        
        # 交叉熵损失（用于对比学习）
        self.criterion_CL = nn.CrossEntropyLoss()
        
        # 将模型移到GPU
        self.cuda()
        
        # ========== 初始化优化器 ==========
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.5, 0.99)
        )
        
        # 初始化训练指标记录器
        self.init_metrics()
    
    def step(self, sample_batched_src, sample_batched_trg, **kwargs):
        """
        执行一步训练或验证
        
        这是MSPAD的核心方法，包含：
        1. 前向传播：同时处理源域和目标域数据
        2. 计算多个损失函数（包括多尺度域对抗损失）
        3. 反向传播更新参数（训练模式）
        
        参数:
            sample_batched_src: 源域数据批次
            sample_batched_trg: 目标域数据批次
            **kwargs: 额外参数（如 count_step）
        """
        
        # ========== 第一步：计算域对抗的自适应权重 alpha ==========
        p = float(kwargs.get("count_step", 0)) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        # ========== 第二步：准备输入数据 ==========
        seq_k_src, seq_k_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        seq_q_src, seq_q_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        
        # ========== 第三步：前向传播，获取所有输出 ==========
        outputs = self.model(
            sample_batched_src['sequence'], seq_k_src, sample_batched_src['label'], 
            sample_batched_src.get('static'),
            sample_batched_trg['sequence'], seq_k_trg, sample_batched_trg.get('static'), 
            alpha,
            sample_batched_src['positive'], sample_batched_src['negative'],
            sample_batched_trg['positive'], sample_batched_trg['negative']
        )
        
        # 解包返回值
        (output_s, target_s, output_t, target_t, output_ts, labels_ts,
         output_disc, target_disc, pred_s, prototype, threshold,
         q_s_repr, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg,
         q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg,
         ms_disc_outputs_s, ms_disc_outputs_t) = outputs
        
        # ========== 第四步：计算各项损失 ==========
        
        # 1️⃣ 单尺度域判别器损失（原有）
        loss_disc = F.binary_cross_entropy(output_disc, target_disc)
        
        # 2️⃣ 多尺度域对抗损失（新增）
        ms_disc_loss = 0
        if len(ms_disc_outputs_s) > 0 and len(ms_disc_outputs_t) > 0:
            domain_labels_s = torch.ones(len(q_s_repr), 1).to(device=q_s_repr.device)
            domain_labels_t = torch.zeros(len(q_t), 1).to(device=q_t.device)
            
            # 获取过滤后的权重（只包含启用的层）
            active_weights = [w for i, w in enumerate(self.model.scale_weights) if self.model.use_layer_mask[i]]
            
            for i, (disc_s, disc_t, weight) in enumerate(
                zip(ms_disc_outputs_s, ms_disc_outputs_t, active_weights)
            ):
                # 拼接源域和目标域的预测和标签
                labels_ms = torch.cat([domain_labels_s, domain_labels_t], dim=0)
                preds_ms = torch.cat([disc_s, disc_t], dim=0)
                
                # 计算该层的域对抗损失
                layer_loss = F.binary_cross_entropy(preds_ms, labels_ms)
                ms_disc_loss += weight * layer_loss
        
        # 3️⃣ 原型网络分类损失（替换Deep SVDD）
        # 获取原型网络间隔参数（margin），默认1.0
        prototypical_margin = getattr(self.args, 'prototypical_margin', 1.0)
        src_cls_loss = self.pred_loss.prototypical_loss(
            pred_s, sample_batched_src['label'], prototype, threshold, margin=prototypical_margin
        )
        
        # 4️⃣ 源域监督对比损失
        src_sup_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(
            p_q_s, p_q_s_pos, p_q_s_neg, sample_batched_src['label'], margin=2
        )
        
        # 5️⃣ 目标域异常注入对比损失
        trg_fake_labels = np.zeros(len(sample_batched_trg['label']))
        trg_inj_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(
            p_q_t, p_q_t_pos, p_q_t_neg, trg_fake_labels, margin=2
        )
        
        # 6️⃣ 额外的平衡损失
        tmp_loss = 0
        if (trg_inj_cont_loss - src_sup_cont_loss) < 0:
            tmp_loss = abs(trg_inj_cont_loss - src_sup_cont_loss)
        
        # ========== 第五步：加权组合所有损失 ==========
        # 权重说明：
        # - weight_loss_disc: 单尺度域对抗损失权重（默认0.5）
        # - weight_loss_ms_disc: 多尺度域对抗损失权重（默认0.3，新增）
        # - weight_loss_pred: 原型网络分类损失权重（默认1.0，替换Deep SVDD）
        # - weight_loss_src_sup: 源域监督对比损失权重（默认0.1）
        # - weight_loss_trg_inj: 目标域注入对比损失权重（默认0.1）
        weight_ms_disc = getattr(self.args, 'weight_loss_ms_disc', 0.3)
        
        loss = (self.args.weight_loss_disc * loss_disc +
                weight_ms_disc * ms_disc_loss +
                self.args.weight_loss_pred * src_cls_loss +
                self.args.weight_loss_src_sup * src_sup_cont_loss +
                self.args.weight_loss_trg_inj * trg_inj_cont_loss +
                tmp_loss)
        
        # ========== 第六步：反向传播（仅在训练模式） ==========
        if self.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # ========== 更新指标记录器 ==========
        self.losses_sup.update(src_sup_cont_loss.item(), seq_q_src.size(0))
        self.losses_inj.update(trg_inj_cont_loss.item(), seq_q_trg.size(0))
        
        acc1 = accuracy_score(
            output_disc.detach().cpu().numpy().flatten() > 0.5,
            target_disc.detach().cpu().numpy().flatten()
        )
        self.losses_disc.update(loss_disc.item(), output_disc.size(0))
        self.losses_ms_disc.update(ms_disc_loss.item(), output_disc.size(0))  # 新增
        self.top1_disc.update(acc1, output_disc.size(0))
        
        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))
        
        pred_meter_src = PredictionMeter(self.args)
        pred_meter_src.update(sample_batched_src['label'], pred_s)
        metrics_pred_src = pred_meter_src.get_metrics()
        
        self.score_pred.update(
            metrics_pred_src[self.main_pred_metric],
            sample_batched_src['sequence'].size(0)
        )
        
        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))
        
        if not self.training:
            # 记录验证集的预测结果
            self.pred_meter_val_src.update(sample_batched_src['label'], pred_s)
            
            pred_t = self.model.predict(
                sample_batched_trg,
                sample_batched_trg.get('static'),
                is_target=True,
                is_eval=False
            )
            self.pred_meter_val_trg.update(sample_batched_trg['label'], pred_t)
    
    def init_metrics(self):
        """初始化训练指标记录器"""
        self.losses_s = AverageMeter('Loss Source', ':.4e')
        self.top1_s = AverageMeter('Acc@1', ':6.2f')
        self.losses_sup = AverageMeter('L_Src_Sup', ':.4e')
        self.top1_sup = AverageMeter('Acc@1', ':6.2f')
        self.losses_inj = AverageMeter('L_Trg_Inj', ':.4e')
        self.top1_inj = AverageMeter('Acc@1', ':6.2f')
        self.losses_t = AverageMeter('Loss Target', ':.4e')
        self.top1_t = AverageMeter('Acc@1', ':6.2f')
        self.losses_ts = AverageMeter('Loss Sour-Tar CL', ':.4e')
        self.top1_ts = AverageMeter('Acc@1', ':6.2f')
        self.losses_disc = AverageMeter('Loss Disc', ':.4e')
        self.losses_ms_disc = AverageMeter('Loss MS Disc', ':.4e')  # 新增
        self.top1_disc = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')
    
    def return_metrics(self):
        """返回训练指标"""
        return [
            self.losses_sup, self.losses_inj, self.losses_disc,
            self.losses_ms_disc, self.losses_pred, self.losses  # 包含多尺度损失
        ]
    
    def save_state(self, experiment_folder_path):
        """保存模型状态"""
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(experiment_folder_path, "model_best.pth.tar"))
    
    def load_state(self, experiment_folder_path):
        """加载模型状态"""
        checkpoint = torch.load(experiment_folder_path + "/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def predict_trg(self, sample_batched):
        """在目标域上进行预测"""
        seq_t = self.concat_mask(
            sample_batched['sequence'],
            sample_batched['sequence_mask'],
            self.args.use_mask
        )
        y_pred_trg = self.model.predict(
            sample_batched,
            sample_batched.get('static'),
            is_target=True,
            is_eval=True
        )
        self.pred_meter_val_trg.update(
            sample_batched['label'],
            y_pred_trg,
            id_patient=sample_batched.get('patient_id'),
            stay_hour=sample_batched.get('stay_hour')
        )
    
    def predict_src(self, sample_batched):
        """在源域上进行预测"""
        seq_s = self.concat_mask(
            sample_batched['sequence'],
            sample_batched['sequence_mask'],
            self.args.use_mask
        )
        y_pred_src = self.model.predict(
            sample_batched,
            sample_batched.get('static'),
            is_target=False,
            is_eval=True
        )
        self.pred_meter_val_src.update(
            sample_batched['label'],
            y_pred_src,
            id_patient=sample_batched.get('patient_id'),
            stay_hour=sample_batched.get('stay_hour')
        )
    
    def get_embedding(self, sample_batched):
        """获取特征嵌入"""
        seq = self.concat_mask(
            sample_batched['sequence'],
            sample_batched['sequence_mask'],
            self.args.use_mask
        )
        feat = self.model.get_encoding(seq)
        return feat

