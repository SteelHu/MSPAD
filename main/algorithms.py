"""
DACAD 算法封装层
==================
功能：将 DACAD 模型和训练逻辑封装成统一的算法接口

主要类：
- Base_Algorithm: 所有算法的基类，定义通用接口
- DACAD: DACAD 算法的具体实现

关键方法：
- step(): 执行一步训练或验证
- predict_src/predict_trg(): 在源域/目标域上进行预测
- save_state/load_state(): 保存/加载模型权重
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
from models.dacad import DACAD_NN
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
        算法实例（目前只支持 DACAD）
    """
    if args.algo_name == "dacad":
        return DACAD(args, input_channels_dim, input_static_dim)
    else:
        return None

def get_num_channels(args):
    return list(map(int, args.num_channels_TCN.split("-")))

class Base_Algorithm(nn.Module):
    """
    算法基类
    =========
    定义所有算法的通用接口和功能
    
    所有具体算法（如 DACAD）都继承自这个类
    """
    
    def __init__(self, args):
        super(Base_Algorithm, self).__init__()

        # 保存参数供后续使用
        self.args = args

        # 记录算法名称和数据集类型
        self.algo_name = args.algo_name
        self.dataset_type = get_dataset_type(args)

        # 初始化损失函数
        self.pred_loss = PredictionLoss(self.dataset_type, args.task, args.weight_ratio)
        self.sup_cont_loss = SupervisedContrastiveLoss(self.dataset_type)

        # 输出维度（分类类别数）
        self.output_dim = get_output_dim(args)

        # TCN 网络的通道数配置（如 [128, 256, 512]）
        self.num_channels = get_num_channels(args)

        # ========== 根据数据集类型选择主要评估指标 ==========
        # 训练时会报告的主要指标
        self.main_pred_metric = ""
        if self.dataset_type == "smd":
            self.main_pred_metric = "avg_prc"      # SMD: 使用 AUPRC
        elif self.dataset_type == "msl":
            self.main_pred_metric = "avg_prc"      # MSL: 使用 AUPRC
        elif self.dataset_type == "boiler":
            self.main_pred_metric = "avg_prc"      # Boiler: 使用 AUPRC
        else:
            self.main_pred_metric = "mac_f1"       # 其他: 使用宏平均 F1

        # 初始化验证集的预测记录器
        self.init_pred_meters_val()

    def init_pred_meters_val(self):
            #We save prediction scores for validation set (for reporting purposes)
            self.pred_meter_val_src = PredictionMeter(self.args)
            self.pred_meter_val_trg = PredictionMeter(self.args)

    def predict_trg(self, sample_batched):
        trg_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_trg = self.classifier(trg_feat, sample_batched.get('static'))

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):
        src_feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        y_pred_src = self.classifier(src_feat, sample_batched.get('static'))

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))
    
    def get_embedding(self, sample_batched):
        feat = self.feature_extractor(sample_batched['sequence'].transpose(1,2))[:,:,-1]
        return feat

    #Score prediction is dataset and task dependent, that's why we write init function here
    def init_score_pred(self):
        if self.dataset_type == "smd":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "msl":
            return AverageMeter('ROC AUC', ':6.2f')
        elif self.dataset_type == "boiler":
            return AverageMeter('ROC AUC', ':6.2f')
        else:
            return AverageMeter('Macro F1', ':6.2f')

    #Helper function to build TCN feature extractor for all related algorithms 
    def build_feature_extractor_TCN(self, args, input_channels_dim, num_channels):
        return TemporalConvNet(num_inputs=input_channels_dim, num_channels=num_channels, kernel_size=args.kernel_size_TCN,
            stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN, dropout=args.dropout)



# ============================================================================
# DACAD 算法实现
# ============================================================================
class DACAD(Base_Algorithm):
    """
    DACAD: Domain Adaptation Contrastive Learning for Anomaly Detection
    =====================================================================
    
    功能：跨域异常检测算法
    
    核心组件：
    1. encoder_q/encoder_k: 查询/键编码器（TCN）
    2. projector: 特征投影器（MLP）
    3. predictor: Deep SVDD 分类器
    4. discriminator: 域判别器
    5. queue: MoCo 队列（存储负样本）
    
    训练策略：
    - 源域: 监督对比学习 + Deep SVDD 分类
    - 目标域: 自监督对比学习（异常注入）
    - 域对抗: 学习域不变特征
    """

    def __init__(self, args, input_channels_dim, input_static_dim):
        """
        初始化 DACAD 算法
        
        参数:
            args: 超参数配置
            input_channels_dim: 时间序列的特征维度
            input_static_dim: 静态特征的维度
        """
        super(DACAD, self).__init__(args)

        self.input_channels_dim = input_channels_dim
        self.input_static_dim = input_static_dim

        # ========== 创建 DACAD 模型 ==========
        # 注意：与其他算法不同，DACAD 将所有组件封装在一个模型中
        self.model = DACAD_NN(
            num_inputs=(1+args.use_mask)*input_channels_dim,  # 输入维度（可选是否使用mask）
            output_dim=self.output_dim,                       # 输出维度（类别数）
            num_channels=self.num_channels,                   # TCN 通道数
            num_static=input_static_dim,                      # 静态特征维度
            mlp_hidden_dim=args.hidden_dim_MLP,               # MLP 隐藏层维度
            use_batch_norm=args.use_batch_norm,               # 是否使用 BatchNorm
            kernel_size=args.kernel_size_TCN,                 # TCN 卷积核大小
            stride=args.stride_TCN,                           # TCN 步长
            dilation_factor=args.dilation_factor_TCN,         # TCN 膨胀因子
            dropout=args.dropout,                             # Dropout 率
            K=args.queue_size,                                # MoCo 队列大小
            m=args.momentum                                   # 动量更新系数
        )

        self.augmenter = None              # 数据增强器（暂未使用）
        self.concat_mask = concat_mask     # mask 拼接函数

        # 交叉熵损失（用于对比学习）
        self.criterion_CL = nn.CrossEntropyLoss()

        # 将模型移到 GPU
        self.cuda()

        # ========== 初始化优化器 ==========
        # 使用 Adam 优化器，betas=(0.5, 0.99) 适用于 GAN 类训练
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
        
        这是 DACAD 的核心方法，包含：
        1. 前向传播：同时处理源域和目标域数据
        2. 计算多个损失函数
        3. 反向传播更新参数（训练模式）
        
        参数:
            sample_batched_src: 源域数据批次（包含 sequence, label, positive, negative）
            sample_batched_trg: 目标域数据批次
            **kwargs: 额外参数（如 count_step）
        """
        
        # ========== 第一步：计算域对抗的自适应权重 alpha ==========
        # alpha 随训练步数逐渐从 0 增加到 1
        # 使得域对抗训练从弱到强，避免初期训练不稳定
        p = float(kwargs.get("count_step")) / 1000
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # ========== 第二步：准备输入数据 ==========
        seq_k_src, seq_k_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        seq_q_src, seq_q_trg = sample_batched_src['sequence'], sample_batched_trg['sequence']
        
        # ========== 第三步：前向传播，获取所有输出 ==========
        # 模型返回值详解：
        # - output_s/target_s: 源域对比学习的输出和目标
        # - output_t/target_t: 目标域对比学习的输出和目标  
        # - output_disc/target_disc: 域判别器的输出和目标
        # - pred_s: 源域的异常检测预测
        # - center/squared_radius: Deep SVDD 的中心和半径
        # - q_s/q_s_pos/q_s_neg: 源域的锚点、正样本、负样本特征
        # - p_q_s/p_q_s_pos/p_q_s_neg: 投影后的源域三元组特征
        # - q_t/q_t_pos/q_t_neg, p_q_t/p_q_t_pos/p_q_t_neg: 目标域对应特征
        output_s, target_s, output_t, target_t, output_ts, target_ts, output_disc, target_disc, pred_s,\
        center, squared_radius, q_s_repr, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg, q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg = \
            self.model(sample_batched_src['sequence'], seq_k_src, sample_batched_src['label'], sample_batched_src.get('static'),
                       sample_batched_trg['sequence'], seq_k_trg, sample_batched_trg.get('static'), alpha,
                       sample_batched_src['positive'], sample_batched_src['negative'], sample_batched_trg['positive'], sample_batched_trg['negative'])

        # ========== 第四步：计算各项损失 ==========
        
        # 1️⃣ 域判别器损失（Domain Adversarial Loss）
        # 目的：让特征提取器学习域不变的特征
        # 判别器试图区分源域和目标域，编码器试图混淆判别器
        loss_disc = F.binary_cross_entropy_with_logits(output_disc, target_disc)

        # 2️⃣ Deep SVDD 分类损失
        # 目的：在特征空间学习一个超球面，正常样本在球内，异常样本在球外
        src_cls_loss = self.pred_loss.deep_svdd_loss(q_s_repr, sample_batched_src['label'], center, squared_radius)

        # 3️⃣ 源域监督对比损失（Supervised Contrastive Triplet Loss）
        # 目的：使用真实标签，拉近同类样本，推远不同类样本
        # 三元组：(锚点, 正样本=同类, 负样本=异常)
        src_sup_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(
            p_q_s, p_q_s_pos, p_q_s_neg, sample_batched_src['label'], margin=2
        )
        
        # 4️⃣ 目标域异常注入对比损失（Injection-based Contrastive Loss）
        # 目的：使用注入的异常作为负样本，进行自监督学习
        # 假设所有目标域样本都是正常的（label=0）
        trg_fake_labels = np.zeros(len(sample_batched_trg['label']))
        trg_inj_cont_loss = self.sup_cont_loss.get_sup_cont_tripletloss(
            p_q_t, p_q_t_pos, p_q_t_neg, trg_fake_labels, margin=2
        )

        # 5️⃣ 额外的平衡损失
        # 目的：防止目标域对比损失过小，确保目标域也得到充分训练
        tmp_loss = 0
        if (trg_inj_cont_loss - src_sup_cont_loss) < 0 :
            tmp_loss = abs(trg_inj_cont_loss - src_sup_cont_loss)

        # ========== 第五步：加权组合所有损失 ==========
        # 权重说明：
        # - weight_loss_disc=0.5: 域对抗损失
        # - weight_loss_pred=1.0: Deep SVDD 分类损失
        # - weight_loss_src_sup=0.1: 源域监督对比损失
        # - weight_loss_trg_inj=0.1: 目标域注入对比损失
        loss = self.args.weight_loss_disc*loss_disc + self.args.weight_loss_pred*src_cls_loss \
               + self.args.weight_loss_src_sup * src_sup_cont_loss + self.args.weight_loss_trg_inj * trg_inj_cont_loss + tmp_loss

        # ========== 第六步：反向传播（仅在训练模式） ==========
        if self.training:
            self.optimizer.zero_grad()  # 清空梯度
            loss.backward()             # 反向传播
            self.optimizer.step()       # 更新参数

        self.losses_sup.update(src_sup_cont_loss.item(), seq_q_src.size(0))
        self.losses_inj.update(trg_inj_cont_loss.item(), seq_q_trg.size(0))

        acc1 = accuracy_score(output_disc.detach().cpu().numpy().flatten()>0.5, target_disc.detach().cpu().numpy().flatten())
        self.losses_disc.update(loss_disc.item(), output_disc.size(0))
        self.top1_disc.update(acc1, output_disc.size(0))

        self.losses_pred.update(src_cls_loss.item(), seq_q_src.size(0))

        pred_meter_src = PredictionMeter(self.args)

        pred_meter_src.update(sample_batched_src['label'], pred_s)

        metrics_pred_src = pred_meter_src.get_metrics()

        self.score_pred.update(metrics_pred_src[self.main_pred_metric], sample_batched_src['sequence'].size(0))

        self.losses.update(loss.item(), sample_batched_src['sequence'].size(0))

        if not self.training:
            #keep track of prediction results (of source) explicitly
            self.pred_meter_val_src.update(sample_batched_src['label'], pred_s)

            #keep track of prediction results (of target) explicitly
            pred_t = self.model.predict(sample_batched_trg, sample_batched_trg.get('static'), is_target=True, is_eval=False)

            self.pred_meter_val_trg.update(sample_batched_trg['label'], pred_t)

    def init_metrics(self):
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
        self.top1_disc = AverageMeter('Acc@1', ':6.2f')
        self.losses_pred = AverageMeter('Loss Pred', ':.4e')
        self.score_pred = self.init_score_pred()
        self.losses = AverageMeter('Loss TOTAL', ':.4e')

    def return_metrics(self):
        return [self.losses_sup, self.losses_inj, self.losses_disc, self.losses_pred, self.losses]

    def save_state(self,experiment_folder_path):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict()}, 
                               os.path.join(experiment_folder_path, "model_best.pth.tar"))

    def load_state(self,experiment_folder_path):
        checkpoint = torch.load(experiment_folder_path+"/model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])

    #We need to overwrite below functions for DACAD
    def predict_trg(self, sample_batched):

        seq_t = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_trg = self.model.predict(sample_batched, sample_batched.get('static'), is_target=True, is_eval=True)

        self.pred_meter_val_trg.update(sample_batched['label'], y_pred_trg, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def predict_src(self, sample_batched):

        seq_s = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        y_pred_src = self.model.predict(sample_batched, sample_batched.get('static'), is_target=False, is_eval=True)

        self.pred_meter_val_src.update(sample_batched['label'], y_pred_src, id_patient = sample_batched.get('patient_id'), stay_hour = sample_batched.get('stay_hour'))

    def get_embedding(self, sample_batched):

        seq = self.concat_mask(sample_batched['sequence'], sample_batched['sequence_mask'], self.args.use_mask)
        feat = self.model.get_encoding(seq)

        return feat

    def get_augmenter(self, sample_batched):

        seq_len = sample_batched['sequence'].shape[1]
        num_channel = sample_batched['sequence'].shape[2]
        cutout_len = math.floor(seq_len / 12)
        if self.input_channels_dim != 1:
            self.augmenter = Augmenter(cutout_length=cutout_len)
        else:
            print("The model only support multi channel time series data")


