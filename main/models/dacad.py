"""
DACAD 模型架构
===============
功能：定义 DACAD 的神经网络结构

主要组件：
1. ReverseLayerF: 梯度反转层（用于域对抗训练）
2. Discriminator: 域判别器
3. DeepSVDD: Deep SVDD 分类器
4. DACAD_NN: 完整的 DACAD 模型

模型架构：
┌─────────────────────────────────────────────┐
│  输入时间序列                                  │
└────────────┬────────────────────────────────┘
             │
         ┌───┴───┐
         │  TCN  │  ← encoder_q (查询编码器)
         └───┬───┘
             │
      ┌──────┼──────┐
      │      │      │
   ┌──┴─┐ ┌─┴──┐ ┌─┴───┐
   │proj││SVDD││Disc│  ← 投影器、分类器、判别器
   └────┘ └────┘ └─────┘
"""

import sys

sys.path.append("../..")
import torch
import torch.nn as nn
from utils.tcn_no_norm import TemporalConvNet
from utils.mlp import MLP
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn.utils import weight_norm

# 导入梯度反转层
from torch.autograd import Function


class ReverseLayerF(Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    =====================================
    
    功能：在前向传播时不改变输入，在反向传播时将梯度反转
    
    用途：域对抗训练
    - 判别器试图区分源域和目标域 → 梯度为正
    - 特征提取器试图混淆判别器 → 梯度反转为负
    
    这样特征提取器会学到域不变的特征表示
    """
    @staticmethod
    def forward(ctx, x, alpha):
        """
        前向传播：直接返回输入，不做任何改变
        
        参数:
            x: 输入特征
            alpha: 梯度反转的权重系数（随训练逐渐增大）
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：将梯度取反并乘以 alpha
        
        参数:
            grad_output: 从后续层传回来的梯度
        
        返回:
            反转后的梯度
        """
        output = grad_output.neg() * ctx.alpha
        return output, None


class Discriminator(nn.Module):
    """
    域判别器 (Domain Discriminator)
    =================================
    
    功能：判断输入特征来自源域还是目标域
    
    架构：3层MLP
    - 输入：特征向量
    - 输出：域标签 (0=源域, 1=目标域)
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # 第一层
            nn.LeakyReLU(0.2),                     # 激活函数（负斜率0.2）
            nn.Linear(hidden_dim, hidden_dim),     # 第二层
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),     # 输出层
            nn.Sigmoid()                           # Sigmoid 输出概率
        )

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, input_dim]
        
        返回:
            域预测概率 [batch_size, 1]，值域 [0,1]
        """
        return self.model(x)


class DACAD_NN(nn.Module):
    """
    DACAD 完整神经网络模型
    =======================
    
    基于 MoCo (Momentum Contrast) 框架的跨域异常检测模型
    
    关键组件：
    1. encoder_q: 查询编码器（可训练）
    2. encoder_k: 键编码器（动量更新）
    3. projector: 特征投影器
    4. predictor: Deep SVDD 分类器
    5. discriminator: 域判别器
    6. queue_s/queue_t: 源域和目标域的特征队列
    
    MoCo 机制：
    - encoder_k 通过动量更新 encoder_q 的参数：θ_k = m*θ_k + (1-m)*θ_q
    - 队列存储大量负样本特征，提升对比学习效果
    """

    def __init__(self, num_inputs, output_dim, num_channels, num_static, mlp_hidden_dim=256,
                 use_batch_norm=True, num_neighbors=1, kernel_size=2, stride=1, dilation_factor=2,
                 dropout=0.2, K=24576, m=0.999, T=0.07):
        """
        初始化 DACAD 模型
        
        参数:
            num_inputs: 输入时间序列的通道数
            output_dim: 输出维度（类别数）
            num_channels: TCN的通道数列表，如 [128, 256, 512]
            num_static: 静态特征的维度
            mlp_hidden_dim: MLP隐藏层维度
            use_batch_norm: 是否使用BatchNorm
            kernel_size: TCN卷积核大小
            stride: TCN步长
            dilation_factor: TCN膨胀因子
            dropout: Dropout率
            K: 队列大小（负样本数量）
            m: 动量更新系数
            T: 温度参数（用于对比学习）
        """
        super(DACAD_NN, self).__init__()

        self.sigmoid = nn.Sigmoid()

        # ========== MoCo 超参数 ==========
        self.K = K                      # 队列大小（如 98304）
        self.m = m                      # 动量更新系数（如 0.999）
        self.T = T                      # 温度参数（如 0.07）
        self.num_neighbors = num_neighbors

        # ========== 1. 编码器（TCN） ==========
        # encoder_q: 查询编码器，通过梯度更新
        self.encoder_q = TemporalConvNet(
            num_inputs=num_inputs, 
            num_channels=num_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            dilation_factor=dilation_factor, 
            dropout=dropout
        )
        
        # encoder_k: 键编码器，通过动量更新（不直接反向传播）
        self.encoder_k = TemporalConvNet(
            num_inputs=num_inputs, 
            num_channels=num_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            dilation_factor=dilation_factor, 
            dropout=dropout
        )

        # ========== 2. 特征投影器 ==========
        # 将编码器的输出投影到对比学习空间
        self.projector = MLP(
            input_dim=num_channels[-1], 
            hidden_dim=mlp_hidden_dim,
            output_dim=num_channels[-1], 
            use_batch_norm=use_batch_norm
        )

        # ========== 3. Deep SVDD 分类器 ==========
        # 学习一个超球面边界来区分正常和异常
        self.predictor = DeepSVDD(
            input_dim=num_channels[-1] + num_static,
            hidden_dim=mlp_hidden_dim,
            output_dim=num_channels[-1] + num_static,
            use_batch_norm=use_batch_norm
        )

        # ========== 4. 域判别器 ==========
        # 用于域对抗训练，学习域不变特征
        self.discriminator = Discriminator(
            input_dim=num_channels[-1], 
            hidden_dim=mlp_hidden_dim,
            output_dim=1
        )

        # ========== 初始化键编码器 ==========
        # encoder_k 的参数从 encoder_q 复制，不通过梯度更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)     # 复制参数
            param_k.requires_grad = False        # 禁用梯度

        # ========== 5. 创建特征队列 ==========
        # 队列用于存储大量负样本特征，提升对比学习效果
        
        # 源域队列：[特征维度, 队列大小]
        self.register_buffer("queue_s", torch.randn(num_channels[-1], K))
        self.queue_s = nn.functional.normalize(self.queue_s, dim=0)

        # 目标域队列
        self.register_buffer("queue_t", torch.randn(num_channels[-1], K))
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)

        # 队列指针：指向下一个要替换的位置
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        动量更新键编码器 (Momentum Update)
        ===================================
        
        功能：用动量方式更新 encoder_k 的参数
        
        公式：θ_k = m * θ_k + (1-m) * θ_q
        
        其中：
        - θ_k: 键编码器的参数
        - θ_q: 查询编码器的参数
        - m: 动量系数（如 0.999）
        
        优点：
        - 键编码器更新更平滑，提供更稳定的特征
        - 避免键特征变化太快，提升对比学习效果
        """
        # 仅在训练模式下更新
        if self.training:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                # 动量更新：大部分保留旧参数，小部分使用新参数
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        """
        队列更新 (Dequeue and Enqueue)
        ================================
        
        功能：将新的键特征加入队列，移除最旧的特征
        
        队列作用：
        - 存储大量负样本特征（如 98304 个）
        - 比仅使用当前 batch 的负样本效果更好
        - 采用FIFO策略：先进先出
        
        参数:
            keys_s: 源域的键特征 [batch_size, feature_dim]
            keys_t: 目标域的键特征 [batch_size, feature_dim]
        """
        # 仅在训练模式下更新队列
        if self.training:
            batch_size = keys_s.shape[0]
            ptr = int(self.queue_ptr)
            
            # 将新特征放入队列的当前指针位置
            # .T 是转置，因为队列形状是 [feature_dim, queue_size]
            self.queue_s[:, ptr:ptr + batch_size] = keys_s.T
            self.queue_t[:, ptr:ptr + batch_size] = keys_t.T

            # 移动指针到下一个位置（循环）
            ptr = (ptr + batch_size) % self.K
            self.queue_ptr[0] = ptr

    def forward(self, sequence_q_s, sequence_k_s, real_s, static_s, sequence_q_t, sequence_k_t, static_t, alpha,
                seq_src_positive, seq_src_negative, seq_trg_positive, seq_trg_negative):

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        q_s = self.encoder_q(sequence_q_s.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s = nn.functional.normalize(q_s, dim=1)

        q_s_pos = self.encoder_q(seq_src_positive.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s_pos = nn.functional.normalize(q_s_pos, dim=1)

        q_s_neg = self.encoder_q(seq_src_negative.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_s_neg = nn.functional.normalize(q_s_neg, dim=1)

        # Project the query
        p_q_s = self.projector(q_s, None)  # queries: NxC
        p_q_s = nn.functional.normalize(p_q_s, dim=1)

        p_q_s_pos = self.projector(q_s_pos, None)  # queries: NxC
        p_q_s_pos = nn.functional.normalize(p_q_s_pos, dim=1)

        p_q_s_neg = self.projector(q_s_neg, None)  # queries: NxC
        p_q_s_neg = nn.functional.normalize(p_q_s_neg, dim=1)
        # TARGET DATASET query computations

        # compute query features
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output
        q_t = self.encoder_q(sequence_q_t.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t = nn.functional.normalize(q_t, dim=1)

        q_t_pos = self.encoder_q(seq_trg_positive.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t_pos = nn.functional.normalize(q_t_pos, dim=1)

        q_t_neg = self.encoder_q(seq_trg_negative.transpose(1, 2))[:, :, -1]  # queries: NxC
        q_t_neg = nn.functional.normalize(q_t_neg, dim=1)

        # Project the query
        p_q_t = self.projector(q_t, None)  # queries: NxC
        p_q_t = nn.functional.normalize(p_q_t, dim=1)

        p_q_t_pos = self.projector(q_t_pos, None)  # queries: NxC
        p_q_t_pos = nn.functional.normalize(p_q_t_pos, dim=1)

        p_q_t_neg = self.projector(q_t_neg, None)  # queries: NxC
        p_q_t_neg = nn.functional.normalize(p_q_t_neg, dim=1)

        l_queue_s = torch.mm(p_q_s, self.queue_s.clone().detach())
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long).to(device=p_q_s.device)
        l_queue_t = torch.mm(p_q_t, self.queue_t.clone().detach())
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long).to(device=p_q_t.device)

        # DOMAIN DISCRIMINATION Loss
        real_s = real_s.squeeze(1)
        q_n_s = q_s[real_s == 0]
        q_s_reversed = ReverseLayerF.apply(q_s, alpha)

        domain_label_s = torch.ones((len(q_s), 1)).to(device=q_s.device)
        domain_label_t = torch.zeros((len(q_t), 1)).to(device=q_t.device)

        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)

        # q_s_reversed = ReverseLayerF.apply(q_s, alpha)
        q_t_reversed = ReverseLayerF.apply(q_t, alpha)

        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        pred_domain = self.discriminator(q_reversed)

        # SOURCE Prediction task
        # y_s = self.predictor(q_s, static_s)
        pred_s, center, squared_radius = self.predictor(q_s, static_s)

        # dequeue and enqueue
        # self._dequeue_and_enqueue(p_k_s, p_k_t)

        logits_s, logits_t, logits_ts, labels_ts = 0, 0, 0, 0

        return logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts, pred_domain, labels_domain, pred_s, center, squared_radius, \
            q_s, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg, q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg

    def get_encoding(self, sequence, is_target=True):
        # compute the encoding of a sequence (i.e. before projection layer)
        # Input is in the shape N*L*C whereas TCN expects N*C*L
        # Since the tcn_out has the shape N*C_out*L, we will get the last timestep of the output

        # We will use the encoder from a given domain (either source or target)

        q = self.encoder_q(sequence.transpose(1, 2))[:, :, -1]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        return q

    def predict(self, sequence, static, is_target=True, is_eval=False):
        # Get the encoding of a sequence from a given domain
        q = self.get_encoding(sequence['sequence'], is_target=is_target)

        # Make the prediction based on the encoding
        # y = self.predictor(q, static)
        dist, center, squared_radius = self.predictor(q, static)

        return dist  # y


class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm):
        super(DeepSVDD, self).__init__()

        # Encoder layers
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        # )

        # Center and radius of the hypersphere
        self.center = nn.Parameter(torch.Tensor(input_dim))
        self.radius = nn.Parameter(torch.Tensor(1))

        # # Decoder layers (optional)
        # self.decoder = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        #
        # # Batch normalization
        # self.use_batch_norm = use_batch_norm
        # if use_batch_norm:
        #     self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        # nn.init.xavier_uniform_(self.encoder[0].weight)
        # nn.init.constant_(self.encoder[0].bias, 0.0)
        # nn.init.xavier_uniform_(self.encoder[2].weight)
        # nn.init.constant_(self.encoder[2].bias, 0.0)
        #
        # if self.use_batch_norm:
        #     nn.init.constant_(self.batch_norm.weight, 1)
        #     nn.init.constant_(self.batch_norm.bias, 0)

        nn.init.constant_(self.center, 0.0)
        nn.init.constant_(self.radius, 0.0)

    def forward(self, x, statics):
        # # Encode the input
        # encoded = self.encoder(x)
        #
        # # Apply batch normalization if enabled
        # if self.use_batch_norm:
        #     encoded = self.batch_norm(encoded)
        #
        # decoded = self.decoder(encoded)
        # encoded = x.clone()
        # decoded = x.clone()
        tmp_x = x.clone()

        # Calculate the distance to the center
        dist = torch.sum((tmp_x - self.center) ** 2, dim=1)

        # Calculate the squared radius
        squared_radius = self.radius ** 2

        # Return the encoded representation, distance, and squared radius
        return dist, self.center, squared_radius
