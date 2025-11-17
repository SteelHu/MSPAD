"""
MSPAD模型实现 (Multi-Scale Domain Adversarial Prototypical Anomaly Detection)
============================================================================
功能：在TCN的多个中间层同时进行域对抗训练，实现多尺度域对齐

核心改进：
1. 多尺度域判别器：在TCN的每个block后添加域判别器
2. 层次化域对齐：从低层到高层逐步对齐域特征
3. 原型网络分类器：使用原型网络替代Deep SVDD
4. 加权多尺度损失：不同层使用不同权重
"""

import sys
import os
sys.path.append("../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from utils.tcn_no_norm import TemporalConvNet, TemporalBlock
from utils.mlp import MLP

# 导入原始DACAD的组件
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'models'))
    from dacad import ReverseLayerF, Discriminator
    # 注意：不再导入DeepSVDD，使用PrototypicalClassifier替代
except ImportError:
    # 如果无法导入，则在这里定义
    class ReverseLayerF(Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)
        
        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha
            return output, None
    
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, output_dim=1):
            super(Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, output_dim),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.model(x)
    
    class DeepSVDD(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, use_batch_norm):
            super(DeepSVDD, self).__init__()
            self.center = nn.Parameter(torch.Tensor(input_dim))
            self.radius = nn.Parameter(torch.Tensor(1))
            self._init_weights()
        
        def _init_weights(self):
            nn.init.constant_(self.center, 0.0)
            nn.init.constant_(self.radius, 0.0)
        
        def forward(self, x, statics):
            tmp_x = x.clone()
            dist = torch.sum((tmp_x - self.center) ** 2, dim=1)
            squared_radius = self.radius ** 2
            return dist, self.center, squared_radius


class PrototypicalClassifier(nn.Module):
    """
    原型网络分类器 (Prototypical Network Classifier)
    ===================================================
    
    功能：学习正常样本的原型，异常样本远离原型
    
    原理：
    1. 通过特征变换网络将输入映射到原型空间
    2. 学习正常样本的原型中心
    3. 计算样本到原型的距离作为异常分数
    4. 正常样本距离小，异常样本距离大
    
    数学描述：
    - 原型：c_normal = mean({f(x_i) : y_i = 0})
    - 距离：d(x, c_normal) = ||f(x) - c_normal||²
    - 异常分数：anomaly_score = d(x, c_normal)
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=None, use_batch_norm=True):
        """
        初始化原型网络分类器
        
        参数:
            input_dim: 输入特征维度（时间序列特征 + 静态特征）
            hidden_dim: 特征变换网络的隐藏层维度
            output_dim: 输出维度（保留兼容性，实际不使用）
            use_batch_norm: 是否使用BatchNorm
        """
        super(PrototypicalClassifier, self).__init__()
        
        # 特征变换网络：将输入特征映射到原型空间
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        self.feature_net = nn.Sequential(*layers)
        
        # 可学习的原型中心（初始化为零向量）
        # 在训练过程中，原型会逐渐学习到正常样本的中心
        self.prototype = nn.Parameter(torch.zeros(hidden_dim))
        
        # 可学习的阈值（用于损失计算）
        # 正常样本应在阈值内，异常样本应在阈值外
        self.threshold = nn.Parameter(torch.tensor(1.0))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        # 使用Xavier初始化特征变换网络
        for m in self.feature_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # 原型中心初始化为零（会在训练中更新）
        nn.init.constant_(self.prototype, 0.0)
        
        # 阈值初始化为1.0
        nn.init.constant_(self.threshold, 1.0)
    
    def forward(self, x, statics):
        """
        前向传播
        
        参数:
            x: 时间序列特征 [batch_size, feature_dim]
            statics: 静态特征 [batch_size, static_dim] 或 None
        
        返回:
            dist: 到原型的距离（异常分数）[batch_size]
            prototype: 原型中心（用于损失计算）
            threshold: 阈值（用于损失计算）
        """
        # 拼接时间序列特征和静态特征
        if statics is not None:
            features = torch.cat([x, statics], dim=1)
        else:
            features = x
        
        # 特征变换：映射到原型空间
        transformed = self.feature_net(features)  # [batch_size, hidden_dim]
        
        # 计算到原型的欧氏距离（平方）
        dist = torch.sum((transformed - self.prototype) ** 2, dim=1)  # [batch_size]
        
        # 返回距离、原型和阈值（保持与DeepSVDD接口兼容）
        return dist, self.prototype, self.threshold
    
    def update_prototype(self, features, labels):
        """
        更新原型中心（可选：使用正常样本的均值更新原型）
        
        参数:
            features: 特征 [batch_size, feature_dim]
            labels: 标签 [batch_size]，0=正常，1=异常
        
        注意：这个方法可以在训练过程中周期性调用，但通常原型会通过梯度下降自动学习
        """
        # 提取正常样本的特征
        normal_mask = (labels.squeeze() == 0)
        if normal_mask.sum() > 0:
            normal_features = features[normal_mask]
            # 计算正常样本的均值作为原型（可选）
            # self.prototype.data = normal_features.mean(dim=0).detach()


class MultiScaleDiscriminator(nn.Module):
    """
    多尺度域判别器
    在TCN的多个中间层进行域判别
    """
    def __init__(self, num_channels_list, hidden_dim=256):
        """
        参数:
            num_channels_list: TCN各层的通道数列表，如 [128, 256, 512]
            hidden_dim: 判别器隐藏层维度
        """
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList()
        
        # 为每个TCN层创建独立的判别器
        for channels in num_channels_list:
            self.discriminators.append(
                nn.Sequential(
                    nn.Linear(channels, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            )
    
    def forward(self, features_list, alpha):
        """
        前向传播
        
        参数:
            features_list: 各层的特征列表 [feat_layer1, feat_layer2, ...]
            alpha: 梯度反转权重
        
        返回:
            outputs: 各层判别器的输出列表
        """
        outputs = []
        for feat, disc in zip(features_list, self.discriminators):
            # 应用梯度反转层
            feat_reversed = ReverseLayerF.apply(feat, alpha)
            outputs.append(disc(feat_reversed))
        return outputs


class TemporalConvNetWithIntermediate(nn.Module):
    """
    支持提取中间层特征的TCN
    继承自TemporalConvNet，但可以返回各层的中间特征
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dilation_factor=2, dropout=0.2):
        super(TemporalConvNetWithIntermediate, self).__init__()
        
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = dilation_factor ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=stride, 
                            dilation=dilation_size, padding=(kernel_size-1) * dilation_size, 
                            dropout=dropout)
            )
    
    def forward(self, x, return_intermediate=False):
        """
        前向传播
        
        参数:
            x: 输入 [N, C, L]
            return_intermediate: 是否返回中间层特征
        
        返回:
            如果return_intermediate=True: (最终输出, 中间特征列表)
            否则: 最终输出
        """
        intermediate_features = []
        
        for layer in self.layers:
            x = layer(x)
            if return_intermediate:
                # 取最后一个时间步的特征
                feat = x[:, :, -1]  # [N, C]
                feat = F.normalize(feat, dim=1)
                intermediate_features.append(feat)
        
        if return_intermediate:
            return x, intermediate_features
        else:
            return x


class MSPAD_NN(nn.Module):
    """
    MSPAD模型实现 (Multi-Scale Domain Adversarial Prototypical Anomaly Detection)
    
    基于原始DACAD，添加多尺度域对抗训练和原型网络分类器
    """
    
    def __init__(self, num_inputs, output_dim, num_channels, num_static, mlp_hidden_dim=256,
                 use_batch_norm=True, num_neighbors=1, kernel_size=2, stride=1, dilation_factor=2,
                 dropout=0.2, K=24576, m=0.999, T=0.07, scale_weights=None, use_layer_mask=None):
        """
        初始化MSPAD模型
        
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
            scale_weights: 多尺度层权重列表，如 [0.1, 0.3, 0.6]，如果为None则自动生成
            use_layer_mask: 层掩码列表，如 [1, 1, 0] 表示使用前两层，如果为None则使用所有层
        """
        super(MSPAD_NN, self).__init__()
        
        self.sigmoid = nn.Sigmoid()
        
        # ========== MoCo 超参数 ==========
        self.K = K
        self.m = m
        self.T = T
        self.num_neighbors = num_neighbors
        self.num_channels = num_channels
        
        # ========== 1. 编码器（支持中间层特征提取的TCN） ==========
        self.encoder_q = TemporalConvNetWithIntermediate(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation_factor=dilation_factor,
            dropout=dropout
        )
        
        self.encoder_k = TemporalConvNetWithIntermediate(
            num_inputs=num_inputs,
            num_channels=num_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation_factor=dilation_factor,
            dropout=dropout
        )
        
        # ========== 2. 特征投影器 ==========
        self.projector = MLP(
            input_dim=num_channels[-1],
            hidden_dim=mlp_hidden_dim,
            output_dim=num_channels[-1],
            use_batch_norm=use_batch_norm
        )
        
        # ========== 3. 原型网络分类器（替换Deep SVDD） ==========
        self.predictor = PrototypicalClassifier(
            input_dim=num_channels[-1] + num_static,
            hidden_dim=mlp_hidden_dim,
            output_dim=num_channels[-1] + num_static,  # 保留兼容性
            use_batch_norm=use_batch_norm
        )
        
        # ========== 4. 单尺度域判别器（保留原有） ==========
        self.discriminator = Discriminator(
            input_dim=num_channels[-1],
            hidden_dim=mlp_hidden_dim,
            output_dim=1
        )
        
        # ========== 5. 多尺度域判别器（新增） ==========
        self.ms_discriminator = MultiScaleDiscriminator(
            num_channels_list=num_channels,
            hidden_dim=mlp_hidden_dim
        )
        
        # ========== 多尺度权重（低层权重小，高层权重大） ==========
        num_layers = len(num_channels)
        if scale_weights is not None:
            # 使用用户指定的权重
            assert len(scale_weights) == num_layers, f"scale_weights长度({len(scale_weights)})必须等于层数({num_layers})"
            self.scale_weights = scale_weights
        else:
            # 根据层数自动生成权重
            if num_layers == 3:
                self.scale_weights = [0.1, 0.3, 0.6]  # 3层TCN
            elif num_layers == 2:
                self.scale_weights = [0.2, 0.8]  # 2层TCN
            else:
                # 自动生成：权重随层数递增
                total_weight = sum(range(1, num_layers + 1))
                self.scale_weights = [i / total_weight for i in range(1, num_layers + 1)]
        
        # ========== 层掩码（控制哪些层参与多尺度域对抗） ==========
        if use_layer_mask is not None:
            assert len(use_layer_mask) == num_layers, f"use_layer_mask长度({len(use_layer_mask)})必须等于层数({num_layers})"
            self.use_layer_mask = use_layer_mask
        else:
            # 默认使用所有层
            self.use_layer_mask = [1] * num_layers
        
        # 归一化权重：只对启用的层归一化
        active_weights = [w for i, w in enumerate(self.scale_weights) if self.use_layer_mask[i]]
        if len(active_weights) > 0 and sum(active_weights) > 0:
            total_active_weight = sum(active_weights)
            normalized_weights = [w / total_active_weight if self.use_layer_mask[i] else 0.0 
                                 for i, w in enumerate(self.scale_weights)]
            self.scale_weights = normalized_weights
        
        # ========== 初始化键编码器 ==========
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # ========== 6. 创建特征队列 ==========
        self.register_buffer("queue_s", torch.randn(num_channels[-1], K))
        self.queue_s = nn.functional.normalize(self.queue_s, dim=0)
        
        self.register_buffer("queue_t", torch.randn(num_channels[-1], K))
        self.queue_t = nn.functional.normalize(self.queue_t, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新键编码器"""
        if self.training:
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys_s, keys_t):
        """队列更新"""
        if self.training:
            batch_size = keys_s.shape[0]
            ptr = int(self.queue_ptr)
            
            self.queue_s[:, ptr:ptr + batch_size] = keys_s.T
            self.queue_t[:, ptr:ptr + batch_size] = keys_t.T
            
            ptr = (ptr + batch_size) % self.K
            self.queue_ptr[0] = ptr
    
    def _get_multiscale_features(self, sequence):
        """
        提取TCN各层的中间特征
        
        参数:
            sequence: 输入序列 [N, L, C]
        
        返回:
            features: 各层特征列表 [feat_layer1, feat_layer2, ...]
        """
        # 转换为TCN输入格式 [N, C, L]
        x = sequence.transpose(1, 2)
        
        # 前向传播并获取中间特征
        _, intermediate_features = self.encoder_q(x, return_intermediate=True)
        
        return intermediate_features
    
    def forward(self, sequence_q_s, sequence_k_s, real_s, static_s, sequence_q_t, sequence_k_t, static_t, alpha,
                seq_src_positive, seq_src_negative, seq_trg_positive, seq_trg_negative):
        """
        前向传播
        
        返回:
            - 原有的所有返回值
            - ms_disc_outputs_s: 源域多尺度判别器输出列表
            - ms_disc_outputs_t: 目标域多尺度判别器输出列表
        """
        
        # ========== 源域特征提取 ==========
        # 获取最终特征
        q_s = self.encoder_q(sequence_q_s.transpose(1, 2))[:, :, -1]
        q_s = F.normalize(q_s, dim=1)
        
        # 获取多尺度特征
        feat_s_list = self._get_multiscale_features(sequence_q_s)
        
        q_s_pos = self.encoder_q(seq_src_positive.transpose(1, 2))[:, :, -1]
        q_s_pos = F.normalize(q_s_pos, dim=1)
        
        q_s_neg = self.encoder_q(seq_src_negative.transpose(1, 2))[:, :, -1]
        q_s_neg = F.normalize(q_s_neg, dim=1)
        
        # 投影
        p_q_s = self.projector(q_s, None)
        p_q_s = F.normalize(p_q_s, dim=1)
        
        p_q_s_pos = self.projector(q_s_pos, None)
        p_q_s_pos = F.normalize(p_q_s_pos, dim=1)
        
        p_q_s_neg = self.projector(q_s_neg, None)
        p_q_s_neg = F.normalize(p_q_s_neg, dim=1)
        
        # ========== 目标域特征提取 ==========
        q_t = self.encoder_q(sequence_q_t.transpose(1, 2))[:, :, -1]
        q_t = F.normalize(q_t, dim=1)
        
        # 获取多尺度特征
        feat_t_list = self._get_multiscale_features(sequence_q_t)
        
        q_t_pos = self.encoder_q(seq_trg_positive.transpose(1, 2))[:, :, -1]
        q_t_pos = F.normalize(q_t_pos, dim=1)
        
        q_t_neg = self.encoder_q(seq_trg_negative.transpose(1, 2))[:, :, -1]
        q_t_neg = F.normalize(q_t_neg, dim=1)
        
        # 投影
        p_q_t = self.projector(q_t, None)
        p_q_t = F.normalize(p_q_t, dim=1)
        
        p_q_t_pos = self.projector(q_t_pos, None)
        p_q_t_pos = F.normalize(p_q_t_pos, dim=1)
        
        p_q_t_neg = self.projector(q_t_neg, None)
        p_q_t_neg = F.normalize(p_q_t_neg, dim=1)
        
        # ========== 队列计算 ==========
        l_queue_s = torch.mm(p_q_s, self.queue_s.clone().detach())
        labels_s = torch.arange(p_q_s.shape[0], dtype=torch.long).to(device=p_q_s.device)
        l_queue_t = torch.mm(p_q_t, self.queue_t.clone().detach())
        labels_t = torch.arange(p_q_t.shape[0], dtype=torch.long).to(device=p_q_t.device)
        
        # ========== 单尺度域判别（原有） ==========
        real_s = real_s.squeeze(1)
        q_n_s = q_s[real_s == 0]
        q_s_reversed = ReverseLayerF.apply(q_s, alpha)
        
        domain_label_s = torch.ones((len(q_s), 1)).to(device=q_s.device)
        domain_label_t = torch.zeros((len(q_t), 1)).to(device=q_t.device)
        
        labels_domain = torch.cat([domain_label_s, domain_label_t], dim=0)
        
        q_t_reversed = ReverseLayerF.apply(q_t, alpha)
        q_reversed = torch.cat([q_s_reversed, q_t_reversed], dim=0)
        pred_domain = self.discriminator(q_reversed)
        
        # ========== 多尺度域判别（新增） ==========
        ms_disc_outputs_s_all = self.ms_discriminator(feat_s_list, alpha)
        ms_disc_outputs_t_all = self.ms_discriminator(feat_t_list, alpha)
        
        # 根据use_layer_mask过滤输出
        ms_disc_outputs_s = [out for i, out in enumerate(ms_disc_outputs_s_all) if self.use_layer_mask[i]]
        ms_disc_outputs_t = [out for i, out in enumerate(ms_disc_outputs_t_all) if self.use_layer_mask[i]]
        
        # ========== 源域预测任务（使用原型网络） ==========
        # 返回：距离、原型中心、阈值（保持接口兼容）
        pred_s, prototype, threshold = self.predictor(q_s, static_s)
        
        logits_s, logits_t, logits_ts, labels_ts = 0, 0, 0, 0
        
        return (logits_s, labels_s, logits_t, labels_t, logits_ts, labels_ts, 
                pred_domain, labels_domain, pred_s, prototype, threshold,
                q_s, q_s_pos, q_s_neg, p_q_s, p_q_s_pos, p_q_s_neg,
                q_t, q_t_pos, q_t_neg, p_q_t, p_q_t_pos, p_q_t_neg,
                ms_disc_outputs_s, ms_disc_outputs_t)  # 新增返回值
    
    def get_encoding(self, sequence, is_target=True):
        """获取编码特征"""
        q = self.encoder_q(sequence.transpose(1, 2))[:, :, -1]
        q = F.normalize(q, dim=1)
        return q
    
    def predict(self, sequence, static, is_target=True, is_eval=False):
        """预测异常分数（使用原型网络）"""
        q = self.get_encoding(sequence['sequence'], is_target=is_target)
        dist, prototype, threshold = self.predictor(q, static)
        return dist  # 返回到原型的距离作为异常分数

