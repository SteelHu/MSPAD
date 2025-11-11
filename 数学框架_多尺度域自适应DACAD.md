# 多尺度域自适应DACAD模型数学框架

## 目录

1. [模型概述](#模型概述)
2. [符号定义](#符号定义)
3. [网络架构](#网络架构)
4. [前向传播过程](#前向传播过程)
5. [损失函数](#损失函数)
6. [优化目标](#优化目标)
7. [训练算法](#训练算法)
8. [关键创新点](#关键创新点)

---

## 模型概述

多尺度域自适应DACAD（Multi-Scale Domain Adversarial DACAD，MSDA-DACAD）是一个用于跨域异常检测的深度学习模型。该模型在原始DACAD基础上引入了**多尺度域对抗训练**机制，通过在时间卷积网络（TCN）的多个中间层同时进行域判别，实现层次化的域特征对齐。

### 核心改进

1. **多尺度域对抗训练**：在TCN的每个block后添加域判别器，实现从低层到高层的层次化域对齐
2. **原型网络分类器**：使用原型网络（Prototypical Network）替代Deep SVDD，学习正常样本的原型表示
3. **加权多尺度损失**：不同层使用不同权重，低层权重小，高层权重大

---

## 符号定义

### 输入数据

- $\mathcal{D}_s = \{(\mathbf{x}_s^{(i)}, y_s^{(i)})\}_{i=1}^{N_s}$：源域数据集，包含标注数据
- $\mathcal{D}_t = \{\mathbf{x}_t^{(i)}\}_{i=1}^{N_t}$：目标域数据集，无标注数据
- $\mathbf{x}_s^{(i)} \in \mathbb{R}^{L \times C}$：源域时间序列样本，长度为$L$，特征维度为$C$
- $\mathbf{x}_t^{(i)} \in \mathbb{R}^{L \times C}$：目标域时间序列样本
- $y_s^{(i)} \in \{0, 1\}$：源域标签，$0$表示正常，$1$表示异常
- $\mathbf{s}_s^{(i)} \in \mathbb{R}^{D_s}$：源域静态特征（可选）
- $\mathbf{s}_t^{(i)} \in \mathbb{R}^{D_t}$：目标域静态特征（可选）

### 网络组件

- $E_q(\cdot)$：查询编码器（Query Encoder），可训练的TCN
- $E_k(\cdot)$：键编码器（Key Encoder），动量更新的TCN
- $P(\cdot)$：特征投影器（Projector），MLP网络
- $F(\cdot)$：原型网络分类器（Prototypical Classifier）
- $D(\cdot)$：单尺度域判别器（Single-Scale Discriminator）
- $D_{ms}^{(l)}(\cdot)$：多尺度域判别器（Multi-Scale Discriminator），第$l$层
- $\mathbf{Q}_s \in \mathbb{R}^{d \times K}$：源域特征队列
- $\mathbf{Q}_t \in \mathbb{R}^{d \times K}$：目标域特征队列

### 超参数

- $K$：MoCo队列大小（负样本数量）
- $m$：动量更新系数（通常$m=0.999$）
- $T$：温度参数（通常$T=0.07$）
- $\alpha$：梯度反转权重（域对抗训练）
- $\lambda_{disc}$：单尺度域对抗损失权重
- $\lambda_{ms}$：多尺度域对抗损失权重
- $\lambda_{pred}$：原型网络分类损失权重
- $\lambda_{sup}$：源域监督对比损失权重
- $\lambda_{inj}$：目标域注入对比损失权重
- $\gamma_l$：第$l$层的多尺度权重

### 中间变量

- $\mathbf{q}_s$：源域查询特征（Query Features）
- $\mathbf{k}_s$：源域键特征（Key Features）
- $\mathbf{p}_s$：源域投影特征（Projected Features）
- $\mathbf{q}_t$：目标域查询特征
- $\mathbf{k}_t$：目标域键特征
- $\mathbf{p}_t$：目标域投影特征
- $\mathbf{f}_s^{(l)}$：源域第$l$层中间特征
- $\mathbf{f}_t^{(l)}$：目标域第$l$层中间特征
- $\mathbf{c}$：原型中心（Prototype Center）
- $\tau$：可学习的阈值参数

---

## 网络架构

### 1. 时间卷积网络（TCN）编码器

TCN编码器由$L$个TemporalBlock组成，每个block的输出维度为$d_l$（$l=1,2,\ldots,L$）。

**第$l$层的TCN Block：**

$$
\mathbf{h}_s^{(l)} = \text{TemporalBlock}^{(l)}(\mathbf{h}_s^{(l-1)})
$$

其中：
- $\mathbf{h}_s^{(0)} = \mathbf{x}_s$（输入序列）
- $\mathbf{h}_s^{(l)} \in \mathbb{R}^{N \times d_l \times L'}$（$L'$为经过卷积后的序列长度）

**最终特征提取：**

$$
\mathbf{q}_s = \text{Normalize}(E_q(\mathbf{x}_s)[:, :, -1]) \in \mathbb{R}^{N \times d_L}
$$

即取最后一个时间步的特征并归一化。

### 2. 多尺度特征提取

对于第$l$层（$l=1,2,\ldots,L$），提取中间特征：

$$
\mathbf{f}_s^{(l)} = \text{Normalize}(\mathbf{h}_s^{(l)}[:, :, -1]) \in \mathbb{R}^{N \times d_l}
$$

### 3. 特征投影器

将编码特征投影到对比学习空间：

$$
\mathbf{p}_s = \text{Normalize}(P(\mathbf{q}_s)) \in \mathbb{R}^{N \times d_L}
$$

### 4. 原型网络分类器

原型网络将特征映射到原型空间并计算到原型的距离：

$$
\mathbf{z}_s = F_{\text{feat}}(\text{Concat}[\mathbf{q}_s, \mathbf{s}_s]) \in \mathbb{R}^{N \times d_h}
$$

$$
d_s^{(i)} = \|\mathbf{z}_s^{(i)} - \mathbf{c}\|^2
$$

其中：
- $F_{\text{feat}}(\cdot)$：特征变换网络（MLP）
- $\mathbf{c} \in \mathbb{R}^{d_h}$：可学习的原型中心
- $d_s^{(i)}$：第$i$个样本到原型的距离（异常分数）

### 5. 域判别器

**单尺度域判别器（最终层）：**

$$
\hat{y}_d = D(\text{ReverseLayer}(\mathbf{q}_s, \alpha), \text{ReverseLayer}(\mathbf{q}_t, \alpha))
$$

**多尺度域判别器（第$l$层）：**

$$
\hat{y}_d^{(l)} = D_{ms}^{(l)}(\text{ReverseLayer}(\mathbf{f}_s^{(l)}, \alpha), \text{ReverseLayer}(\mathbf{f}_t^{(l)}, \alpha))
$$

其中$\text{ReverseLayer}(\mathbf{x}, \alpha)$表示梯度反转层，在前向传播时不变，反向传播时梯度乘以$-\alpha$。

---

## 前向传播过程

### 源域前向传播

1. **编码阶段：**
   $$
   \mathbf{q}_s = E_q(\mathbf{x}_s)[:, :, -1], \quad \mathbf{q}_s = \frac{\mathbf{q}_s}{\|\mathbf{q}_s\|_2}
   $$

2. **多尺度特征提取：**
   $$
   \mathbf{f}_s^{(l)} = \text{ExtractIntermediate}(\mathbf{x}_s, l), \quad l=1,2,\ldots,L
   $$

3. **投影阶段：**
   $$
   \mathbf{p}_s = P(\mathbf{q}_s), \quad \mathbf{p}_s = \frac{\mathbf{p}_s}{\|\mathbf{p}_s\|_2}
   $$

4. **原型网络预测：**
   $$
   d_s, \mathbf{c}, \tau = F(\mathbf{q}_s, \mathbf{s}_s)
   $$

5. **域判别：**
   $$
   \hat{y}_d = D(\text{ReverseLayer}(\mathbf{q}_s, \alpha))
   $$
   $$
   \hat{y}_d^{(l)} = D_{ms}^{(l)}(\text{ReverseLayer}(\mathbf{f}_s^{(l)}, \alpha)), \quad l=1,2,\ldots,L
   $$

### 目标域前向传播

目标域的前向传播过程与源域类似，但无标签信息：

$$
\mathbf{q}_t = E_q(\mathbf{x}_t)[:, :, -1], \quad \mathbf{q}_t = \frac{\mathbf{q}_t}{\|\mathbf{q}_t\|_2}
$$

$$
\mathbf{f}_t^{(l)} = \text{ExtractIntermediate}(\mathbf{x}_t, l), \quad l=1,2,\ldots,L
$$

$$
\mathbf{p}_t = P(\mathbf{q}_t), \quad \mathbf{p}_t = \frac{\mathbf{p}_t}{\|\mathbf{p}_t\|_2}
$$

### MoCo机制

**动量更新键编码器：**

$$
\theta_k \leftarrow m \cdot \theta_k + (1-m) \cdot \theta_q
$$

**队列更新：**

$$
\mathbf{Q}_s[:, \text{ptr}:\text{ptr}+B] = \mathbf{k}_s^T
$$

$$
\mathbf{Q}_t[:, \text{ptr}:\text{ptr}+B] = \mathbf{k}_t^T
$$

其中$\mathbf{k}_s = E_k(\mathbf{x}_s)[:, :, -1]$，$\mathbf{k}_t = E_k(\mathbf{x}_t)[:, :, -1]$。

---

## 损失函数

### 1. 单尺度域对抗损失

单尺度域对抗损失在最终层特征上进行域判别：

$$
\mathcal{L}_{disc} = \frac{1}{N_s + N_t} \sum_{i=1}^{N_s} \text{BCE}(D(\text{ReverseLayer}(\mathbf{q}_s^{(i)}, \alpha)), 1) + \sum_{j=1}^{N_t} \text{BCE}(D(\text{ReverseLayer}(\mathbf{q}_t^{(j)}, \alpha)), 0)
$$

其中$\text{BCE}(p, y) = -y\log(p) - (1-y)\log(1-p)$是二元交叉熵损失。

**数学形式：**

$$
\mathcal{L}_{disc} = -\frac{1}{N_s + N_t} \left[ \sum_{i=1}^{N_s} \log(D(\text{ReverseLayer}(\mathbf{q}_s^{(i)}, \alpha))) + \sum_{j=1}^{N_t} \log(1-D(\text{ReverseLayer}(\mathbf{q}_t^{(j)}, \alpha))) \right]
$$

### 2. 多尺度域对抗损失

多尺度域对抗损失在TCN的多个中间层同时进行域判别：

$$
\mathcal{L}_{ms} = \sum_{l=1}^{L} \gamma_l \cdot \mathcal{L}_{disc}^{(l)}
$$

其中第$l$层的域对抗损失为：

$$
\mathcal{L}_{disc}^{(l)} = -\frac{1}{N_s + N_t} \left[ \sum_{i=1}^{N_s} \log(D_{ms}^{(l)}(\text{ReverseLayer}(\mathbf{f}_s^{(l,i)}, \alpha))) + \sum_{j=1}^{N_t} \log(1-D_{ms}^{(l)}(\text{ReverseLayer}(\mathbf{f}_t^{(l,j)}, \alpha))) \right]
$$

**多尺度权重设置：**

对于$L=3$层TCN，权重设置为：
$$
\gamma_1 = 0.1, \quad \gamma_2 = 0.3, \quad \gamma_3 = 0.6
$$

对于$L=2$层TCN：
$$
\gamma_1 = 0.2, \quad \gamma_2 = 0.8
$$

对于一般情况，权重随层数递增：
$$
\gamma_l = \frac{l}{\sum_{k=1}^{L} k} = \frac{2l}{L(L+1)}
$$

### 3. 原型网络分类损失

原型网络损失确保正常样本靠近原型，异常样本远离原型：

$$
\mathcal{L}_{pred} = \mathcal{L}_{normal} + \mathcal{L}_{anomal}
$$

**正常样本损失：**

$$
\mathcal{L}_{normal} = w_0 \cdot \frac{1}{|\mathcal{N}_s|} \sum_{i \in \mathcal{N}_s} \max(0, d_s^{(i)} - \tau)
$$

其中$\mathcal{N}_s = \{i : y_s^{(i)} = 0\}$是正常样本索引集合。

**异常样本损失：**

$$
\mathcal{L}_{anomal} = w_1 \cdot \frac{1}{|\mathcal{A}_s|} \sum_{i \in \mathcal{A}_s} \max(0, \tau + \text{margin} - d_s^{(i)})
$$

其中$\mathcal{A}_s = \{i : y_s^{(i)} = 1\}$是异常样本索引集合，$\text{margin}$是间隔参数（通常为1.0）。

**类别权重：**

$$
w_0 = \frac{1}{2p}, \quad w_1 = w_0 \cdot \frac{p}{1-p}
$$

其中$p = 1 - \frac{1}{\text{weight\_ratio} + 1}$是异常样本比例。

### 4. 源域监督对比损失

源域监督对比损失使用三元组损失（Triplet Loss），拉近同类样本，推远不同类样本：

$$
\mathcal{L}_{sup} = \frac{1}{|\mathcal{N}_s|} \sum_{i \in \mathcal{N}_s} \max(0, d_{pos}^{(i)} - d_{neg}^{(i)} + \text{margin})
$$

其中：
- $d_{pos}^{(i)} = \|\mathbf{p}_s^{(i)} - \mathbf{p}_{s,pos}^{(i)}\|_2^2$：锚点到正样本的距离
- $d_{neg}^{(i)} = \min_j \|\mathbf{p}_s^{(i)} - \mathbf{p}_{s,neg}^{(j)}\|_2^2$：锚点到负样本的最小距离
- $\mathbf{p}_{s,pos}^{(i)}$：与$\mathbf{p}_s^{(i)}$同类的正样本（通过数据增强生成）
- $\mathbf{p}_{s,neg}^{(j)}$：与$\mathbf{p}_s^{(i)}$不同类的负样本（异常样本或注入的异常）

**详细形式：**

$$
\mathcal{L}_{sup} = \frac{1}{|\mathcal{N}_s|} \sum_{i \in \mathcal{N}_s} \max\left(0, \|\mathbf{p}_s^{(i)} - \mathbf{p}_{s,pos}^{(i)}\|_2^2 - \min_{j: y_s^{(j)}=1} \|\mathbf{p}_s^{(i)} - \mathbf{p}_{s,neg}^{(j)}\|_2^2 + 2\right)
$$

### 5. 目标域异常注入对比损失

目标域异常注入对比损失假设所有目标域样本都是正常的，使用注入的异常作为负样本：

$$
\mathcal{L}_{inj} = \frac{1}{N_t} \sum_{i=1}^{N_t} \max(0, d_{pos}^{(i)} - d_{neg}^{(i)} + \text{margin})
$$

其中：
- $d_{pos}^{(i)} = \|\mathbf{p}_t^{(i)} - \mathbf{p}_{t,pos}^{(i)}\|_2^2$：目标域样本到正样本的距离
- $d_{neg}^{(i)} = \|\mathbf{p}_t^{(i)} - \mathbf{p}_{t,neg}^{(i)}\|_2^2$：目标域样本到注入异常的距离

### 6. 平衡损失

为了防止目标域对比损失过小，添加平衡损失：

$$
\mathcal{L}_{balance} = \max(0, \mathcal{L}_{sup} - \mathcal{L}_{inj})
$$

---

## 优化目标

### 总损失函数

模型的总体优化目标为：

$$
\mathcal{L}_{total} = \lambda_{disc} \cdot \mathcal{L}_{disc} + \lambda_{ms} \cdot \mathcal{L}_{ms} + \lambda_{pred} \cdot \mathcal{L}_{pred} + \lambda_{sup} \cdot \mathcal{L}_{sup} + \lambda_{inj} \cdot \mathcal{L}_{inj} + \mathcal{L}_{balance}
$$

**默认权重设置：**

- $\lambda_{disc} = 0.5$：单尺度域对抗损失权重
- $\lambda_{ms} = 0.3$：多尺度域对抗损失权重
- $\lambda_{pred} = 1.0$：原型网络分类损失权重
- $\lambda_{sup} = 0.1$：源域监督对比损失权重
- $\lambda_{inj} = 0.1$：目标域注入对比损失权重

### 优化问题

**编码器和投影器的优化目标：**

$$
\min_{E_q, P, F} \lambda_{pred} \cdot \mathcal{L}_{pred} + \lambda_{sup} \cdot \mathcal{L}_{sup} + \lambda_{inj} \cdot \mathcal{L}_{inj} - \lambda_{disc} \cdot \mathcal{L}_{disc} - \lambda_{ms} \cdot \mathcal{L}_{ms}
$$

注意：域对抗损失前有负号，因为编码器要**欺骗**判别器，使源域和目标域特征不可区分。

**域判别器的优化目标：**

$$
\min_{D, \{D_{ms}^{(l)}\}} \lambda_{disc} \cdot \mathcal{L}_{disc} + \lambda_{ms} \cdot \mathcal{L}_{ms}
$$

域判别器要**正确区分**源域和目标域。

### 对抗训练机制

通过梯度反转层（Gradient Reversal Layer）实现对抗训练：

**前向传播：**
$$
\text{ReverseLayer}(\mathbf{x}, \alpha) = \mathbf{x}
$$

**反向传播：**
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = -\alpha \cdot \frac{\partial \mathcal{L}}{\partial \text{ReverseLayer}(\mathbf{x}, \alpha)}
$$

这样，编码器在最小化域判别损失时，实际上是在最大化域判别损失（通过负梯度），从而实现域对齐。

---

## 训练算法

### 算法流程

**输入：**
- 源域数据集$\mathcal{D}_s$，目标域数据集$\mathcal{D}_t$
- 超参数：$\lambda_{disc}, \lambda_{ms}, \lambda_{pred}, \lambda_{sup}, \lambda_{inj}, m, K, T, \alpha$

**初始化：**
1. 初始化编码器$E_q$、投影器$P$、原型网络$F$、域判别器$D$和$\{D_{ms}^{(l)}\}$
2. 初始化键编码器：$\theta_k \leftarrow \theta_q$
3. 初始化特征队列$\mathbf{Q}_s$和$\mathbf{Q}_t$（随机初始化）

**训练循环：**

对于每个epoch：

1. **采样批次：**
   - 从$\mathcal{D}_s$采样批次$(\mathbf{X}_s, \mathbf{Y}_s)$
   - 从$\mathcal{D}_t$采样批次$\mathbf{X}_t$

2. **数据增强：**
   - 生成正样本：$\mathbf{X}_{s,pos}, \mathbf{X}_{t,pos}$（通过时间窗口滑动）
   - 生成负样本：$\mathbf{X}_{s,neg}, \mathbf{X}_{t,neg}$（通过异常注入或采样异常样本）

3. **前向传播：**
   - 计算$\mathbf{q}_s, \mathbf{q}_t, \mathbf{f}_s^{(l)}, \mathbf{f}_t^{(l)}$（$l=1,\ldots,L$）
   - 计算$\mathbf{p}_s, \mathbf{p}_t$
   - 计算原型网络输出$d_s, \mathbf{c}, \tau$
   - 计算域判别器输出$\hat{y}_d, \hat{y}_d^{(l)}$

4. **计算损失：**
   $$
   \mathcal{L}_{total} = \lambda_{disc} \cdot \mathcal{L}_{disc} + \lambda_{ms} \cdot \mathcal{L}_{ms} + \lambda_{pred} \cdot \mathcal{L}_{pred} + \lambda_{sup} \cdot \mathcal{L}_{sup} + \lambda_{inj} \cdot \mathcal{L}_{inj} + \mathcal{L}_{balance}
   $$

5. **反向传播：**
   - 更新编码器、投影器、原型网络：$\theta_q, \theta_P, \theta_F \leftarrow \theta_q - \eta \nabla_{\theta_q, \theta_P, \theta_F} \mathcal{L}_{total}$
   - 更新域判别器：$\theta_D, \theta_{D_{ms}} \leftarrow \theta_D - \eta \nabla_{\theta_D, \theta_{D_{ms}}} (\lambda_{disc} \mathcal{L}_{disc} + \lambda_{ms} \mathcal{L}_{ms})$

6. **动量更新：**
   $$
   \theta_k \leftarrow m \cdot \theta_k + (1-m) \cdot \theta_q
   $$

7. **队列更新：**
   - 计算键特征：$\mathbf{k}_s = E_k(\mathbf{X}_s)[:, :, -1]$, $\mathbf{k}_t = E_k(\mathbf{X}_t)[:, :, -1]$
   - 更新队列：$\mathbf{Q}_s[:, \text{ptr}:\text{ptr}+B] = \mathbf{k}_s^T$, $\mathbf{Q}_t[:, \text{ptr}:\text{ptr}+B] = \mathbf{k}_t^T$

### 伪代码

```
Algorithm: MSDA-DACAD Training
Input: D_s, D_t, hyperparameters
Initialize: E_q, E_k, P, F, D, {D_ms^(l)}, Q_s, Q_t

for epoch = 1 to num_epochs do
    for batch (X_s, Y_s) in D_s, X_t in D_t do
        # 数据增强
        X_s_pos, X_s_neg = augment(X_s, Y_s)
        X_t_pos, X_t_neg = augment(X_t)
        
        # 前向传播
        q_s = E_q(X_s)[:, :, -1]
        q_t = E_q(X_t)[:, :, -1]
        f_s^(l) = extract_intermediate(X_s, l) for l=1..L
        f_t^(l) = extract_intermediate(X_t, l) for l=1..L
        
        p_s = P(q_s)
        p_t = P(q_t)
        
        d_s, c, τ = F(q_s, s_s)
        
        # 域判别
        y_d = D(ReverseLayer(q_s, α), ReverseLayer(q_t, α))
        y_d^(l) = D_ms^(l)(ReverseLayer(f_s^(l), α), ReverseLayer(f_t^(l), α)) for l=1..L
        
        # 计算损失
        L_disc = BCE(y_d, domain_labels)
        L_ms = Σ_l γ_l · BCE(y_d^(l), domain_labels)
        L_pred = prototypical_loss(d_s, Y_s, c, τ)
        L_sup = triplet_loss(p_s, p_s_pos, p_s_neg, Y_s)
        L_inj = triplet_loss(p_t, p_t_pos, p_t_neg)
        L_balance = max(0, L_sup - L_inj)
        
        L_total = λ_disc·L_disc + λ_ms·L_ms + λ_pred·L_pred + 
                  λ_sup·L_sup + λ_inj·L_inj + L_balance
        
        # 反向传播
        ∇_E_q,P,F L_total → update E_q, P, F
        ∇_D,{D_ms^(l)} (λ_disc·L_disc + λ_ms·L_ms) → update D, {D_ms^(l)}
        
        # 动量更新
        θ_k ← m·θ_k + (1-m)·θ_q
        
        # 队列更新
        k_s = E_k(X_s)[:, :, -1]
        k_t = E_k(X_t)[:, :, -1]
        Q_s[:, ptr:ptr+B] = k_s^T
        Q_t[:, ptr:ptr+B] = k_t^T
        ptr = (ptr + B) mod K
    end for
end for
```

---

## 关键创新点

### 1. 多尺度域对抗训练

**动机：** 不同层次的TCN特征捕获不同抽象级别的信息。低层特征捕获局部模式（如短期依赖），高层特征捕获全局模式（如长期依赖）。域差异可能在不同层次都有体现。

**实现：** 在TCN的每个block后添加独立的域判别器，同时进行域对齐。

**数学表达：**

$$
\mathcal{L}_{ms} = \sum_{l=1}^{L} \gamma_l \cdot \mathcal{L}_{disc}^{(l)}
$$

其中权重$\gamma_l$随层数递增，强调高层特征的对齐。

**优势：**
- 层次化域对齐，从局部到全局逐步对齐
- 更鲁棒的特征表示
- 减少不同抽象层次的域差异

### 2. 原型网络分类器

**动机：** Deep SVDD需要学习一个固定的超球中心，而原型网络通过可学习的原型中心更灵活地适应数据分布。

**数学表达：**

$$
d(\mathbf{x}) = \|F_{\text{feat}}(\text{Concat}[\mathbf{q}, \mathbf{s}]) - \mathbf{c}\|^2
$$

**优势：**
- 原型中心可学习，适应数据分布
- 损失函数更简单，训练更稳定
- 异常分数直接为到原型的距离，解释性强

### 3. 加权多尺度损失

**权重设计原则：**
- 低层权重小：低层特征更通用，域差异较小
- 高层权重大：高层特征更任务相关，域差异较大

**数学表达：**

$$
\gamma_l = \frac{2l}{L(L+1)}
$$

对于3层TCN：$\gamma_1 = 0.1, \gamma_2 = 0.3, \gamma_3 = 0.6$

**优势：**
- 平衡不同层次的重要性
- 避免低层特征过度对齐导致的信息丢失
- 强调高层语义特征的对齐

### 4. MoCo机制

**动量对比学习：**
- 使用动量更新的键编码器生成稳定的负样本
- 队列机制存储大量负样本，提升对比学习效果

**数学表达：**

$$
\theta_k \leftarrow m \cdot \theta_k + (1-m) \cdot \theta_q
$$

**优势：**
- 稳定的负样本表示
- 大批量对比学习效果
- 减少内存占用（队列替代大批量）

---

## 总结

多尺度域自适应DACAD模型通过以下机制实现跨域异常检测：

1. **多尺度域对抗训练**：在TCN的多个层次同时进行域对齐，实现层次化特征对齐
2. **原型网络分类**：学习正常样本的原型表示，异常样本远离原型
3. **对比学习**：使用MoCo框架和三元组损失学习判别性特征
4. **加权损失组合**：平衡不同损失项的重要性，实现最优性能

该模型在保持原始DACAD优势的基础上，通过多尺度域对抗训练显著提升了跨域异常检测的性能。

---

## 参考文献

1. DACAD: Domain Adaptation for Cross-Domain Anomaly Detection
2. MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
3. Prototypical Networks for Few-shot Learning
4. Temporal Convolutional Networks for Action Segmentation and Detection

