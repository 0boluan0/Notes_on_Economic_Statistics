# 1. 第9章：因子分析（Factor Analysis）

>[!summary] 本章主线
> 因子分析可以看成 PCA 之后更进一步的问题：不仅想降维，还想用少数不可观测的公共因子解释变量之间的协方差结构。

## 1.1. 引言

因子分析的目的：用少量随机因子描述多个变量之间的协方差结构。

核心思想：

- 如果变量可以按相关性分组，组内变量高度相关，组间变量相关较低；
- 每个组可由一个公共因子代表；
- 每个变量还保留自己独有的特殊部分。

## 1.2. 正交因子模型（Orthogonal Factor Model）

### 1.2.1. 模型设定

给定 $p\times1$ 随机向量 $X$，均值为 $\mu$，协方差矩阵为 $\Sigma$。

模型写作
$$
X-\mu=LF+\epsilon.
$$

其中：

- $L$ 是 $p\times m$ 因子载荷矩阵；
- $F$ 是 $m\times1$ 公共因子向量，不可观测；
- $\epsilon$ 是 $p\times1$ 特殊因子向量。

### 1.2.2. 假设条件

$$
E(F)=0,\qquad \operatorname{Cov}(F)=I_m.
$$

$$
E(\epsilon)=0,\qquad \operatorname{Cov}(\epsilon)=\Psi.
$$

其中 $\Psi$ 是对角矩阵。它为对角矩阵的原因是：变量之间的共同相关性已经由公共因子 $F$ 解释，剩下的是每个变量独有的特殊部分。

此外：
$$
\operatorname{Cov}(\epsilon,F)=0.
$$

### 1.2.3. 协方差分解

由模型可得
$$
\Sigma=LL'+\Psi.
$$

其中：

- $LL'$ 是公共因子贡献的协方差；
- $\Psi$ 是特殊因子的协方差。

>[!note] 复习核心
> 因子分析最重要的公式就是 $\Sigma=LL'+\Psi$。PCA 没有这个“公共部分 + 特殊部分”的模型分解。

## 1.3. 公共度与特殊方差

第 $i$ 个变量的公共度为
$$
h_i^2=\sum_{j=1}^m l_{ij}^2.
$$

特殊方差为
$$
\psi_i=\sigma_{ii}-h_i^2.
$$

所以
$$
\sigma_{ii}=h_i^2+\psi_i.
$$

>[!warning] Heywood case
> 如果估计出 $\hat\psi_i<0$，通常说明模型或因子数设定有问题，不应机械接受。

## 1.4. 因子载荷的非唯一性

因子载荷矩阵 $L$ 不是唯一的。

如果 $T$ 是正交矩阵，则
$$
L^*=LT
$$
也满足同样的协方差结构，因为
$$
(LT)(LT)'=LTT'L'=LL'.
$$

因此因子分析常配合旋转，让载荷矩阵更容易解释。

## 1.5. 参数估计方法

### 1.5.1. 主成分法（Principal Component Method）

给定样本协方差矩阵 $S$，先做特征值分解：
$$
S=\sum_{j=1}^p\lambda_j e_je_j'.
$$

保留最大的 $m$ 个特征值，近似为
$$
\Sigma\approx LL'+\Psi.
$$

因子载荷矩阵估计为
$$
\hat L=
\left[
\sqrt{\lambda_1}e_1,\sqrt{\lambda_2}e_2,\ldots,\sqrt{\lambda_m}e_m
\right].
$$

特殊方差估计为
$$
\hat\psi_i=s_{ii}-\sum_{j=1}^m\hat l_{ij}^2.
$$

>[!example] 做题顺序
> 先求特征值和特征向量，再取前 $m$ 个构造 $\hat L$，最后逐个变量算公共度和特殊方差。

### 1.5.2. 极大似然法（Maximum Likelihood Method）

若假设
$$
X\sim N_p(\mu,\Sigma),
$$
可在约束 $L'\Psi^{-1}L$ 为对角矩阵下估计 $L$ 和 $\Psi$。

>[!note] 课堂提示
> 旧笔记标注“考试不会考”。本轮整理只保留识别信息，不展开推导。

## 1.6. 因子数量选择

常见依据：

1. 碎石图。
2. 累计方差解释率。
3. 残差矩阵 $S-(LL'+\Psi)$。
4. 信息准则，如 AIC 和 BIC。
5. 似然比检验。

>[!warning] 解释优先
> 因子数量不是越多越好；因子数量过多会失去“用少数潜在维度解释结构”的意义。

## 1.7. 关联卡片

- [[Factor Analysis]]
- [[Factor Analysis PC Method]]
- [[Factor Loadings]]
- [[Communality]]
- [[Specific Variance]]
- [[PCA vs Factor Analysis]]
