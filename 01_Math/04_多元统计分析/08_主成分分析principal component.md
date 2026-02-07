
# 第8章：主成分分析（Principal component）

**1. 主成分分析（PCA）简介**

• 定义与目标
• 数据降维与信息解释

**2. 理论基础**

• 总体主成分
• 主成分的定义与性质
• 总变异性公式
• 标准化变量的主成分
• 标准化方法
• 总方差性质

**3. 样本主成分**

• 样本协方差矩阵计算
• 样本主成分的性质
• 标准化变量的样本主成分

**4. 主成分的选择**

• 累积方差贡献率标准
• 碎石图法（Scree Plot）

**5. 大样本性质**

• 特征值与特征向量的分布性质
• 特征值置信区间公式

---

# **1.Principal Component Analysis (PCA)**

本节概览 PCA 的目的与应用场景（降维与解释），形式化推导见后续小节。

# **2. Population Principal Components**

## 2.1做法

• 给定 $X = \begin{pmatrix}X_1  \\ X_2  \\  \vdots  \\  X_p\end{pmatrix}$，具有均值 $\mu$ 和协方差矩阵 $\Sigma$。
• 特征值：$\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_p$。
• 特征向量：$e_1, e_2, \dots, e_p$。
• 主成分定义为：
$$ Y_i = e_i’X = \sum_{j=1}^p e_{ij}X_j, \quad i = 1, 2, \dots, p $$

按特征值从大到小排序；第一个主成分对应的特征向量为 $e_1$，$Y_1=e_1'X$，以此类推。

• 相关性质：

• $E(Y_i) = e_i’\mu$
• $\text{Var}(Y_i) = e_i’\Sigma e_i = \lambda_i$（每个主成分的方差等于相应特征值）
• $\text{Cov}(Y_i, Y_k) = 0, \quad i \neq k$（主成分两两不相关）
• 总变异：$\text{tr}(\Sigma) = \sum_{i=1}^p \lambda_i = \sum_{i=1}^p \text{Var}(Y_i)$。 实务中通常选取前几个主成分以达到较高的累计方差解释率。

==课后题考到了$\rho_{Y_iZ_j} = w_{ij} \cdot \sqrt{\lambda_i}$,但是上课没教==


**总变异比例**

• 第 $k$ 个主成分的变异比例为：
$$ \frac{\lambda_k}{\sum_{i=1}^p \lambda_i} $$
• 通常前几个主成分能解释大部分变异。

## **2.2Principal Components of Standardized Variables**

基于协方差矩阵与基于相关矩阵的 PCA 结果不同；前者受量纲影响，后者在标准化后消除量纲差异。

• 标准化变量：$Z_j = \frac{X_j - \mu_j}{\sqrt{\sigma_{jj}}}$。
• 标准化后的协方差矩阵为相关矩阵 $\rho$。

• 主成分定义为：
$$ Y_i = e_i’Z = e_i’(X - \mu) $$

## **2.3Sample Principal Components**

用样本协方差矩阵：$S = \frac{1}{n-1}(X - \bar{X})(X - \bar{X})’$。代替$\Sigma$,其余不变

## **2.4决定主成分数量**

• 使用样本协方差矩阵时，总变异比例为：
$$ \frac{\sum_{i=1}^k \lambda_i}{\text{trace}(S)} $$
• 使用相关矩阵时，总变异比例为：
$$ \frac{\sum_{i=1}^k \lambda_i}{p} $$
• **Scree Plot**（碎石图）：绘制特征值 $\lambda_i$ 与索引 $i$ 的关系图，通过观察“肘部”确定主成分数量。

# **大样本性质**

• 当样本量 $n$ 足够大且在正态等正则条件下，样本特征值具备渐近正态性：
• $\sqrt{n}(\hat{\lambda}_i - \lambda_i) \overset{a}{\sim} N\!\big(0,\,2\lambda_i^2\big)$。
• 特征值/特征向量的联合性质依赖分布假设，独立性一般需额外条件。
• $\lambda_i$ 的近似置信区间：
$$ \hat{\lambda}_i \pm Z_{\alpha/2} \sqrt{\tfrac{2\lambda_i^2}{n}} $$
==大样本性质不考==
