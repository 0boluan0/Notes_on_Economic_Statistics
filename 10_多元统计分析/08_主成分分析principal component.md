
# chapter 8 principal component

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

简介略

# **2. Population Principal Components**

## 2.1做法

• 给定 $X = \begin{pmatrix}X_1  \\ X_2  \\  \vdots  \\  X_p\end{pmatrix}$，具有均值 $\mu$ 和协方差矩阵 $\Sigma$。
• 特征值：$\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_p$。
• 特征向量：$e_1, e_2, \dots, e_p$。
• 主成分定义为：
$$ Y_i = e_i’X = \sum_{j=1}^p e_{ij}X_j, \quad i = 1, 2, \dots, p $$

就是按从大到小的顺序对特征值排序,然后第一个特征值对应的特征向量乘以x1,x2……xn得到第一个主成分Y1.以此类推.

• 相关性质：

• $E(Y_i) = e_i’\mu$
• $\text{Var}(Y_i) = e_i’\Sigma e_i = \lambda_i$==每个主成分对应的var就是对应的特征值==
• $\text{Cov}(Y_i, Y_k) = 0, \quad \text{for } i \neq k$所有主成分正交
• 总变异：$\text{tr}(\Sigma) = \sum_{i=1}^p \lambda_i = \sum_{i=1}^p \text{Var}(Y_i)$意思是所有的变异都能被新的变量解释.但是实际上还是选尽可能少的.

==课后题考到了$\rho_{Y_iZ_j} = w_{ij} \cdot \sqrt{\lambda_i}$,但是上课没教==



**总变异比例**

• 第 $k$ 个主成分的变异比例为：
$$ \frac{\lambda_k}{\sum_{i=1}^p \lambda_i} $$
• 通常前几个主成分能解释大部分变异。

## **2.2Principal Components of Standardized Variables**

用协方差矩阵算和用相关系数矩阵算出来的不一样.效果也不一样.

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

**大样本性质**

  

• 当样本量 $n$ 足够大时，特征值和特征向量的分布性质：

• $\sqrt{n}(\lambda_i - \hat{\lambda}_i) \sim N(0, 2\lambda_i^2)$。

• 特征值和特征向量之间独立分布。

• 特征值 $\lambda_i$ 的置信区间：

$$ \hat{\lambda}_i \pm Z_{\alpha/2} \sqrt{\frac{2\lambda_i^2}{n}} $$
