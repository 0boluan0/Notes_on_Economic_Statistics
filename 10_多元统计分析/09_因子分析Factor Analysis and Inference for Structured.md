 
可视为PCA的扩展
# 1.introduction
- **目的**: 使用少量 随机因子描述变量间的协方差结构。
- **核心思想**: 如果变量可按相关性分组，则组内变量高度相关，组间变量相关性较低。每组用一个因子表示。
- **与主成分分析（PCA）的关系**: 因子分析可以看作是PCA的扩展。

---

# 2.正交因子模型

## 2.1模型假设

- 给定 $X_{p \times 1}$ 的随机向量，其均值为 $\mu$，协方差矩阵为 $\Sigma$。
- 模型表示为：
  $$X - \mu = LF + \epsilon$$

  其中：
  - $L$: 因子载荷矩阵，维度为 $p \times m$。
  - $F$: 公因子向量，维度为 $m \times 1$，不可观测。
  - $\epsilon$: 特殊因子向量，维度为 $p \times 1$。

### 协方差分解

- 假设条件：
  - $E(F) = 0$, $Cov(F) = I_m$。
  - $E(\epsilon) = 0$, $Cov(\epsilon) = \Psi$，其中 $\Psi$ 是对角矩阵。对角矩阵是因为所有的相关性都被拿走了,体现在了公因子里了,只有独特的部分被保留
  - $Cov(\epsilon, F) = 0$。
- 协方差矩阵可表示为：
  $$\Sigma = LL' + \Psi$$

  其中：
  - $LL'$ 是公因子贡献的协方差。
  - $\Psi$ 是特殊因子的协方差。

### 公共度与特殊方差

- 公共度 $h_i^2$:
  $$h_i^2 = \sum_{j=1}^m l_{ij}^2$$
  表示第 $i$ 个变量由公因子解释的方差部分。
- 特殊方差 $\psi_i$:
  $$\psi_i = \sigma_{ii} - h_i^2$$

  表示未被公因子解释的方差。

## 2.2因子载荷的非唯一性

- 因子载荷矩阵 $L$ 不是唯一的。
- 如果 $T$ 是正交矩阵，则 $L^* = LT$ 也满足模型。

# 3.参数估计方法

## 3.1主成分（PC）法

1. 使用样本协方差矩阵 $S$。
2. 特征值分解:
   $$\Sigma = \sum_{j=1}^p \lambda_j e_j e_j'$$
   - $\lambda_j$: 特征值，$e_j$: 对应特征向量。
3. 近似:
   $$\Sigma \approx LL'$$
   - 保留最大的 $m$ 个特征值。

4. 因子载荷:
   $$L = [\sqrt{\lambda_1}e_1, \sqrt{\lambda_2}e_2, \dots, \sqrt{\lambda_m}e_m]$$

5. 特殊方差:
   $$\psi_i = \sigma_{ii} - \sum_{j=1}^m l_{ij}^2$$
### **例题**

**题目:**
设有三个变量$X_1,X_2,X_3$，其样本协方差矩阵记为
$$
S = \begin{pmatrix}
2   & 0.8 & 0.6  \\ 
0.8 & 1   & 0.5  \\ 
0.6 & 0.5 & 1.5
\end{pmatrix}.
$$
请使用“主成分法”（Principal Components，简称PC法）来做因子分析，假设我们希望提取$m=2$个因子。请完成以下步骤：
1. 对矩阵$S$做特征值分解，得到其特征值$\lambda_j$和对应的特征向量$e_j$。
2. 从大到小选取最大的2个特征值，得到对应的特征向量，并计算因子载荷矩阵
$$L = \big[\sqrt{\lambda_1}  e_1; \sqrt{\lambda_2}  e_2\big].$$
3. 给出近似分解
$$\Sigma \approx LL’,$$
并计算各变量的特殊方差
$$\psi_i = \sigma_{ii} - \sum_{j=1}^m l_{ij}^2,$$
其中$\sigma_{ii}$是$S$（或$\Sigma$）对角线元素，$l_{ij}$是$L$矩阵中元素。

**解题思路与步骤演示**

下面演示如何一步步完成上述题目的解答。为简明起见，以下数值仅为“演示”所用，实际操作中可借助软件或手动计算得到精确结果。

**步骤1：特征值分解**

对样本协方差矩阵
$$
S = \begin{pmatrix}
2   & 0.8 & 0.6  \\ 
0.8 & 1   & 0.5  \\ 
0.6 & 0.5 & 1.5
\end{pmatrix}
$$
做特征值分解，即求解
$$
S e_j = \lambda_j e_j,
$$
得到3个特征值 $\lambda_1,\lambda_2,\lambda_3$及其对应的特征向量 $e_1,e_2,e_3$。假设计算结果如下（数值示例）：
• $\lambda_1 \approx 2.3,\quad e_1 \approx \begin{pmatrix} 0.67  \\  0.51  \\  0.53 \end{pmatrix}$
• $\lambda_2 \approx 1.2,\quad e_2 \approx \begin{pmatrix} 0.45  \\  -0.85  \\  0.27 \end{pmatrix}$
• $\lambda_3 \approx 1.0,\quad e_3 \approx \begin{pmatrix} 0.59  \\  0.12  \\  -0.80 \end{pmatrix}$
**说明：**
1. 特征值从大到小排列（$\lambda_1 \ge \lambda_2 \ge \lambda_3$）。
2. 特征向量 $e_j$ 一般取单位向量（范数为1）。

**步骤2：选取前$m=2$个特征值并计算因子载荷矩阵**

既然题目要求保留$m=2$个因子，则我们从上面选取最大两个特征值 $\lambda_1,\lambda_2$ 及对应特征向量 $e_1,e_2$。然后根据公式
$$
L =
\big[\sqrt{\lambda_1}  e_1,; \sqrt{\lambda_2}  e_2\big],
$$
构造因子载荷矩阵。  
• 首先计算 $\sqrt{\lambda_1} \approx \sqrt{2.3} \approx 1.52$，$\sqrt{\lambda_2} \approx \sqrt{1.2} \approx 1.10$。
• 分别将特征向量 $e_1,e_2$ 乘以对应的特征值的平方根：
$$
\sqrt{\lambda_1}, e_1
= 1.52 \times \begin{pmatrix} 0.67  \\  0.51  \\  0.53 \end{pmatrix}
\approx \begin{pmatrix} 1.02  \\  0.78  \\  0.81 \end{pmatrix},
$$
$$
\sqrt{\lambda_2}, e_2
= 1.10 \times \begin{pmatrix} 0.45 \ -0.85 \ 0.27 \end{pmatrix}
\approx \begin{pmatrix} 0.50 \ -0.94 \ 0.30 \end{pmatrix}.
$$
• 将这两个向量并列起来，就得到因子载荷矩阵$L$
$$
L
= \begin{pmatrix}
1.02 & 0.50  \\ 
0.78 & -0.94  \\ 
0.81 & 0.30
\end{pmatrix}.
$$
**步骤3：近似分解 $\Sigma \approx LL’$ 并计算特殊方差**

1. **近似分解：**
如果我们只用2个因子来近似原协方差矩阵$S$，则可以写成
$$
\Sigma \approx LL’,
$$
在样本层面上通常拿 $S$ 代替 $\Sigma$ 进行估计。
2. **计算特殊方差：**

对每个变量$i$（对应矩阵$S$的第$i$行/列），特殊方差
$$
\psi_i = \sigma_{ii} - \sum_{j=1}^m l_{ij}^2,
$$
其中 $\sigma_{ii}$ 是 $S$ 的对角线元素（即变量 $X_i$ 的方差），$l_{ij}$ 是$L$矩阵中第$i$行、第$j$列的元素。
• 对 $X_1$ 来说：
$$
\sigma_{11} = 2,
\quad
l_{11}^2 + l_{12}^2 = 1.02^2 + 0.50^2 \approx 1.0404 + 0.25 = 1.2904,
$$
因此
$$
\psi_1 = 2 - 1.2904 = 0.7096.
$$

• 对 $X_2$ 来说：
$$
\sigma_{22} = 1,
\quad
l_{21}^2 + l_{22}^2 = 0.78^2 + (-0.94)^2 \approx 0.6084 + 0.8836 = 1.492,
$$
但这里会出现 $l_{21}^2 + l_{22}^2$ 大于 $\sigma_{22}$ 的情况，说明只用2个因子可能无法很好地逼近第二个变量的方差（或许此例中还需检验数据是否有误差、或考虑提取更多因子）。
$$
\psi_2 = 1 - 1.492 = -0.492 \quad (\text{出现负值仅作演示}).
$$
• 对 $X_3$ 来说：
$$
\sigma_{33} = 1.5,
\quad
l_{31}^2 + l_{32}^2 = 0.81^2 + 0.30^2 \approx 0.6561 + 0.09 = 0.7461,
$$
因此
$$
\psi_3 = 1.5 - 0.7461 = 0.7539.
$$

## 3.2极大似然（ML）法

==“大家看一下就好了,考试不会考大家“,彭教授如是说==

- 假设 $X \sim N_p(\mu, \Sigma)$。
- 在约束条件 $L'\Psi^{-1}L$ 为对角矩阵下，估计 $L$ 和 $\Psi$。
- 因子 $j$ 解释的总方差比例:
  $$\text{比例} = \frac{\sum_{i=1}^p l_{ij}^2}{\text{总方差}}$$

## 3.3确定因子数量

==基本没讲,不会考的==

- **残差分析**: 检查 $S - (LL' + \Psi)$。
- **检验方法**:
  - 似然比检验:
    $$-2\ln\Lambda \sim \chi^2$$
  - 信息准则: AIC 和 BIC。
