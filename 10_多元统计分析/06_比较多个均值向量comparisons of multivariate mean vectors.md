
# 1.比较多个均值

## 1.1. **研究背景与目标**

- 比较不同实验条件下的多个变量的均值，观察它们是否有显著差异。
- 样本测量值的结构：
    - $x_{1j1}$: 处理1下的变量1
    - $x_{2j1}$: 处理2下的变量1

## 1.2. **均值差异的数学表示**

- 差值向量定义： $$D_j = \begin{pmatrix} D_{j1} \\ D_{j2} \\ \vdots \\ D_{jp} \end{pmatrix} = \begin{pmatrix} x_{1j1} - x_{2j1} \\ x_{1j2} - x_{2j2} \\ \vdots \\ x_{1jp} - x_{2jp} \end{pmatrix}$$
- 差值向量的期望和协方差： $$(D_j) = \delta = \begin{pmatrix} \delta_1 \\ \delta_2 \\ \vdots \\ \delta_p \end{pmatrix}, \quad Cov(D_j) = \Sigma_d$$

## 1.3. **Hotelling $T^2$ 检验**

- 目标：对比 $\delta = 0$ 的假设 $H_0$（无显著差异）。
- 检验统计量： $$T^2 = n (\bar{D} - \delta)' S_d^{-1} (\bar{D} - \delta)$$ 其中：
    - $\bar{D} = \frac{1}{n} \sum_{j=1}^n D_j$
    - $S_d = \frac{1}{n-1} \sum_{j=1}^n (D_j - \bar{D})(D_j - \bar{D})'$
- $T^2$ 的分布：
- #重点
    ==当样本量足够大时，$T^2 \sim \chi^2_p$。==
    ==精确分布为 $T^2 \sim \frac{p(n-1)}{n-p}F_{p,n-p}$。==
    
## 1.4. **置信区间**

- 单变量差异的同时置信区间： $$\bar{d}_i \pm \sqrt{\frac{p(n-1)}{n-p} F_{p,n-p}(\alpha)} \sqrt{\frac{S_{di}^2}{n}}$$ 其中：
    - $\bar{d}_i$ 是差值向量中第$i$个变量的均值。
    - $S_{di}^2$ 是差值协方差矩阵 $S_d$ 的第$i$个对角元素。

## 1.5.案例

### **1.5.1. 题干背景**
• **数据记录：**

| Sample $j$ | Commercial Lab ($x_{1j1}$, $x_{1j2}$) | State Lab ($x_{2j1}$, $x_{2j2}$) |
| ---------- | ------------------------------------- | -------------------------------- |
| 1          | (6, 27)                               | (25, 15)                         |
| 2          | (6, 23)                               | (28, 13)                         |
| 3          | (18, 64)                              | (36, 22)                         |
| 4          | (8, 44)                               | (35, 29)                         |
| 5          | (11, 30)                              | (15, 31)                         |
| 6          | (34, 75)                              | (44, 64)                         |
| 7          | (28, 26)                              | (42, 30)                         |
| 8          | (71, 124)                             | (54, 64)                         |
| 9          | (43, 54)                              | (54, 64)                         |
| 10         | (33, 30)                              | (34, 56)                         |
| 11         | (20, 14)                              | (39, 21)                         |

• **变量定义：**

• Variable 1: 生化需氧量 (Biochemical Oxygen Demand, BOD)。
• Variable 2: 悬浮固体 (Suspended Solids, SS)。

• **实验设计：**

• Treatment 1: 商业实验室 (Commercial Lab)。
• Treatment 2: 国家实验室 (State Lab)。

• 样本数：$n = 11$，每个样本在两个实验室进行平行检测。

• **实验目标：**

• 检验国家实验室在 BOD 和 SS 的测量结果是否显著不同于商业实验室。

### 1.5.2参数计算

• **差异计算：**

• 对每个样本计算两个实验室测量值的差异：

$$

d_{j1} = x_{1j1} - x_{2j1}, \quad d_{j2} = x_{1j2} - x_{2j2}.

$$

• 差异结果：

• $d_{j1}: -19, -22, -18, -27, -4, -10, -14, 17, 9, 4, -19$。
• $d_{j2}: 12, 10, 42, 15, -1, 11, -4, -60, -2, 10, -7$。

**2. 差异均值向量与协方差矩阵**

• **差异均值向量：**
$$
\bar{\mathbf{d}} =
\begin{bmatrix}
\bar{d}_1 \\
\bar{d}_2
\end{bmatrix} =
\begin{bmatrix}
-9.36 \\
13.27
\end{bmatrix}.
$$
• **协方差矩阵：**
$$
S_d =
\begin{bmatrix}
199.26 & 88.38 \\
88.38 & 418.61
\end{bmatrix}, \quad S_d^{-1} =
\begin{bmatrix}
0.0055 & 0.0012 \\
-0.0012 & 0.0026
\end{bmatrix}.
$$
### **1.5.3. $T^2$ 检验统计量计算**

• **公式：**
$$
T^2 = n \cdot \bar{\mathbf{d}}’ S_d^{-1} \bar{\mathbf{d}}.
$$

• **计算步骤：**
$$
T^2 = 11 \cdot
\begin{bmatrix}
-9.36 & 13.27
\end{bmatrix}
\begin{bmatrix}
0.0055 & 0.0012 \\
-0.0012 & 0.0026
\end{bmatrix}
\begin{bmatrix}
-9.36 \\
13.27
\end{bmatrix}.
$$
计算得：
$$
T^2 = 13.6.
$$

### **1.5.4. 检验结果与置信区间**

• **显著性水平：$\alpha = 0.05$**
• F 分布临界值：
$$
c_\alpha = \frac{p(n-1)}{n-p} F_{p, n-p}(0.05) = \frac{2 \cdot 10}{11 - 2} F_{2, 9}(0.05) = 9.47.
$$
• **结论：**
$$
T^2 = 13.6 > c_\alpha = 9.47,
$$
因此拒绝原假设 $H_0$，说明两个实验室测量结果在均值上存在显著差异。

• **95% 同时置信区间：**

对于均值差异 $\delta_1$ 和 $\delta_2$，置信区间计算为：
$$
\delta_1 : \bar{d}_1 \pm \sqrt{c_\alpha \cdot \frac{S_{d,11}}{n}} = -9.36 \pm \sqrt{9.47 \cdot \frac{199.26}{11}} = (-22.46, 3.74),
$$

$$
\delta_2 : \bar{d}_2 \pm \sqrt{c_\alpha \cdot \frac{S_{d,22}}{n}} = 13.27 \pm \sqrt{9.47 \cdot \frac{418.61}{11}} = (-5.71, 32.25).
$$

• **观察：**

• 虽然置信区间包括 0，但 $T^2$ 检验拒绝了 $H_0$。
• 这是因为 $T^2$ 方法的置信椭球覆盖更大的置信区域。
### 1.5.5**Transformed Parametric Functions**

• **给定矩阵 $C$：**
$$
C =
\begin{bmatrix}
1 & 0 & -1 & 0 \\
0 & 1 & 0 & -1
\end{bmatrix}, \quad \text{Rank of } C = 2.
$$
• **变换定义：**
$$
\psi = C\mu =
\begin{bmatrix}
\mu_1 - \mu_3 \\
\mu_2 - \mu_4
\end{bmatrix}.
$$
• **检验目标：**
检验：
$$
H_0 : \psi = 0 \quad \text{对比} \quad H_1 : \psi \neq 0.
$$
**2. 已知数据**
• **样本均值向量：**
$$
\bar{X} =
\begin{bmatrix}
25.2727 \\
46.4545 \\
34.6364 \\
33.1818
\end{bmatrix}.
$$
• **样本协方差矩阵：**
$$
S =
\begin{bmatrix}
387.4 & 489.4 & 1014.1 & 148.7 \\
489.4 & 225.4 & 109.3 & 296.0 \\
1014.1 & 109.3 & 479.6 & 120.4  \\ 
148.7 & 296.0 & 120.4 & 363.8
\end{bmatrix}.
$$
• **$\psi$ 的估计值：**
$$
\hat{\psi} = C\bar{X} =
\begin{bmatrix}
-9.3636 \\
13.2727
\end{bmatrix}.
$$
• **$CSC’$ 矩阵：**
$$
CSC’ =
\begin{bmatrix}
199.2545 & 88.3091 \\
88.3091 & 418.6182
\end{bmatrix}.
$$
**3. $T^2$ 检验统计量**
**公式：**
$$
T^2 = n\hat{\psi}’(CSC’)^{-1}\hat{\psi}.
$$
• **计算：**
$$
T^2 = 11
\begin{bmatrix}
-9.3636 & 13.2727
\end{bmatrix}
\begin{bmatrix}
0.0055 & -0.0012 \\
-0.0012 & 0.0026
\end{bmatrix}
\begin{bmatrix}
-9.3636 \\
13.2727
\end{bmatrix} = 13.6.
$$
• **临界值：**
==注意这里的F统计量的分布变了==
$$
C_\alpha = \frac{m(n-1)}{n-m} F_{m,n-m}(\alpha),
$$
当 $\alpha = 0.05, m = 2, n = 11$ 时：
$$
C_\alpha = \frac{2(11-1)}{11-2} F_{2,9}(0.05) = 9.47.
$$
• **结论：**
由于 $T^2 = 13.6 > C_\alpha = 9.47$，拒绝 $H_0$，认为 $\psi$ 显著不为 0。

# 2.比较来自两个总体的均值向量

## 2.1协方差矩阵相等的情况


• **Population 1**: $x_{11}, x_{12}, \dots, x_{1n_1} \sim N_p(\mu_1, \Sigma_1)$
• **Population 2**: $x_{21}, x_{22}, \dots, x_{2n_2} \sim N_p(\mu_2, \Sigma_2)$

• 假设两总体均值和协方差矩阵如下：

• $\mu_1$ 和 $\mu_2$ 为均值向量；
• $\Sigma_1 = \Sigma_2 = \Sigma$（假设协方差矩阵相等）。

**2. 样本统计量定义**

• **样本均值向量：**
$$
\bar{X}_1 = \frac{1}{n_1} \sum_{j=1}^{n_1} x_{1j}, \quad \bar{X}_2 = \frac{1}{n_2} \sum_{j=1}^{n_2} x_{2j}.
$$
• **样本协方差矩阵：**
$$
S_1 = \frac{1}{n_1 - 1} \sum_{j=1}^{n_1} (x_{1j} - \bar{X}_1)(x_{1j} - \bar{X}_1)’
$$

$$S_2 = \frac{1}{n_2 - 1} \sum_{j=1}^{n_2} (x_{2j} - \bar{X}_2)(x_{2j} - \bar{X}_2)’.
$$
• **合并样本协方差矩阵（Pooled Covariance Matrix）：**
$$
S_{pooled} = S = \frac{(n_1 - 1)S_1 + (n_2 - 1)S_2}{n_1 + n_2 - 2}.
$$
**3. 假设检验**

• 检验均值向量差异：
$$
H_0 : \mu_1 = \mu_2 \quad \text{对比} \quad H_1 : \mu_1 \neq \mu_2,
$$
或：
$$
H_0 : \mu_1 - \mu_2 = \delta_0 \quad \text{对比} \quad H_1 : \mu_1 - \mu_2 \neq \delta_0.
$$
• **均值差的期望与方差：**
$$
E(\bar{X}_1 - \bar{X}_2) = \mu_1 - \mu_2, \quad \text{Var}(\bar{X}_1 - \bar{X}_2) = \left(\frac{1}{n_1} + \frac{1}{n_2}\right)\Sigma.
$$
• $\Sigma$ 使用 $S_{pooled}$ 估计。
• **Hotelling $T^2$ 统计量：**

$$

T^2 = (\bar{X}_1 - \bar{X}_2 - \delta_0)’ \left(\frac{1}{n_1} + \frac{1}{n_2}\right) S_{pooled}^{-1} (\bar{X}_1 - \bar{X}_2 - \delta_0).

$$

• **$T^2$ 的分布：**

$$

T^2 \sim \frac{p(n_1 + n_2 - 2)}{n_1 + n_2 - 1 - p} F_{p, n_1 + n_2 - 1 - p}.

$$

**4. 拒绝域与置信区间**

• **拒绝域：**

$$
H_0 \text{ 被拒绝当且仅当 } T^2 > \frac{p(n_1 + n_2 - 2)}{n_1 + n_2 - 1 - p} F_{p, n_1 + n_2 - 1 - p}(\alpha).
$$
• **$T^2$ 同时置信区间：**
对于任意向量 $a$，线性组合的置信区间为：
$$
a’(\bar{X}_1 - \bar{X}2) \pm C\alpha \sqrt{a’\left(\frac{1}{n_1} + \frac{1}{n_2}\right)S_{pooled}a},
$$
其中：
$$
C_\alpha^2 = \frac{p(n_1 + n_2 - 2)}{n_1 + n_2 - 1 - p} F_{p, n_1 + n_2 - 1 - p}(\alpha).
$$
• **每个均值差的置信区间：**
对于第 $i$ 个变量：
$$
(\mu_{1i} - \mu_{2i}) \in (\bar{x}_{1i} - \bar{x}_{2i}) \pm C_\alpha \sqrt{\left(\frac{1}{n_1} + \frac{1}{n_2}\right)s_{ii, pooled}},
$$
其中 $s_{ii, pooled}$ 是 $S_{pooled}$ 的第 $i$ 行第 $i$ 列元素。

• **Bonferroni 同时置信区间：**
$$
(\mu_{1i} - \mu_{2i}) \in (\bar{x}_{1i} - \bar{x}_{2i}) \pm t_{n_1 + n_2 - 2}\left(\frac{\alpha}{2p}\right)\sqrt{\left(\frac{1}{n_1} + \frac{1}{n_2}\right)s_{ii, pooled}}.
$$

## 2.2协方差矩阵不同的情况

就讲了30分钟,中间还扯了15分钟自己的故事.大概率不考

当比较来自两个总体的均值向量时，如果**两个总体的协方差矩阵不同**，不能直接使用**合并协方差矩阵**（$S_{pooled}$）的方法，而需要采用其他方法来进行检验和构造置信区间。

**1. 检验问题**

假设我们要比较两个总体的均值向量：

• **总体 1**: $x_{11}, x_{12}, \dots, x_{1n_1} \sim N_p(\mu_1, \Sigma_1)$。
• **总体 2**: $x_{21}, x_{22}, \dots, x_{2n_2} \sim N_p(\mu_2, \Sigma_2)$。

检验：
$$
H_0: \mu_1 = \mu_2 \quad \text{对比} \quad H_1: \mu_1 \neq \mu_2。
$$
由于 $\Sigma_1 \neq \Sigma_2$，无法假设协方差矩阵相等，需要基于各自样本协方差矩阵 $S_1$ 和 $S_2$ 来处理。

**2. 方法：通用Hotelling $T^2$ 检验**

当协方差矩阵不同（$\Sigma_1 \neq \Sigma_2$），可以使用**广义Hotelling $T^2$ 检验**，公式如下：

$$
T^2 = [(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)]’ \left[\frac{1}{n_1} S_1 + \frac{1}{n_2} S_2\right]^{-1} [(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)]。
$$

• $S_1$ 和 $S_2$ 分别是样本协方差矩阵。
• $\left( \frac{S_1}{n_1} + \frac{S_2}{n_2} \right)$ 是均值差的估计协方差矩阵。

**3. 分布**
$T^2$ 统计量的分布在小样本情况下不再严格服从 $F$ 分布。可以使用以下方法：

• **渐近方法**：当样本量足够大时，$T^2$ 统计量近似服从 $\chi^2_p$ 分布（$p$ 是变量数）。
• **Multivariate Behrens-Fisher Problem**
	当比较两个总体的均值向量时，如果两个总体的协方差矩阵不相等（$\Sigma_1 \neq \Sigma_2$）且样本量较小，这属于**多变量 Behrens-Fisher 问题**。
• **分布：**
$$
T^2 \sim \frac{\nu p}{\nu - p + 1} F_{p, \nu - p + 1}，
$$

其中 $\nu$ 是自由度，其计算复杂。

$$
\nu = \frac{p + p^2}{
    \sum_{i=1}^2 \frac{1}{n_i} \left\{ 
    \text{tr} \left[ \frac{1}{n_i} S_i \left( \frac{1}{n_1} S_1 + \frac{1}{n_2} S_2 \right)^{-1} \right]^2 
    + \left( \text{tr} \left[ \frac{1}{n_i} S_i \left( \frac{1}{n_1} S_1 + \frac{1}{n_2} S_2 \right)^{-1} \right] \right)^2 
    \right\}
}
$$
• $\nu$ 的范围：

$$

\min(n_1, n_2) \leq \nu \leq n_1 + n_2。

$$

**4. 同时置信区间**

• 均值差 $\mu_1 - \mu_2$ 的置信区间可以通过以下公式构造：
$$
(\mu_1 - \mu_2)_i \in (\bar{x}_{1i} - \bar{x}_{2i}) \pm C_\alpha \sqrt{\left( \frac{S_1}{n_1} + \frac{S_2}{n_2} \right)_{ii}},
$$

其中 $C_\alpha$ 为临界值，通常基于 $\chi^2$ 或重抽样方法。

# 3.比较多个多变量总体均值(单因子多元方差分析-One-Way MANOVA )

==理解思路就行,不会考察背诵==

## **3.1. 背景**

• **问题**：比较多个多元正态分布总体的均值向量是否相等。
• **假设**：有 $g$ 个多元正态分布总体，每个总体为：
$$
x_{l1}, x_{l2}, \dots, x_{ln_l} \sim N_p(\mu_l, \Sigma) \quad (l = 1, 2, \dots, g)。
$$

假设这些总体：
==• 具有相同的协方差矩阵 $\Sigma$；==
• 相互独立。
• **检验目标**：
$$
H_0: \mu_1 = \mu_2 = \cdots = \mu_g \quad \text{对比} \quad H_1: \text{至少有一个均值不同}。
$$

## **3.2. 模型表达**

• **观测值模型：**
$$
x_{lj} = \mu + \gamma_e + e_{lj},
$$
其中：
• $\mu$ 是总体均值；
• $\gamma_e$ 是第 $l$ 个总体的处理效应；
• $e_{lj} \sim N_p(0, \Sigma)$ 是残差，满足 $\sum_{l=1}^g n_e \gamma_e = 0$。
• **分解形式：**
$$
x_{lj} = \bar{X} + (\bar{X}_l - \bar{X}) + (x_{lj} - \bar{X}_l),
$$
其中：
• $\bar{X}$：总体样本均值；
• $\bar{X}_l - \bar{X}$：处理效应的估计；
• $x_{lj} - \bar{X}_l$：残差。

## **3.3 平方和与交叉积矩阵（SSP）**

• 总平方和与交叉积：
$$
\sum_{l=1}^g \sum_{j=1}^{n_g} (x_{lj} - \bar{X})(x_{lj} - \bar{X})’。
$$
• 分解为：
• **处理平方和与交叉积矩阵（Between）：**
$$
B = \sum_{l=1}^g n_l (\bar{X}_l - \bar{X})(\bar{X}_l - \bar{X})’。
$$
• **残差平方和与交叉积矩阵（Within）：**
$$
W = \sum_{l=1}^g \sum_{j=1}^{n_g} (x_{lj} - \bar{X}_l)(x_{lj} - \bar{X}_l)’。
$$
• **总平方和与交叉积矩阵：**
$$
T = B + W。
$$

## **3.4. Wilks Lambda 检验**

• **统计量：**
$$
\Delta^* = \frac{|W|}{|T|}。
$$
• $\Delta^*$ 称为 Wilks Lambda，值越小表明组间差异越显著。
• **自由度与显著性检验：**

如果样本量较大：
$$
\left(n - 1 - \frac{p + g}{2}\right)\ln\Delta^* \sim \chi^2_{p(g-1)}。
$$
拒绝域：
$$
\left(n - 1 - \frac{p + g}{2}\right)\ln\Delta^* > \chi^2_{p(g-1)}(\alpha)。
$$

## **3.5. MANOVA 表**

| **Source of Variation** | **SSP 矩阵**                                                                    | **自由度**                |
| ----------------------- | ----------------------------------------------------------------------------- | ---------------------- |
| Treatment (Between)     | $B = \sum_{l=1}^g n_l (\bar{X}_l - \bar{X})(\bar{X}_l - \bar{X})'$            | $g-1$                  |
| Residual (Within)       | $W = \sum_{l=1}^g \sum_{j=1}^{n_g} (x_{lj} - \bar{X}_l)(x_{lj} - \bar{X}_l)'$ | $\sum_{l=1}^g n_l - g$ |
| Total (Corrected)       | $T = B + W$                                                                   | $\sum_{l=1}^g n_l - 1$ |

## **3.6. 采样分布**

• $\Delta^*$ 的采样分布基于 Wilks Lambda 的分布，取决于变量数 $p$ 和组数 $g$：
• 当 $p = 1, g \geq 2$：$\Delta^* \sim F_{g-1, \sum n_e - g}$；
• 当 $p = 2, g \geq 2$：$\Delta^* \sim F_{2(g-1), 2(\sum n_e - g)}$。
