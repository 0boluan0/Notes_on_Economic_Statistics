
# 1.比较多个均值

## 1. **研究背景与目标**

- 比较不同实验条件下的多个变量的均值，观察它们是否有显著差异。
- 样本测量值的结构：
    - $x_{1j1}$: 处理1下的变量1
    - $x_{2j1}$: 处理2下的变量1

#### 2. **均值差异的数学表示**

- 差值向量定义： $$D_j = \begin{pmatrix} D_{j1} \\ D_{j2} \\ \vdots \\ D_{jp} \end{pmatrix} = \begin{pmatrix} x_{1j1} - x_{2j1} \\ x_{1j2} - x_{2j2} \\ \vdots \\ x_{1jp} - x_{2jp} \end{pmatrix}$$
- 差值向量的期望和协方差： $$(D_j) = \delta = \begin{pmatrix} \delta_1 \\ \delta_2 \\ \vdots \\ \delta_p \end{pmatrix}, \quad Cov(D_j) = \Sigma_d$$

#### 3. **Hotelling $T^2$ 检验**

- 目标：对比 $\delta = 0$ 的假设 $H_0$（无显著差异）。
- 检验统计量： $$T^2 = n (\bar{D} - \delta)' S_d^{-1} (\bar{D} - \delta)$$ 其中：
    - $\bar{D} = \frac{1}{n} \sum_{j=1}^n D_j$
    - $S_d = \frac{1}{n-1} \sum_{j=1}^n (D_j - \bar{D})(D_j - \bar{D})'$
- $T^2$ 的分布：
    - 当样本量足够大时，$T^2 \sim \chi^2_p$。
    - 精确分布为 $T^2 \sim \frac{p(n-1)}{n-p}F_{p,n-p}$。
    
#### 4. **置信区间**

- 单变量差异的同时置信区间： $$\bar{d}_i \pm \sqrt{\frac{p(n-1)}{n-p} F_{p,n-p}(\alpha)} \sqrt{\frac{S_{di}^2}{n}}$$ 其中：
    - $\bar{d}_i$ 是差值向量中第$i$个变量的均值。
    - $S_{di}^2$ 是差值协方差矩阵 $S_d$ 的第$i$个对角元素。

