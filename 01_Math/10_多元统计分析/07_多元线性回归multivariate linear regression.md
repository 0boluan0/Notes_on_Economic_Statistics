
# 1. OLS

## **1.1. 模型形式**

- 经典线性回归模型的数学形式为： $$Y = \beta_0 + \beta_1x_1 + \cdots + \beta_kx_k + \epsilon$$
    - $Y$: 响应变量
    - $\beta_0, \beta_1, \dots, \beta_k$: 回归系数
    - $\epsilon$: 随机误差项
- 矩阵形式表示： $$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}$$
    - $\mathbf{Y}$: $n \times 1$ 响应变量向量
    - $\mathbf{X}$: $n \times (k+1)$ 设计矩阵 (含截距项列)
    - $\boldsymbol{\beta}$: $(k+1) \times 1$ 回归系数向量
    - $\boldsymbol{\epsilon}$: $n \times 1$ 误差向量

## **1.2. 模型假设**

1. 线性关系：响应变量 $Y$ 和自变量 $\mathbf{X}$ 之间呈线性关系。
2. 误差项独立同分布：
    - $E(\boldsymbol{\epsilon}) = 0$
    - $Var(\boldsymbol{\epsilon}) = \sigma^2\mathbf{I}_n$
3. 自变量无完全多重共线性。

## **1.3. 最小二乘估计**

- 回归系数的估计量： $$\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$$
- 估计值的性质：
    - 无偏性：$E(\hat{\boldsymbol{\beta}}) = \boldsymbol{\beta}$
    - 协方差矩阵：$Var(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}$
- 拟合值与残差：
    - 拟合值：$\hat{\mathbf{Y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$
    - 残差：$\hat{\boldsymbol{\epsilon}} = \mathbf{Y} - \hat{\mathbf{Y}}$

## **1.4. 总平方和分解**

- 总平方和 (TSS)：$TSS = \mathbf{Y}'\mathbf{Y} - n\bar{Y}^2$
- 回归平方和 (RSS)：$RSS = \hat{\mathbf{Y}}'\hat{\mathbf{Y}} - n\bar{Y}^2$
- 残差平方和 (ESS)：$ESS = \hat{\boldsymbol{\epsilon}}'\hat{\boldsymbol{\epsilon}}$
- 分解公式： TSS = RSS + ESS

## **1.5. 决定系数 ($R^2$)**

- 定义： $$R^2 = \frac{RSS}{TSS} = 1 - \frac{ESS}{TSS}$$
    - 衡量模型拟合优度，取值范围 $[0, 1]$。
    - $R^2$ 越接近 $1$，模型拟合效果越好。
## **1.6. 假设检验**

1. **总体显著性检验** (F检验)：
    - 检验假设：$H_0: \beta_1 = \beta_2 = \cdots = \beta_k = 0$
    - F统计量： $$F = \frac{\frac{RSS}{k}}{\frac{ESS}{n-k-1}}$$
    - $F \sim F_{k, n-k-1}$。
2. **单个变量检验** (t检验)：
    - 检验假设：$H_0: \beta_j = 0$
    - t统计量： $$t = \frac{\hat{\beta}_j}{\sqrt{\sigma^2 (\mathbf{X}'\mathbf{X})^{-1}_{jj}}}$$
    - $t \sim t_{n-k-1}$。

## **1.7. 大样本性质**

- 在大样本下，$\hat{\boldsymbol{\beta}}$ 的分布近似为正态分布： $$\hat{\boldsymbol{\beta}} \sim N(\boldsymbol{\beta}, \sigma^2 (\mathbf{X}'\mathbf{X})^{-1})$$

