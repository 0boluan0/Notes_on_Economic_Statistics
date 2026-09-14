---
aliases:
  - "共享设计下多响应最小二乘等于逐响应 OLS"
  - Multivariate least squares equals response-wise OLS under a shared design
  - response-wise OLS equivalence
student_os: knowledge-atom
atom_id: STAT-MVR-004
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应线性回归]]"
  - "[[普通最小二乘]]"
  - "[[Frobenius范数]]"
leads_to:
  - "[[多响应残差与误差SSP]]"
related:
  - "[[满列秩与OLS唯一性]]"
  - "[[矩阵正态误差结构]]"
  - "[[相关行下的多响应GLS]]"
  - "[[相关行不容独立推断]]"
---

# 共享设计下多响应最小二乘等于逐响应 OLS
<!-- bilingual-en:start -->
*Under a shared design, multivariate least squares equals response-wise OLS*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 若所有 $q$ 个响应使用同一个 $n\times k$ 设计矩阵、同一组观测行与同一未加权平方损失，且没有跨响应的系数约束或惩罚，则
> $$\|Y-XB\|_F^2=\sum_{j=1}^q\|y_j-Xb_j\|_2^2$$
> 可逐列最小化。$X$ 满列秩时，唯一解是
> $$\hat B=(X^TX)^{-1}X^TY,$$
> 第 $j$ 列恰是单独回归 $y_j$ 对 $X$ 的 OLS 系数。
> <!-- bilingual-en:start -->
> With the same rows, the same design matrix, ordinary unweighted squared loss, and no cross-response restrictions or penalty, the multivariate objective separates by response. Full column rank gives $\hat B=(X^TX)^{-1}X^TY$.
> <!-- bilingual-en:end -->

这里的 $\|Y-XB\|_F$ 是把残差矩阵全部元素一起计量的 [[Frobenius范数]]；把其平方按列展开，才得到右侧各响应的残差平方和。

同一行内响应相关的 $\Sigma$ 不改变这个点估计。在 Gaussian 矩阵正态似然中，对 $B$ 的一阶条件是 $X^T(Y-XB)\Sigma^{-1}=0$；$\Sigma\succ0$ 时仍化为普通正规方程。更一般地，固定正定的响应方向权重 $W$ 只把这个方程右乘 $W$，也不会改变解。但 $\Sigma$ 会进入 $\hat B$ 的联合协方差和 $CBA=D$ 的多元检验，所以“点估计相同”不等于“可以把联合推断拆成互不相关的检验”。

这里要分清两种“等价”。只要研究者选择的是共享 $X$ 的 Frobenius 平方损失，上面的**目标函数按列分解**不依赖真实误差协方差；相关性不会改变“所选 OLS 如何计算”。但 OLS 是否同时等于 Gaussian 似然估计或有效估计，取决于误差协方差。$\Omega\propto I_n$ 时有效 GLS 与 OLS 重合；一般已知 $\Omega$ 下的估计规则由 [[相关行下的多响应GLS]] 单独给出。非可分离的整体协方差还可能让有效 GLS 在响应之间耦合。

不同响应有不同缺失行、不同 $X_j$ 或不同观测行权重，意味着单一共享矩阵目标已不再描述实际资料；一般联合损失，或低秩、稀疏、相等系数等跨响应约束/惩罚，也会破坏逐列分离。秩亏时逐列最小值仍对应同一拟合投影，但系数向量不唯一，必须说明广义逆或其他选解规则。相关行为什么会破坏独立行推断，见 [[相关行不容独立推断]]。

> [!question]- 自检
> 为什么“响应彼此相关”不改变共享设计下的 OLS 点估计，却会改变联合检验？
>
> **答案：** 普通平方损失按响应列可分；但不同列系数估计的抽样协方差由响应误差协方差 $\Sigma$ 决定。

## 来源与核验

- [statsmodels, Multivariate Linear Model](https://www.statsmodels.org/stable/examples/notebooks/generated/multivariate_ls.html)：明确核对共享 explanatory variables 时参数估计对应各 dependent variable 的 separate OLS，而多响应模型的优势在联合推断。
- [[普通最小二乘]]、[[满列秩与OLS唯一性]]：核对逐列目标与满列秩唯一解。
- [statsmodels `mv_test`](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.multivariate_ols.MultivariateLSResults.mv_test.html)：核对点估计之后仍需使用 $L B M=C$ 的联合假设。
- [SciPy `matrix_normal`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html)：核对可分离行/响应协方差下的矩阵正态密度；对 $B$ 求一阶条件得到上面的 OLS/GLS 分界。
<!-- bilingual-en:start -->
- The official statsmodels example states the separate-OLS equivalence under common explanatory variables; the OLS definition and full-column-rank condition fix the loss and uniqueness boundaries.
<!-- bilingual-en:end -->
