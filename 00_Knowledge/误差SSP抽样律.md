---
aliases:
  - "独立 Gaussian 行下回归误差 SSP 服从自由度 n-rank(X) 的 Wishart 分布"
  - The regression error SSP has a Wishart distribution with n-rank(X) degrees of freedom under independent Gaussian rows
  - residual SSP sampling law
student_os: knowledge-atom
atom_id: STAT-MVR-011
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应残差与误差SSP]]"
  - "[[矩阵正态误差结构]]"
  - "[[Wishart分布]]"
related:
  - "[[误差SSP秩边界]]"
  - "[[Wishart秩与可逆性]]"
---

# 独立 Gaussian 行下回归误差 SSP 服从自由度 n-rank(X) 的 Wishart 分布
<!-- bilingual-en:start -->
*Under independent Gaussian rows, the regression error SSP has a Wishart distribution with $n-\operatorname{rank}(X)$ degrees of freedom*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 令 $r=\operatorname{rank}(X)$。若
> $$Y\mid X\sim MN_{n,q}(XB,I_n,\Sigma),\qquad \Sigma\succ0,$$
> 并用最小二乘拟合 $B$，则
> $$E_{SSP}=(Y-X\hat B)^T(Y-X\hat B)\sim W_q(\Sigma,n-r).$$
> 回归消耗了观测空间中 $r$ 个方向，所以 Wishart 自由度是剩余的 $n-r$，不是机械地写成 $n-k$；只有 $X$ 满列秩时二者才相同。
>
> <!-- bilingual-en:start -->
> If $Y\mid X\sim MN_{n,q}(XB,I_n,\Sigma)$ and $r=\operatorname{rank}(X)$, least-squares residualisation leaves $n-r$ Gaussian directions. Hence $E_{SSP}\sim W_q(\Sigma,n-r)$.
> <!-- bilingual-en:end -->

当 $n-r>0$ 时，这个抽样律立刻给出
$$
E\!\left(\frac{E_{SSP}}{n-r}\right)=\Sigma.
$$
因此 $E_{SSP}/(n-r)$ 是响应误差协方差的无偏估计。它回答的是重复抽样下的无偏性，不是极大化 Gaussian 似然时的分母选择。

在同一个 Gaussian 模型中，若 $E_{SSP}\succ0$，对 $B$ 和正定 $\Sigma$ 的内部极大似然估计为
$$
\hat B=(X^TX)^{-1}X^TY,
\qquad
\hat\Sigma_{ML}=\frac{E_{SSP}}{n}
$$
（这里写了满列秩 $X$ 的常规形式）。MLE 使用 $n$，因为似然包含 $n$ 个 $q$ 维观测行；无偏估计使用 $n-r$，因为拟合已经消耗了 $r$ 个残差方向。二者采用不同准则，不能交换分母后仍称为同一个估计量。

若 $E_{SSP}$ 奇异，$E_{SSP}/n$ 不在正定协方差参数空间内。沿着没有残差变化的方向把协方差特征值逼近 $0$，可使剖面似然继续上升；因此不能把这个奇异矩阵称为该正定参数空间内已经存在的 MLE。何时必然或几乎必然奇异，见 [[误差SSP秩边界]]。

> [!question]- 自检
> $n=80$、$\operatorname{rank}(X)=6$ 时，无偏协方差估计和 Gaussian MLE 各用什么分母？
>
> **答案：** 无偏估计用 $80-6=74$；在 $E_{SSP}\succ0$ 且内部 MLE 存在时，MLE 用 $80$。

## 来源与核验

- [R `SSD` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/SSD.html)：核对多响应线性模型的 residual sums-of-squares-and-products、残差自由度与协方差估计对象。
- [[Wishart分布]]、[[Wishart秩与可逆性]]：核对 Gaussian 外积和的 Wishart 律、自由度、期望与可逆性门槛。
- [[多响应残差与误差SSP]]：核对 $E_{SSP}$ 的对象定义；本卡只承担加入抽样假设之后的分布与估计结论。
<!-- bilingual-en:start -->
- R's official `SSD` documentation supports the residual SSP and residual degrees of freedom; the Wishart atoms support its Gaussian sampling law, expectation, and rank boundary.
<!-- bilingual-en:end -->
