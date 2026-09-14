---
aliases:
  - "Gaussian 协方差的 n 分母似然解只有在中心化散布满秩时才是正定参数空间中的有限 MLE"
  - "Existence boundary for the Gaussian covariance MLE"
  - "Singular Gaussian covariance likelihood"
student_os: knowledge-atom
atom_id: STAT-MVD-021
atom_set: multivariate-data-foundations
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[样本协方差矩阵]]"
  - "[[Gaussian密度与奇异支撑]]"
related:
  - "[[样本协方差无偏分母]]"
  - "[[样本协方差秩边界]]"
part_of:
  - "[[多元数据与协方差.canvas]]"
---

# Gaussian 协方差的 n 分母似然解只有在中心化散布满秩时才是正定参数空间中的有限 MLE
<!-- bilingual-en:start -->
*The $n$-divisor Gaussian covariance solution is a finite MLE in the positive-definite parameter space only when the centred scatter matrix has full rank*
<!-- bilingual-en:end -->

> [!summary] 存在性边界
> 在 iid $N_p(\mu,\Sigma)$ 模型中同时估计 $\mu$ 与 $\Sigma\succ0$。令
> $$
> C=\sum_{i=1}^n(x_i-\bar x)(x_i-\bar x)^T.
> $$
> 若 $C\succ0$，协方差 MLE 为 $\widehat\Sigma=C/n$。若 $C$ 奇异，正定参数空间中的对数似然没有有限最大值；此时 $C/n$ 不能无条件称为正定协方差的 MLE。
> <!-- bilingual-en:start -->
> Full-rank centred scatter yields the usual $C/n$ maximiser. Singular scatter leaves an unpenalised residual direction along which the positive-definite covariance can collapse and the likelihood can increase without attaining a finite maximum.
> <!-- bilingual-en:end -->

## 奇异时为什么没有有限最大值

代入 $\widehat\mu=\bar x$ 后，与 $\Sigma$ 有关的对数似然为

$$
\ell(\Sigma)
=-\frac n2\log|\Sigma|
-\frac12\operatorname{tr}(\Sigma^{-1}C)
+\text{常数}.
$$

若 $C$ 在某个非零方向上没有残差变化，就可以让 $\Sigma$ 在该方向的正特征值趋近 0。迹项不会产生对应惩罚，而 $-\frac n2\log|\Sigma|$ 继续上升，所以最大值不会在 $\Sigma\succ0$ 内取得。

[[样本协方差秩边界]]说明 $p\ge n$ 时 $C$ 必奇异；即使 $p<n$，精确共线或常数列仍可能造成同样问题。

> [!warning] 参数模型与估计口径要一起说
> $n$ 分母来自 Gaussian likelihood，不表示它在一般有限二阶矩模型下无偏；两种口径由不同问题产生。

> [!question]- 自检
> $p\ge n$ 且均值未知时，能否不加条件地称 $C/n$ 为正定协方差的 Gaussian MLE？
>
> **答案：** 不能。中心化散布秩至多 $n-1<p$，正定参数空间中的有限最大值不存在。

## 来源与核验

- [Stanford STATS 305C, *One sample problem*, “Maximizing likelihood”](https://web.stanford.edu/class/stats305c/lectures/Onesample.html)：直接核对 Gaussian likelihood 的 $n$ 分母解和正定条件。
- [Duke, *Multivariate Statistical Analysis* notes, PDF pp.80–81](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf)：核对多元正态一总体的 likelihood 解。
- [[样本协方差秩边界]]与[[Gaussian密度与奇异支撑]]：分别核对散布秩限制和正定协方差密度的参数边界。

> [!success] 审核状态
> 本卡的满秩解与奇异时不取极值已分别核验；掌握状态仍为 `unassessed`。
