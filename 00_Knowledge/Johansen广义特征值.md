---
aliases:
  - "Johansen 秩检验解的是残差典型相关的广义特征值问题"
  - Johansen generalized eigenvalue problem
  - Johansen 典型相关问题
student_os: knowledge-atom
atom_id: TS-CI-019
atom_set: cointegration-error-correction
atom_type: method-mechanism
status: source-checked
mastery_state: unassessed
requires:
  - "[[Johansen检验]]"
  - "[[VECM的Π秩]]"
  - "[[αβ分解非唯一性]]"
related:
  - "[[Johansen秩检验]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# Johansen 秩检验解的是残差典型相关的广义特征值问题
<!-- bilingual-en:start -->
*Johansen rank testing solves a generalized-eigenvalue problem for residual canonical correlations*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Johansen 方法先把 $\Delta x_t$ 与 $x_{t-1}$ 分别对滞后差分和确定性项残差化，再寻找两组残差之间最强的线性长期联系；检验所用 $\hat\lambda_i$ 来自样本矩阵的广义特征值或典型相关问题，不是 $\Pi$ 的普通特征值。

若以 $R_{0t}$ 表示短期部分剔除后的 $\Delta x_t$ 残差，以 $R_{1t}$ 表示相同控制项剔除后的 $x_{t-1}$ 残差，并记样本乘积矩阵为 $S_{ij}$，典型方程可写成
$$
\left|\lambda S_{11}-S_{10}S_{00}^{-1}S_{01}\right|=0.
$$
$\hat\lambda_i\in[0,1]$ 衡量对应典型方向中水平残差与差分残差的联系强度，并进入似然比统计量。

对一般方阵，$\operatorname{rank}(\Pi)$ 等于非零奇异值的个数，却**不等于**非零普通特征值的个数。例如非零幂零矩阵可以有正秩，但全部普通特征值都为零。因此，直接数估计 $\Pi$ 的普通特征值，既不是一般有效的秩计算，也不产生 Johansen 的检验分布。Johansen 程序检验的是总体长期矩阵的秩限制，计算上用的则是残差乘积矩阵构成的约化秩回归问题。

> [!question]- 自检
> Johansen 输出的 $\hat\lambda_i$ 是否就是估计矩阵 $\hat\Pi$ 的普通特征值？
>
> **答案：** 不是。它们来自残差样本矩阵的广义特征值／典型相关问题。

## 来源与核验

- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对高斯约化秩似然与广义特征值构造。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程特别注明的“不是 $\Pi$ 普通特征值”边界。
