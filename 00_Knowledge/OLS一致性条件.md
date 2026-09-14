---
aliases:
  - "OLS 一致性依赖总体正交与大样本稳定性"
  - OLS consistency from population orthogonality
  - OLS consistency conditions
  - OLS 一致性条件
student_os: knowledge-atom
atom_id: ECON-OLS-011
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
requires:
  - "[[普通最小二乘]]"
  - "[[依概率收敛]]"
  - "[[连续映射定理]]"
contrasts_with:
  - "[[零条件均值无偏性]]"
---

# OLS 一致性依赖总体正交与大样本稳定性
<!-- bilingual-en:start -->
*OLS consistency depends on population orthogonality and stable large-sample moments*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 在线性参数模型 $Y_i=x_i'\beta+u_i$ 中，若 $E(x_i u_i)=0$，$E(x_i x_i')$ 有限且非奇异，并且相应样本矩依概率收敛到这些总体矩，则
> $$
> \hat\beta-\beta
> =\left(\frac1n\sum_i x_ix_i'\right)^{-1}
> \left(\frac1n\sum_i x_i u_i\right)
> \xrightarrow{p}0.
> $$
> <!-- bilingual-en:start -->
> In the linear-in-parameters model $Y_i=x_i'\beta+u_i$, if $E(x_i u_i)=0$, $E(x_i x_i')$ is finite and nonsingular, and the corresponding sample moments converge in probability to their population counterparts, then $\hat\beta-\beta=(n^{-1}\sum_i x_ix_i')^{-1}(n^{-1}\sum_i x_i u_i)\xrightarrow{p}0$.
> <!-- bilingual-en:end -->

## 自然解释与边界
<!-- bilingual-en:start -->
*Natural interpretation and boundary*
<!-- bilingual-en:end -->

一致性说的是样本扩大时估计量会集中到目标参数，不是每个有限样本的期望都等于真值。零条件均值 $E(u_i\mid x_i)=0$ 足以推出 $E(x_i u_i)=0$，但对一致性而言通常比“总体正交”更强；反过来，只有正交还不够，设计矩阵必须在总体中识别参数，且样本平均必须有稳定的大样本极限。时间序列、聚类或重尾数据需要与其依赖结构相匹配的收敛条件，不能机械套用独立同分布版本。
<!-- bilingual-en:start -->
Consistency says that the estimator concentrates on its target as the sample grows; it does not say that its finite-sample expectation equals the truth. Zero conditional mean, $E(u_i\mid x_i)=0$, is sufficient for $E(x_i u_i)=0$ but is generally stronger than the population orthogonality needed for consistency. Orthogonality alone is not enough: the population design must identify the parameter and sample averages must possess stable large-sample limits. Time-series, clustered, or heavy-tailed data require convergence conditions matched to their dependence and tail structure rather than a mechanical iid argument.
<!-- bilingual-en:end -->

## 最小对照
<!-- bilingual-en:start -->
*Minimal contrast*
<!-- bilingual-en:end -->

某估计量可以在小样本中略有偏差，却随着 $n$ 增大而一致；也可以对每个 $n$ 都无偏但方差不收缩，从而不一致。因此“无偏”和“一致”回答的是不同问题，不能互相替代。
<!-- bilingual-en:start -->
An estimator may be slightly biased in small samples yet consistent as $n$ grows. It may also be unbiased for every $n$ but have a variance that does not shrink, making it inconsistent. Unbiasedness and consistency answer different questions and do not substitute for each other.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么 $E(x_i u_i)=0$ 仍不足以单独证明 OLS 一致？
> <!-- bilingual-en:start -->
> Why is $E(x_i u_i)=0$ alone insufficient to establish OLS consistency?
> <!-- bilingual-en:end -->
>
> **答案：** 还要保证 $E(x_i x_i')$ 能识别参数，并保证样本矩收敛到总体矩；否则逆矩阵或概率极限可能不存在。
> <!-- bilingual-en:start -->
> **Answer:** The population second-moment matrix must identify the parameter, and sample moments must converge to population moments; otherwise the inverse or probability limit may not exist.
> <!-- bilingual-en:end -->

## 继续

- [[零条件均值无偏性]]：比较有限样本无偏所需的更强条件。
- [[动态回归规格流程]]：把正交、稳定性与样本矩条件放进动态、可能有依赖的数据环境。
<!-- bilingual-en:start -->
- [[零条件均值无偏性|Zero conditional mean makes OLS conditionally unbiased]] provides the stronger condition used for finite-sample unbiasedness.
- [[动态回归规格流程|The dynamic-regression specification workflow]] places orthogonality, stability, and sample-moment requirements in a dynamic, potentially dependent-data setting.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 5 §5.1：核验 OLS 一致性的概率极限推导，以及零均值/零协方差条件弱于零条件均值。
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.2. 大样本性质|本地课程：大样本性质]]：核对样本矩分解、概率极限和一致性记号。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 5 §5.1, supports the probability-limit argument for OLS consistency and the fact that zero mean and zero covariance are weaker than zero conditional mean.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.2. 大样本性质|The local course section on large-sample properties]] fixes the sample-moment decomposition, probability-limit notation, and consistency result.
<!-- bilingual-en:end -->
