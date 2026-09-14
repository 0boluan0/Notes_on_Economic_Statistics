---
aliases:
  - "恰好识别时 GMM 权重不改变点估计，过度识别时权重决定矩条件的组合与效率"
  - "广义矩估计的权重矩阵"
  - "Weight-matrix role in generalized method of moments"
student_os: knowledge-atom
atom_id: ECON-IV-015
atom_set: endogeneity-iv-gmm
atom_type: estimator-mechanism
status: source-checked
mastery_state: unassessed
part_of: "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[GMM矩条件]]"
related:
  - "[[过度识别检验边界]]"
  - "[[标准误口径匹配]]"
  - "[[2SLS标准误]]"
---

# 恰好识别时 GMM 权重不改变点估计，过度识别时权重决定矩条件的组合与效率
<!-- bilingual-en:start -->
*GMM weighting leaves the exactly identified point estimate unchanged but determines the combination and efficiency of moments under overidentification*
<!-- bilingual-en:end -->

> [!summary] 原子机制
> GMM 最小化
> $$
> Q_n(\theta)=\bar g_n(\theta)'W_n\bar g_n(\theta).
> $$
> 若矩条件数 $q$ 等于参数数 $k$，且 $\bar g_n(\theta)=0$ 有唯一解，则任何正定 $W_n$ 都在同一解处得到 $Q_n=0$，所以权重不改变点估计。若 $q>k$，样本矩通常不能同时为零；$W_n$ 决定哪些方向的偏离代价更高，因而会改变点估计及其效率。
>
> <!-- bilingual-en:start -->
> In exact identification, a unique root of the sample moments makes every positive-definite weighting matrix attain zero at the same estimate. Under overidentification, all sample moments generally cannot be zero together, so the weight matrix determines their compromise and the estimator's efficiency.
> <!-- bilingual-en:end -->

令

$$
S=\lim_{n\to\infty}\operatorname{Var}\!\left(\sqrt n\,\bar g_n(\theta_0)\right)
$$

是矩条件的长期协方差矩阵，$G=E[\partial m(W_i,\theta_0)/\partial\theta']$ 是矩对参数的导数。一般权重 $W$ 下，GMM 渐近方差为

$$
V(W)=(G'WG)^{-1}G'WSWG(G'WG)^{-1}.
$$

在矩条件正确且正则条件成立时，选择 $W=S^{-1}$ 得到基于这些矩条件的有效 GMM，方差化简为

$$
V(S^{-1})=(G'S^{-1}G)^{-1}.
$$

这里的“有效”只是在同一组有效矩条件构成的估计量类中具有更小渐近方差，不等于识别假设更可信，也不等于有限样本一定更好。
<!-- bilingual-en:start -->
The optimal large-sample weight is the inverse long-run covariance of the moments. Optimality is conditional on the same valid moments and regularity conditions; it means lower asymptotic variance within that class, not stronger identification assumptions or guaranteed finite-sample superiority.
<!-- bilingual-en:end -->

因为 $S$ 未知，两步 GMM 先用一个给定权重得到一致的初步估计，再用初步残差估计 $\hat S$，最后以 $\hat S^{-1}$ 重新估计。独立异方差数据、聚类数据和时间相关数据对应不同的 $S$：分别需要 heteroskedasticity-robust、cluster 或 HAC 的矩协方差估计。最终协方差矩阵必须与最终权重、矩导数及依赖结构匹配。
<!-- bilingual-en:start -->
Two-step GMM uses a preliminary consistent estimate to estimate the moment covariance and then re-estimates with its inverse. Independent heteroskedastic, clustered, and serially dependent data require different estimators of that covariance, and the reported VCE must match the final estimator and dependence structure.
<!-- bilingual-en:end -->

权重越“精密”越不能补救无效矩条件。若某个工具违反排除限制，给它较大权重只会让错误限制对估计影响更大。相反，权重矩阵估计本身在小样本中很噪时，两步有效性的渐近优势也可能伴随有限样本不稳定；因此应同时报告矩条件来源、样本结构和使用的权重/VCE 口径。

> [!question]- 最小自检
> 一个模型恰好识别。把单位权重改成估计的最优权重后，点估计是否应变化？
>
> **答案：** 在样本矩有唯一精确根且权重正定的标准情形下不应变化；两个权重都在同一 $\bar g_n(\theta)=0$ 处达到目标函数零。变化通常提示实现、矩集合、样本或数值处理也发生了变化。

## 来源与核验

- Hansen (1982), [“Large Sample Properties of Generalized Method of Moments Estimators”](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/), *Econometrica* 50(4), 1029–1054：原始建立一般权重下 GMM 渐近方差与最优权重结果。
- MIT OpenCourseWare 14.382, [Lecture 3: Structural Equations Models and GMM](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/f6b8cd36eee8df2b259c979a1fd2673e_MIT14_382S17_lec3.pdf)：明确推导恰好识别时权重消失、过度识别时权重影响估计，以及 $S^{-1}$ 的渐近有效性。
- StataCorp, [`ivregress` manual](https://www.stata.com/manuals/rivregress.pdf)：核验线性 IV-GMM 的加权二次型、两步 $\hat S^{-1}$ 实现和 robust/cluster/HAC 权重选择。
