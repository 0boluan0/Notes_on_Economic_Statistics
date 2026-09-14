---
aliases:
  - "回归 t 统计量以相匹配的标准误标准化单个线性限制，并须按经典精确或渐近参考分布校准"
  - Regression t test
  - 单个回归限制检验
student_os: knowledge-atom
atom_id: ECON-OLS-018
atom_set: regression-inference
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[标准误口径匹配]]"
  - "[[BLUE正态性边界]]"
leads_to:
  - "[[p值条件解释]]"
  - "[[置信区间覆盖率]]"
related:
  - "[[嵌套模型F检验]]"
  - "[[Wald联合检验]]"
---

# 回归 t 统计量以相匹配的标准误标准化单个线性限制，并须按经典精确或渐近参考分布校准
<!-- bilingual-en:start -->
*A regression t statistic standardizes one linear restriction by a matching standard error and must be calibrated against the appropriate exact or asymptotic reference distribution*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 对单个系数原假设 $H_0:\beta_j=\beta_{j,0}$，
> $$
> t=\frac{\hat\beta_j-\beta_{j,0}}{se(\hat\beta_j)}.
> $$
> 分子给出估计值离原假设多远，分母把距离换成该抽样设计和协方差口径下的标准误单位。统计量本身还不是结论；必须说明它使用哪一种参考分布与校准。
>
> <!-- bilingual-en:start -->
> The statistic expresses the distance from the null in standard-error units. It becomes an inferential result only after the covariance estimator and reference distribution have been specified.
> <!-- bilingual-en:end -->

更一般的单个线性限制 $H_0:a'\beta=c$ 使用
$$
t=\frac{a'\hat\beta-c}{\sqrt{a'\widehat V(\hat\beta)a}}.
$$
这里的协方差传播见 [[线性组合推断]]。只检验一个系数是 $a$ 只有一个非零元素的特例。

在固定 $X$ 或条件于 $X$ 的经典正态同方差模型中，使用 $\hat\sigma^2=RSS/(n-p)$ 得到的统计量有精确 $t_{n-p}$ 分布。若使用异方差稳健、HAC 或聚类协方差，或者只依赖大样本正态性，通常得到的是渐近校准；临界值、自由度修正和有效独立单元数必须与该口径配套。不能把经典 $t_{n-p}$ 标签原样贴在任意“稳健标准误”上。
<!-- bilingual-en:start -->
Under the classical normal homoskedastic model, the usual residual-variance estimator yields an exact $t_{n-p}$ reference distribution. Robust covariance estimators generally rely on asymptotic or specially adjusted calibration, so their critical values and degrees of freedom must match the estimator and sampling structure.
<!-- bilingual-en:end -->

双侧检验拒绝的是“与 $\beta_{j,0}$ 的差异大到难以由该原假设下的抽样波动解释”。它不自动说明效应大、因果成立或模型正确。尾概率的正确读法见 [[p值条件解释]]，与实际重要性的区别见 [[显著性解释边界]]。

当只有一个限制、使用同一经典线性模型、同一样本、同一协方差估计和相匹配的自由度时，经典嵌套模型检验满足 $F=t^2$。更一般地，单限制 Wald 统计量在使用同一个 $\widehat V$ 时满足 $W=t^2$；软件把 $W$ 缩放成哪一种 $F$ 形式仍需另查输出口径。

> [!question]- 自检
> 同一系数用经典标准误得到 $t=2.4$，换成聚类标准误后得到 $t=1.6$。能否继续用原来的 $t=2.4$，只把结果标成“聚类稳健”？
>
> **答案：** 不能。分子虽相同，推断使用的分母已经改变；必须用 $1.6$ 并按聚类口径相匹配的参考分布或自由度校准解释。

## 来源与核验

- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.4. 参数的约束检验|本地课程：一元回归参数检验]] 与 [[02_Economy/01_Econometrics/03_多元线性回归.md#3.1. 单参数检验：t 检验|多元回归单参数检验]]：核对统计量及单限制语境。
- [[BLUE正态性边界]]：核对 BLUE 不需要正态性，而经典有限样本 $t/F$ 的精确分布需要额外正态条件。
- [MIT OpenCourseWare 14.310x, Lecture 17](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf)：交叉核验回归抽样分布、方差估计和有限样本推断结构。
<!-- bilingual-en:start -->
- The local regression notes verify the statistic and its single-restriction use; MIT OCW and the linked normality-boundary atom support the distinction between exact finite-sample and asymptotic calibration.
<!-- bilingual-en:end -->
