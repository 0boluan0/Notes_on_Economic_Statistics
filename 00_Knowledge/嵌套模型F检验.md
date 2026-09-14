---
aliases:
  - "经典嵌套模型 F 检验用同一样本和响应下受限与不受限 RSS 的增量检验预先给定的线性限制"
  - Classical nested-model F test
  - 受限与不受限回归 F 检验
student_os: knowledge-atom
atom_id: ECON-OLS-020
atom_set: regression-inference
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归推断.canvas|回归推断]]"
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[线性组合推断]]"
  - "[[OLS经典协方差估计]]"
  - "[[BLUE正态性边界]]"
related:
  - "[[模型比较可比性]]"
  - "[[模型选择目标]]"
  - "[[回归t检验]]"
leads_to:
  - "[[回归联合显著边界]]"
  - "[[Chow结构稳定性检验]]"
---

# 经典嵌套模型 F 检验用同一样本和响应下受限与不受限 RSS 的增量检验预先给定的线性限制
<!-- bilingual-en:start -->
*The classical nested-model F test uses the increase in RSS between restricted and unrestricted regressions on the same response and sample to test prespecified linear restrictions*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 若受限模型由无约束模型施加 $q$ 个独立线性限制得到，且两者使用同一响应、同一组观测和同一经典线性模型，则
> $$
> F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-p_U)}.
> $$
> 分子衡量限制迫使模型放弃多少样本内拟合，分母用无约束模型的残差方差把它标准化。
>
> <!-- bilingual-en:start -->
> For $q$ independent linear restrictions, the statistic compares the restriction-induced increase in residual sum of squares with the unrestricted residual variance. The restricted and unrestricted regressions must be genuinely nested and fitted to the same response and observations.
> <!-- bilingual-en:end -->

检验 $H_0:R\beta=r$ 时，$p_U$ 是无约束模型实际估计的参数数。经典正态、球形误差条件下，原假设成立时统计量具有精确 $F_{q,n-p_U}$ 分布。没有正态性时，Gauss–Markov 的 BLUE 结论仍可能成立，但这个精确有限样本参考分布不再由该定理保证，见 [[BLUE正态性边界]]。

使用 RSS 公式前必须满足三个可比条件：

1. 受限模型能由无约束模型施加明确线性限制得到，而不是两个非嵌套函数形式。
2. 两个模型使用同一响应尺度和完全相同的观测行；缺失值导致样本不同就不能直接比较 RSS。
3. 权重、误差模型和估计准则一致；普通 OLS 的 RSS 公式不能直接替代异方差稳健联合检验。

<!-- bilingual-en:start -->
The RSS comparison is invalid for nonnested models, different responses or samples, or incompatible fitting criteria. Robust covariance settings should use a matching joint test rather than relabel the classical RSS statistic.
<!-- bilingual-en:end -->

当 $q=1$，且单限制 $t$ 检验与 F 检验使用同一个经典模型、样本、$\hat\sigma^2$ 和自由度时，$F=t^2$。这是一条有条件的等价关系，不是“任何软件中的 F 都是某个 t 的平方”。

F 检验回答的是预先给定的一组限制是否与数据相容。它不会从大量候选模型中自动发现真模型；是否应使用它先由 [[模型选择目标]] 决定，反复逐步筛选后再把最终 F 检验当成一次预先设定的检验，则会忽略选择过程带来的额外不确定性，见 [[选择后推断]]。

> [!question]- 自检
> 两个回归都预测工资，但因为一个变量缺失，受限模型用 500 行、无约束模型只用 470 行。能否把二者 RSS 代入经典嵌套 F 公式？
>
> **答案：** 不能。RSS 差异同时混入了样本变化，不再只代表施加限制损失的拟合；必须先在同一 470 行上重新估计两者。

## 来源与核验

- [[02_Economy/01_Econometrics/03_多元线性回归.md#3.2. 多参数检验：F 检验|本地课程：F 检验]]：核对受限/无约束 RSS、限制数与自由度公式。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#3.2.4. 具体步骤示例|规模报酬限制示例]]：核对一般线性限制不必只是把若干系数设为零。
- [MIT OpenCourseWare 14.310x, Lecture 17](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf)：交叉核验经典回归抽样分布和 F 检验语境。
<!-- bilingual-en:start -->
- The local course provides the restricted-versus-unrestricted formula and examples; MIT OCW supplies an independent course-level check on the classical sampling framework.
<!-- bilingual-en:end -->
