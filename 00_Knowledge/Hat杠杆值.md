---
aliases:
  - "Hat 矩阵对角元只由设计矩阵决定并量化观测的杠杆潜力，高杠杆不等于实际影响"
  - "高杠杆点"
  - "Hat values and leverage"
student_os: knowledge-atom
atom_id: ECON-SPEC-010
atom_set: regression-influence-diagnostics
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
requires:
  - "[[异常杠杆影响区分]]"
  - "[[满列秩与OLS唯一性]]"
leads_to:
  - "[[Cook距离]]"
  - "[[DFFITS]]"
---

# Hat 矩阵对角元只由设计矩阵决定并量化观测的杠杆潜力，高杠杆不等于实际影响
<!-- bilingual-en:start -->
*The diagonal of the hat matrix depends only on the design matrix and measures an observation's leverage potential; high leverage is not the same as actual influence*
<!-- bilingual-en:end -->

> [!summary] 杠杆是 $X$ 的几何性质
> 在满列秩线性 OLS 中，Hat 矩阵把观测结果 $y$ 投影到设计矩阵 $X$ 的列空间。其第 $i$ 个对角元 $h_{ii}$ 只取决于解释变量的布局，表示该行对自身拟合值的杠杆潜力；它不使用 $y_i$，因而不能单独判定实际影响。

设 $X\in\mathbb R^{n\times p}$ 满列秩，$X$ 已包含模型中的截距列（若有）。则

$$
\hat y=Hy,\qquad H=X(X'X)^{-1}X',
$$

且若 $x_i'$ 是 $X$ 的第 $i$ 行，

$$
h_{ii}=x_i'(X'X)^{-1}x_i.
$$

$H$ 是对称幂等的正交投影矩阵，因此

$$
0\le h_{ii}\le 1,\qquad
\sum_{i=1}^n h_{ii}=\operatorname{tr}(H)=p,
$$

平均杠杆值为 $p/n$，其中 $p$ 是实际估计的参数数目，截距若存在也算一个参数。当模型含截距时，还有 $h_{ii}\ge 1/n$；没有截距时，这个 $1/n$ 下界不再有保证。

直观上，高 $h_{ii}$ 意味着该解释变量组合远离样本的主要设计区域。它不必是某个单独变量的极端值：每个分量都处在常见范围内的稀有组合，仍可能高杠杆。

## 边界

- 高杠杆表示“有能力改变拟合”，不表示“已经显著改变拟合”；实际影响还要结合残差和目标。
- 常见的倍数平均杠杆线只是筛查规则，统一放在 [[影响诊断阈值]] 中，不是这个定义的一部分。
- 若 $X$ 秩亏，$(X'X)^{-1}$ 不存在，上述普通逆公式不可使用。用伪逆可写出列空间的唯一投影，此时 $\operatorname{tr}(H)=\operatorname{rank}(X)$；但系数向量本身不唯一，必须另行限定可估函数。

> [!question]- 最小自检
> 保持整个 $X$ 不变，只把第 $i$ 个观测的 $y_i$ 改成一个极端值，$h_{ii}$ 会改变吗？
>
> **答案：** 不会。$h_{ii}$ 只由 $X$ 决定。$y_i$ 的改变可能改变残差和实际影响，但不改变杠杆值。

## 来源与核验

- Penn State STAT 501, [Lesson 11.2: Using Leverages to Help Identify Extreme X Values](https://online.stat.psu.edu/stat501/Lesson11)：核验 $H=X(X'X)^{-1}X'$ 、$h_{ii}$ 的设计空间含义、取值范围、和与平均值。
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.1. OLS|本地课程：矩阵 OLS]]与 [[满列秩投影公式]]：核对满列秩下的 OLS 估计和列空间投影公式。
- NIST Dataplot, [Regression Diagnostics](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/regrdiag.htm)：仅交叉核验杠杆对角元、范围和平均值；该页一处 Hat 矩阵排版漏了逆号，本卡不以该处作公式依据。
- 作者逐项核验日：2026-08-30；本卡已完成独立模型复核。
