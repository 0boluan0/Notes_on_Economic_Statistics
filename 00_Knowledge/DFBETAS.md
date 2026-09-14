---
aliases:
  - "DFBETAS 分别标准化删去观测前后每个回归系数的变化，因此能定位总体影响由哪个参数承担"
  - Standardized difference in coefficients
  - DFBETAS
student_os: knowledge-atom
atom_id: ECON-SPEC-015
atom_set: regression-influence-diagnostics
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
requires:
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
related:
  - "[[Cook距离]]"
  - "[[DFFITS]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
---

# DFBETAS 分别标准化删去观测前后每个回归系数的变化，因此能定位总体影响由哪个参数承担
<!-- bilingual-en:start -->
*DFBETAS standardize the deletion-induced change in each regression coefficient separately, revealing which parameter carries the overall influence*
<!-- bilingual-en:end -->

> [!summary] 逐参数影响
> Cook 距离可以提示一个观测是否改变整组拟合，DFBETAS 则把问题拆到每个系数：它用删点误差尺度 $s_{(i)}$ 和全样本设计尺度 $\sqrt{c_{jj}}$ 标准化第 $j$ 个系数的删点变化。

先区分未标准化与标准化统计量。未标准化的 DFBETA 是

$$
DFBETA_{ij}=\hat\beta_j-\hat\beta_{j(i)},
$$

其中 $\hat\beta_{j(i)}$ 是删去观测 $i$ 后第 $j$ 个系数的估计。它仍保留系数原单位，不能跨参数直接比较。标准化后的 DFBETAS 是

$$
DFBETAS_{ij}
=\frac{\hat\beta_j-\hat\beta_{j(i)}}
{s_{(i)}\sqrt{c_{jj}}},
\qquad
c_{jj}=\left[(X'X)^{-1}\right]_{jj},
$$

其中 $s_{(i)}=\sqrt{MSE_{(i)}}$，而 $c_{jj}$ 来自全样本设计矩阵。每个观测会产生一整行 DFBETAS，每个系数对应一列。

## 解释边界

- $DFBETA_{ij}$ 是原单位变化；$DFBETAS_{ij}$ 才是以 $s_{(i)}\sqrt{c_{jj}}$ 缩放后的变化。两者名称和阈值不能混用。
- DFBETAS 的正负号表示全样本中的观测把该系数向哪个方向推动；筛查幅度时使用 $|DFBETAS_{ij}|$。
- 同一个观测可以强烈推动截距而几乎不改变斜率，也可以只影响某个交互项；因此“总体有影响”不等于“每个系数都受影响”。
- $2/\sqrt n$ 等经验线只适用于标准化的 DFBETAS，并且仍只是筛查线；具体边界见 [[影响诊断阈值]]。
- 以上公式针对普通线性 OLS。其他模型中的同名函数可能使用模型特定的工作残差、权重或近似，需按该模型文档解释。
- 该 OLS 公式还要求 $X$ 满列秩、删去 $i$ 后仍满列秩（因而 $h_{ii}<1$），且删点模型有正的残差自由度和 $s_{(i)}>0$；否则 $\hat\beta_{(i)}$ 或它的标准化量可能没有唯一或有限定义。

> [!question]- 最小自检
> 为什么不能把 `dfbeta()` 的输出直接与 $2/\sqrt n$ 比较？
>
> **答案：** `dfbeta()` 返回未标准化的系数原单位变化，而 $2/\sqrt n$ 是针对标准化 $|DFBETAS|$ 的经验筛查线。要使用该筛查线，应先确认拿到的是 `dfbetas()` 所定义的量。

## 来源与核验

- R Core Team, [`stats::influence.measures`](https://search.r-project.org/R/refmans/stats/html/influence.measures.html)：核验 `dfbeta()` 与 `dfbetas()` 是不同接口，并核对线性模型与其他模型的方法边界。
- University of Virginia Library, [Detecting Influential Points in Regression with DFBETA(S)](https://library.virginia.edu/data/articles/detecting-influential-points-in-regression-with-dfbetas)：核验 raw DFBETA、标准化 DFBETAS 的公式、符号解释与 $2/\sqrt n$ 筛查线。

作者逐项核验日 2026-08-30；本卡已完成独立模型复核。
