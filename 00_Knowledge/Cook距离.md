---
aliases:
  - "Cook 距离汇总删去一个观测后整组拟合值的标准化变化，因此同时受该观测的残差和杠杆作用"
  - "Cook's Distance"
  - "Cook's D"
student_os: knowledge-atom
atom_id: ECON-SPEC-013
atom_set: regression-influence-diagnostics
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
requires:
  - "[[Hat杠杆值]]"
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
related:
  - "[[DFFITS]]"
  - "[[DFBETAS]]"
---

# Cook 距离汇总删去一个观测后整组拟合值的标准化变化，因此同时受该观测的残差和杠杆作用
<!-- bilingual-en:start -->
*Cook's distance aggregates the standardized change in all fitted values after deleting one observation and therefore depends jointly on that observation's residual and leverage*
<!-- bilingual-en:end -->

> [!summary] 全局拟合敏感性，不是删点命令
> Cook 距离比较全样本拟合与删去观测 $i$ 后的拟合，并用参数数目和全样本误差方差进行缩放。它回答“整组拟合被这个观测推动多少”，不直接说明哪个系数变化，也不自动证明观测应被删除。

在满列秩、同方差线性 OLS 中，令 $p$ 为实际估计的参数数目（含截距，若有），$\mathrm{MSE}$ 为全样本回归的残差均方。还需要删去观测 $i$ 后的同一模型仍可估，对应于这里的 $h_{ii}<1$。将删点模型在所有原设计点上求值，得到 $\hat y_{j(i)}$，则

$$
D_i
=\frac{\sum_{j=1}^n(\hat y_j-\hat y_{j(i)})^2}
{p\,\mathrm{MSE}}.
$$

这也可以写成整组系数变化在 $X'X$ 几何下的长度：

$$
D_i
=\frac{(\hat\beta-\hat\beta_{(i)})'
X'X(\hat\beta-\hat\beta_{(i)})}
{p\,\mathrm{MSE}}.
$$

对普通最小二乘，它还等价于

$$
D_i
=\frac{e_i^2}{p\,\mathrm{MSE}}
\frac{h_{ii}}{(1-h_{ii})^2}.
$$

最后一个公式显示，Cook 距离把结果方向的偏离 $e_i$ 与设计空间的杠杆 $h_{ii}$ 结合起来。高杠杆观测若恰好满足 $e_i=0$，则 $D_i=0$；大残差若杠杆很低，也不必然产生极大的 Cook 距离。

## 边界

- Cook 距离是全局汇总量。要定位某个系数，转到 [[DFBETAS]]；要看观测自身设计点的拟合变化，转到 [[DFFITS]]。
- $D_i$ 大只表示当前模型对该观测敏感。它不区分录入错误、模型设定失配与真实稀有信息，不能代替 [[异常观测处理原则]]。
- Cook 距离的经验线只用于筛查；具体口径统一见 [[影响诊断阈值]]。
- 上述等价式依赖所述线性 OLS 口径且删点后模型仍可估。若 $h_{ii}=1$，删去该行会使设计矩阵秩亏，常规删点 Cook 公式不再有定义。广义线性模型和其他模型的 Cook 类距离也可使用不同近似和缩放，不应直接套用这组精确公式。

> [!question]- 最小自检
> 一个观测的 $h_{ii}$ 很大但小于 1，且它恰好落在全样本拟合面上，使 $e_i=0$。根据线性 OLS 公式，它的 Cook 距离是多少？
>
> **答案：** $D_i=0$。高杠杆只表示影响潜力；在该拟合下残差为零时，删点不会产生这个全局拟合差异。

## 来源与核验

- Penn State STAT 501, [Lesson 11.5: Identifying Influential Data Points](https://online.stat.psu.edu/stat501/Lesson11)：核验 Cook 距离通过逐一删点比较拟合，并同时反映残差和杠杆。
- NIST Dataplot, [Regression Diagnostics](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/regrdiag.htm)：核验 $D_i=e_i^2[p\,\mathrm{MSE}]^{-1}h_{ii}(1-h_{ii})^{-2}$ 以及 Cook 距离是整组回归系数/拟合变化的汇总诊断。
- R Core Team, [`stats::cooks.distance`](https://search.r-project.org/R/refmans/stats/html/influence.measures.html)：交叉核验线性与广义线性模型的 Cook 距离实现需区分缩放和近似。
- 作者逐项核验日：2026-08-30；本卡已完成独立模型复核。
