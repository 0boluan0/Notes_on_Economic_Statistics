---
aliases:
  - "DFFITS 衡量删去观测 i 后其自身设计点的拟合值改变了多少个删点标准误"
  - Difference in fits
  - DFFITS
student_os: knowledge-atom
atom_id: ECON-SPEC-014
atom_set: regression-influence-diagnostics
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hat杠杆值]]"
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
related:
  - "[[Cook距离]]"
  - "[[DFBETAS]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
---

# DFFITS 衡量删去观测 i 后其自身设计点的拟合值改变了多少个删点标准误
<!-- bilingual-en:start -->
*DFFITS measures how many deleted-model standard errors the fitted value at observation i changes when that observation is omitted*
<!-- bilingual-en:end -->

> [!summary] 诊断对象
> DFFITS 只回答一个局部问题：删去观测 $i$ 后，在同一个设计点 $x_i$ 上的拟合值改变了多少。它把这个变化除以删点模型下的标准误，因此不同响应尺度上的结果可以比较。

令 $\hat\beta_{(i)}$ 为删去观测 $i$ 后重新估计的系数，并令

$$
\hat y_{i(i)}=x_i'\hat\beta_{(i)}
$$

表示删点模型在原设计点 $x_i$ 上的预测。在普通同方差线性 OLS 中，

$$
DFFITS_i
=\frac{\hat y_i-\hat y_{i(i)}}{\sqrt{MSE_{(i)}h_{ii}}}
=t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}},
$$

其中 $MSE_{(i)}$ 是删去观测 $i$ 后的残差均方，$t_i$ 是外部学生化残差，$h_{ii}$ 是全样本设计矩阵的杠杆值。这个恒等式说明 DFFITS 同时需要结果方向偏离和杠杆：相同的 $t_i$ 在更高杠杆处会产生更大的拟合敏感性。

## 解释边界

- DFFITS 有正负号；符号表示全样本拟合值相对删点拟合值移动的方向。筛查时看的是 $|DFFITS_i|$。
- 它衡量 $x_i$ 这一处的拟合改变，不等于所有拟合值的总体改变；后者由 [[Cook距离]] 汇总。
- 它也不指出哪个回归系数承担了变化；逐系数诊断见 [[DFBETAS]]。
- 大 DFFITS 只说明该观测值得检查，不能证明记录错误，也不能单独授权删除。
- 上式属于普通线性 OLS 的删点诊断；加权、广义线性或相关数据模型需要使用相应模型的定义或近似，不能直接套用。
- 该等式还要求全样本设计矩阵满列秩、删去 $i$ 后仍满列秩（因而 $h_{ii}<1$），且删点模型有正的残差自由度和 $MSE_{(i)}>0$。第一个分式还要求 $h_{ii}>0$；当 $h_{ii}=0$ 时，可按右侧恒等式的连续延拓记为 0。其他情形下，DFFITS 不按这组公式定义。

> [!question]- 最小自检
> 一个观测的外部学生化残差不算很大，但 $h_{ii}$ 非常接近 1，它的 DFFITS 为什么仍可能很大？
>
> **答案：** 因为 $DFFITS_i=t_i\sqrt{h_{ii}/(1-h_{ii})}$。杠杆因子在 $h_{ii}$ 接近 1 时迅速增大，所以中等的结果方向偏离也可能显著改变该设计点的拟合值；这仍只是筛查信号。

## 来源与核验

- Penn State STAT 501, [Lesson 11: Influential Points](https://online.stat.psu.edu/stat501/Lesson11)：核验 DFFITS 的删点定义、分母以及 $t_i\sqrt{h_{ii}/(1-h_{ii})}$ 恒等式。
- R Core Team, [`stats::influence.measures`](https://search.r-project.org/R/refmans/stats/html/influence.measures.html)：核验 `dffits()` 属于回归删点诊断接口，并核对它与其他影响量的对象边界。

作者逐项核验日 2026-08-30；本卡已完成独立模型复核。
