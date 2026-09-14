---
aliases:
  - "面板 pooled OLS 把所有单位时期观测堆叠后用复合误差估计共同线性斜率"
  - Pooled OLS in panel data
  - 面板混合最小二乘
student_os: knowledge-atom
atom_id: ECON-PANEL-003
atom_set: panel-data
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[面板误差分解]]"
leads_to:
  - "[[Pooled OLS正交条件]]"
contrasts_with:
  - "[[个体固定效应]]"
  - "[[随机效应]]"
---

# 面板 pooled OLS 把所有单位时期观测堆叠后用复合误差估计共同线性斜率

<!-- bilingual-en:start -->
*Panel pooled OLS stacks all unit-period observations and estimates a common linear slope with a composite error*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在
> $$
> y_{it}=\alpha+x_{it}'\beta+c_i+u_{it}
> $$
> 中，pooled OLS 把每个 $(i,t)$ 当作一行普通线性回归，并把 $v_{it}=c_i+u_{it}$ 整体放入误差项，以一个共同的 $\beta$ 拟合所有单位和时期。
>
> <!-- bilingual-en:start -->
> In $y_{it}=\alpha+x_{it}'\beta+c_i+u_{it}$, pooled OLS treats each $(i,t)$ as one row of an ordinary linear regression, places the composite term $v_{it}=c_i+u_{it}$ in the error, and fits one common $\beta$ across units and periods.
> <!-- bilingual-en:end -->

它同时使用单位之间和单位内部的变异，没有通过去均值或单位 indicators 消去 $c_i$。因此 pooled OLS 可以是一个描述性线性投影，也可以在相应条件下估计结构斜率；“把数据堆起来”本身不提供后一种解释，放行条件见 [[Pooled OLS正交条件]]。
<!-- bilingual-en:start -->
It uses both between-unit and within-unit variation and does not eliminate $c_i$ by demeaning or unit indicators. Pooled OLS may be a descriptive linear projection or, under suitable conditions, an estimator of a structural slope. Stacking the data alone does not justify the latter interpretation; the required moment belongs to the separate orthogonality atom.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> pooled OLS 与截面 OLS 的计算外形相似，但面板里误差为什么更特别？
>
> **答案：** 同一单位各期共享 $c_i$，所以复合误差不仅可能与回归量相关，也通常在单位内相关。

## 来源与核验

- MIT OpenCourseWare, [14.382 Lecture 8](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf), §2.4：把 pooled 方法写成以 $v_{it}=a_i+\epsilon_{it}$ 为复合误差的普通回归。
- [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 15 章 PDF 页 169–170：核对课程的 pooled 估计记号与顺序。
