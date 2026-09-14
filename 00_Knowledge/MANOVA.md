---
aliases:
  - "MANOVA 是在共享设计的多响应线性模型中联合检验预设系数与响应假设的方法"
  - "单因素 MANOVA 检验两个或更多组的总体均值向量是否相等"
  - "Multivariate Analysis of Variance"
  - "多元方差分析"
student_os: knowledge-atom
atom_id: STAT-MAN-001
atom_set: manova
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[多响应线性回归]]"
  - "[[多响应一般线性假设]]"
related:
  - "[[多响应与多解释变量回归]]"
  - "[[pooled Hotelling T²]]"
  - "[[多响应联合显著边界]]"
  - "[[多响应因果边界]]"
---

# MANOVA 是在共享设计的多响应线性模型中联合检验预设系数与响应假设的方法
<!-- bilingual-en:start -->
*MANOVA jointly tests prespecified coefficient and response hypotheses in a multivariate linear model with a shared design*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 在多响应线性模型 $Y=XB+U$ 中，同一观测的多个响应共享设计矩阵 $X$ 的同一行。MANOVA 用假设 SSCP $H$ 与误差 SSCP $E$ 联合检验预先写明的系数—响应假设；一般形式可写成 [[多响应一般线性假设|$H_0:CBA=D$]]。
>
> 最小特例是单因素设计。第 $i$ 组第 $j$ 个独立观测的 $p$ 维响应写作
> $$Y_{ij}=\mu_i+\varepsilon_{ij},\qquad i=1,\ldots,g.$$
> 此时零假设是
> $$H_0:\mu_1=\mu_2=\cdots=\mu_g,$$
> 备择是至少一对总体均值向量不同。MANOVA 检验的是**预先选定的一整组响应**，不是把 $p$ 个单变量检验结果简单拼在一起。
> <!-- bilingual-en:start -->
> MANOVA is a joint hypothesis framework for several responses sharing one design. One-way equality of group mean vectors is its simplest special case, not its full definition.
> <!-- bilingual-en:end -->

也可以写成 $Y_{ij}=\mu+\tau_i+\varepsilon_{ij}$，但 $\mu$ 与 $\tau_i$ 必须配合一种识别约束，例如 $\sum_i n_i\tau_i=0$。在这种参数化下，“无组别效应”与均值向量相等是同一个**可估检验**；不能在没有约束时把每个 $\tau_i$ 当成独立可识别参数。

把组别编码进设计矩阵 $X$，单因素 MANOVA 就回到上述多响应线性模型。一般因子、协变量和预设响应组合继续写成 [[多响应一般线性假设|$CBA=D$]]；因此 “MANOVA” 指联合假设的结构，不等于“预测变量只能是分类变量”。

两个边界很有用：

- $p=1$ 时，问题退化为普通单因素 ANOVA；
- $g=2$ 且采用共同协方差的经典模型时，组均值向量相等的检验与 [[pooled Hotelling T²|pooled 两样本 Hotelling $T^2$]] 等价。实务上常把 MANOVA 留给三组及以上，但这不是定义门槛。

例如，三种教学方案各产生 $(\text{数学},\text{阅读},\text{写作})^T$。零假设是三组的三个坐标**作为一个向量**完全相同。拒绝它只说明至少一个组均值对比在某个响应方向上不同；它不自动推出三科都不同，见 [[多响应联合显著边界]]。能否进一步说教学方案造成了差异，取决于分配与识别设计，见 [[多响应因果边界]]。

> [!question]- 自检
> 两组学生、每人有三个成绩时，能否称为 MANOVA 问题？
>
> **答案：** 可以。$g=2$ 已满足定义；在独立组、共同协方差的经典条件下，它与 pooled 两样本 Hotelling $T^2$ 给出同一均值向量检验。

## 来源与核验

- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5. 多个总体均值向量比较：单因子 MANOVA|本地多元统计课程 §1.5]]：核对单因素模型、均值向量零假设和课程题型。
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对 one-way MANOVA 的组别—向量观测结构、$H_0:\mu_1=\cdots=\mu_g$ 与备择。
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm)：核对 MANOVA 是共享设计的多响应线性模型联合检验，并可对系数方向和响应方向同时作线性对比。
<!-- bilingual-en:start -->
- The local course fixes the one-way formulation; Penn State and the SAS GLM documentation verify the mean-vector hypothesis and its general multivariate-linear-model embedding.
<!-- bilingual-en:end -->
