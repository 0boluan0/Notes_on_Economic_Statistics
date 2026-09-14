---
aliases:
  - "面板 pooled OLS 只有在复合误差与回归量总体正交时才一致"
  - Pooled OLS orthogonality in panel data
student_os: knowledge-atom
atom_id: ECON-PANEL-016
atom_set: panel-data
atom_type: assumption
status: source-checked
mastery_state: unassessed
part_of:
  - "[[面板数据：Pooled OLS、固定效应与随机效应.canvas]]"
requires:
  - "[[面板Pooled OLS]]"
  - "[[OLS一致性条件]]"
related:
  - "[[遗漏变量偏差]]"
---

# 面板 pooled OLS 只有在复合误差与回归量总体正交时才一致

<!-- bilingual-en:start -->
*Panel pooled OLS is consistent only when regressors are population-orthogonal to the composite error*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 令 $v_{it}=c_i+u_{it}$。模型显式含截距时，共同斜率的一项核心总体矩条件是
> $$
> \operatorname{Cov}(x_{it},v_{it})
> =E[(x_{it}-E x_{it})(v_{it}-E v_{it})]=0.
> $$
> 若把常数并入 $w_{it}=(1,x_{it}')'$ 并把复合误差归一化为零均值，完整正规方程可写成 $E[w_{it}v_{it}]=0$。还需要回归量二阶矩可逆和相应的样本矩收敛。
>
> <!-- bilingual-en:start -->
> Let $v_{it}=c_i+u_{it}$. With an explicit intercept, a core population moment for the common slope is $\operatorname{Cov}(x_{it},v_{it})=0$. If the constant is included in $w_{it}=(1,x_{it}')'$ and the composite error is normalised to have mean zero, the full normal equations can be written as $E[w_{it}v_{it}]=0$. An invertible regressor second moment and the corresponding sample-moment convergence are also required.
> <!-- bilingual-en:end -->

若稳定能力 $c_i$ 同时提高教育 $x_{it}$ 和工资 $y_{it}$，pooled OLS 会把一部分能力差异归给教育。增加行数、随机抽取工人或只更换标准误都不会恢复这个总体正交条件；这正是 [[遗漏变量偏差]] 在面板复合误差中的表现。
<!-- bilingual-en:start -->
If stable ability $c_i$ raises both education $x_{it}$ and wages $y_{it}$, pooled OLS attributes part of the ability difference to education. More rows, random sampling of workers, or a different standard error does not restore population orthogonality; this is omitted-variable bias operating through the panel composite error.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么“样本中的人是随机抽来的”仍不足以保证 pooled OLS 一致？
>
> **答案：** 随机抽样说明怎样从总体抽单位，不保证总体中 $c_i$ 与 $x_{it}$ 正交。

## 来源与核验

- MIT OpenCourseWare, [14.382 Lecture 8](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf), §2.4：把 pooled 复合误差写成 $v_{it}=a_i+\epsilon_{it}$，并明确要求其与回归量正交。
- [[OLS一致性条件]]：复用总体正交、二阶矩可逆与样本矩收敛的通用结论。
