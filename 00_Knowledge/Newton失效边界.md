---
aliases:
  - "Newton 迭代没有括区间不变量，可能因零或近零导数、不合适的初值、其他吸引域与周期而失效"
  - Newton iteration has no bracketing invariant and can fail through zero derivatives, poor initial values, other basins, or cycles
student_os: knowledge-atom
atom_id: CS-NR-009
atom_set: numerical-root-finding
atom_type: failure-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Newton迭代]]"
related:
  - "[[Newton局部收敛]]"
  - "[[数值迭代停止条件]]"
  - "[[条件性与算法稳定性]]"
  - "[[求根残差]]"
  - "[[残差控制根误差]]"
leads_to:
  - "[[混合求根]]"
  - "[[求根策略]]"
part_of:
  - "[[数值求根.canvas|数值求根]]"
  - "[[导数的应用.canvas]]"
---

# Newton 迭代没有括区间不变量，可能因零或近零导数、不合适的初值、其他吸引域与周期而失效
<!-- bilingual-en:start -->
*Newton iteration has no bracketing invariant and can fail through a zero or near-zero derivative, a poor initial value, another basin of attraction, or a cycle*
<!-- bilingual-en:end -->

> [!summary] 原子失败边界
> Newton 每轮只依赖当前局部切线，没有持续保存“某个根仍在可信区间内”的证书。于是，算出一个有限的下一点不代表根存在，也不代表它靠近原先想找的根。失败可能表现为公式无定义、巨大跳步、发散、周期、越界，或稳定地收敛到另一个根。
>
> <!-- bilingual-en:start -->
> Each Newton step uses only the current local tangent; it does not preserve a certificate that a root remains inside a trusted interval. A finite next point therefore proves neither that a root exists nor that the point is approaching the intended root. Failure can appear as an undefined update, a huge jump, divergence, a cycle, a domain escape, or stable convergence to a different root.
> <!-- bilingual-en:end -->

## 四类机制要分开诊断

1. **非根处的零或近零导数。** 若 $f(x_k)\ne0$ 而 $f'(x_k)=0$，更新式没有定义；若导数绝对值很小，$-f(x_k)/f'(x_k)$ 可能成为远离局部模型的巨大步长。若 $f(x_k)=0$，则应先按已找到根返回，不再做除法。
2. **初值落在别的吸引域。** 多根函数会把不同初值送往不同根。收敛到另一个根在数值上未必发散，却可能没有回答任务指定的问题。
3. **周期或发散。** 对

   $$
   f(x)=x^3-2x+2,
   $$

   从 $x_0=0$ 出发有 $x_1=1$，再由 $x_1=1$ 得 $x_2=0$，形成 $0\leftrightarrow1$ 的二周期。
4. **定义域与机器数失败。** 候选可能越出函数定义域，或产生 NaN、overflow 和舍入停滞。程序继续吐出数值不能把它们改造成收敛证据。

单轮残差变小也不是单调保证；Newton 并不普遍最小化 $|f(x)|$。同理，$x_{k+1}\approx x_k$ 可能只是舍入后走不动。运行时应把 [[求根残差|残差]]、导数、步长、有限值、迭代上限和失败状态交给 [[数值迭代停止条件]]，并用 [[残差控制根误差]] 判断函数值缺口能否解释为位置误差。

## 保护机制改变的是路径，不是定理前提

换初值、阻尼或限制步长可能改善某个问题，却不自动提供全局证书。若有连续异号括区间，[[混合求根]] 可以拒绝越界或缺乏进展的 Newton 候选并退回二分；它之所以更稳，是因为括区间不变量一直保留，而不是因为 Newton 的局部定理被扩成了全局定理。

> [!question]- 自检
> Newton 从某个初值稳定收敛到根 $r_2$，而任务原先要找同一函数的另一个根 $r_1$。这算“算法收敛成功”吗？
>
> **答案：** 对“找到任意一个根”的规格可以成功；对“找到 $r_1$”的规格则失败。必须先写清目标根怎样由区间、符号或其他条件识别，不能只看数值序列是否稳定。

## 来源与核验

- MIT 18.330, [*Introduction to Numerical Analysis, Chapter 4: Nonlinear equations*](https://ocw.mit.edu/courses/18-330-introduction-to-numerical-analysis-spring-2012/5b325bfa56a599794c7196de926844b0_MIT18_330S12_Chapter4.pdf)：核对 Newton 的初值依赖、局部性和收敛失败。
- SciPy, [`newton`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)：核对初值、导数、步长停止不保证根以及有括区间时更安全算法的官方边界。
- [[01_Math/01_calculus/02_Applications_of_Differentiation.md#Exam 2 第 5 题：Newton 法为什么失败|微积分课程反例]]：核对水平切线和迭代失败的课程语境。

> [!success] 独立内容审核通过
> 非根处零导数、吸引域、周期反例、机器数失败、关系与来源均已通过第二位模型复审；`status: source-checked`。学习证据尚未评估，`mastery_state: unassessed`。
