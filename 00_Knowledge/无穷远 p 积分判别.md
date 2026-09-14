---
aliases:
  - "尾部 p 积分从一到无穷当且仅当 p 大于一收敛"
  - p-integral at infinity
  - 无穷远 p 积分判别
student_os: knowledge-atom
atom_id: CALC-IMP-004
atom_set: improper-integrals
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[反常积分]]"
part_of:
  - "[[反常积分.canvas]]"
---

# 尾部 p 积分从一到无穷当且仅当 p 大于一收敛
<!-- bilingual-en:start -->
*The tail $p$-integral from one to infinity converges exactly when $p>1$*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> $$\int_1^\infty x^{-p}\,dx\text{ 收敛}\iff p>1.$$
> 当 $p>1$ 时值为 $1/(p-1)$；$p=1$ 是对数发散的临界点；$p<1$ 时截断积分按幂次发散，其中 $p\le0$ 时被积函数甚至不趋于零。
> <!-- bilingual-en:start -->
>
> &nbsp;
> $\int_1^\infty x^{-p}\,dx$ converges exactly when $p>1$. For $p>1$ its value is $1/(p-1)$. The case $p=1$ is the logarithmically divergent boundary. For $p<1$ the truncated integral diverges by a power; when $p\le0$, the integrand does not even tend to zero.
> <!-- bilingual-en:end -->

对 $p\ne1$，
$$\int_1^b x^{-p}dx=\frac{b^{1-p}-1}{1-p}.$$
若 $p>1$，$b^{1-p}\to0$；若 $p<1$，该幂不趋于有限值。$p=1$ 必须单独用 $\int_1^b dx/x=\log b$ 处理，不能代入含 $1-p$ 分母的公式。
<!-- bilingual-en:start -->
For $p\ne1$, $\int_1^b x^{-p}dx=(b^{1-p}-1)/(1-p)$. If $p>1$, $b^{1-p}\to0$; if $p<1$, that power has no finite limit. The boundary $p=1$ must be treated through $\int_1^b dx/x=\log b$, not substituted into a formula containing $1-p$ in the denominator.
<!-- bilingual-en:end -->

这个结果主要用作比较基准。若 $f(x)$ 在无穷远近似 $C/x^p$ 且 $C>0$，真正要识别的是指数 $p$ 是否越过 $1/x$ 的临界尺度，而不是函数最终“看起来很小”。
<!-- bilingual-en:start -->
The result is chiefly a comparison benchmark. If $f(x)$ behaves like $C/x^p$ with $C>0$ at infinity, the decisive question is whether $p$ crosses the critical $1/x$ scale, not whether the function eventually looks small.
<!-- bilingual-en:end -->

> [!question]- 自检
> $\int_{10}^{\infty}x^{-3/2}dx$ 是否因下限不是 1 而需要新判别？
>
> **答案：** 不需要。有限起点的改变不影响尾部敛散；$3/2>1$，所以收敛。

## 来源与核验

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses91e_Lecture_Notes.pdf|MIT 18.01SC Session 91e]]：核对三种 $p$ 情形与收敛值。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf|MIT 18.01SC Session 93c]]：核对无穷远与零附近的临界方向对照。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses91e_Lecture_Notes.pdf|MIT 18.01SC Session 91e]] was checked for all three $p$ cases and the convergent value.
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/05_Infinite_Series_and_Improper_Integrals/Ses93c_Lecture_Notes.pdf|MIT 18.01SC Session 93c]] was checked for the contrast between behaviour at infinity and near zero.
<!-- bilingual-en:end -->
