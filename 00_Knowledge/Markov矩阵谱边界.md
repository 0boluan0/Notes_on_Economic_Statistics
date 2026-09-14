---
aliases:
  - "有限 Markov 矩阵的全部特征值都位于闭单位圆内"
  - Spectral boundary of stochastic matrices
  - Markov 矩阵单位圆谱边界
student_os: knowledge-atom
atom_id: LA-EIG-040
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵]]"
  - "[[谱半径]]"
related:
  - "[[Markov矩阵必有特征值一]]"
  - "[[Markov单位圆特征值半单]]"
  - "[[Markov稳态唯一判据]]"
  - "[[Markov矩阵幂收敛判据]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 有限 Markov 矩阵的全部特征值都位于闭单位圆内
<!-- bilingual-en:start -->
*Every eigenvalue of a finite Markov matrix lies in the closed unit disk*
<!-- bilingual-en:end -->

> [!summary] 随机性给出的谱约束
> 对有限行随机或列随机矩阵 $P$，每个特征值都满足
> $$|\lambda|\le1.$$
> <!-- bilingual-en:start -->
> Every eigenvalue of a finite stochastic matrix lies in the closed unit disk.
> <!-- bilingual-en:end -->

先看列随机约定。$P^k$ 仍是列随机矩阵，因而
$$
\|P^k\|_1=1\qquad(k\ge0).
$$
若 $\lambda$ 是 $P$ 的特征值，则 $\lambda^k$ 是 $P^k$ 的特征值，因此
$$
|\lambda|^k\le \|P^k\|_1=1.
$$
这对每个 $k$ 都成立，只能有 $|\lambda|\le1$。行随机约定同理，只需改用 $\|P^k\|_\infty=1$。

这里的结论是“闭单位圆”，不是“开单位圆”。置换矩阵也是 Markov 矩阵，它可以拥有 $-1$ 或其他单位根；这些模式不增长，却可能让矩阵幂持续振荡。是否真正收敛，需要更强的[[Markov矩阵幂收敛判据|谱条件]]。
<!-- bilingual-en:start -->
Stochasticity keeps every matrix power norm-bounded. Since $\lambda^k$ is an eigenvalue of $P^k$, this excludes eigenvalues outside the unit disk. Other unit roots may still occur and obstruct convergence.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> Markov 矩阵除 $1$ 外的所有特征值是否都必须严格满足 $|\lambda|<1$？
>
> **答案：** 不必须。周期链可以有 $-1$ 或其他单位根；随机性只保证它们不越出闭单位圆。
>
> <!-- bilingual-en:start -->
> Must every eigenvalue of a Markov matrix other than one satisfy $|\lambda|<1$?
>
> **Answer:** No. A periodic chain may have $-1$ or other roots of unity; stochasticity only keeps them inside the closed unit disk.
> <!-- bilingual-en:end -->

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：直接核对闭单位圆谱边界。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对列随机矩阵幂保持随机性与范数有界。
<!-- bilingual-en:start -->
- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/) was checked for the closed-unit-disk spectral bound.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]] was checked for preservation of stochasticity and norm-bounded powers under the column-stochastic convention.
<!-- bilingual-en:end -->
