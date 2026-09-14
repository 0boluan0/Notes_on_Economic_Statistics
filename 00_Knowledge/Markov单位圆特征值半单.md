---
aliases:
  - "有限 Markov 矩阵的每个单位圆特征值都是半单的"
  - Semisimplicity of unit-circle eigenvalues of stochastic matrices
  - Markov 单位圆谱半单性
student_os: knowledge-atom
atom_id: LA-EIG-049
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov矩阵]]"
  - "[[离散系统有界判据]]"
related:
  - "[[Markov矩阵谱边界]]"
  - "[[Markov矩阵幂收敛判据]]"
  - "[[Markov稳态唯一判据]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 有限 Markov 矩阵的每个单位圆特征值都是半单的
<!-- bilingual-en:start -->
*Every unit-circle eigenvalue of a finite Markov matrix is semisimple*
<!-- bilingual-en:end -->

> [!summary] 单位圆上的 Jordan 边界
> 对有限行随机或列随机矩阵 $P$，若 $|\lambda|=1$，则 $\lambda$ 对应的 Jordan 块全为 $1\times1$；等价地，$\lambda$ 是半单特征值。
>
> <!-- bilingual-en:start -->
> If $P$ is a finite row- or column-stochastic matrix and $|\lambda|=1$, then every Jordan block associated with $\lambda$ has size one; equivalently, $\lambda$ is semisimple.
> <!-- bilingual-en:end -->

列随机矩阵的每个幂 $P^k$ 仍是列随机矩阵，所以 $\|P^k\|_1=1$；行随机约定下则有 $\|P^k\|_\infty=1$。因此两种约定都使矩阵幂一致有界。

若某个满足 $|\lambda|=1$ 的特征值带有大于 $1\times1$ 的 Jordan 块，[[离散系统有界判据]]中的块幂公式会产生 $k\lambda^{k-1}$ 等多项式增长项，使 $\|P^k\|$ 无界。这与随机矩阵幂的有界性矛盾，所以单位圆上的 Jordan 块只能是一阶。
<!-- bilingual-en:start -->
Every power of a column-stochastic matrix has one-norm equal to one, and every power of a row-stochastic matrix has infinity-norm equal to one. A nontrivial Jordan block at $|\lambda|=1$ would instead create polynomially growing terms such as $k\lambda^{k-1}$. Power-boundedness therefore forces every unit-circle eigenvalue to be semisimple.
<!-- bilingual-en:end -->

半单性不意味着单位圆上只有特征值 $1$。置换矩阵仍可拥有 $-1$ 或其他单位根；半单性只排除了沿这些模式的多项式增长，不排除持续振荡。
<!-- bilingual-en:start -->
Semisimplicity does not mean that one is the only eigenvalue on the unit circle. Permutation matrices may still have $-1$ or other roots of unity; semisimplicity removes polynomial growth, not persistent oscillation.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 为什么有限 Markov 矩阵不可能在特征值 $1$ 处有一个 $2\times2$ Jordan 块？
>
> **答案：** 该块的 $k$ 次幂会出现与 $k$ 成正比的项，使矩阵幂无界；但 Markov 矩阵的每个幂仍是随机矩阵，在相应的 $1$-范数或 $\infty$-范数下始终有界。
>
> <!-- bilingual-en:start -->
> Why can a finite Markov matrix not have a $2\times2$ Jordan block at eigenvalue one?
>
> **Answer:** Powers of that block contain a term proportional to $k$ and would be unbounded, whereas every power of a stochastic matrix remains bounded in the appropriate one- or infinity-norm.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对列随机矩阵的幂仍保持随机性，因此在 $1$-范数下一致有界。
- [[离散系统有界判据]]：核对单位圆上的非平凡 Jordan 块会导致矩阵幂出现多项式增长，以及有界幂必排除这种块。
<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]] was checked for preservation of stochasticity under powers and the resulting uniform one-norm bound.
- [[离散系统有界判据|Power-boundedness criterion for discrete systems]] was checked for the polynomial growth caused by nontrivial Jordan blocks on the unit circle and their exclusion under bounded powers.
<!-- bilingual-en:end -->
