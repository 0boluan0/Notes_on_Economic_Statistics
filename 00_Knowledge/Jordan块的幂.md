---
aliases:
  - "Jordan 块的非负整数幂由有限二项式展开给出"
  - Powers of a Jordan block
  - Jordan 块的幂
student_os: knowledge-atom
atom_id: LA-EIG-013
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan块]]"
related:
  - "[[对角化计算矩阵幂]]"
  - "[[Jordan块的指数]]"
  - "[[矩阵幂趋零判据]]"
  - "[[离散系统谱稳定性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 块的非负整数幂由有限二项式展开给出
<!-- bilingual-en:start -->
*Nonnegative integer powers of a Jordan block are given by a finite binomial expansion*
<!-- bilingual-en:end -->

> [!summary] 核心公式
> 对 $r\times r$ Jordan 块 $J=\lambda I+N$，其中 $N^r=0$，以及任意整数 $m\ge0$，
> $$J^m=\sum_{j=0}^{\min(m,r-1)}\binom mj\lambda^{m-j}N^j.$$
> 当 $m=0$ 时，和式只有 $j=0$ 一项，给出 $J^0=I$。
> <!-- bilingual-en:start -->
> The nilpotent part truncates the binomial expansion after $N^{r-1}$.
> <!-- bilingual-en:end -->

因为 $\lambda I$ 与 $N$ 可交换，普通二项式公式可以直接用于 $(\lambda I+N)^m$；又因为 $N^r=0$，所有 $j\ge r$ 的项消失。若 $\lambda\ne0$，每个 $N^j$ 外的系数含有一个关于 $m$ 的至多 $j$ 次多项式因子，因此块大小决定了矩阵幂可能出现的多项式阶数。

对二阶块，当 $m\ge1$ 时，
$$
J=\begin{bmatrix}\lambda&1\\0&\lambda\end{bmatrix}
\quad\Longrightarrow\quad
J^m=\begin{bmatrix}
\lambda^m&m\lambda^{m-1}\\
0&\lambda^m
\end{bmatrix}.
$$
$m=0$ 必须单独使用 $J^0=I$；这样就不会出现没有意义的 $0\cdot\lambda^{-1}$。
当 $m=1$ 且 $\lambda=0$ 时，上式的超对角元按 $\lambda^0=1$ 计算，因此仍得到 $J$；这里没有把一般的 $0^0$ 当作独立运算，而是在二项式中读取零次幂项。

当 $|\lambda|<1$ 时，指数衰减最终压过这些固定次数多项式；当 $|\lambda|=1$ 且块大小大于一时，多项式因子会使矩阵幂无界。例如 $J_2(1)^m=\begin{bmatrix}1&m\\0&1\end{bmatrix}$。
<!-- bilingual-en:start -->
Inside the unit circle, exponential decay dominates the polynomial factor. On the unit circle, a nontrivial block causes polynomial growth.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> $J=\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 的 $J^0$ 与 $J^3$ 分别是什么？
>
> **答案：** $J^0=I$，而 $J^3=\begin{bmatrix}1&3\\0&1\end{bmatrix}$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对 Jordan 块结构。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#Jordan块的幂与Jordan块的指数|课程 Jordan 块计算]]：核对二项式展开与单位圆边界的多项式增长。
