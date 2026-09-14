---
aliases:
  - "Jordan 块的矩阵指数等于标量指数乘以有限阶多项式"
  - Matrix exponential of a Jordan block
  - Jordan 块的指数
student_os: knowledge-atom
atom_id: LA-EIG-032
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan块]]"
  - "[[矩阵指数]]"
related:
  - "[[Jordan块的幂]]"
  - "[[对角化计算矩阵指数]]"
  - "[[连续系统谱稳定性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 块的矩阵指数等于标量指数乘以有限阶多项式
<!-- bilingual-en:start -->
*The exponential of a Jordan block is a scalar exponential times a finite polynomial*
<!-- bilingual-en:end -->

> [!summary] 核心公式
> 对 $r\times r$ Jordan 块 $J=\lambda I+N$，其中 $N^r=0$，
> $$e^{tJ}=e^{\lambda t}\sum_{j=0}^{r-1}\frac{t^j}{j!}N^j.$$
> 因此，一个大小为 $r$ 的块最多带来 $t^{r-1}e^{\lambda t}$ 这一阶的时间因子。
> <!-- bilingual-en:start -->
> The nilpotent part truncates the exponential series after degree $r-1$.
> <!-- bilingual-en:end -->

因为 $\lambda I$ 与 $N$ 可交换，
$$e^{tJ}=e^{t(\lambda I+N)}=e^{\lambda t}e^{tN}.$$
而 $N^r=0$，所以
$$e^{tN}=I+tN+\frac{t^2}{2!}N^2+\cdots+\frac{t^{r-1}}{(r-1)!}N^{r-1}$$
是有限和。这个多项式不是近似截断，而是精确公式。

对二阶块，
$$
J=\begin{bmatrix}\lambda&1\\0&\lambda\end{bmatrix}
\quad\Longrightarrow\quad
e^{tJ}=e^{\lambda t}\begin{bmatrix}1&t\\0&1\end{bmatrix}.
$$
若 $\operatorname{Re}\lambda<0$，指数衰减最终压过固定次数多项式；若 $\operatorname{Re}\lambda=0$ 且块大小大于一，多项式因子会使某些轨迹无界。这正是连续时间稳定性在虚轴边界上必须检查 Jordan 块的原因。
<!-- bilingual-en:start -->
Negative real part dominates the polynomial factor, whereas a nontrivial block on the imaginary axis produces polynomial growth.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 若 $N^3=0$，$e^{tN}$ 的精确展开到哪一项停止？
>
> **答案：** $e^{tN}=I+tN+\tfrac12t^2N^2$；从 $N^3$ 起全部为零。

## 来源与核验

- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#Jordan块的幂与Jordan块的指数|课程 Jordan 块计算]]：核对幂零部分的有限指数展开与连续时间边界。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S10_Lecture_Lecture_23_Differential_Equations_and_expAt.pdf|MIT Lecture 23 transcript]]：核对矩阵指数与常系数线性系统模式。
