---
aliases:
  - "Moore-Penrose pseudoinverse"
  - "Moore–Penrose inverse"
  - "Pseudoinverse"
  - "伪逆"
student_os: knowledge-atom
atom_id: LA-PINV-004
atom_set: pseudoinverse-one-sided-inverses
atom_type: definition
status: source-checked
requires:
  - "[[广义逆]]"
  - "[[Hermitian 矩阵]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
leads_to:
  - "[[伪逆SVD公式]]"
  - "[[AA+列空间投影]]"
  - "[[A+A行空间投影]]"
related:
  - "[[伪逆存在唯一性]]"
---

<!-- bilingual-en:start -->
# Moore–Penrose pseudoinverse
<!-- bilingual-en:end -->

*Moore–Penrose 伪逆*

<!-- bilingual-en:start -->
An ordinary [[逆矩阵|inverse]] recovers an input exactly from its output. That can fail when a matrix loses some input information or cannot produce every possible output. The **Moore–Penrose pseudoinverse** extends the idea: it keeps the attainable part of a target and returns the shortest input that produces that part. “Shortest” uses the usual [[欧氏范数|Euclidean length]] in the stated coordinates.
<!-- bilingual-en:end -->

普通[[逆矩阵]]从输出精确恢复输入。如果矩阵丢失了部分输入信息，或不能产生所有输出，这件事便可能做不到。**Moore–Penrose 伪逆**把这个想法扩展为：先保留目标中能够产生的部分，再返回能产生该部分的最短输入。“最短”按既定坐标中的[[欧氏范数|欧氏长度]]衡量。

> [!example] Recover a sum, not the lost difference / 恢复和，无法恢复丢失的差
> <!-- bilingual-en:start -->
> Let $A=\begin{bmatrix}1&1\\0&0\end{bmatrix}$. It sends $(x_1,x_2)^T$ to $(x_1+x_2,0)^T$, retaining the sum and losing the difference. For target $b=(2,3)^T$, the second output can never be 3. The nearest attainable output is $(2,0)^T$. Among inputs with $x_1+x_2=2$, the shortest is $(1,1)^T$. The pseudoinverse performs this choice for every target:
>
> $$A^+=\begin{bmatrix}1/2&0\\1/2&0\end{bmatrix},\qquad A^+b=(1,1)^T.$$
>
> Here $T$ denotes [[矩阵转置|transpose]], used to write coordinate lists as columns. The superscript $+$ names the pseudoinverse; it is not an addition operation.
> <!-- bilingual-en:end -->
>
> 上述 $A$ 把输入变成 $(x_1+x_2,0)^T$：保留两项之和，丢失它们之差。目标为 $(2,3)^T$ 时，第二输出无法等于3；最近的可达输出是 $(2,0)^T$。在所有和为2的输入中，最短的是 $(1,1)^T$。显示的 $A^+$ 对任意目标都执行这个选择。$T$ 是[[矩阵转置|转置]]，上标 $+$ 是伪逆记号，不是加法。

<!-- bilingual-en:start -->
## Four equations specify this choice
<!-- bilingual-en:end -->

*四个方程怎样刻画这种选择*

<!-- bilingual-en:start -->
Let $A$ have $m$ rows and $n$ columns, with real or complex entries and $m,n\ge1$. A candidate inverse-like map $G$ must have $n$ rows and $m$ columns, reversing the direction from output coordinates to input coordinates. Write $^*$ for **conjugate transpose**: transpose, then replace each $a+bi$ by $a-bi$, where $i^2=-1$. For real matrices, it is ordinary transpose. The defining conditions are
<!-- bilingual-en:end -->

设 $A$ 有 $m$ 行、$n$ 列，$m,n\ge1$，元素可为实数或复数。候选反向映射 $G$ 必须有 $n$ 行、$m$ 列，把输出坐标送回输入坐标。$^*$ 表示**共轭转置**：先转置，再把每个 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$；实矩阵时就是普通转置。定义要求

$$
AGA=A,\qquad GAG=G,\qquad
(AG)^*=AG,\qquad (GA)^*=GA.
$$

<!-- bilingual-en:start -->
The first equation says that applying $A$, then $G$, then $A$ preserves every output originally attainable through $A$. The second requires the analogous consistency for $G$. They make the square products $AG$ and $GA$ [[幂等矩阵|idempotent]]: applying either twice has the same effect as applying it once. The last two equations make them [[Hermitian 矩阵|Hermitian]], meaning equal to their conjugate transpose. Together, these properties make the products [[正交投影矩阵判别|orthogonal projectors]], which keep a subspace component and remove its perpendicular component.
<!-- bilingual-en:end -->

第一式说：先用 $A$、再用 $G$、最后再用 $A$，不会改变原先由 $A$ 产生的输出。第二式对 $G$ 要求相应的一致性。它们使方阵 $AG$ 和 $GA$ 都[[幂等矩阵|幂等]]，即作用两次与一次相同。后两式再要求它们为 [[Hermitian 矩阵|Hermitian 矩阵]]，也就是等于各自的共轭转置。合在一起，这些条件使两乘积成为[[正交投影矩阵判别|正交投影矩阵]]：保留一个子空间中的分量，去掉垂直分量。

<!-- bilingual-en:start -->
For $G=A^+$, the output-space product [[AA+列空间投影|$AA^+$]] projects onto the [[列空间|column space]] $C(A)$, the set of all attainable outputs. The input-space product [[A+A行空间投影|$A^+A$]] projects onto $C(A^*)=N(A)^\perp$. Here the [[零空间|zero-output space]] $N(A)$ contains inputs erased by $A$, and $\perp$ denotes the [[正交补|orthogonal complement]]. Over the reals, $C(A^T)$ is the row space written as column vectors.
<!-- bilingual-en:end -->

令 $G=A^+$，输出空间中的乘积 [[AA+列空间投影|$AA^+$]] 投影到[[列空间]] $C(A)$，即全部可产生输出的集合。输入空间中的乘积 [[A+A行空间投影|$A^+A$]] 投影到 $C(A^*)=N(A)^\perp$。这里[[零空间]] $N(A)$ 包含被 $A$ 消去的输入，$\perp$ 表示[[正交补]]；实数域中的 $C(A^T)$ 就是把行向量改写为列向量后的行空间。

$$
AA^+=\begin{bmatrix}1&0\\0&0\end{bmatrix},
\qquad
A^+A=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}
\quad\text{for the example above.}
$$

<!-- bilingual-en:start -->
The first keeps $(b_1,0)^T$; the second replaces $(x_1,x_2)^T$ by two copies of their average. Both are symmetric and idempotent, but they act on different spaces. Only full row rank gives $AA^+=I_m$, and only full column rank gives $A^+A=I_n$, where $I_k$ is the [[单位矩阵|identity matrix]] of size $k$.
<!-- bilingual-en:end -->

第一乘积保留 $(b_1,0)^T$；第二乘积把 $(x_1,x_2)^T$ 换成两项相同的平均值。它们都对称且幂等，却作用在不同空间。只有满行秩时 $AA^+=I_m$，只有满列秩时 $A^+A=I_n$；$I_k$ 是 $k$ 阶[[单位矩阵]]。

> [!warning] One equation allows other choices / 一个方程允许别的选择
> <!-- bilingual-en:start -->
> For the same $A$, $G=\begin{bmatrix}1&0\\0&0\end{bmatrix}$ satisfies $AGA=A$ and even $GAG=G$. But $GA=\begin{bmatrix}1&1\\0&0\end{bmatrix}$ is not symmetric, so $G$ is not $A^+$. It returns $(2,0)^T$ for the target, whose length is larger than that of $(1,1)^T$. A [[广义逆|generalized inverse]] defined only by $AGA=A$ need not make the Moore–Penrose choice.
> <!-- bilingual-en:end -->
>
> 同一个 $A$ 下，上述 $G$ 满足 $AGA=A$，甚至满足 $GAG=G$，但 $GA$ 不对称，所以不是 $A^+$。它对目标返回 $(2,0)^T$，长度大于 $(1,1)^T$。只要求 $AGA=A$ 的[[广义逆]]并不自动作出 Moore–Penrose 选择。

<!-- bilingual-en:start -->
## Definition and existence are separate
<!-- bilingual-en:end -->

## 定义与定理的边界

<!-- bilingual-en:start -->
The four equations define the object. That every finite real or complex matrix has exactly one such object is the [[伪逆存在唯一性|existence-and-uniqueness theorem]]. The [[伪逆SVD公式|SVD formula]] constructs it by reversing positive singular stretches. A square invertible matrix has $A^+=A^{-1}$; a zero $m\times n$ matrix has the zero $n\times m$ pseudoinverse. No probability model or full-rank assumption is needed.
<!-- bilingual-en:end -->

四个方程给出定义；每个有限维实或复矩阵都恰好拥有一个这样的对象，是[[伪逆存在唯一性|存在唯一性定理]]。[[伪逆SVD公式]]通过撤销正奇异伸缩给出构造。可逆方阵满足 $A^+=A^{-1}$；$m\times n$ 零矩阵的伪逆是 $n\times m$ 零矩阵。这些结论不需要概率模型或满秩条件。

<!-- bilingual-en:start -->
The opening “nearest output, then shortest input” description is proved in [[最小范数最小二乘解|the minimum-norm least-squares theorem]]. It does not recover erased information. Also, exact uniqueness does not guarantee insensitivity to noise: [[小奇异值放大噪声|small positive singular values can amplify errors]].
<!-- bilingual-en:end -->

开头“先找最近输出，再选最短输入”的描述由[[最小范数最小二乘解]]证明。它无法恢复已经丢失的信息；精确唯一也不保证对噪声不敏感，[[小奇异值放大噪声|小正奇异值可以放大误差]]。

## 来源与核验

<!-- bilingual-en:start -->
- R. Penrose, [“A generalized inverse for matrices” (1955), pp.406–408](https://doi.org/10.1017/S0305004100030401): four complex-matrix equations, existence/uniqueness and the invertible special case; the original PDF was checked.
- [Axler, *Linear Algebra Done Right*, 6.67–6.70, PDF pp.234–236](https://linear.axler.net/LADR4e.pdf#page=234): the geometric inverse between the zero-output complement and the attainable-output space, and minimum-norm fitting.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8, PDF pp.1–2]] and [[03_Positive Definite Matrices and Applications#3.8.4 Moore–Penrose 四条件|course §3.8.4]]: local notation and the two projection roles. The two small matrices above are verified directly.
<!-- bilingual-en:end -->

- Penrose 原论文第406–408页核对四条件、实复任意形状矩阵的存在唯一性及可逆特例。
- Axler 第6.67–6.70条核对零空间正交补与可达空间之间的逆及最小范数拟合。
- MIT Session 3.8 第1–2页与课程第3.8.4节核对记号与两类投影；两个小矩阵由直接乘法核验。
