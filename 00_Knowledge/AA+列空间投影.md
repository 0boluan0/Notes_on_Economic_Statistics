---
aliases:
  - "AA plus column-space projector"
  - "Pseudoinverse column-space projection identity"
  - "AA+ 投影公式"
student_os: knowledge-atom
atom_id: LA-PROJ-008
atom_set: orthogonal-projection-least-squares
atom_type: identity
status: source-checked
requires:
  - "[[正交投影矩阵判别]]"
  - "[[Moore–Penrose 伪逆]]"
related:
  - "[[A+A行空间投影]]"
  - "[[最小二乘解唯一性]]"
leads_to:
  - "[[最小范数最小二乘解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

<!-- bilingual-en:start -->
# Column-space projection with AA⁺
<!-- bilingual-en:end -->

*AA⁺ 列空间投影*

<!-- bilingual-en:start -->
A matrix $A$ can produce only certain output vectors: its [[列空间|column space]] $C(A)$ is the set of all $Ax$. Given a target $b$ that may lie outside this space, $AA^+b$ keeps the part that can be produced and removes the perpendicular mismatch. Thus $AA^+$ acts on the **output space**, where the target lives. The symbol $A^+$ denotes the [[Moore–Penrose 伪逆|Moore–Penrose pseudoinverse]].
<!-- bilingual-en:end -->

矩阵 $A$ 只能产生某些输出向量；它的[[列空间]] $C(A)$ 就是全部 $Ax$ 组成的集合。目标 $b$ 可能不在其中，$AA^+b$ 会保留能够产生的部分，去掉与这个空间垂直的不匹配部分。因此，$AA^+$ 作用在目标所在的**输出空间**。$A^+$ 表示 [[Moore–Penrose 伪逆]]。

> [!example] Two coefficients, one output direction / 两个系数，一个输出方向
> <!-- bilingual-en:start -->
> Take
>
> $$A=\begin{bmatrix}1&1\\2&2\end{bmatrix},\qquad b=\begin{bmatrix}2\\1\end{bmatrix},\qquad A^+=\frac1{10}\begin{bmatrix}1&2\\1&2\end{bmatrix}.$$
>
> Every output is $(s,2s)^T$, where $s=x_1+x_2$. The target $(2,1)^T$ cannot have this form. Multiplication gives
>
> $$AA^+=\frac15\begin{bmatrix}1&2\\2&4\end{bmatrix},\qquad
> p=AA^+b=\begin{bmatrix}4/5\\8/5\end{bmatrix},\qquad
> r=b-p=\begin{bmatrix}6/5\\-3/5\end{bmatrix}.$$
>
> The fitted vector $p$ lies on the output line, and the [[向量点积|dot product]] $(1,2)r=6/5-6/5=0$ shows that the residual $r$ is perpendicular to it.
> <!-- bilingual-en:end -->
>
> 上述 $A$ 的每个输出都形如 $(s,2s)^T$，其中 $s=x_1+x_2$，所以目标 $(2,1)^T$ 无法精确产生。矩阵相乘给出显示的 $p,r$：拟合向量 $p$ 在输出直线上，残差 $r$ 与方向 $(1,2)^T$ 的[[向量点积|点积]]为零，故与整条直线垂直。

<!-- bilingual-en:start -->
For any finite real or complex $m\times n$ matrix, with positive integers $m,n$, the identity is
<!-- bilingual-en:end -->

对任意有限维实或复 $m\times n$ 矩阵，其中 $m,n$ 为正整数，都有

$$
AA^+=P_{C(A)}.
$$

<!-- bilingual-en:start -->
$P_{C(A)}$ denotes [[正交投影|orthogonal projection]] onto $C(A)$ under the usual Euclidean inner product. Since $A^+$ is $n\times m$, the product $AA^+$ is $m\times m$, so it takes an $m$-coordinate target to another $m$-coordinate vector. It is not the coefficient vector $A^+b$, which has $n$ coordinates.
<!-- bilingual-en:end -->

$P_{C(A)}$ 表示通常欧氏内积下向 $C(A)$ 的[[正交投影]]。$A^+$ 是 $n\times m$，因此 $AA^+$ 是 $m\times m$：输入是有 $m$ 个坐标的目标，输出也是 $m$ 个坐标。它不是有 $n$ 个坐标的系数向量 $A^+b$。

<!-- bilingual-en:start -->
## Seeing the retained directions
<!-- bilingual-en:end -->
## SVD 直接显出列空间投影

<!-- bilingual-en:start -->
For [[矩阵秩|rank]] $r>0$, the [[紧SVD|compact SVD]] and [[伪逆SVD公式|pseudoinverse formula]] give
<!-- bilingual-en:end -->

当[[矩阵秩|秩]] $r>0$ 时，[[紧SVD]]与[[伪逆SVD公式]]给出

$$
A=U_r\Sigma_rV_r^*,\qquad A^+=V_r\Sigma_r^{-1}U_r^*,\qquad
AA^+=U_rU_r^*.
$$

<!-- bilingual-en:start -->
The columns of $U_r$ are an [[标准正交组|orthonormal list]] spanning $C(A)$. In $U_rU_r^*b$, the right factor reads the coordinates of $b$ along those directions and the left factor reconstructs that component. This is the [[标准正交基投影公式|orthonormal-basis projection formula]]. The middle stretches cancel because $\Sigma_r$ contains only positive singular values and $V_r^*V_r=I_r$, where $I_r$ is the [[单位矩阵|identity matrix]].
<!-- bilingual-en:end -->

$U_r$ 的列是一组张成 $C(A)$ 的[[标准正交组]]。在 $U_rU_r^*b$ 中，右侧因子读出 $b$ 沿这些方向的坐标，左侧因子再合成这一部分；这就是[[标准正交基投影公式]]。中间的伸缩量能够消去，是因为 $\Sigma_r$ 只含正奇异值，并且 $V_r^*V_r=I_r$，其中 $I_r$ 是[[单位矩阵]]。

<!-- bilingual-en:start -->
The star $*$ means [[矩阵转置|conjugate transpose]], replacing each $a+bi$ by $a-bi$ after transposition, with $i^2=-1$. For real matrices it is $T$. Consequently $AA^+$ is [[Hermitian 矩阵|Hermitian]] and [[幂等矩阵|idempotent]]: repeating the projection changes nothing. Its range is $C(A)$, and its nullspace is $C(A)^\perp=N(A^*)$, the output directions perpendicular to all attainable outputs. Here $\perp$ denotes the [[正交补|orthogonal complement]], and $N(A^*)$ is the [[零空间|nullspace]] of $A^*$.
<!-- bilingual-en:end -->

$*$ 表示[[矩阵转置|共轭转置]]，转置后把每个 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$；实矩阵时就是 $T$。因此 $AA^+$ 为 [[Hermitian 矩阵]]且[[幂等矩阵|幂等]]，重复投影不再改变结果。其值域为 $C(A)$，零空间为 $C(A)^\perp=N(A^*)$，即垂直于全部可达输出的方向；$\perp$ 表示[[正交补]]，$N(A^*)$ 是 $A^*$ 的[[零空间]]。

<!-- bilingual-en:start -->
Why is this the best fit? If $p=AA^+b$ and $r=b-p$, then every other attainable output $y\in C(A)$ satisfies
<!-- bilingual-en:end -->

为什么它给出最佳拟合？令 $p=AA^+b$、$r=b-p$，则任何其他可达输出 $y\in C(A)$ 都满足

$$
\|b-y\|_2^2=\|r\|_2^2+\|p-y\|_2^2.
$$

<!-- bilingual-en:start -->
The [[最佳逼近与正交残差|perpendicular-error identity]] separates an unavoidable error from an additional nonnegative error, which vanishes only at $y=p$. The symbol $\|\cdot\|_2$ denotes [[内积|Euclidean length]], using squared absolute values for complex coordinates. Thus the fitted output is unique. In the opening example, both $(4/5,0)^T$ and $(2/5,2/5)^T$ produce $p$, so the coefficients need not be unique. Choosing the shortest coefficient vector is the separate [[最小范数最小二乘解|minimum-norm theorem]].
<!-- bilingual-en:end -->

[[最佳逼近与正交残差|垂直误差恒等式]]把总误差分成无法消除的部分与额外非负部分；后者仅在 $y=p$ 为零。$\|\cdot\|_2$ 表示[[内积|欧氏长度]]，复坐标使用绝对值平方。所以拟合输出唯一。开头的 $(4/5,0)^T$ 与 $(2/5,2/5)^T$ 却都能产生同一个 $p$，说明系数未必唯一；从中选最短系数是另一个[[最小范数最小二乘解|最小范数定理]]的职责。

<!-- bilingual-en:start -->
## When every target is retained
<!-- bilingual-en:end -->
## 满秩与单位矩阵的边界

<!-- bilingual-en:start -->
The identity $AA^+=I_m$ holds exactly when $C(A)$ fills the whole output space, that is, when $\operatorname{rank}(A)=m$ (**full row rank**). The [[单位矩阵|identity matrix]] retains every target. Full column rank alone is insufficient: for $A=(1,0)^T$, $AA^+=\operatorname{diag}(1,0)$, not $I_2$. Conversely, the full-row-rank matrix $A=[1\ 1]$ has $AA^+=[1]$ despite having two input coordinates.
<!-- bilingual-en:end -->

$AA^+=I_m$ 当且仅当 $C(A)$ 填满整个输出空间，也就是 $\operatorname{rank}(A)=m$，即**满行秩**。此时[[单位矩阵]]保留每个目标。仅满列秩并不足够：$A=(1,0)^T$ 满列秩，但 $AA^+=\operatorname{diag}(1,0)$，不是 $I_2$。反过来，满行秩矩阵 $A=[1\ 1]$ 虽然有两个输入坐标，仍满足 $AA^+=[1]$。

<!-- bilingual-en:start -->
If $A=0$, its column space is $\{0\}$ and $AA^+=0$: every target projects to zero. Rank deficiency never prevents this projector from existing. The companion product [[A+A行空间投影|$A^+A$]] instead acts on the input space and removes directions invisible to $A$. In floating-point work, discarding positive singular values defines a different matrix $A_\tau$; the resulting projector is $A_\tau A_\tau^+$, and should be described as the projector onto that retained subspace.
<!-- bilingual-en:end -->

若 $A=0$，列空间为 $\{0\}$，$AA^+=0$，所有目标都投到零。秩亏并不妨碍这个投影存在。配对乘积 [[A+A行空间投影|$A^+A$]] 则作用在输入空间，去掉 $A$ 看不见的方向。浮点计算若删去正奇异值，就改用了另一个矩阵 $A_\tau$；所得投影是 $A_\tau A_\tau^+$，应说明它投向的是该保留子空间。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 6.69(b) and 6.70, PDF pp.235–236](https://linear.axler.net/LADR4e.pdf#page=235): the output-space projector and unique least-squares fit for arbitrary rank.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8, PDF pp.1–2]]: SVD inverse construction and real row/column-space projection interpretation.
- [[03_Positive Definite Matrices and Applications#3.8.3 SVD 定义伪逆|Course §3.8.3]] and [[03_Positive Definite Matrices and Applications#3.8.5 伪逆同时解决两类“最佳解”|§3.8.5]]: course notation and two best-solution roles; [[最小二乘解唯一性|least-squares uniqueness]] supplies the coefficient/fit distinction. The numerical example and rank boundaries were checked directly.
<!-- bilingual-en:end -->

- Axler 第6.69(b)、6.70条核对任意秩下的输出空间投影与唯一最佳拟合。
- MIT Session 3.8 第1–2页核对 SVD 逆构造及实数行、列空间的投影解释。
- [[03_Positive Definite Matrices and Applications#3.8.3 SVD 定义伪逆|课程第3.8.3节]]与[[03_Positive Definite Matrices and Applications#3.8.5 伪逆同时解决两类“最佳解”|第3.8.5节]]核对记号与两种最佳解职责；[[最小二乘解唯一性]]核对系数与拟合的区别。算例和秩边界由直接乘法核验。
