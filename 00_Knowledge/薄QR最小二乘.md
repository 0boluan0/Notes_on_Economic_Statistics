---
aliases:
  - Thin QR least-squares solve
  - Reduced QR least squares
  - QR 最小二乘
student_os: knowledge-atom
atom_id: LA-PROJ-010
atom_set: orthogonal-projection-least-squares
atom_type: procedure
status: source-checked
requires:
  - "[[薄QR分解]]"
  - "[[线性最小二乘]]"
related:
  - "[[Gram-Schmidt数值失稳]]"
  - "[[正规方程条件数平方]]"
contrasts_with:
  - "[[秩亏最小二乘算法]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---
<!-- bilingual-en:start -->
# Least squares with thin QR
<!-- bilingual-en:end -->

*薄 QR 最小二乘*

<!-- bilingual-en:start -->
Thin QR solves a [[线性最小二乘|linear least-squares problem]] by changing the coordinates of the output. It first reads the part of the target that the matrix can reproduce, then solves a triangular system for the original coefficients. The part perpendicular to the column space remains as the unavoidable residual.
<!-- bilingual-en:end -->

薄 QR 通过改变输出坐标来求解[[线性最小二乘]]：先读出目标中矩阵能够重建的那部分，再解一个三角方程组，找回原来的组合系数。垂直于列空间的部分，则成为无法消除的残差。

<!-- bilingual-en:start -->
For example, let
<!-- bilingual-en:end -->

例如，取

$$
A=\begin{bmatrix}1&1\\0&1\\0&0\end{bmatrix},\qquad
b=\begin{bmatrix}3\\2\\4\end{bmatrix},\qquad
Ax=\begin{bmatrix}x_1+x_2\\x_2\\0\end{bmatrix}.
$$

<!-- bilingual-en:start -->
The two entries of $x=(x_1,x_2)^T$ are the unknown column-combination coefficients; $T$ denotes [[矩阵转置|transpose]]. No choice of them can make the third output equal 4. We can match the first two outputs by taking $x_2=2$ and then $x_1=1$. Thin QR organizes exactly this separation, including when the reachable plane is tilted rather than aligned with coordinate axes.
<!-- bilingual-en:end -->

$x=(x_1,x_2)^T$ 的两个元素是未知的列组合系数；$T$ 表示[[矩阵转置|转置]]。无论怎么选系数，第三输出都不可能等于4；但可以先取 $x_2=2$，再取 $x_1=1$，使前两项恰好匹配。薄 QR 把这种分离组织成通用计算步骤，即使能够到达的平面是倾斜的，也适用。

> [!example] Read coordinates, then solve backwards / 先读坐标，再回代
> <!-- bilingual-en:start -->
> For this $A$, take the [[薄QR分解|thin factorization]] $A=QR$ below. The columns of $Q$ are perpendicular unit directions. Reading $b$ along them gives $Q^Tb=(3,2)^T$; the third coordinate is omitted. Solve the last equation $\hat x_2=2$, then the first equation $\hat x_1+\hat x_2=3$. This gives $\hat x=(1,2)^T$, fitted vector $A\hat x=(3,2,0)^T$, and residual $r=b-A\hat x=(0,0,4)^T$.
> <!-- bilingual-en:end -->
>
> $$Q=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix},\quad R=\begin{bmatrix}1&1\\0&1\end{bmatrix},\quad R\hat x=Q^Tb=\begin{bmatrix}3\\2\end{bmatrix}.$$
>
> 对这个 $A$，采用上面的[[薄QR分解]]。$Q$ 的列是彼此垂直的单位方向，沿它们读取 $b$ 得到 $Q^Tb=(3,2)^T$，第三坐标不在其中。先解最后一式 $\hat x_2=2$，再解第一式 $\hat x_1+\hat x_2=3$，便有 $\hat x=(1,2)^T$。拟合向量为 $(3,2,0)^T$，残差为 $r=b-A\hat x=(0,0,4)^T$。

<!-- bilingual-en:start -->
The squared error is $(x_1+x_2-3)^2+(x_2-2)^2+16$. Its minimum is 16, since the first two squares can both be zero. The residual length is 4; do not confuse that length with its square.
<!-- bilingual-en:end -->

误差平方为 $(x_1+x_2-3)^2+(x_2-2)^2+16$。前两个平方可以同时为零，因此最小值为16。残差长度则是4，要区分长度与长度平方。

<!-- bilingual-en:start -->
## Why the triangular system gives the minimum
<!-- bilingual-en:end -->

## 为什么三角方程就是最小二乘

<!-- bilingual-en:start -->
Now let $A\in\mathbb R^{m\times n}$, $m\ge n\ge1$, have [[线性无关|independent columns]], and let $b\in\mathbb R^m$ be the target. Independence means full column rank and makes the least-squares coefficient [[最小二乘解唯一性|unique]]. In a thin QR, $A=QR$, $Q$ has size $m\times n$ with $Q^TQ=I_n$, and $R$ is an invertible $n\times n$ [[三角矩阵|upper-triangular matrix]]. Here $I_n$ is the [[单位矩阵|identity matrix]], and $\|\cdot\|_2$ denotes [[欧氏范数|Euclidean length]].
<!-- bilingual-en:end -->

现在设 $A\in\mathbb R^{m\times n}$、$m\ge n\ge1$，其各列[[线性无关]]，目标为 $b\in\mathbb R^m$。列独立就是满列秩，也使最小二乘系数[[最小二乘解唯一性|唯一]]。在薄 QR 中，$A=QR$，$Q$ 的尺寸为 $m\times n$ 且 $Q^TQ=I_n$，$R$ 是可逆的 $n\times n$ [[三角矩阵|上三角矩阵]]。这里 $I_n$ 为[[单位矩阵]]，$\|\cdot\|_2$ 表示[[欧氏范数|欧氏长度]]。

<!-- bilingual-en:start -->
Complete the columns of $Q$ to a square [[正交矩阵|orthogonal matrix]] $[Q\ Q_\perp]$. The $m-n$ columns of $Q_\perp$ are perpendicular unit directions in the [[正交补|orthogonal complement]] of the [[列空间|column space]] $C(A)$, the set of all possible outputs $Ax$. Changing to this full orthonormal coordinate system preserves length and yields
<!-- bilingual-en:end -->

把 $Q$ 的列补成方形[[正交矩阵]] $[Q\ Q_\perp]$。$Q_\perp$ 的 $m-n$ 列，是[[列空间]] $C(A)$ 的[[正交补]]中的彼此垂直单位方向；$C(A)$ 包含全部可能的输出 $Ax$。换到这一整组标准正交坐标保持长度，于是

$$
\begin{aligned}
\|Ax-b\|_2^2
&=\left\|\begin{bmatrix}Q^T\\Q_\perp^T\end{bmatrix}(QRx-b)\right\|_2^2\\
&=\|Rx-Q^Tb\|_2^2+\|Q_\perp^Tb\|_2^2.
\end{aligned}
$$

<!-- bilingual-en:start -->
The first term measures error in the reachable directions; the second measures the target's perpendicular part and cannot be changed by choosing $x$. Since $R$ is invertible, exactly one $\hat x$ makes the first term zero. Therefore the computational rule is
<!-- bilingual-en:end -->

第一项测量可到达方向上的误差；第二项测量目标的垂直部分，无论怎样选 $x$ 都不能改变。由于 $R$ 可逆，恰好一个 $\hat x$ 能使第一项为零，所以计算规则是

$$
\boxed{R\hat x=Q^Tb.}
$$

<!-- bilingual-en:start -->
Solve this system by **back substitution**: the last row determines the last coefficient, and each preceding row determines the next coefficient once the later ones are known. There is no need to form $R^{-1}$. Nor must a program store the full $m\times m$ matrix: it needs the first $n$ transformed coordinates $Q^Tb$, and Householder implementations can apply the transformations without explicitly building $Q$.
<!-- bilingual-en:end -->

用**回代**解该系统：最后一行先确定最后一个系数，再由后往前，把已知系数代入上一行。不必显式形成 $R^{-1}$，程序也不必存储完整 $m\times m$ 方阵：它需要的是前 $n$ 个变换后坐标 $Q^Tb$，Householder 实现还可以直接施加变换而不显式构造 $Q$。

<!-- bilingual-en:start -->
Rebuilding the matched coordinates gives the [[标准正交基投影公式|orthogonal projection]] $p=A\hat x=QQ^Tb$. The residual and its length are
<!-- bilingual-en:end -->

把匹配后的坐标重新合成，就得到[[标准正交基投影公式|正交投影]] $p=A\hat x=QQ^Tb$。残差及其长度为

$$
r=b-QQ^Tb=Q_\perp Q_\perp^Tb\in C(A)^\perp,\qquad
\|r\|_2=\|Q_\perp^Tb\|_2.
$$

<!-- bilingual-en:start -->
Here $C(A)^\perp$ means all vectors perpendicular to every reachable output. When $m=n$, there is no complementary block: the second term is zero, and the invertible square system fits every $b$ exactly. For a tall matrix, the residual can also be zero when $b$ happens to lie in $C(A)$.
<!-- bilingual-en:end -->

$C(A)^\perp$ 表示垂直于每个可到达输出的向量集合。若 $m=n$，没有正交补块，第二项为零；可逆方形系统对任意 $b$ 都能精确拟合。高矩阵情形下，若 $b$ 恰好在 $C(A)$ 中，残差也可以为零。

<!-- bilingual-en:start -->
## Rank and numerical limits
<!-- bilingual-en:end -->

## 使用边界

<!-- bilingual-en:start -->
This rule requires full column rank. With exactly dependent columns, a thin factorization may still exist, but $R$ is singular and the triangular system need not have a solution or a unique one. For example, replacing $R$ above by $\begin{bmatrix}1&1\\0&0\end{bmatrix}$ makes its second equation $0=2$. The least-squares problem still has solutions, but demanding a zero first term in the displayed split is then impossible. Use a [[秩亏最小二乘算法|rank-deficient least-squares method]] and state how a coefficient vector is selected from the minimizers.
<!-- bilingual-en:end -->

这条规则要求满列秩。列精确相关时，薄分解仍可能存在，但 $R$ 奇异，三角方程组可能无解，或解不唯一。例如把上例的 $R$ 换成 $\begin{bmatrix}1&1\\0&0\end{bmatrix}$，第二式就成为 $0=2$。最小二乘问题仍有解，只是此时不能要求分解中的第一项降到零。应使用[[秩亏最小二乘算法]]，并说明在多个最小化系数中怎样选择。

<!-- bilingual-en:start -->
For a nearly dependent but still independent matrix, the algebra remains valid. A stable Householder QR implementation avoids explicitly forming $A^TA$ and the [[正规方程条件数平方|squaring of the matrix condition number]] in that route. It cannot remove sensitivity already present in the data: small input changes may cause large coefficient changes, and [[条件性与算法稳定性|a stable algorithm and a well-conditioned problem]] are different requirements. The residual's size also matters in general least-squares forward-error bounds. Classical Gram–Schmidt may [[Gram-Schmidt数值失稳|lose orthogonality in finite precision]], so the name “QR” alone is not a numerical guarantee. Declaring a small direction to be zero is a separate [[数值秩|numerical-rank decision]] with a stated tolerance.
<!-- bilingual-en:end -->

若矩阵只是接近列相关、仍然列独立，上述代数依旧成立。稳定实现的 Householder QR 避免显式形成 $A^TA$，也避开该路线中的[[正规方程条件数平方|矩阵条件数平方]]；但它不能消除数据本身的敏感性，小输入变化仍可能造成大系数变化。[[条件性与算法稳定性|算法稳定与问题良态]]是不同要求，一般最小二乘前向误差界还涉及残差大小。经典 Gram–Schmidt 在有限精度下可能[[Gram-Schmidt数值失稳|丢失正交性]]，所以仅说“用了 QR”并非数值保证。把一个很小的方向当成零，则是需要声明容差的独立[[数值秩]]决定。

<!-- bilingual-en:start -->
For complex $A,b$ with the usual Euclidean inner product, replace transpose by conjugate transpose throughout: $Q^*Q=I_n$, $R\hat x=Q^*b$, and $p=QQ^*b$. The squared norm is the sum of squared absolute values. The same full-column-rank requirement and perpendicular-error decomposition apply.
<!-- bilingual-en:end -->

对采用通常欧氏内积的复数 $A,b$，把转置全部换成共轭转置：$Q^*Q=I_n$、$R\hat x=Q^*b$、$p=QQ^*b$；范数平方为绝对值平方和。满列秩要求和垂直误差分解保持不变。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK Users’ Guide, QR Factorization](https://www.netlib.org/lapack/lug/node40.html): the transformed objective, triangular solve, residual norm, and implicit application of orthogonal transformations. [Linear Least Squares Problems](https://www.netlib.org/lapack/lug/node27.html) distinguishes full-rank QR/LQ from rank-deficient solvers.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary, PDF pp.1–3]] and [[02_Least Squares, Determinants and Eigenvalues#从 Gram–Schmidt 到 QR 分解|course section: From Gram–Schmidt to QR]]: orthonormal coordinates, triangular factors and the local least-squares notation.
- [MIT 18.335J Week 4, Lectures 9–10](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-4/) and [LAPACK least-squares error bounds](https://www.netlib.org/lapack/lug/node83.html): Householder stability, Gram–Schmidt limitations and residual-dependent sensitivity. The small matrix example and rank-deficient failure are direct calculations.
<!-- bilingual-en:end -->

- LAPACK QR 章节核对换坐标后的目标、三角求解、残差范数及隐式施加正交变换；最小二乘章节区分满秩 QR/LQ 与秩亏求解器。
- MIT Session 2.4（PDF第1–3页）及课程“从 Gram–Schmidt 到 QR 分解”核对标准正交坐标、上三角因子及本地最小二乘记号。
- MIT 18.335J 第9–10讲和 LAPACK 最小二乘误差界核对 Householder 稳定性、Gram–Schmidt 边界及残差相关的敏感性；小矩阵算例与秩亏失效例直接计算核验。
