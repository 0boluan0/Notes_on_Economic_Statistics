---
aliases:
  - "A plus A row-space projector"
  - "Pseudoinverse row-space projection identity"
  - "Pseudoinverse input-space projector"
student_os: knowledge-atom
atom_id: LA-PROJ-014
atom_set: orthogonal-projection-least-squares
atom_type: identity
status: source-checked
requires:
  - "[[正交投影矩阵判别]]"
  - "[[Moore–Penrose 伪逆]]"
related:
  - "[[AA+列空间投影]]"
leads_to:
  - "[[最小范数最小二乘解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

<!-- bilingual-en:start -->
# Input-space projection with A⁺A
<!-- bilingual-en:end -->

*A⁺A 行空间投影*

<!-- bilingual-en:start -->
Two different inputs can give the same output under a matrix $A$. Their difference is then invisible to $A$. The product $A^+A$ removes that invisible part from an input, while preserving its output. It acts on the **input space**, whereas [[AA+列空间投影|$AA^+$]] acts on the output space. Here $A^+$ is the [[Moore–Penrose 伪逆|Moore–Penrose pseudoinverse]].
<!-- bilingual-en:end -->

两个不同输入经过矩阵 $A$ 后，可能给出相同输出；两者的差就是 $A$ 看不见的方向。$A^+A$ 会从输入中删去这部分，同时保留原来的输出。它作用在**输入空间**，而 [[AA+列空间投影|$AA^+$]] 作用在输出空间；$A^+$ 是 [[Moore–Penrose 伪逆]]。

> [!example] Keeping the sum, removing the difference / 保留总和，去掉差异
> <!-- bilingual-en:start -->
> Let $A=[1\ 1]$. It sees only $x_1+x_2$, so the input $x=(3,1)^T$ produces 4. Since $A^+=(1/2,1/2)^T$,
>
> $$A^+A=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix},\qquad
> A^+Ax=\begin{bmatrix}2\\2\end{bmatrix},\qquad
> x-A^+Ax=\begin{bmatrix}1\\-1\end{bmatrix}.$$
>
> Both $(3,1)^T$ and $(2,2)^T$ have sum 4. The removed part has sum zero and is perpendicular to $(2,2)^T$. Their squared lengths are $10$ and $8$: removing an invisible component has shortened the input without changing the output. The superscript $T$ denotes [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> $A=[1\ 1]$ 只能看见 $x_1+x_2$，所以 $(3,1)^T$ 产生输出4。上述乘积把它变成 $(2,2)^T$，删去的 $(1,-1)^T$ 总和为零，并与保留部分垂直。两个输入都产生4，但长度平方从10降到8：输出没变，输入中的不可见分量消失了。$T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
For a finite real or complex $m\times n$ matrix, $m,n\ge1$, the exact identity is
<!-- bilingual-en:end -->

对任意有限维实或复 $m\times n$ 矩阵，$m,n\ge1$，精确恒等式为

$$
A^+A=P_{C(A^*)}=P_{N(A)^\perp}.
$$

<!-- bilingual-en:start -->
The [[零空间|nullspace]] $N(A)=\{z:Az=0\}$ collects the invisible inputs. Its [[正交补|orthogonal complement]] $N(A)^\perp$ contains all vectors perpendicular to them. This is the [[列空间|column space]] of $A^*$; $P$ denotes [[正交投影|orthogonal projection]] under the usual Euclidean inner product. The star means [[矩阵转置|conjugate transpose]], so for real matrices this becomes $C(A^T)$, the row space represented by column vectors. The product is $n\times n$: it takes an input to another input.
<!-- bilingual-en:end -->

[[零空间]] $N(A)=\{z:Az=0\}$ 收集全部不可见输入。其[[正交补]] $N(A)^\perp$ 包含与这些方向都垂直的向量，恰好等于 $A^*$ 的[[列空间]]。$P$ 表示通常欧氏内积下的[[正交投影]]；星号表示[[矩阵转置|共轭转置]]。实矩阵时目标空间就是 $C(A^T)$，即把行向量写作列向量后的行空间。乘积尺寸为 $n\times n$，把输入映到另一个输入。

<!-- bilingual-en:start -->
## Why the right singular directions remain
<!-- bilingual-en:end -->
## SVD 直接显出行空间投影

<!-- bilingual-en:start -->
For positive [[矩阵秩|rank]] $r$, use the [[紧SVD|compact SVD]] and the [[伪逆SVD公式|pseudoinverse formula]]:
<!-- bilingual-en:end -->

对正[[矩阵秩|秩]] $r$，使用[[紧SVD]]和[[伪逆SVD公式]]：

$$
A=U_r\Sigma_rV_r^*,\qquad A^+=V_r\Sigma_r^{-1}U_r^*,\qquad
A^+A=V_rV_r^*.
$$

<!-- bilingual-en:start -->
The $r$ columns of $V_r$ are the input directions attached to positive singular values. They form an [[标准正交组|orthonormal list]] spanning $C(A^*)=N(A)^\perp$. The product $V_r^*x$ reads the coordinates along these directions; $V_r$ reconstructs their part of $x$. The [[标准正交基投影公式|orthonormal-basis projection formula]] therefore gives the identity above. In particular, $A^+A$ is [[Hermitian 矩阵|Hermitian]] (equal to its conjugate transpose) and [[幂等矩阵|idempotent]], its range is $C(A^*)$, and its nullspace is exactly $N(A)$.
<!-- bilingual-en:end -->

$V_r$ 的 $r$ 列是正奇异值对应的输入方向，构成张成 $C(A^*)=N(A)^\perp$ 的[[标准正交组]]。$V_r^*x$ 读出沿这些方向的坐标，$V_r$ 再合成这一部分，因此[[标准正交基投影公式]]给出上述恒等式。特别地，$A^+A$ 为 [[Hermitian 矩阵]]（等于自身共轭转置）且[[幂等矩阵|幂等]]，值域为 $C(A^*)$，零空间恰好为 $N(A)$。

<!-- bilingual-en:start -->
Write $q=A^+Ax$ and $z=x-q$. Then $Az=Ax-AA^+Ax=0$, using the Penrose equation $AA^+A=A$. Thus
<!-- bilingual-en:end -->

令 $q=A^+Ax$、$z=x-q$，由 Penrose 方程 $AA^+A=A$，有 $Az=Ax-AA^+Ax=0$。所以

$$
x=q+z,\qquad Aq=Ax,\qquad q\perp z,\qquad
\|x\|_2^2=\|q\|_2^2+\|z\|_2^2.
$$

<!-- bilingual-en:start -->
The [[欧氏范数|Euclidean norm]] measures length; for complex coordinates its square is the sum of squared absolute values. Every input producing $Ax$ differs from $q$ by a nullspace vector, so $q$ is the unique shortest one. This is the exact-fit instance of the [[最小范数最小二乘解|minimum-norm least-squares theorem]]. It does not recover a true but invisible component of the original input. Adding any $v\in N(A)$ to $x$ leaves $A^+Ax$ unchanged.
<!-- bilingual-en:end -->

[[欧氏范数]]衡量长度，复坐标时长度平方是各坐标绝对值平方的和。每个产生 $Ax$ 的输入，都与 $q$ 相差一个零空间向量，因此 $q$ 是其中唯一最短者。这是[[最小范数最小二乘解|最小范数最小二乘定理]]在能够精确拟合时的情形；它并没有找回原输入中真实但不可见的分量。向 $x$ 添加任意 $v\in N(A)$，都不会改变 $A^+Ax$。

> [!warning] Complex rows must be conjugated / 复数行需要共轭
> <!-- bilingual-en:start -->
> For $A=[1\ i]$, where $i^2=-1$, $A^*=(1,-i)^T$ and
>
> $$A^+A=\frac12\begin{bmatrix}1&i\\-i&1\end{bmatrix}.$$
>
> Its retained space is the span of $(1,-i)^T$. Merely turning the original row into the unconjugated column $(1,i)^T$ gives the wrong space: that vector is actually in $N(A)$, because $1+i^2=0$. The general statement is $C(A^*)$, not $C(A^T)$.
> <!-- bilingual-en:end -->
>
> 对 $A=[1\ i]$，有 $i^2=-1$、$A^*=(1,-i)^T$，投影矩阵如上。保留空间由 $(1,-i)^T$ 张成。若只把原行转成未经共轭的列 $(1,i)^T$，便找错了空间；该向量实际上属于 $N(A)$，因为 $1+i^2=0$。一般复数结论必须写成 $C(A^*)$，不能直接写 $C(A^T)$。

<!-- bilingual-en:start -->
## When no input direction is discarded
<!-- bilingual-en:end -->
## 最小例子与边界

<!-- bilingual-en:start -->
The simpler matrix $A=[1\ 0]$ has $A^+=(1,0)^T$, hence $A^+A=\operatorname{diag}(1,0)$. It keeps the first coordinate and removes the second. Although $A$ has full row rank, this product is not $I_2$. In general,
<!-- bilingual-en:end -->

更简单的 $A=[1\ 0]$ 满足 $A^+=(1,0)^T$，所以 $A^+A=\operatorname{diag}(1,0)$，保留第一坐标、删去第二坐标。尽管 $A$ 满行秩，这个乘积也不是 $I_2$。一般地，

$$
A^+A=I_n\quad\Longleftrightarrow\quad N(A)=\{0\}
\quad\Longleftrightarrow\quad \operatorname{rank}(A)=n.
$$

<!-- bilingual-en:start -->
This is **full column rank**: no nonzero input is invisible, so the [[单位矩阵|identity]] retains every input. It differs from the full-row-rank condition for $AA^+=I_m$. At the other extreme, $A=0$ gives $A^+A=0$, discarding every input direction. These are exact statements for the original matrix; replacing small positive singular values by zero changes the retained input space.
<!-- bilingual-en:end -->

这就是**满列秩**：没有非零输入不可见，[[单位矩阵]]保留每个输入。它与 $AA^+=I_m$ 所需的满行秩条件不同。另一个极端是 $A=0$，此时 $A^+A=0$，所有输入方向都被删去。以上针对原矩阵的精确恒等式；若把很小的正奇异值改成零，保留的输入空间也会改变。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 6.69(c) and 6.70, PDF pp.235–236](https://linear.axler.net/LADR4e.pdf#page=235): projection onto the nullspace's orthogonal complement and unique minimum-norm selection over real or complex inner-product spaces.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8, PDF pp.1–2]] and [[03_Positive Definite Matrices and Applications#3.8.3 SVD 定义伪逆|course §3.8.3]]: real row-space/nullspace interpretation and SVD construction. All three small matrices and the conjugation distinction were checked by direct multiplication.
<!-- bilingual-en:end -->

- Axler 第6.69(c)、6.70条核对实、复内积空间中向零空间正交补的投影与唯一最小范数选择。
- MIT Session 3.8 第1–2页及[[03_Positive Definite Matrices and Applications#3.8.3 SVD 定义伪逆|课程第3.8.3节]]核对实数行空间／零空间解释与 SVD 构造；三个小矩阵及共轭区别由直接乘法核验。
