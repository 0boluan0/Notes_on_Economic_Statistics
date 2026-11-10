---
aliases:
  - "Unitary matrix"
  - "酉矩阵"
student_os: knowledge-atom
atom_id: LA-SPD-004
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
related:
  - "[[Unitary不等于Hermitian]]"
  - "[[正交矩阵]]"
  - "[[标准正交基投影公式]]"
  - "[[Normal 矩阵谱定理]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Unitary matrices
<!-- bilingual-en:end -->
*酉矩阵*

<!-- bilingual-en:start -->
A **unitary matrix** changes complex coordinates without stretching vectors or altering their inner products. The transformation can be reversed by transposing the matrix and conjugating its entries. It plays the role that an [[正交矩阵|orthogonal matrix]] plays for real vectors.
<!-- bilingual-en:end -->

**酉矩阵**改变复数坐标，却不拉伸向量，也不改变向量间的内积。把矩阵转置并对元素取共轭，就能反向还原这个变换。它在复向量中的作用，对应实向量中的[[正交矩阵]]。

<!-- bilingual-en:start -->
For a first example, $U=\operatorname{diag}(i,1)$ multiplies the first coordinate by $i$, where $i^2=-1$, and leaves the second coordinate unchanged. It sends $x=(3,4)^T$ to $(3i,4)^T$. The squared length remains $|3i|^2+|4|^2=25$. Multiplying the first coordinate by $-i$ reverses the change.
<!-- bilingual-en:end -->

先看 $U=\operatorname{diag}(i,1)$：它把第一坐标乘以 $i$，其中 $i^2=-1$，第二坐标不变。向量 $x=(3,4)^T$ 被送到 $(3i,4)^T$，长度平方仍为 $|3i|^2+|4|^2=25$。把第一坐标乘以 $-i$，就能还原。

<!-- bilingual-en:start -->
## Which complex length is being preserved?
<!-- bilingual-en:end -->
*保持的是哪一种复向量长度*

<!-- bilingual-en:start -->
For a complex number $a+bi$ with real $a,b$, its conjugate is $a-bi$ and its squared modulus is $a^2+b^2$. The star $*$ means [[Hermitian 矩阵#Transpose, then conjugate|conjugate transpose]]: exchange rows and columns, then conjugate every entry. The standard complex [[内积|inner product]] and its associated [[欧氏范数|length]] are
<!-- bilingual-en:end -->

对实数 $a,b$ 组成的复数 $a+bi$，共轭为 $a-bi$，模平方为 $a^2+b^2$。星号 $*$ 表示[[Hermitian 矩阵#Transpose, then conjugate|共轭转置]]：先交换行列，再对每个元素取共轭。标准复[[内积]]及其对应的[[欧氏范数|长度]]为

$$x^*y=\sum_{j=1}^n\overline{x_j}y_j,\qquad
\|x\|_2^2=x^*x=\sum_{j=1}^n|x_j|^2.$$

<!-- bilingual-en:start -->
Here $x,y$ are columns with $n$ complex coordinates; the bar means conjugation. We use conjugate-linearity in the first input and linearity in the second. Plain transpose $T$ is insufficient: for $z=(1,i)^T$, $z^Tz=0$ even though its squared length is $z^*z=2$.
<!-- bilingual-en:end -->

其中 $x,y$ 是有 $n$ 个复坐标的列向量，横线表示共轭。本卡采用第一个输入共轭线性、第二个输入线性的约定。仅用普通转置 $T$ 不够：取 $z=(1,i)^T$，有 $z^Tz=0$，但长度平方为 $z^*z=2$。

> [!important] Definition · 定义
> <!-- bilingual-en:start -->
> Let $n\ge1$ be a finite integer. A square matrix $U\in\mathbb C^{n\times n}$ is **unitary** when
> <!-- bilingual-en:end -->
>
> 设 $n\ge1$ 为有限整数。方阵 $U\in\mathbb C^{n\times n}$ 称为**酉矩阵**，如果
>
> $$\boxed{U^*U=I_n.}$$
>
> <!-- bilingual-en:start -->
> The [[单位矩阵|identity matrix]] $I_n$ has ones on its diagonal and zeros elsewhere. The condition uses the standard coordinate inner product just defined.
> <!-- bilingual-en:end -->
>
> [[单位矩阵]] $I_n$ 的对角元为一，其余元素为零。这个条件使用刚定义的标准坐标内积。

<!-- bilingual-en:start -->
## Read the equation column by column
<!-- bilingual-en:end -->
*逐列理解这个等式*

<!-- bilingual-en:start -->
Write $U=[u_1\ \cdots\ u_n]$. Entry $(j,k)$ of $U^*U$ is $u_j^*u_k$. Thus the diagonal entries require every column to have length one, and the other entries require distinct columns to be [[正交|orthogonal]], meaning their inner product is zero. The columns form a [[标准正交组|standard orthonormal set]]; because there are $n$ of them in $\mathbb C^n$, they are a basis.
<!-- bilingual-en:end -->

写成 $U=[u_1\ \cdots\ u_n]$，则 $U^*U$ 的第 $(j,k)$ 项为 $u_j^*u_k$。所以对角线要求每列长度为一，其他位置要求不同列[[正交]]，即内积为零。各列构成[[标准正交组]]；在 $\mathbb C^n$ 中共有 $n$ 个，因此还是一组基。

<!-- bilingual-en:start -->
This also proves that the transformation is reversible. If $Ux=0$, then $x=U^*Ux=0$, so its columns are [[线性无关|linearly independent]]. A square matrix with independent columns has an [[逆矩阵|inverse]]. Multiplying $U^*U=I_n$ by that inverse gives $U^*=U^{-1}$, hence also $UU^*=I_n$. Conversely, starting from $UU^*=I_n$ applies the same argument to $U^*$, giving $U^*U=I_n$. Therefore
<!-- bilingual-en:end -->

这也说明变换可逆。如果 $Ux=0$，则 $x=U^*Ux=0$，所以各列[[线性无关]]。列线性无关的方阵存在[[逆矩阵]]。在 $U^*U=I_n$ 后乘该逆矩阵，得到 $U^*=U^{-1}$，于是也有 $UU^*=I_n$。反过来，从 $UU^*=I_n$ 出发，对 $U^*$ 使用同一论证，也得到 $U^*U=I_n$。因此

$$U^*U=I_n\quad\Longleftrightarrow\quad UU^*=I_n
\quad\Longleftrightarrow\quad U^{-1}=U^*.$$

<!-- bilingual-en:start -->
The equality $UU^*=I_n$ says that the rows also form an orthonormal set. Unit length matters: $2I_n$ has perpendicular columns, but doubles lengths and satisfies $(2I_n)^*(2I_n)=4I_n$.
<!-- bilingual-en:end -->

等式 $UU^*=I_n$ 说明各行也构成标准正交组。单位长度不能省略：$2I_n$ 的各列虽然垂直，却把长度翻倍，并满足 $(2I_n)^*(2I_n)=4I_n$。

<!-- bilingual-en:start -->
## Why every inner product and length survives
<!-- bilingual-en:end -->
*为什么每个内积与长度都不变*

<!-- bilingual-en:start -->
For any $x,y\in\mathbb C^n$, associating the matrix products gives
<!-- bilingual-en:end -->

对任意 $x,y\in\mathbb C^n$，按矩阵乘法的结合律有

$$(Ux)^*(Uy)=x^*U^*Uy=x^*y.$$

<!-- bilingual-en:start -->
Setting $y=x$ proves $\|Ux\|_2=\|x\|_2$. Applying that equality to $x-y$ also preserves distances. Conversely, preservation of all inner products gives the defining equation by taking $x,y$ to be coordinate unit vectors, so that each entry of $U^*U$ is tested.
<!-- bilingual-en:end -->

取 $y=x$，得到 $\|Ux\|_2=\|x\|_2$。对 $x-y$ 使用同一等式，距离也保持不变。反过来，如果全部内积都保持，令 $x,y$ 分别取坐标单位向量，就逐项检验了 $U^*U$，从而得到定义等式。

<!-- bilingual-en:start -->
Preserving the length of every complex vector is enough as well. To see this directly, set $C=U^*U-I_n$, which is Hermitian, and suppose $x^*Cx=0$ for every $x$. The coordinate vector $e_j$, with one in position $j$ and zeros elsewhere, gives $C_{jj}=0$. For $j\ne k$, the vectors $e_j+e_k$ and $e_j+i e_k$ then give $2\operatorname{Re}C_{jk}=0$ and $-2\operatorname{Im}C_{jk}=0$. Both parts of every entry vanish, so $C=0$. Testing only one vector's length would not establish this.
<!-- bilingual-en:end -->

只要每个复向量的长度都保持，也已经足够。直接证明如下：令 $C=U^*U-I_n$，它是 Hermitian 矩阵，并假设对每个 $x$ 都有 $x^*Cx=0$。取坐标向量 $e_j$（第 $j$ 项为一，其余为零），得到 $C_{jj}=0$。对 $j\ne k$，再取 $e_j+e_k$ 与 $e_j+i e_k$，得到 $2\operatorname{Re}C_{jk}=0$、$-2\operatorname{Im}C_{jk}=0$。每个元素的实部与虚部都为零，所以 $C=0$。只检验某一个向量的长度，不能得到这个结论。

> [!example] Coordinates can mix without stretching · 混合坐标也可以不拉伸
> <!-- bilingual-en:start -->
> A unitary matrix need not be diagonal. Consider
> <!-- bilingual-en:end -->
>
> 酉矩阵不一定是对角矩阵。例如
>
> $$U=\frac1{\sqrt2}\begin{pmatrix}1&i\\i&1\end{pmatrix},
> \qquad U^*=\frac1{\sqrt2}\begin{pmatrix}1&-i\\-i&1\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> Multiplication gives $U^*U=I_2$. Its two columns have squared length $(1+1)/2=1$ and inner product $(i-i)/2=0$. For $x=(1,i)^T$, $Ux=(0,\sqrt2\,i)^T$; both vectors have squared length $2$. Here $U^T=U$, but $U^TU=U^2=\begin{pmatrix}0&i\\i&0\end{pmatrix}$, not $I_2$. Conjugation is essential when reversing this transformation.
> <!-- bilingual-en:end -->
>
> 直接相乘得 $U^*U=I_2$。两列的长度平方都是 $(1+1)/2=1$，内积为 $(i-i)/2=0$。对 $x=(1,i)^T$，有 $Ux=(0,\sqrt2\,i)^T$，两个向量的长度平方都为 $2$。这里 $U^T=U$，但 $U^TU=U^2=\begin{pmatrix}0&i\\i&0\end{pmatrix}$，不是 $I_2$。反向还原时，共轭不可省略。

<!-- bilingual-en:start -->
## Why the square-matrix condition matters
<!-- bilingual-en:end -->
*为什么要求方阵*

<!-- bilingual-en:start -->
A tall matrix $Q\in\mathbb C^{m\times r}$ with $r<m$ can satisfy $Q^*Q=I_r$. It preserves lengths of inputs in $\mathbb C^r$, but its outputs occupy only the space spanned by its columns. For example, $Q=\begin{pmatrix}1\\i\end{pmatrix}/\sqrt2$ sends a scalar to a vector in one complex line. Here
<!-- bilingual-en:end -->

高矩阵 $Q\in\mathbb C^{m\times r}$ 在 $r<m$ 时也可能满足 $Q^*Q=I_r$。它保持 $\mathbb C^r$ 中输入的长度，但输出只占据各列张成的空间。例如 $Q=\begin{pmatrix}1\\i\end{pmatrix}/\sqrt2$ 把一个标量送到某条复直线上的向量。此时

$$Q^*Q=[1],\qquad QQ^*=\frac12\begin{pmatrix}1&-i\\i&1\end{pmatrix}\ne I_2.$$

<!-- bilingual-en:start -->
The matrix $QQ^*$ is the [[标准正交基投影公式|orthogonal projector onto the column space]]. It fixes $Q$ but sends $(1,-i)^T$ to zero. Thus $Q$ has no two-sided inverse and is not a unitary matrix under the square-matrix convention used here.
<!-- bilingual-en:end -->

矩阵 $QQ^*$ 是[[标准正交基投影公式|列空间上的正交投影矩阵]]：它保持 $Q$ 不变，却把 $(1,-i)^T$ 映成零。因此 $Q$ 没有双侧逆，按本库使用的方阵约定，它不是酉矩阵。

<!-- bilingual-en:start -->
When every entry is real, conjugation changes nothing, so the unitary condition becomes exactly the [[正交矩阵|orthogonal-matrix condition]]. For complex matrices, unitarity is different from [[Hermitian 矩阵|Hermitian symmetry]]: the latter says $A^*=A$, not $A^*=A^{-1}$. See [[Unitary不等于Hermitian|their distinction and overlap]]. Unitary matrices also provide the orthonormal coordinate changes used in the [[Normal 矩阵谱定理|complex spectral theorem]].
<!-- bilingual-en:end -->

若全部元素为实数，共轭不改变元素，酉条件就恰好变成[[正交矩阵]]条件。对复矩阵，酉性质与 [[Hermitian 矩阵|Hermitian 对称性]]不同：后者要求 $A^*=A$，不是 $A^*=A^{-1}$，详见[[Unitary不等于Hermitian|两者的区别与交集]]。[[Normal 矩阵谱定理|复谱定理]]中的标准正交坐标变换也由酉矩阵提供。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, Linear Algebra Done Right, fourth edition, §7D, 7.44–7.57, PDF272–277](https://linear.axler.net/LADR4e.pdf#page=272): isometries, the finite square condition, equivalent row/column/inverse characterizations, and preservation of norms and inner products. Axler's inner-product argument uses the opposite linearity convention; the calculations here consistently use $x^*y$.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|MIT 18.06SC Session3.2, PDF1–2]], and [[03_Positive Definite Matrices and Applications#3.2.1 复向量的长度与 Hermitian 内积|permanent course §3.2.1]]: complex lengths, conjugate transpose, and square unitary matrices. All displayed examples and the entrywise converse are checked directly.
<!-- bilingual-en:end -->

- Axler 第7D节上述各条，PDF272–277页：核对等距映射、有限方阵前提、行列与逆的等价刻画，以及长度、内积保持。原书内积采用相反的线性约定，本卡始终使用 $x^*y$。
- MIT 第3.2节摘要PDF1–2页与永久课程§3.2.1：核对复长度、共轭转置及方阵酉条件；全部例子与逐元素逆向证明直接核算。
