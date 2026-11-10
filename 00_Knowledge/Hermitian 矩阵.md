---
aliases:
  - Hermitian matrix
  - 自伴矩阵
  - 厄米矩阵
student_os: knowledge-atom
atom_id: LA-SPD-003
atom_set: symmetric-positive-definite
atom_type: definition
status: source-checked
related:
  - "[[实对称矩阵]]"
  - "[[Unitary不等于Hermitian]]"
  - "[[Hermitian 谱定理]]"
  - "[[Hermitian二次型实值判据]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# Hermitian matrices
<!-- bilingual-en:end -->
*Hermitian（厄米、自伴）矩阵*

<!-- bilingual-en:start -->
A **Hermitian matrix** has conjugate entries in positions reflected across its main diagonal. Complex conjugation replaces $a+bi$ by $a-bi$, where $a,b$ are real and $i^2=-1$. Thus the entries in a reflected pair have the same real part and opposite imaginary parts. For example,
<!-- bilingual-en:end -->

**Hermitian 矩阵**在主对角线两侧的镜像位置上，填着互为共轭的数。复共轭把 $a+bi$ 换成 $a-bi$，其中 $a,b$ 为实数，$i^2=-1$。因此，一对镜像元素的实部相同、虚部互为相反数。例如，

$$A=\begin{pmatrix}2&i\\-i&3\end{pmatrix}.$$

<!-- bilingual-en:start -->
The entries $i$ and $-i$ match by conjugation. Each diagonal entry is paired with itself, so it must be real; the two diagonal entries need not equal each other. This extends the familiar entry pattern of a [[实对称矩阵|real symmetric matrix]].
<!-- bilingual-en:end -->

$i$ 与 $-i$ 通过共轭配成一对。每个对角元与自己配对，所以必须为实数；两个对角元不必彼此相同。这推广了[[实对称矩阵]]的元素排列关系。

<!-- bilingual-en:start -->
## Transpose, then conjugate
<!-- bilingual-en:end -->
*先转置，再取共轭*

<!-- bilingual-en:start -->
The [[矩阵转置|transpose]] $A^T$ exchanges rows and columns without changing any entry. The **conjugate transpose** $A^*=\overline A^{\,T}$ additionally conjugates every entry; a bar means conjugation. Some courses write $A^H$ for the same operation. For the opening matrix,
<!-- bilingual-en:end -->

[[矩阵转置|普通转置]] $A^T$ 交换行列，却不改变元素本身。**共轭转置** $A^*=\overline A^{\,T}$ 还对每个元素取共轭；横线表示共轭。有些课程把同一操作记作 $A^H$。对开头的矩阵，

$$A^T=\begin{pmatrix}2&-i\\i&3\end{pmatrix},\qquad
A^*=\begin{pmatrix}2&i\\-i&3\end{pmatrix}=A.$$

> [!important] The defining equality · 定义等式
> <!-- bilingual-en:start -->
> Let $n\ge1$ be an integer and $A=(a_{jk})\in\mathbb C^{n\times n}$, where $a_{jk}$ denotes row $j$, column $k$. The matrix is **Hermitian** exactly when
> <!-- bilingual-en:end -->
>
> 设 $n\ge1$ 为整数，$A=(a_{jk})\in\mathbb C^{n\times n}$，其中 $a_{jk}$ 表示第 $j$ 行、第 $k$ 列的元素。矩阵为 **Hermitian**，恰好是指
>
> $$A=A^*\quad\Longleftrightarrow\quad
> a_{jk}=\overline{a_{kj}}\quad\text{for all }j,k.$$

<!-- bilingual-en:start -->
Setting $j=k$ gives $a_{jj}=\overline{a_{jj}}$, so diagonal entries have zero imaginary part. If every entry is real, conjugation changes nothing and the condition becomes $A=A^T$. A one-entry matrix $[a+bi]$ is Hermitian precisely when $b=0$. The zero matrix is Hermitian. In contrast, $\begin{pmatrix}2&i\\i&3\end{pmatrix}$ equals its ordinary transpose but is not Hermitian: the lower-left entry should be $-i$.
<!-- bilingual-en:end -->

令 $j=k$，得到 $a_{jj}=\overline{a_{jj}}$，所以对角元虚部为零。若全部元素为实数，共轭不改变任何元素，条件就变成 $A=A^T$。单元素矩阵 $[a+bi]$ 恰在 $b=0$ 时为 Hermitian；零矩阵也是 Hermitian。相反，$\begin{pmatrix}2&i\\i&3\end{pmatrix}$ 虽等于普通转置，却不是 Hermitian，因为左下角应为 $-i$。

<!-- bilingual-en:start -->
## Why conjugation matters for length
<!-- bilingual-en:end -->
*共轭为什么与长度有关*

<!-- bilingual-en:start -->
For complex column vectors, the standard [[内积|inner product]] is $\langle z,w\rangle=z^*w=\sum_j\overline{z_j}w_j$. It is conjugate-linear in the first input and linear in the second. Its self-product is $z^*z=\sum_j|z_j|^2$, the squared length; $|a+bi|^2=a^2+b^2$. For $z=(1,i)^T$,
<!-- bilingual-en:end -->

对复列向量，标准[[内积]]为 $\langle z,w\rangle=z^*w=\sum_j\overline{z_j}w_j$，第一个输入共轭线性、第二个输入线性。自身内积 $z^*z=\sum_j|z_j|^2$ 是长度平方，其中 $|a+bi|^2=a^2+b^2$。取 $z=(1,i)^T$，

$$z^Tz=1+i^2=0,\qquad z^*z=1+(-i)i=2.$$

<!-- bilingual-en:start -->
Thus ordinary transposition alone cannot produce the standard squared length on complex vectors. Under this standard inner product, matrix multiplication gives $\langle Az,w\rangle=\langle z,A^*w\rangle$. Consequently $A=A^*$ says that applying $A$ to either slot has the same effect: $\langle Az,w\rangle=\langle z,Aw\rangle$. This is the meaning of **self-adjoint** here. The identification with conjugate transpose uses these standard coordinates, or an [[标准正交组|orthonormal basis]] for the chosen inner product.
<!-- bilingual-en:end -->

因此，仅用普通转置无法给出复向量的标准长度平方。在这一标准内积下，矩阵乘法给出 $\langle Az,w\rangle=\langle z,A^*w\rangle$。于是 $A=A^*$ 表示把 $A$ 作用于哪一个输入都得到相同结果，即 $\langle Az,w\rangle=\langle z,Aw\rangle$；这就是这里**自伴**的含义。把伴随写成共轭转置，使用的是这些标准坐标，或选定内积下的[[标准正交组|标准正交基]]。

> [!warning] Two legitimate complex conventions · 两种合法的复数约定
> <!-- bilingual-en:start -->
> An algebraic complex [[二次型|quadratic form]] $z^TBz$ uses ordinary transpose. It comes from a bilinear expression $z^TBw$, meaning linear in each input separately. It is a valid object, but it need not describe length or take real values. The expression $z^*Bw$ instead conjugates coefficients in its first input. At equal inputs, the scaling rules for a complex scalar $c$ are
> <!-- bilingual-en:end -->
>
> 代数意义的复[[二次型]] $z^TBz$ 使用普通转置，来自双线性表达式 $z^TBw$，即对两个输入分别线性。它是合法对象，却不必表示长度或取实数值。表达式 $z^*Bw$ 则会对第一个输入的系数取共轭。两个输入相同时，对复数 $c$ 有不同的缩放规律：
>
> $$(cz)^TB(cz)=c^2z^TBz,\qquad
> (cz)^*B(cz)=|c|^2z^*Bz.$$
>
> <!-- bilingual-en:start -->
> These formulas apply to any complex $n\times n$ matrix $B$. When $B$ is Hermitian, the second expression is called a **Hermitian quadratic form** and is real-valued. Read the stated convention before using a formula.
> <!-- bilingual-en:end -->
>
> 上式对任意复 $n\times n$ 矩阵 $B$ 都成立。当 $B$ 为 Hermitian 时，第二种表达式称为 **Hermitian 二次型**，并且取实数值。使用公式前要先确认采用哪一种约定。

<!-- bilingual-en:start -->
## Real values do not require positive values
<!-- bilingual-en:end -->
*取实值不等于取正值*

<!-- bilingual-en:start -->
For Hermitian $A$, the scalar $z^*Az$ equals its conjugate and is therefore real. The [[Hermitian二次型实值判据|real-valuedness criterion]] also proves the converse when every complex vector is tested. Reality alone does not give positivity: $D=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ is Hermitian, but $z^*Dz=|z_1|^2-|z_2|^2$ takes both signs. [[半正定矩阵|Positive semidefiniteness]] adds nonnegativity for all $z$, and [[正定矩阵|positive definiteness]] adds strict positivity for nonzero $z$.
<!-- bilingual-en:end -->

若 $A$ 为 Hermitian，标量 $z^*Az$ 等于自己的共轭，因此为实数。[[Hermitian二次型实值判据]]还证明：若对全部复向量取实值，反过来也能推出 Hermitian。但取实值并不保证正性：$D=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ 为 Hermitian，$z^*Dz=|z_1|^2-|z_2|^2$ 却能正能负。[[半正定矩阵|半正定]]另要求对所有 $z$ 非负；[[正定矩阵|正定]]另要求对非零 $z$ 严格为正。

<!-- bilingual-en:start -->
Hermitian symmetry also differs from [[Unitary不等于Hermitian|unitarity]]: $A^*=A$ is not the condition $A^*A=I$, where $I$ is the [[单位矩阵|identity matrix]]. The consequences for eigenvalues and eigenvectors are developed separately in the [[Hermitian 谱定理|Hermitian spectral theorem]].
<!-- bilingual-en:end -->

Hermitian 与 [[Unitary不等于Hermitian|unitary]] 也是不同性质：$A^*=A$ 不是 $A^*A=I$ 这一条件，其中 $I$ 为[[单位矩阵]]。特征值与特征向量的结论由 [[Hermitian 谱定理]]单独展开。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=1|MIT 18.06SC Session 3.2 summary, pp. 1–2]] and [[03_Positive Definite Matrices and Applications#3.2.1 复向量的长度与 Hermitian 内积|permanent course notes, Section 3.2.1]]: conjugate transpose, complex length, Hermitian entry pattern and the real-symmetric special case.
- [Axler, Linear Algebra Done Right, fourth edition, definitions 7.7 and 7.10, theorem 7.9; printed pp. 231–233](https://linear.axler.net/LADR4e.pdf#page=245): the adjoint matrix in orthonormal coordinates and the self-adjoint condition. The displayed inner-product identity follows directly using this card's conjugate-first convention.
- [Kahan, UC Berkeley Math H110, How to Recognize a Quadratic Form, p. 1, final paragraph](https://people.eecs.berkeley.edu/~wkahan/MathH110/qf.pdf#page=1): distinguishes algebraic complex quadratic forms from Hermitian forms. The examples and scaling rules are verified by multiplication.
<!-- bilingual-en:end -->

- MIT Session 3.2 第 1–2 页及永久课程 §3.2.1：核对共轭转置、复向量长度、Hermitian 元素关系和实对称特例。
- Axler 第 4 版定义 7.7、7.10 及定理 7.9，印刷第 231–233 页（PDF 第 245–247 页）：核对标准正交坐标下的伴随矩阵与自伴条件；内积恒等式按本卡“第一个输入共轭线性”的约定直接核验。
- Kahan 第 1 页末段：区分代数复二次型与 Hermitian 二次型；例子和缩放规律直接乘法核验。
