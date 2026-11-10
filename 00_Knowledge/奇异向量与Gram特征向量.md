---
aliases:
- Singular vectors and Gram eigenvectors
- Gram-matrix characterisation of singular vectors
student_os: knowledge-atom
atom_id: LA-SVD-026
atom_set: singular-value-decomposition-low-rank
atom_type: equivalence
status: source-checked
requires:
- '[[奇异向量]]'
- '[[奇异值与Gram谱]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[SVD手算流程]]'
related:
- '[[SVD与四个基本子空间]]'
---

<!-- bilingual-en:start -->
# Singular vectors from Gram eigenvectors
<!-- bilingual-en:end -->

*奇异向量与 Gram 特征向量*

<!-- bilingual-en:start -->
To find [[奇异向量|singular directions]], we need inputs whose images fit together as perpendicular output directions. The [[Gram矩阵正定判据|Gram matrix]] $A^*A$ helps find them: it compares the columns of $A$ through their inner products, and acts entirely within the input space. Its positive [[特征对|eigenpairs]] identify directions that are stretched by a definite amount. Applying the original matrix $A$ then supplies the correctly matched output direction.
<!-- bilingual-en:end -->

寻找[[奇异向量|奇异方向]]，需要选出一组输入，使它们的像成为彼此垂直的输出方向。[[Gram矩阵正定判据|Gram 矩阵]] $A^*A$ 通过内积比较 $A$ 的各列，并始终在输入空间内作用，因而能帮助找到这些方向。它的正[[特征对]]指出按确定大小伸缩的方向；再作用原矩阵 $A$，便得到正确配对的输出方向。

<!-- bilingual-en:start -->
For a real or complex $m\times n$ matrix, with $m,n\ge1$, $G=A^*A$ is $n\times n$ and $H=AA^*$ is $m\times m$. The star denotes [[矩阵转置|conjugate transpose]], replacing $a+bi$ by $a-bi$ after transposition, with $i^2=-1$. Both Gram matrices are [[Hermitian 矩阵|Hermitian]] and [[半正定矩阵|positive semidefinite]]; for any input vector $x$, for example, $x^*Gx=\|Ax\|_2^2\ge0$. The symbol $\|\cdot\|_2$ means [[欧氏范数|Euclidean length]], using absolute-value squares for complex coordinates.
<!-- bilingual-en:end -->

对实或复 $m\times n$ 矩阵，$m,n\ge1$，$G=A^*A$ 为 $n\times n$，$H=AA^*$ 为 $m\times m$。星号表示[[矩阵转置|共轭转置]]，转置后把 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$。两个 Gram 矩阵都为 [[Hermitian 矩阵]]且[[半正定矩阵|半正定]]，例如对任意输入向量 $x$，有 $x^*Gx=\|Ax\|_2^2\ge0$。$\|\cdot\|_2$ 是[[欧氏范数|欧氏长度]]，复坐标使用绝对值平方。

<!-- bilingual-en:start -->
## From a positive eigenpair to a matched direction
<!-- bilingual-en:end -->

*从正特征对构造配对方向*

> [!important] Normalize the image of a unit eigenvector / 归一化单位特征向量的像
> <!-- bilingual-en:start -->
> Suppose $\|v\|_2=1$ and $Gv=\lambda v$ with $\lambda>0$. Then
> $$\sigma=\sqrt{\lambda},\qquad u=\frac{Av}{\sigma}$$
> gives a unit left singular vector paired with the right singular vector $v$. In the other direction, start from a unit eigenvector $u$ of $H$ for $\lambda>0$ and set $v=A^*u/\sqrt{\lambda}$. The positive square root is a [[奇异值|singular value]], not the eigenvalue $\lambda$ itself.
> <!-- bilingual-en:end -->
>
> 若 $\|v\|_2=1$、$Gv=\lambda v$ 且 $\lambda>0$，取 $\sigma=\sqrt{\lambda}$、$u=Av/\sigma$，就得到与右奇异向量 $v$ 配对的单位左奇异向量。反向也成立：从 $H$ 的单位正特征向量 $u$ 出发，取 $v=A^*u/\sqrt{\lambda}$。[[奇异值]]是正平方根，不是特征值 $\lambda$ 本身。

<!-- bilingual-en:start -->
For instance, a unit $v$ with $Gv=16v$ gives $\sigma=4$ and $u=Av/4$. The unit-length requirement matters. For the stated unit vector,
<!-- bilingual-en:end -->

例如，单位向量 $v$ 若满足 $Gv=16v$，就有 $\sigma=4$、$u=Av/4$。单位长度条件不可省略。对上述单位向量，

$$
\|u\|_2^2=\frac{v^*A^*Av}{\lambda}
=\frac{\lambda v^*v}{\lambda}=1,\qquad
A^*u=\frac{A^*Av}{\sqrt{\lambda}}=\sqrt{\lambda}\,v.
$$

<!-- bilingual-en:start -->
The definition of $u$ already gives $Av=\sigma u$, so both pairing equations hold. If the starting eigenvector had length 2, the same formula would also produce a vector of length 2; normalize it first. Conversely, a positive singular pair satisfies
<!-- bilingual-en:end -->

$u$ 的定义已经给出 $Av=\sigma u$，所以两个配对方程都成立。若起点特征向量长为2，同一公式也只会产生长为2的向量，必须先归一化。反过来，正奇异方向对满足

$$
A^*Av=\sigma^2v,\qquad AA^*u=\sigma^2u,
$$

<!-- bilingual-en:start -->
by applying $A^*$ or $A$ once more. This proves the [[奇异值与Gram谱|positive Gram spectral correspondence]] together with the actual pairing, rather than just matching two numerical eigenvalues.
<!-- bilingual-en:end -->

这是再作用一次 $A^*$ 或 $A$ 的结果。它既给出[[奇异值与Gram谱|正 Gram 谱对应]]，也给出真正的方向配对，而不只是把两张矩阵的特征值按数值对上。

<!-- bilingual-en:start -->
## Following the construction in small matrices
<!-- bilingual-en:end -->
## 最小例子

<!-- bilingual-en:start -->
For $A=\operatorname{diag}(3,1)$, meaning the diagonal matrix with entries 3 and 1, both Gram matrices are $\operatorname{diag}(9,1)$. The coordinate unit vectors $e_1=(1,0)^T$ and $e_2=(0,1)^T$ give $\sigma_1=3,\sigma_2=1$ and $u_1=e_1,u_2=e_2$. Here $T$ is ordinary transpose. This simple case allows the same basis on both sides, but that is not a general rule.
<!-- bilingual-en:end -->

若 $A=\operatorname{diag}(3,1)$，即对角元为3、1的对角矩阵，则两个 Gram 矩阵都是 $\operatorname{diag}(9,1)$。坐标单位向量 $e_1=(1,0)^T$、$e_2=(0,1)^T$ 分别给出 $\sigma_1=3,\sigma_2=1$，且 $u_1=e_1,u_2=e_2$；这里 $T$ 为普通转置。这个简单例子允许两侧使用相同基，但一般不能这样假设。

<!-- bilingual-en:start -->
Now take
$$A=\begin{bmatrix}0&-3\\1&0\end{bmatrix},\qquad
G=\begin{bmatrix}1&0\\0&9\end{bmatrix},\qquad
H=\begin{bmatrix}9&0\\0&1\end{bmatrix}.$$
The largest Gram eigenvalue is 9. Its right eigenvector is $v_1=e_2$, so $\sigma_1=3$ and $u_1=Av_1/3=-e_1$. The smaller value gives $v_2=e_1$, $\sigma_2=1$, $u_2=e_2$. Thus the order and sign on the left are set by applying $A$; taking $u_1=e_1$ would violate $Av_1=3u_1$.
<!-- bilingual-en:end -->

对上面第二个矩阵，最大 Gram 特征值为9，右特征向量为 $v_1=e_2$，所以 $\sigma_1=3$、$u_1=Av_1/3=-e_1$。较小值给出 $v_2=e_1$、$\sigma_2=1$、$u_2=e_2$。左侧的顺序与符号都由作用 $A$ 来确定；若直接取 $u_1=e_1$，就不满足 $Av_1=3u_1$。

<!-- bilingual-en:start -->
The same rule works over the complex numbers. For $A=[i]$, $G=[1]$; choose $v=1$, then $\sigma=1$ and $u=i$. Conjugation is essential: $A^*u=(-i)i=1=v$.
<!-- bilingual-en:end -->

复数中也适用同一规则。若 $A=[i]$，则 $G=[1]$；取 $v=1$，得到 $\sigma=1,u=i$。共轭不可省略，因为 $A^*u=(-i)i=1=v$。

<!-- bilingual-en:start -->
## Repeated eigenvalues still require matched bases
<!-- bilingual-en:end -->

*特征值重复时仍须匹配两侧的基*

<!-- bilingual-en:start -->
A [[Hermitian 谱定理|Hermitian spectral decomposition]] supplies an orthonormal eigenbasis of $G$. Mapping its positive directions by $u_i=Av_i/\sigma_i$ preserves orthonormality:
$$u_i^*u_j
=\frac{v_i^*Gv_j}{\sigma_i\sigma_j}
=\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}$$
This also works for repeated positive values. It constructs compatible left vectors; it does not license independent choices on the two sides.
<!-- bilingual-en:end -->

[[Hermitian 谱定理]]给出 $G$ 的标准正交特征基。把正方向按 $u_i=Av_i/\sigma_i$ 映过去，上面的内积恒等式保证左向量仍标准正交，即使正值重复也成立。这是在构造彼此匹配的左向量，并不允许两侧随意独立选择。

> [!warning] Equal eigenvalues alone do not make a pair / 特征值相同还不构成配对
> <!-- bilingual-en:start -->
> For $A=I_2$, the [[单位矩阵|identity]], both Gram matrices are $I_2$. The choices $v=e_1$ and $u=e_2$ are unit eigenvectors for eigenvalue 1, but $Av=e_1\ne e_2$. They are not a pair. Choose the right basis first, then map it with $A/\sigma$; see [[重奇异值下方向不唯一|repeated singular values]] for the remaining coordinated freedom.
> <!-- bilingual-en:end -->
>
> 若 $A=I_2$ 为[[单位矩阵]]，两个 Gram 矩阵也都是 $I_2$。$v=e_1$、$u=e_2$ 都是特征值1的单位特征向量，却有 $Av=e_1\ne e_2$，并非一对。应先选右侧基，再按 $A/\sigma$ 映到左侧；剩余的协调自由见[[重奇异值下方向不唯一]]。

<!-- bilingual-en:start -->
## When the eigenvalue is zero
<!-- bilingual-en:end -->
## 零谱边界

<!-- bilingual-en:start -->
If $Gv=0$, then $\|Av\|_2^2=v^*Gv=0$, hence $Av=0$. The formula $Av/\sqrt{\lambda}$ would be $0/0$, so it cannot produce a left unit vector. Instead, choose an orthonormal basis of the [[零空间|right nullspace]] $N(A)$ and, independently, one of the [[左零空间|left nullspace]] $N(A^*)$. Their dimensions are $n-r$ and $m-r$, where $r$ is the [[矩阵秩|rank]], and can differ.
<!-- bilingual-en:end -->

若 $Gv=0$，则 $\|Av\|_2^2=v^*Gv=0$，所以 $Av=0$。这时 $Av/\sqrt{\lambda}$ 变成 $0/0$，不能产生左单位向量。应分别选取[[零空间|右零空间]] $N(A)$ 与[[左零空间]] $N(A^*)$ 的标准正交基，两侧独立选择。其维数为 $n-r$、$m-r$，$r$ 是[[矩阵秩|秩]]，两者可能不同。

<!-- bilingual-en:start -->
These vectors complete a full SVD without dividing by zero. Positive directions determine the [[紧SVD|compact factorization]], while the zero completion has its [[SVD零块补基不唯一|separate freedom]]. For actual hand computation, normalize first, pair positive directions through the original matrix, and verify the product; the fuller procedure is in [[SVD手算流程|the hand-calculation guide]]. A proof using Gram matrices is not a recommendation to form them for every floating-point computation.
<!-- bilingual-en:end -->

这些向量无需除以零，就能补齐完整 SVD。正方向确定[[紧SVD]]，零方向补基则有[[SVD零块补基不唯一|独立的自由]]。实际手算时，先归一化，再用原矩阵配对正方向，最后核对乘积；完整步骤见 [[SVD手算流程]]。利用 Gram 矩阵证明结论，并不意味着每次数值计算都应显式形成它。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.64 and 7.70–7.77, PDF pp.284 and 287–289](https://linear.axler.net/LADR4e.pdf#page=284): Gram positivity/nullspaces, normalized positive images, orthonormality and adjoint pairing. The displayed converse and counterexamples were checked directly.
- [LAPACK Users’ Guide, SVD](https://netlib.org/lapack/lug/node32.html): the coupled left/right equations in the real and complex cases.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5]], PDF pp.2–4: actual Gram eigenvalue calculation, square roots, the left-vector sign correction and nullspace completion.
<!-- bilingual-en:end -->

- Axler 第7.64、7.70–7.77条核对 Gram 正性与零空间、正像归一化、标准正交性及伴随配对；反向推导和反例由直接乘法核验。
- LAPACK 核对实、复情形中左右相互耦合的两个方程。
- MIT Session 3.5 第2–4页核对实际 Gram 特征值计算、平方根、左向量符号修正和零空间补基。
