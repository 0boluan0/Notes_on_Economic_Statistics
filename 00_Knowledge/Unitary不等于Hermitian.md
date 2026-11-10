---
aliases:
  - "Unitary versus Hermitian matrices"
  - "酉矩阵与 Hermitian 矩阵的区别"
student_os: knowledge-atom
atom_id: LA-SPD-030
atom_set: symmetric-positive-definite
atom_type: concept-distinction
status: source-checked
requires:
  - "[[Unitary 矩阵]]"
  - "[[Hermitian 矩阵]]"
related:
  - "[[Normal 矩阵]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# Unitary and Hermitian matrices
<!-- bilingual-en:end -->
*Unitary 与 Hermitian 矩阵的区别*

<!-- bilingual-en:start -->
**Unitary** and **Hermitian** describe two different relationships with the conjugate transpose. A [[Unitary 矩阵|unitary matrix]] uses its conjugate transpose to undo its action, so lengths and inner products are preserved. A [[Hermitian 矩阵|Hermitian matrix]] equals its conjugate transpose; this gives real quadratic values and real eigenvalues, but does not require lengths to stay fixed. Neither property alone implies the other.
<!-- bilingual-en:end -->

**Unitary（酉）**与 **Hermitian** 描述的是矩阵与共轭转置的两种不同关系。[[Unitary 矩阵|酉矩阵]]用共轭转置撤销自己的作用，因而保持长度与内积；[[Hermitian 矩阵]]等于自己的共轭转置，由此得到实二次型值与实特征值，却不要求长度保持。任何一种性质都不能单独推出另一种。

<!-- bilingual-en:start -->
Throughout, $A$ is a finite square complex matrix of size $n\ge1$. The star $*$ means transpose followed by complex conjugation, which changes $a+bi$ to $a-bi$ for real $a,b$. The [[单位矩阵|identity]] is $I_n$, and $A^{-1}$ denotes the [[逆矩阵|inverse]], when it exists:
<!-- bilingual-en:end -->

以下 $A$ 始终为有限阶复方阵，阶数 $n\ge1$。星号 $*$ 表示先转置再取复共轭；对实数 $a,b$，共轭把 $a+bi$ 变为 $a-bi$。[[单位矩阵]]记作 $I_n$，存在时的[[逆矩阵]]记作 $A^{-1}$：

$$\text{unitary: }A^*A=I_n\ \Longleftrightarrow\ A^*=A^{-1},
\qquad
\text{Hermitian: }A^*=A.$$

<!-- bilingual-en:start -->
The first condition is a statement about reversing a transformation. The second is a symmetry condition. In particular, a Hermitian matrix can be singular, such as the zero matrix, whereas a unitary matrix is always invertible.
<!-- bilingual-en:end -->

第一个条件描述怎样反向还原变换，第二个描述对称性。特别是，Hermitian 矩阵可以奇异，例如零矩阵；酉矩阵则一定可逆。

<!-- bilingual-en:start -->
## Two minimal counterexamples
<!-- bilingual-en:end -->
## 两个最小反例

<!-- bilingual-en:start -->
A one-by-one matrix is already enough to separate the two properties. Let $i^2=-1$ and take $U=[i]$. Its conjugate transpose is $U^*=[-i]$, so
<!-- bilingual-en:end -->

一阶矩阵就足以区分两种性质。设 $i^2=-1$，取 $U=[i]$，它的共轭转置为 $U^*=[-i]$，于是

$$U^*U=[(-i)i]=[1],\qquad U^*\ne U.$$

<!-- bilingual-en:start -->
Thus $U$ is unitary but not Hermitian. It multiplies every complex scalar by $i$, changing its phase without changing its modulus. In the other direction, $A=[2]$ satisfies
<!-- bilingual-en:end -->

所以 $U$ 为酉矩阵，却不是 Hermitian。它把每个复标量乘以 $i$，改变相位而不改变模长。反方向取 $A=[2]$，有

$$A^*=A,\qquad A^*A=[4]\ne[1].$$

<!-- bilingual-en:start -->
This matrix is Hermitian but not unitary: it doubles lengths. Even positive definiteness would not rescue the implication, because $[2]$ is positive definite. The two scalar examples disprove the two different implications separately.
<!-- bilingual-en:end -->

这个矩阵为 Hermitian，却不为酉矩阵，因为它把长度翻倍。即使增加正定条件也不能推出酉性质，因为 $[2]$ 本来就正定。两个标量例子分别反驳两个不同方向的推理。

<!-- bilingual-en:start -->
## Their eigenvalues obey different restrictions
<!-- bilingual-en:end -->
*特征值受到的限制不同*

<!-- bilingual-en:start -->
An [[特征对|eigenpair]] satisfies $Av=\lambda v$ with $v\ne0$: the nonzero vector $v$ is scaled by $\lambda$. If $A$ is unitary, preservation of length gives
<!-- bilingual-en:end -->

[[特征对|特征对]]满足 $Av=\lambda v$ 且 $v\ne0$，即非零向量 $v$ 被缩放为 $\lambda v$。若 $A$ 为酉矩阵，由长度保持得到

$$\|v\|_2=\|Av\|_2=\|\lambda v\|_2=|\lambda|\,\|v\|_2,
\qquad |\lambda|=1.$$

<!-- bilingual-en:start -->
The absolute-value signs denote complex modulus. Such an eigenvalue lies on the unit circle; it need not be real, as $[i]$ shows. For a Hermitian matrix, the [[Hermitian 谱定理|Hermitian spectral theorem]] instead guarantees real eigenvalues and an orthonormal eigenbasis. The real eigenvalues can have arbitrary signs and magnitudes, including zero.
<!-- bilingual-en:end -->

绝对值符号表示复数的模。因此该特征值位于单位圆上，却不一定为实数，$[i]$ 就是例子。对 Hermitian 矩阵，[[Hermitian 谱定理]]则保证特征值为实数，并存在标准正交特征基。实特征值的符号、大小没有这样的单位模限制，也可以为零。

<!-- bilingual-en:start -->
Both classes are [[Normal 矩阵|normal]], meaning $A^*A=AA^*$. For unitary matrices both products equal $I_n$; for Hermitian matrices both equal $A^2$. The [[Normal 矩阵谱定理|complex normal spectral theorem]] therefore supplies an orthonormal eigenbasis for either class. Within normal matrices, unit-modulus eigenvalues characterize unitarity, while real eigenvalues characterize Hermitian symmetry: writing $A=V\Lambda V^*$ in that basis gives $A^*A=V\Lambda^*\Lambda V^*$ and $A^*=V\Lambda^*V^*$.
<!-- bilingual-en:end -->

两类矩阵都为 [[Normal 矩阵|正规矩阵]]，即 $A^*A=AA^*$。酉矩阵的两边都等于 $I_n$，Hermitian 矩阵的两边都等于 $A^2$。因此[[Normal 矩阵谱定理|复正规谱定理]]为两类都提供标准正交特征基。在正规矩阵范围内，特征值模长全为一刻画酉性质，特征值全为实数刻画 Hermitian 性质：在该基下写成 $A=V\Lambda V^*$，就有 $A^*A=V\Lambda^*\Lambda V^*$ 与 $A^*=V\Lambda^*V^*$。

<!-- bilingual-en:start -->
Here $V$ is the unitary matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues. The matrix $V$ is a coordinate change; it should not be confused with the matrix $A$ being classified. A common normality property does not make the two classes identical: $\operatorname{diag}(2,i)$ is normal but belongs to neither.
<!-- bilingual-en:end -->

其中 $V$ 是由特征向量组成的酉矩阵，$\Lambda$ 为特征值对角矩阵。$V$ 提供坐标变换，不要把它与正在分类的 $A$ 混为一谈。共有正规性不代表两个类别相同：$\operatorname{diag}(2,i)$ 就是正规矩阵，却不属于其中任何一类。

<!-- bilingual-en:start -->
## When the two properties overlap
<!-- bilingual-en:end -->
## 两种性质重合时

<!-- bilingual-en:start -->
If $A$ is both Hermitian and unitary, substituting $A^*=A$ into $A^*A=I_n$ gives $A^2=I_n$. Applying the transformation twice restores every vector. For an eigenpair, this also gives $\lambda^2v=v$, so $\lambda$ is either $1$ or $-1$. In the orthonormal eigenbasis, some coordinates keep their sign and the others reverse it.
<!-- bilingual-en:end -->

如果 $A$ 同时为 Hermitian 与酉矩阵，把 $A^*=A$ 代入 $A^*A=I_n$，就得到 $A^2=I_n$：作用两次还原每个向量。对特征对，又有 $\lambda^2v=v$，因此 $\lambda$ 只能为 $1$ 或 $-1$。在标准正交特征基下，一部分坐标保持符号，另一部分翻转符号。

<!-- bilingual-en:start -->
Conversely, if a Hermitian matrix satisfies $A^2=I_n$, then $A^*A=A^2=I_n$, so it is unitary. Equivalently, a Hermitian matrix is unitary exactly when all its eigenvalues are in $\{1,-1\}$; the spectral theorem turns their squares into the identity. For example,
<!-- bilingual-en:end -->

反过来，若一个 Hermitian 矩阵满足 $A^2=I_n$，则 $A^*A=A^2=I_n$，所以它为酉矩阵。等价地，Hermitian 矩阵为酉矩阵，当且仅当全部特征值都在 $\{1,-1\}$ 中；谱定理把这些特征值的平方组合成单位矩阵。例如

$$R=\begin{pmatrix}1&0\\0&-1\end{pmatrix},
\qquad R^*=R,\quad R^*R=I_2.$$

<!-- bilingual-en:start -->
This reflection preserves the first coordinate and flips the second. The identity and its negative are also in the intersection. Thus the classes overlap, but neither contains the other.
<!-- bilingual-en:end -->

这个反射保持第一坐标，翻转第二坐标。单位矩阵及其负矩阵也在交集中。因此，两类有重合，但彼此都不包含另一类。

> [!warning] Keep the symmetry or normality assumption · 保留对称性或正规性前提
> <!-- bilingual-en:start -->
> The equation $A^2=I$ alone does not establish either property. For
> <!-- bilingual-en:end -->
>
> 仅有 $A^2=I$，不能确定上述任一性质。取
>
> $$N=\begin{pmatrix}1&1\\0&-1\end{pmatrix},
> \qquad N^2=I_2,\qquad
> N^*N=\begin{pmatrix}1&1\\1&2\end{pmatrix}\ne I_2.$$
>
> <!-- bilingual-en:start -->
> This real matrix is not symmetric, hence not Hermitian, and the displayed product shows it is not unitary. Its eigenvalues are nevertheless $1$ and $-1$. Eigenvalues alone cannot replace the normality or Hermitian assumptions in the characterizations above.
> <!-- bilingual-en:end -->
>
> 这个实矩阵不对称，因此不为 Hermitian；展示的乘积也说明它不为酉矩阵。但它的特征值仍为 $1$ 与 $-1$。仅凭特征值，不能替换上述刻画中的正规性或 Hermitian 前提。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, Linear Algebra Done Right, fourth edition, §7A–7B, PDF247–250,260, and §7D, PDF274–276](https://linear.axler.net/LADR4e.pdf#page=247): self-adjoint/normal definitions, the complex spectral theorem, unitary conditions and unit-modulus eigenvalues. Intersection and counterexamples above follow directly by multiplication and the stated spectral representation.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.2sum.pdf#page=2|MIT 18.06SC Session3.2, PDF2]], and [[03_Positive Definite Matrices and Applications#3.2.1 复向量的长度与 Hermitian 内积|permanent course §3.2.1]]: the two matrix definitions and the course's conjugate-transpose notation.
<!-- bilingual-en:end -->

- Axler 第7A–7B节PDF247–250、260页及第7D节PDF274–276页：核对自伴随、正规定义、复谱定理、酉条件及单位模特征值。交集与反例由乘法和所述谱表示直接核验。
- MIT 第3.2节摘要PDF2页与永久课程§3.2.1：核对两类矩阵定义及课程共轭转置记号。
