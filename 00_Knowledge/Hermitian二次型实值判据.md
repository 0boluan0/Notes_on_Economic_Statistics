---
aliases:
  - Real-valued quadratic-form criterion for Hermitian matrices
  - Hermitian quadratic-form test
student_os: knowledge-atom
atom_id: LA-SPD-034
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
requires:
  - "[[Hermitian 矩阵]]"
  - "[[Hermitian 谱定理]]"
related:
  - "[[复二次型实部]]"
  - "[[二次型]]"
  - "[[二次型对称化]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# Recognizing Hermitian matrices from real quadratic values
<!-- bilingual-en:end -->
*Hermitian 二次型实值判据*

<!-- bilingual-en:start -->
A complex matrix can be tested through the scalar values it produces. Form a column vector $z$, multiply it by $A$, and compare the result with $z$ using the standard [[内积|complex inner product]]. The resulting number is $q_A(z)=z^*Az$. If this number is real for **every complex input**, then the matrix must be [[Hermitian 矩阵|Hermitian]]. The word “every” and the conjugate transpose are essential.
<!-- bilingual-en:end -->

复矩阵可以通过它产生的标量值来检验。把列向量 $z$ 乘以 $A$，再用标准[[内积|复内积]]把结果与 $z$ 比较，得到 $q_A(z)=z^*Az$。若对**全部复数输入**，这个数都为实数，那么矩阵必为 [[Hermitian 矩阵]]。这里“全部”及共轭转置都不可省略。

<!-- bilingual-en:start -->
For a quick failure, let $A=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ and $z=(1,i)^T$, where $i^2=-1$. The star denotes [[矩阵转置|conjugate transpose]], so $z^*=(1,-i)$. Then $Az=(1+i,i)^T$ and $z^*Az=2+i$, which is not real. One such input disproves Hermitian symmetry; successful tests on arbitrarily chosen inputs do not by themselves prove it.
<!-- bilingual-en:end -->

例如，取 $A=\begin{pmatrix}1&1\\0&1\end{pmatrix}$、$z=(1,i)^T$，其中 $i^2=-1$。星号表示[[矩阵转置|共轭转置]]，故 $z^*=(1,-i)$。此时 $Az=(1+i,i)^T$，$z^*Az=2+i$ 不是实数。一个这样的输入就能否定 Hermitian；任意挑选一些输入都测试通过，本身还不足以证明它。

> [!important] The complex-field equivalence · 复数域上的充要条件
> <!-- bilingual-en:start -->
> Let $n\ge1$ be an integer and let $A\in\mathbb C^{n\times n}$. Then
> <!-- bilingual-en:end -->
>
> 设 $n\ge1$ 为整数，$A\in\mathbb C^{n\times n}$，则
>
> $$\boxed{A=A^*\quad\Longleftrightarrow\quad
> z^*Az\in\mathbb R\ \text{for every }z\in\mathbb C^n.}$$
>
> <!-- bilingual-en:start -->
> No positivity or invertibility is assumed. The zero input always gives zero, so the condition carries information through nonzero inputs. For $n=1$, it simply says that a scalar $a$ is real exactly when $\overline{z}az=a|z|^2$ is real for every complex $z$.
> <!-- bilingual-en:end -->
>
> 不预设正性或可逆性。零输入总给出零，所以信息来自非零输入。$n=1$ 时，结论就是：标量 $a$ 为实数，当且仅当 $\overline{z}az=a|z|^2$ 对所有复数 $z$ 为实数。

<!-- bilingual-en:start -->
## Why Hermitian symmetry gives real values
<!-- bilingual-en:end -->
*为什么 Hermitian 会给出实值*

<!-- bilingual-en:start -->
A scalar is real exactly when it equals its conjugate. Conjugate-transposing a product reverses the order, and conjugate-transposing twice returns the original factor. Therefore, if $A=A^*$,
<!-- bilingual-en:end -->

一个标量为实数，恰好是它等于自身共轭。对乘积取共轭转置会反转因子顺序，而取两次共轭转置会回到原因子。因此，若 $A=A^*$，

$$\overline{z^*Az}=(z^*Az)^*=z^*A^*z=z^*Az.$$

<!-- bilingual-en:start -->
This proves the forward implication for every $z$. It does not say that the real value is nonnegative.
<!-- bilingual-en:end -->

这就对每个 $z$ 证明了正向结论，却没有断言这个实数值非负。

<!-- bilingual-en:start -->
## Recovering the entry relation
<!-- bilingual-en:end -->
*怎样反推出元素的共轭关系*

<!-- bilingual-en:start -->
Now assume every $q_A(z)$ is real. Write $A=(a_{jk})$ and let $e_j$ be the coordinate column with a $1$ in position $j$ and zeros elsewhere. First, $q_A(e_j)=a_{jj}$ shows that all diagonal entries are real. For distinct indices $j,k$, two further inputs compare the off-diagonal entries:
<!-- bilingual-en:end -->

反过来，假设每个 $q_A(z)$ 都为实数。写 $A=(a_{jk})$，令 $e_j$ 为第 $j$ 个坐标是 $1$、其余为 $0$ 的坐标列向量。先由 $q_A(e_j)=a_{jj}$ 得到全部对角元为实数。对于不同指标 $j,k$，再用两个输入比较非对角元：

$$
\begin{aligned}
q_A(e_j+e_k)&=a_{jj}+a_{kk}+a_{jk}+a_{kj},\\
q_A(e_j+i e_k)&=a_{jj}+a_{kk}+i(a_{jk}-a_{kj}).
\end{aligned}
$$

<!-- bilingual-en:start -->
The first line being real gives $\operatorname{Im}a_{jk}+\operatorname{Im}a_{kj}=0$; $\operatorname{Im}$ denotes imaginary part. The second line being real gives $\operatorname{Re}a_{jk}-\operatorname{Re}a_{kj}=0$; $\operatorname{Re}$ denotes real part. Hence the two entries have equal real parts and opposite imaginary parts, so $a_{kj}=\overline{a_{jk}}$. Together with the diagonal condition, this is $A=A^*$.
<!-- bilingual-en:end -->

第一行取实值，给出 $\operatorname{Im}a_{jk}+\operatorname{Im}a_{kj}=0$，其中 $\operatorname{Im}$ 表示虚部。第二行取实值，给出 $\operatorname{Re}a_{jk}-\operatorname{Re}a_{kj}=0$，其中 $\operatorname{Re}$ 表示实部。因此两元素实部相同、虚部相反，即 $a_{kj}=\overline{a_{jk}}$。加上对角元条件，便得到 $A=A^*$。

<!-- bilingual-en:start -->
This proof also gives a sufficient finite test family: all $e_j$, and $e_j+e_k$ and $e_j+i e_k$ for every $j<k$. Coordinate vectors alone check only the diagonal. The opening non-Hermitian example has real diagonal entries and passes those coordinate tests, but fails the combined complex test.
<!-- bilingual-en:end -->

证明还给出一组充分的有限测试向量：全部 $e_j$，以及每对 $j<k$ 对应的 $e_j+e_k$ 和 $e_j+i e_k$。只用坐标向量，仅能检查对角元。开头的非 Hermitian 例子对角元为实数，能通过坐标测试，却无法通过组合后的复向量测试。

> [!note]- The spectral proof gives the same conclusion · 谱论证明得到同一结论
> <!-- bilingual-en:start -->
> Another proof splits the matrix into two Hermitian matrices:
> <!-- bilingual-en:end -->
>
> 另一种证明把矩阵拆成两个 Hermitian 矩阵的组合：
>
> $$A=H+iS,\qquad H=\frac{A+A^*}{2},\qquad S=\frac{A-A^*}{2i}.$$
>
> <!-- bilingual-en:start -->
> Directly taking adjoints gives $H^*=H$ and $S^*=S$. Both $z^*Hz$ and $z^*Sz$ are real by the forward implication already proved. If $z^*Az$ is always real, its imaginary part $z^*Sz$ is always zero. The [[Hermitian 谱定理|Hermitian spectral theorem]] gives an orthonormal eigenbasis for $S$. For each basis vector $v$ with $Sv=\lambda v$ and $v^*v=1$, we have $0=v^*Sv=\lambda$. Thus $S$ is zero on a basis, hence $S=0$ and $A=H=A^*$. The [[复二次型实部|real-part identity]] describes the role of $H$ for a general matrix.
> <!-- bilingual-en:end -->
>
> 直接取共轭转置可得 $H^*=H$、$S^*=S$。由刚证明的正向结论，$z^*Hz$ 与 $z^*Sz$ 都为实数。若 $z^*Az$ 总为实数，其虚部 $z^*Sz$ 就恒为零。[[Hermitian 谱定理]]给出 $S$ 的标准正交特征基。对其中满足 $Sv=\lambda v$、$v^*v=1$ 的每个基向量，有 $0=v^*Sv=\lambda$。所以 $S$ 在整组基上为零，即 $S=0$，于是 $A=H=A^*$。一般矩阵中 $H$ 的作用见[[复二次型实部]]。

<!-- bilingual-en:start -->
## Why real inputs and ordinary transpose are insufficient
<!-- bilingual-en:end -->
*为什么不能只用实输入或改用普通转置*

<!-- bilingual-en:start -->
Consider the real [[实反对称矩阵|skew-symmetric matrix]]
<!-- bilingual-en:end -->

考虑实[[实反对称矩阵|反对称矩阵]]

$$K=\begin{pmatrix}0&1\\-1&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
For every real column $x=(x_1,x_2)^T$, we get $x^TKx=x_1x_2-x_2x_1=0$, yet $K^*=-K\ne K$. The complex input $z=(1,i)^T$ detects the failure:
<!-- bilingual-en:end -->

对每个实列向量 $x=(x_1,x_2)^T$，都有 $x^TKx=x_1x_2-x_2x_1=0$，但 $K^*=-K\ne K$。复输入 $z=(1,i)^T$ 能发现问题：

$$Kz=(i,-1)^T,\qquad z^*Kz=i+(-i)(-1)=2i.$$

<!-- bilingual-en:start -->
Moreover, the ordinary-transpose expression $z^TKz=z_1z_2-z_2z_1$ is zero even for every complex $z$. Thus replacing $z^*$ by $z^T$ destroys this criterion. Ordinary complex bilinear forms are legitimate; they answer a different question. Over a real vector space, every real matrix already gives real $x^TAx$, so real-valuedness alone cannot characterize symmetry.
<!-- bilingual-en:end -->

而普通转置表达式 $z^TKz=z_1z_2-z_2z_1$ 甚至对全部复向量也为零。因此，把 $z^*$ 换成 $z^T$ 会使本判据失效。普通复双线性形式是合法对象，只是回答不同问题。在实向量空间中，任意实矩阵的 $x^TAx$ 本来就为实数，故仅靠取实值无法刻画对称性。

<!-- bilingual-en:start -->
Finally, the theorem tests reality, not sign. The Hermitian matrix $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ gives $|z_1|^2-|z_2|^2$, which can be positive, zero or negative. Requiring all values to be nonnegative is the stronger condition of [[半正定矩阵|positive semidefiniteness]].
<!-- bilingual-en:end -->

最后，本定理检查实值性，不检查正负。Hermitian 矩阵 $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ 给出 $|z_1|^2-|z_2|^2$，可以为正、零或负。要求全部值非负，是更强的[[半正定矩阵|半正定]]条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Axler, Linear Algebra Done Right, fourth edition, §7A, results 7.13–7.14, printed p. 234](https://linear.axler.net/LADR4e.pdf#page=248): the real-valuedness equivalence over the complex field and its failure over the real field. The coordinate proof above establishes the same result directly. Definition 7.10 and the complex spectral theorem 7.31 (PDF pp. 247 and 260) support the retained spectral proof.
- [Kahan, UC Berkeley Math H110, How to Recognize a Quadratic Form, p. 1, final paragraph](https://people.eecs.berkeley.edu/~wkahan/MathH110/qf.pdf#page=1): distinguishes ordinary complex quadratic forms from Hermitian forms; it is not used as a proof of the full converse. All displayed examples and the decomposition $A=H+iS$ are checked directly.
<!-- bilingual-en:end -->

- Axler 第 4 版 §7A，7.13–7.14，印刷第 234 页（PDF 第 248 页）：支持复数域上的实值充要条件及实数域边界；正文坐标证明直接建立同一结论。定义 7.10 与复谱定理 7.31（PDF 第 247、260 页）支持保留的谱论证明。
- Kahan 第 1 页末段：支持普通复二次型与 Hermitian 二次型的区分，不把它当作完整逆向证明；全部例子及 $A=H+iS$ 分解直接计算核验。
