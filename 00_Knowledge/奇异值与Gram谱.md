---
aliases:
- Singular values and Gram spectra
- Gram spectrum identity for singular values
student_os: knowledge-atom
atom_id: LA-SVD-025
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
requires:
- '[[奇异值]]'
- '[[Hermitian 谱定理]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
leads_to:
- '[[SVD手算流程]]'
- '[[正规方程条件数平方]]'
related:
- '[[奇异向量与Gram特征向量]]'
- '[[Frobenius范数与奇异值]]'
---

<!-- bilingual-en:start -->
# Singular values and Gram spectra
<!-- bilingual-en:end -->

*奇异值与 Gram 谱*

<!-- bilingual-en:start -->
A rectangular matrix has two natural square companions: one compares its input columns, the other compares its output rows. Their sizes can differ, yet their positive eigenvalues agree, including repetitions. Those values are the squares of the original matrix's positive [[奇异值|singular values]]. This lets a square eigenvalue problem reveal the stretch amounts of a rectangular transformation.
<!-- bilingual-en:end -->

长方矩阵有两个自然的方阵伙伴：一侧比较输入列，另一侧比较输出行。它们的尺寸可以不同，正特征值却相同，连重复次数也一致。这些值恰好是原矩阵正[[奇异值]]的平方，因此可用方阵的特征值问题，读出长方变换的伸缩量。

<!-- bilingual-en:start -->
## A minimal example
<!-- bilingual-en:end -->

## 最小例子

> [!example] The extra output direction / 多出来的输出方向
> <!-- bilingual-en:start -->
> The matrix below stretches the first input direction by 3 and the second by 1, then adds a zero third coordinate. Its positive singular values are $3,1$. Multiplying by the transpose in each order gives
> <!-- bilingual-en:end -->
>
> 下列矩阵把第一输入方向伸长3倍、第二方向保持原长，再添一个为零的第三坐标。它的正奇异值为 $3,1$；与转置按两个顺序相乘，得到
>
> $$
> A=\begin{pmatrix}3&0\\0&1\\0&0\end{pmatrix},\qquad
> A^TA=\operatorname{diag}(9,1),\qquad AA^T=\operatorname{diag}(9,1,0).
> $$
>
> <!-- bilingual-en:start -->
> Here $\operatorname{diag}$ places the listed numbers on the diagonal and zeros elsewhere. Both square matrices have positive eigenvalues $9,1$, giving square roots $3,1$. The extra zero in $AA^T$ belongs to the third output direction: $A^T(0,0,1)^T=0$. It does not add a third singular value to the usual two-value list for a $3\times2$ matrix.
> <!-- bilingual-en:end -->
>
> $\operatorname{diag}$ 表示对角线为所列数值、其余项为零。两方阵都有正特征值 $9,1$，开方得到 $3,1$。$AA^T$ 多出的零对应第三输出方向：$A^T(0,0,1)^T=0$；它不会给通常的 $3\times2$ 矩阵奇异值列表添上第三项。

<!-- bilingual-en:start -->
## What the two square matrices measure
<!-- bilingual-en:end -->

## 两个方阵各自测量什么

<!-- bilingual-en:start -->
Let $A$ be an $m\times n$ real or complex matrix, with positive integers $m,n$. Write $A^*$ for its [[矩阵转置|conjugate transpose]], equal to $A^T$ in the real case. The column [[Gram矩阵正定判据|Gram matrix]] $G=A^*A$ is $n\times n$; entry $(i,j)$ is the inner product of columns $i,j$. The other Gram matrix $H=AA^*$ is $m\times m$ and compares rows. Both are [[Hermitian 矩阵|Hermitian]] (equal to their conjugate transpose) and [[半正定矩阵|positive semidefinite]], since
<!-- bilingual-en:end -->

设 $A$ 是 $m\times n$ 的实或复矩阵，$m,n$ 为正整数。$A^*$ 表示[[矩阵转置|共轭转置]]，实数情形就是 $A^T$。列[[Gram矩阵正定判据|Gram 矩阵]] $G=A^*A$ 为 $n\times n$，第 $(i,j)$ 项是第 $i,j$ 列的内积；另一侧 $H=AA^*$ 为 $m\times m$，比较各行。二者都为 [[Hermitian 矩阵]]，即等于自己的共轭转置，并且[[半正定矩阵|半正定]]，因为

$$
x^*Gx=\|Ax\|_2^2\ge0\quad(x\in\mathbb F^n),\qquad
y^*Hy=\|A^*y\|_2^2\ge0\quad(y\in\mathbb F^m),
$$

<!-- bilingual-en:start -->
where $\mathbb F$ is the real or complex field and $\|\cdot\|_2$ is [[欧氏范数|Euclidean length]]. Thus their [[特征对|eigenvalues]] are real and nonnegative. The word **spectrum** here means the eigenvalue list, with each eigenvalue counted according to its multiplicity.
<!-- bilingual-en:end -->

其中 $\mathbb F$ 表示实数域或复数域，$\|\cdot\|_2$ 为[[欧氏范数|欧氏长度]]。因此它们的[[特征对|特征值]]都是非负实数。这里的**谱**指特征值列表，并按重数重复列出每个值。

<!-- bilingual-en:start -->
Use a full [[奇异值分解|SVD]] $A=U\Sigma V^*$, where $U,V$ have orthonormal columns and sizes $m\times m,n\times n$, respectively. They satisfy $U^*U=I_m$ and $V^*V=I_n$, with $I$ the [[单位矩阵|identity matrix]]. The rectangular diagonal factor $\Sigma$ contains $p=\min(m,n)$ singular values. Multiplying the factors gives
<!-- bilingual-en:end -->

取完整[[奇异值分解]] $A=U\Sigma V^*$，其中 $U,V$ 分别是 $m\times m,n\times n$ 的标准正交列矩阵，满足 $U^*U=I_m$、$V^*V=I_n$，$I$ 为[[单位矩阵]]。长方对角因子 $\Sigma$ 放入 $p=\min(m,n)$ 个奇异值。相乘得到

$$
\boxed{A^*A=V\Sigma^*\Sigma V^*,\qquad
AA^*=U\Sigma\Sigma^*U^*.}
$$

<!-- bilingual-en:start -->
The middle factors are square diagonal matrices. Their positive entries are the same squared stretches; extra dimensions only add zeros. Therefore, if $r=\operatorname{rank}(A)$ and $\sigma_1\ge\cdots\ge\sigma_r>0$ are the positive singular values, both Gram matrices have the positive eigenvalue list $\sigma_1^2,\ldots,\sigma_r^2$, including repetitions. Taking the **nonnegative** square roots recovers the stretches. This is an identity about spectra; the complete calculation procedure belongs to [[SVD手算流程|computing an SVD by hand]].
<!-- bilingual-en:end -->

中间因子都是方形对角矩阵，正的对角项是同一组伸缩量的平方，多出的尺寸只增加零。因此若 $r=\operatorname{rank}(A)$，正奇异值为 $\sigma_1\ge\cdots\ge\sigma_r>0$，两侧 Gram 的正特征值列表都是 $\sigma_1^2,\ldots,\sigma_r^2$，含重复值。取**非负**平方根便恢复伸缩量。本卡负责谱的恒等关系；完整计算步骤见[[SVD手算流程]]。

<!-- bilingual-en:start -->
The correspondence can also be seen one direction at a time. If $v\ne0$ and $Gv=\lambda v$ with $\lambda>0$, then $Av\ne0$ and $H(Av)=A(Gv)=\lambda Av$. Conversely, a nonzero $u$ with $Hu=\lambda u$ gives $G(A^*u)=\lambda A^*u$. On these eigenspaces, the maps $v\mapsto Av$ and $u\mapsto A^*u/\lambda$ undo each other. Hence the corresponding eigenspaces have the same dimension, which explains matching multiplicities; see [[奇异向量与Gram特征向量|the paired-vector relationship]].
<!-- bilingual-en:end -->

也可逐方向理解：若 $v\ne0$、$Gv=\lambda v$ 且 $\lambda>0$，则 $Av\ne0$，并有 $H(Av)=A(Gv)=\lambda Av$。反过来，非零 $u$ 满足 $Hu=\lambda u$ 时，得到 $G(A^*u)=\lambda A^*u$。在这些特征空间上，$v\mapsto Av$ 与 $u\mapsto A^*u/\lambda$ 互为逆映射，所以空间维数相同，这解释了重数一致；配对向量关系见[[奇异向量与Gram特征向量]]。

<!-- bilingual-en:start -->
## Count the zero directions separately
<!-- bilingual-en:end -->

## 尺寸边界

<!-- bilingual-en:start -->
The [[矩阵秩|rank]] $r$ counts independent directions that survive the transformation. The [[零空间|nullspaces]] obey $N(A^*A)=N(A)$, which follows from $\|Ax\|_2^2=0$ exactly when $Ax=0$; similarly $N(AA^*)=N(A^*)$. Therefore the complete lists contain
<!-- bilingual-en:end -->

[[矩阵秩]] $r$ 计算变换后仍保留的独立方向个数。由 $\|Ax\|_2^2=0$ 当且仅当 $Ax=0$，得到[[零空间]]关系 $N(A^*A)=N(A)$；类似地 $N(AA^*)=N(A^*)$。因此完整列表分别包含

$$
\begin{array}{c|c|c}
 &\text{positive eigenvalues}&\text{zero eigenvalues}\\\hline
 A^*A\;(n\times n)&\sigma_1^2,\ldots,\sigma_r^2&n-r\\
 AA^*\;(m\times m)&\sigma_1^2,\ldots,\sigma_r^2&m-r
\end{array}
$$

<!-- bilingual-en:start -->
The usual singular-value list has $p$ entries, of which $p-r$ are zero. For a rank-one $2\times3$ matrix, this list has one positive value and one zero; $A^*A$ has two zero eigenvalues, while $AA^*$ has one. If $A=0$, the positive lists are empty and all these counts still apply. For square matrices the complete Gram spectra agree; for rectangular matrices only the positive parts agree with multiplicity.
<!-- bilingual-en:end -->

通常的奇异值列表有 $p$ 项，其中 $p-r$ 项为零。例如秩1的 $2\times3$ 矩阵，奇异值列表含一正一零；$A^*A$ 有两个零特征值，$AA^*$ 有一个。若 $A=0$，正值列表为空，计数仍成立。方阵的两侧完整 Gram 谱相同；长方矩阵则仅正谱连同重数一致。

<!-- bilingual-en:start -->
Extra zeros matter when interpreting invertibility or [[正规方程条件数平方|condition numbers]]. In the opening example, $A^TA$ is invertible with condition number $9$, while $AA^T$ is singular. Their shared positive eigenvalues do not make both matrices invertible. Likewise, this result concerns the Gram eigenvalues, not the eigenvalues of $A$ itself: $A=\begin{pmatrix}0&1\\0&0\end{pmatrix}$ has only zero eigenvalues, but $A^TA=\operatorname{diag}(0,1)$ and its singular values are $1,0$.
<!-- bilingual-en:end -->

解释可逆性或[[正规方程条件数平方|条件数]]时，额外零值不能忽略。开头例子中 $A^TA$ 可逆、条件数为9，$AA^T$ 却奇异；共有正特征值不意味着两侧都可逆。这里谈的也不是 $A$ 自己的特征值：$A=\begin{pmatrix}0&1\\0&0\end{pmatrix}$ 的特征值全为零，但 $A^TA=\operatorname{diag}(0,1)$，奇异值为 $1,0$。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.64–7.70 and 7.75, PDF pp.284–289](https://linear.axler.net/LADR4e.pdf#page=284): nonnegative Gram eigenvalues, nullspaces/ranks, positive singular values and adjoint pairing. Axler's input-dimension convention includes additional zero singular values for wide maps; the $\min(m,n)$ list used here follows LAPACK, while the positive spectra coincide under either convention.
- [LAPACK Users' Guide, Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html): full rectangular factors and the $\min(m,n)$ convention; the displayed Gram identities and zero multiplicities follow by multiplying those factors.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session3.5, PDF pp.2–4]]: course calculation from Gram eigenvalues, sign pairing and a rank-deficient example. The matrices and eigenvalue counts above are checked directly.
<!-- bilingual-en:end -->

- Axler 第7.64–7.70、7.75条，PDF284–289：核对 Gram 非负谱、零空间与秩、正奇异值及伴随配对。该书按输入维数计数，宽映射另含零奇异值；本卡依 LAPACK 使用 $\min(m,n)$ 项列表，两种约定的正谱一致。
- LAPACK：核对完整长方因子与 $\min(m,n)$ 口径；正文 Gram 恒等式与零值重数由这些因子相乘得到。
- MIT18.06SC Session3.5 PDF2–4：核对课程中的 Gram 特征值计算、符号配对及秩亏例子；正文各矩阵与特征值计数直接计算。
