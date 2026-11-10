---
aliases:
- Existence of Schur decompositions
- Schur 分解存在性
student_os: knowledge-atom
atom_id: LA-EIG-044
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
- '[[Schur分解]]'
- '[[特征多项式底层域]]'
related:
- '[[Normal 矩阵谱定理]]'
- '[[Jordan标准形的域条件]]'
- '[[Jordan、Schur与SVD用途边界]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Existence of Schur decompositions
<!-- bilingual-en:end -->

*Schur 分解存在性*

<!-- bilingual-en:start -->
Some matrices have too few independent eigenvectors to be diagonalized. **Schur's theorem** still gives them a useful coordinate system: an orthonormal basis in which the matrix is triangular over complex numbers, or triangular in one- and two-dimensional blocks over real numbers. The proof needs one invariant direction or plane at a time, rather than a full eigenvector basis at the start.
<!-- bilingual-en:end -->

有些矩阵没有足够多的线性无关特征向量，因而无法对角化。**Schur 定理**仍保证能找到有用的坐标系统：复数域中有标准正交基使矩阵呈三角形，实数域中则得到由一维、二维块组成的三角形。证明一次只需要一个不变方向或平面，不必一开始就拥有整组特征向量基。

> [!example] One eigenvector direction can be enough to start / 一个特征方向就能开始
> <!-- bilingual-en:start -->
> Take $A=\begin{pmatrix}2&1\\0&2\end{pmatrix}$. Its only eigenvalue is $2$, and $Av=2v$ forces the second coordinate of $v$ to be zero. Therefore it has only one independent eigenvector direction and cannot be [[对角化|diagonalized]]. Yet it is already upper triangular: $Q=I_2$, $T=A$ is a [[Schur分解|Schur decomposition]], with $I_2$ the two-dimensional [[单位矩阵|identity matrix]]. The extra off-diagonal entry is allowed to remain.
> <!-- bilingual-en:end -->
>
> 取 $A=\begin{pmatrix}2&1\\0&2\end{pmatrix}$。它唯一的特征值为 $2$，而 $Av=2v$ 迫使 $v$ 的第二坐标为零，所以只有一个独立特征方向，不能[[对角化]]。但它已是上三角矩阵：$Q=I_2$、$T=A$ 就给出 [[Schur分解]]，其中 $I_2$ 为二维[[单位矩阵]]。非对角线上的额外元素允许保留下来。

<!-- bilingual-en:start -->
## What exists over each field
<!-- bilingual-en:end -->

*每个数域分别保证什么*

<!-- bilingual-en:start -->
For every complex $n\times n$ matrix $A$, where $n\ge1$, there are a [[Unitary 矩阵|unitary]] matrix $Q$ and an upper [[三角矩阵|triangular]] matrix $T$ such that $A=QTQ^*$. The star is conjugate transpose, and $Q^*Q=I_n$. For every real $A$, there are a real [[正交矩阵|orthogonal]] $Q$ and a real block upper-triangular $T$ such that $A=QTQ^T$. Its diagonal blocks have size $1\times1$ or $2\times2$, with two-by-two blocks carrying nonreal conjugate eigenvalue pairs. No invertibility or distinct-eigenvalue assumption is required.
<!-- bilingual-en:end -->

对每个复数 $n\times n$ 方阵 $A$，$n\ge1$，都存在[[Unitary 矩阵|酉矩阵]] $Q$ 与上[[三角矩阵|三角矩阵]] $T$，使 $A=QTQ^*$；星号是共轭转置，且 $Q^*Q=I_n$。对每个实方阵 $A$，都存在实[[正交矩阵]] $Q$ 与实块上三角矩阵 $T$，使 $A=QTQ^T$。对角块的尺寸为 $1\times1$ 或 $2\times2$，二阶块承载非实共轭特征值对。不要求可逆，也不要求特征值互异。

<!-- bilingual-en:start -->
The field matters because a nonconstant complex polynomial always has a complex root. Thus the [[特征多项式|characteristic polynomial]] of a positive-size complex matrix supplies an eigenvalue, and a nonzero [[特征对|eigenvector]] exists. Over the reals, $\lambda^2+1$ has no real root; a real matrix may have no real eigenvector at all. The [[特征多项式底层域|field condition]] explains why the real theorem must allow two-dimensional blocks.
<!-- bilingual-en:end -->

数域很关键：非恒定复多项式必有复根，所以正尺寸复矩阵的[[特征多项式]]会提供一个特征值，进而存在非零[[特征对|特征向量]]。实数域中，$\lambda^2+1$ 没有实根；实矩阵可能完全没有实特征向量。[[特征多项式底层域|数域条件]]解释了实数版本为何必须允许二维块。

<!-- bilingual-en:start -->
## The complex proof isolates one column
<!-- bilingual-en:end -->

*复数证明先隔离一列*

<!-- bilingual-en:start -->
For $n=1$ there is nothing to eliminate. For larger $n$, choose an eigenvector $v$ with $Av=\lambda v$ and normalize it to $q_1=v/\|v\|_2$, where $\|v\|_2=\sqrt{\sum_j|v_j|^2}$ is its positive length from the [[内积|standard complex inner product]], and $|v_j|$ is the modulus of coordinate $j$. The new vector has length one. Extend it to an [[标准正交组|orthonormal basis]], for example by extending to a basis and applying [[Gram-Schmidt正交化|Gram–Schmidt]]. Write $Q_0=[q_1\ Q_\perp]$, where $Q_\perp$ contains the other $n-1$ columns. Direct block multiplication gives
<!-- bilingual-en:end -->

$n=1$ 时无需消去任何元素。对更大的 $n$，选择满足 $Av=\lambda v$ 的特征向量，归一化为 $q_1=v/\|v\|_2$，其中 $\|v\|_2=\sqrt{\sum_j|v_j|^2}$ 是由[[内积|标准复内积]]得到的正长度，$|v_j|$ 是第 $j$ 个坐标的模；新向量的长度为一。把它扩充成[[标准正交组|标准正交基]]，例如先扩充成基再做 [[Gram-Schmidt正交化]]。记 $Q_0=[q_1\ Q_\perp]$，其中 $Q_\perp$ 含剩余 $n-1$ 列。直接按块相乘得到

$$
Q_0^*AQ_0
=\begin{pmatrix}
q_1^*Aq_1&q_1^*AQ_\perp\\
Q_\perp^*Aq_1&Q_\perp^*AQ_\perp
\end{pmatrix}
=\begin{pmatrix}\lambda&b\\0&C\end{pmatrix}.
$$

<!-- bilingual-en:start -->
Here $b=q_1^*AQ_\perp$ is a row of length $n-1$, and $C=Q_\perp^*AQ_\perp$ is an $(n-1)\times(n-1)$ matrix. The lower-left block vanishes because $Aq_1=\lambda q_1$ and $Q_\perp^*q_1=0$. We have isolated one invariant line and reduced the remaining task to a smaller square matrix.
<!-- bilingual-en:end -->

这里 $b=q_1^*AQ_\perp$ 是含 $n-1$ 个元素的行，$C=Q_\perp^*AQ_\perp$ 是 $(n-1)\times(n-1)$ 矩阵。左下块为零，因为 $Aq_1=\lambda q_1$ 且 $Q_\perp^*q_1=0$。至此隔离出一条不变直线，剩余任务缩小成一个较小的方阵。

<!-- bilingual-en:start -->
By induction, choose unitary $W$ with $W^*CW=R$, where $R$ is upper triangular. Only change the remaining coordinates:
<!-- bilingual-en:end -->

由归纳假设，可以选择酉矩阵 $W$，使 $W^*CW=R$，其中 $R$ 上三角。只改变剩余坐标：

$$
Q=Q_0\begin{pmatrix}1&0\\0&W\end{pmatrix},\qquad
Q^*AQ=
\begin{pmatrix}\lambda&bW\\0&R\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The matrix $Q$ is still unitary, and the right-hand matrix is upper triangular. This completes the induction. The new columns need not be eigenvectors of the original $A$: each step only requires an eigenvector of the smaller matrix currently being treated.
<!-- bilingual-en:end -->

$Q$ 仍为酉矩阵，右边已经上三角，归纳证明完成。新列不必都是原矩阵 $A$ 的特征向量：每一步只需当前正在处理的较小矩阵的一个特征向量。

> [!warning] The orthogonal complement need not be invariant / 正交补未必不变
> <!-- bilingual-en:start -->
> The other columns span the [[正交补|orthogonal complement]] of $q_1$, but $A$ may send a vector there partly back along $q_1$. That contribution is exactly the row $b$. In the opening example, $q_1=(1,0)^T$ and $q_2=(0,1)^T$, yet $Aq_2=(1,2)^T$ leaves the second axis. The proof applies induction to the **compression matrix** $C=Q_\perp^*AQ_\perp$: enter that complement, apply $A$, and keep only the complement coordinates. It does not assume that $A$ maps the complement into itself.
> <!-- bilingual-en:end -->
>
> 其余列张成 $q_1$ 的[[正交补]]，但 $A$ 可能把其中向量的一部分送回 $q_1$ 方向，这部分贡献恰由行 $b$ 记录。开头例子中，$q_1=(1,0)^T$、$q_2=(0,1)^T$，但 $Aq_2=(1,2)^T$ 离开第二条轴。证明把归纳用在**压缩矩阵** $C=Q_\perp^*AQ_\perp$ 上：进入正交补，施加 $A$，再只取正交补的坐标；没有假设 $A$ 把正交补送回自身。

<!-- bilingual-en:start -->
## The real proof isolates a line or a plane
<!-- bilingual-en:end -->

*实数证明隔离直线或平面*

<!-- bilingual-en:start -->
If a real matrix has a real eigenvalue, it has a real eigenvector: from a complex eigenvector, at least one of its real or imaginary parts is a nonzero real eigenvector. The same construction then removes a one-by-one block using real orthogonal coordinates.
<!-- bilingual-en:end -->

实矩阵若有实特征值，就存在实特征向量：如果先得到复特征向量，它的实部、虚部中至少一个是非零实特征向量。随后沿用上面的构造，以实正交坐标隔离一个一阶块。

<!-- bilingual-en:start -->
Otherwise choose a nonreal eigenvalue $\lambda=a+ib$ and an eigenvector $v=x+iy$, with $a,b$ real, $b\ne0$, $x,y\in\mathbb R^n$ and $i^2=-1$. Separating real and imaginary parts of $Av=\lambda v$ gives
<!-- bilingual-en:end -->

否则，选一个非实特征值 $\lambda=a+ib$ 和特征向量 $v=x+iy$，其中 $a,b$ 为实数，$b\ne0$，$x,y\in\mathbb R^n$，且 $i^2=-1$。将 $Av=\lambda v$ 的实部、虚部分开，得到

$$Ax=ax-by,\qquad Ay=bx+ay.$$

<!-- bilingual-en:start -->
The vectors $x,y$ are linearly independent. If they lay on one real line, $v$ would be a nonzero complex scalar times a nonzero real vector $w$, giving $Aw=\lambda w$; its imaginary part would force $bw=0$, a contradiction. Thus $x,y$ span a real two-dimensional [[不变子空间|invariant subspace]]: both displayed images stay in their span.
<!-- bilingual-en:end -->

$x,y$ 线性无关。如果它们位于同一条实直线上，$v$ 就能写成某个非零复标量乘非零实向量 $w$，从而 $Aw=\lambda w$；比较虚部会迫使 $bw=0$，矛盾。因此 $x,y$ 张成实二维[[不变子空间]]，上面两个像都留在它们张成的平面内。

<!-- bilingual-en:start -->
Choose an orthonormal basis for this plane and complete it to a real orthonormal basis of the whole space. In those coordinates the matrix has the form $\begin{pmatrix}B&D\\0&C\end{pmatrix}$, with a real $2\times2$ block $B$. In the original plane basis $[x\ y]$, the action is $\begin{pmatrix}a&b\\-b&a\end{pmatrix}$, whose eigenvalues are $a\pm ib$. Changing the plane's basis is a [[相似矩阵|similarity]], so $B$ has the same pair; it need not have exactly those four displayed entries.
<!-- bilingual-en:end -->

先为这个平面选取标准正交基，再扩充成整个空间的实标准正交基。新坐标中的矩阵呈 $\begin{pmatrix}B&D\\0&C\end{pmatrix}$，其中 $B$ 为实 $2\times2$ 块。在原平面基 $[x\ y]$ 下，作用矩阵是 $\begin{pmatrix}a&b\\-b&a\end{pmatrix}$，特征值为 $a\pm ib$。改变平面基属于[[相似矩阵|相似换基]]，因此 $B$ 具有相同的一对特征值，但四个元素不必恰好等于所显示的四个数。

<!-- bilingual-en:start -->
Apply the same argument to the smaller real matrix $C$. It has size $n-2$ after a plane is removed, or $n-1$ after a line. Stop when no coordinates remain. Each step keeps the previous lower-left zero block, so the assembled form is block upper triangular. As in the complex proof, the remaining orthogonal complement need not itself be invariant under the original matrix.
<!-- bilingual-en:end -->

对较小的实矩阵 $C$ 重复同样论证：移除平面后其尺寸为 $n-2$，移除直线后为 $n-1$；没有剩余坐标时停止。每一步都保留之前的左下零块，最终拼成块上三角形。与复数证明相同，不要求剩余正交补本身在原矩阵作用下不变。

<!-- bilingual-en:start -->
For the quarter-turn $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$, the whole real plane is the first block, and nothing remains. Over complex numbers, a one-dimensional eigenvector step is available instead. In either version, the proof guarantees Schur form, not diagonal form; unitary diagonalization has the extra [[Normal 矩阵谱定理|normal-matrix condition]]. Its leading invariant spaces use full blocks in the real case, as explained in the [[Schur分解|definition and examples]].
<!-- bilingual-en:end -->

对四分之一圈旋转 $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$，整个实平面就是第一个块，无需继续；换到复数域后则可选择一维特征向量步骤。两种版本都只保证 Schur 形，不保证对角形；酉对角化还需要 [[Normal 矩阵谱定理|normal 条件]]。实数情形的前导不变空间须取完整块，详见[[Schur分解|定义与例子]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, Linear Algebra Done Right, 4th ed., 6.36–6.38](https://linear.axler.net/LADR4e.pdf#page=217), printed pp.203–204 / PDF pp.217–218: completing an orthonormal list and complex Schur existence. The recursive block proof here is checked by direct multiplication.
- [Demmel, Berkeley Ma221, Lecture 15, “Proof of Theorem” through “Proof of Real Schur Form”](https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Lectures/Lecture_15.html): eigenvector recursion, the invariant-subspace block lemma, and the real/imaginary-part construction for a conjugate pair. Conjugate transposes are retained throughout the complex calculation.
- [LAPACK Schur factorization](https://netlib.org/lapack/lug/node50.html) and [LAPACK95 nonsymmetric eigenproblems](https://netlib.org/lapack95/lug95/node34.html): real one-by-one/two-by-two and complex triangular forms. Whole real blocks are required for leading invariant subspaces.
<!-- bilingual-en:end -->

- Axler 6.36–6.38，印刷203–204页／PDF217–218页，核对标准正交组扩基与复 Schur 存在性；本卡的递归分块证明另由矩阵乘法直接核算。
- Demmel 的 Berkeley Ma221 Lecture15，从复数定理证明至实 Schur 证明，核对特征向量递归、不变子空间分块引理，以及共轭对的实虚部构造。复数计算始终保留共轭转置。
- LAPACK 两份原说明核对实数一阶／二阶块与复三角形式。实数前导不变子空间必须尊重完整对角块。
