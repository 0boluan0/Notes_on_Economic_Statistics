---
aliases:
- Linear independence of vectors in a Jordan chain
- Jordan 链向量线性无关
student_os: knowledge-atom
atom_id: LA-EIG-045
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
- '[[Jordan链]]'
- '[[线性无关]]'
related:
- '[[Jordan块]]'
- '[[Jordan标准形]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Linear independence of a Jordan chain
<!-- bilingual-en:end -->

*Jordan 链向量线性无关*

<!-- bilingual-en:start -->
Each successive position in a [[Jordan链|Jordan chain]] contributes a new independent direction. None of its vectors can be reconstructed from the others by a linear combination. This is why a length-$r$ chain can serve as a basis for an $r$-dimensional part of the matrix action. The guarantee concerns a complete, valid chain; distinct-looking vectors or several separately valid chains do not by themselves provide the same guarantee.
<!-- bilingual-en:end -->

[[Jordan链]]中每向后一个位置，就增加一个独立方向；其中任何一个向量都不能由其余向量线性组合得到。这使长度为 $r$ 的链能成为矩阵作用中某个 $r$ 维部分的基。保证针对一条完整、有效的链；向量看起来不同，或几条链分别有效，都不足以自动得到同样的结论。

> [!example] Independent directions need not be perpendicular / 独立方向不必垂直
> <!-- bilingual-en:start -->
> Let $A=\begin{pmatrix}2&1&0\\0&2&1\\0&0&2\end{pmatrix}$ and subtract $2I_3$, where $I_3$ is the [[单位矩阵|identity matrix]]. Then $N=A-2I_3$ sends a column $(x,y,z)^T$ to $(y,z,0)^T$; $T$ denotes [[矩阵转置|transpose]], turning the written coordinate row into a column. The following columns form a chain:
> <!-- bilingual-en:end -->
>
> 令 $A=\begin{pmatrix}2&1&0\\0&2&1\\0&0&2\end{pmatrix}$，减去两倍[[单位矩阵]] $I_3$ 后，$N=A-2I_3$ 把列向量 $(x,y,z)^T$ 送到 $(y,z,0)^T$；$T$ 表示[[矩阵转置|转置]]，把写成一行的坐标改成列向量。下列三个列向量构成一条链：
>
> $$v_1=\begin{pmatrix}1\\0\\0\end{pmatrix},\qquad
> v_2=\begin{pmatrix}1\\1\\0\end{pmatrix},\qquad
> v_3=\begin{pmatrix}1\\1\\1\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> Indeed $Nv_1=0$, $Nv_2=v_1$ and $Nv_3=v_2$. A combination $c_1v_1+c_2v_2+c_3v_3$ has coordinates $(c_1+c_2+c_3,c_2+c_3,c_3)^T$. For it to be zero, the third coordinate first forces $c_3=0$, then the second forces $c_2=0$, and finally $c_1=0$. This is [[线性无关|linear independence]]. The vectors are not mutually perpendicular: the [[向量点积|dot product]] of $v_1$ and $v_2$ is $1$.
> <!-- bilingual-en:end -->
>
> 直接计算可得 $Nv_1=0$、$Nv_2=v_1$、$Nv_3=v_2$。线性组合 $c_1v_1+c_2v_2+c_3v_3$ 的坐标为 $(c_1+c_2+c_3,c_2+c_3,c_3)^T$。要等于零，第三坐标先迫使 $c_3=0$，第二坐标再迫使 $c_2=0$，最后得到 $c_1=0$；这就是[[线性无关]]。这些向量并不两两垂直，例如 $v_1$ 与 $v_2$ 的[[向量点积]]为 $1$。

<!-- bilingual-en:start -->
## The recurrence gives the same conclusion in any coordinates
<!-- bilingual-en:end -->

*换成其他坐标，递推仍保证同一结论*

<!-- bilingual-en:start -->
Let $A$ be an $n\times n$ real or complex matrix with $n\ge1$, let $\lambda$ be a scalar in that field, and write $N=A-\lambda I_n$. Suppose $r$ is a positive integer and
<!-- bilingual-en:end -->

设 $A$ 是实或复 $n\times n$ 方阵，$n\ge1$，$\lambda$ 是相应数域中的标量，记 $N=A-\lambda I_n$。假设 $r$ 为正整数，且

$$v_1\ne0,\qquad Nv_1=0,\qquad Nv_{j+1}=v_j\quad(1\le j<r).$$

<!-- bilingual-en:start -->
Then $v_1,\ldots,v_r$ are linearly independent. Here independence means that, for scalar coefficients $c_1,\ldots,c_r$,
<!-- bilingual-en:end -->

则 $v_1,\ldots,v_r$ 线性无关。具体地，对相应数域中的标量系数 $c_1,\ldots,c_r$，有

$$c_1v_1+\cdots+c_rv_r=0
\quad\Longrightarrow\quad c_1=\cdots=c_r=0.$$

<!-- bilingual-en:start -->
No distinct-eigenvalue assumption is involved: all vectors in this chain belong to the same $\lambda$. Nor is an inner product or perpendicularity assumption needed. The initial condition already gives $Av_1=\lambda v_1$, so $\lambda$ is an eigenvalue.
<!-- bilingual-en:end -->

这里不涉及特征值互异假设：整条链属于同一个 $\lambda$，也不要求内积或垂直关系。首项条件本身已经给出 $Av_1=\lambda v_1$，因此 $\lambda$ 确实是特征值。

<!-- bilingual-en:start -->
## Applying powers removes the coefficients from the tail
<!-- bilingual-en:end -->

*施加幂，从链尾逐个消去系数*

<!-- bilingual-en:start -->
Repeated application of the recurrence gives
<!-- bilingual-en:end -->

重复使用递推关系，得到

$$N^kv_j=
\begin{cases}
v_{j-k},&0\le k<j,\\
0,&k\ge j,
\end{cases}$$

<!-- bilingual-en:start -->
where $k$ is a nonnegative integer and $N^0=I_n$. Thus $N^{r-1}$ sends all earlier vectors to zero but sends $v_r$ to the nonzero $v_1$. Apply that power to a possible dependence relation:
<!-- bilingual-en:end -->

其中 $k$ 为非负整数，$N^0=I_n$。于是 $N^{r-1}$ 把前面的向量全部送到零，却把 $v_r$ 送到非零 $v_1$。对可能存在的依赖关系施加这个幂：

$$0=N^{r-1}(c_1v_1+\cdots+c_rv_r)=c_rv_1.$$

<!-- bilingual-en:start -->
Since $v_1\ne0$, the scalar $c_r$ must be zero. Delete that term and apply $N^{r-2}$ to the remaining relation; this gives $c_{r-1}v_1=0$, hence $c_{r-1}=0$. Continue backwards until every coefficient is zero. For a three-vector chain, $N^2$ isolates $c_3$ first, then $N$ isolates $c_2$, and the final relation gives $c_1=0$. For $r=1$, nonzero $v_1$ proves the result immediately.
<!-- bilingual-en:end -->

由于 $v_1\ne0$，标量 $c_r$ 只能为零。删掉这一项，对剩余关系施加 $N^{r-2}$，便得到 $c_{r-1}v_1=0$，所以 $c_{r-1}=0$。继续向前，直到全部系数为零。三向量链中，先用 $N^2$ 确定 $c_3$，再用 $N$ 确定 $c_2$，最后得到 $c_1=0$。若 $r=1$，非零 $v_1$ 已直接证明结论。

<!-- bilingual-en:start -->
Consequently a length-$r$ chain spans an $r$-dimensional subspace, and $r\le n$. Its recurrence also makes that span invariant under $A$, so the chain is a [[基|basis]] for that restricted action and yields one [[Jordan块|Jordan block]]. A larger whole-space [[Jordan标准形|Jordan representation]] requires further compatible chains.
<!-- bilingual-en:end -->

因此长度为 $r$ 的链张成 $r$ 维子空间，且 $r\le n$。递推还保证这个空间在 $A$ 下不变，所以链是该受限作用的一组[[基]]，对应一个 [[Jordan块]]。要得到整个空间的 [[Jordan标准形]]，还需选择其他能共同组成基的链。

> [!warning] Two valid chains may reuse the same directions / 两条有效链可能重复使用同一批方向
> <!-- bilingual-en:start -->
> For $A=\begin{pmatrix}2&1\\0&2\end{pmatrix}$, put $N=A-2I_2$ and let $e_1=(1,0)^T$, $e_2=(0,1)^T$. Both $(e_1,e_2)$ and $(2e_1,e_1+2e_2)$ are length-two chains: the latter satisfies $N(e_1+2e_2)=2e_1$. Each pair is independent. But their combined list is dependent, since its first and third vectors already satisfy $2e_1-(2e_1)=0$ with nonzero coefficients. Individual-chain independence does not select a full Jordan basis for you.
> <!-- bilingual-en:end -->
>
> 对 $A=\begin{pmatrix}2&1\\0&2\end{pmatrix}$，令 $N=A-2I_2$，并取 $e_1=(1,0)^T$、$e_2=(0,1)^T$。$(e_1,e_2)$ 与 $(2e_1,e_1+2e_2)$ 都是长度为二的链，因为后者满足 $N(e_1+2e_2)=2e_1$。每一对分别线性无关，但合并后的列表线性相关：第一、第三个向量已通过非零系数关系 $2e_1-(2e_1)=0$ 相互抵消。单条链的独立性，不会自动替你选出完整 Jordan 基。

<!-- bilingual-en:start -->
Likewise, allowing a zero first vector destroys the theorem. For the zero matrix $N$, the list $(0,v)$ with $v\ne0$ satisfies the recurrence $Nv=0$, but it is dependent because it contains zero. This is why the chain definition requires $v_1\ne0$ rather than just the recurrence equations.
<!-- bilingual-en:end -->

同样，允许首向量为零会破坏定理。对零矩阵 $N$，列表 $(0,v)$（$v\ne0$）满足递推 $Nv=0$，却因为包含零向量而线性相关。这正是链定义除了递推方程，还要求 $v_1\ne0$ 的原因。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, §8A exercise 2, printed p. 306 / PDF p. 320](https://linear.axler.net/LADR4e.pdf#page=320): independence of successive powers when the last nonzero image precedes the first zero. Reversing that list gives the chain convention here; the coefficient-isolation proof is supplied in full.
- [Axler, §8C, printed pp. 321–323 / PDF pp. 335–337](https://linear.axler.net/LADR4e.pdf#page=335): chain bases, their block representation and the need to choose compatible pieces for a full basis. Both displayed examples are checked directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=3|MIT Session 3.4 summary, pp. 3–4]]: the course's Jordan-block context, not a standalone proof of chain independence.
- [[03_Positive Definite Matrices and Applications#3.4.4 Jordan链与Jordan标准形|Permanent course §3.4.4]]: the same eigenvector-first recurrence and basis ordering.
<!-- bilingual-en:end -->

- Axler 第8A节习题2，印刷306页／PDF320页：核对最后非零像之后首次归零时，连续幂向量的独立性；反转列表便对应本卡的链约定，正文完整给出逐项隔离系数的证明。
- Axler 第8C节，印刷321–323页／PDF335–337页：核对链基、块表示及为完整基选取兼容部分的必要性。两个显示例子均直接核算。
- MIT Session3.4 摘要第3–4页：保留课程的 Jordan 块背景，不将其当成链独立性的单独证明。
- 永久课程 §3.4.4：核对先列普通特征向量的递推及基顺序。
