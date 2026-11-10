---
aliases:
  - "CPQR minimum-norm boundary"
  - "Basic and minimum-norm solutions in pivoted QR"
  - "主元 QR 基本解与最小范数解"
student_os: knowledge-atom
atom_id: LA-PROJ-029
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
requires:
  - "[[CPQR秩揭示边界]]"
  - "[[最小范数最小二乘解]]"
  - "[[QR分解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
leads_to:
  - "[[秩亏最小二乘算法]]"
---

<!-- bilingual-en:start -->
# Basic and minimum-norm solutions in pivoted QR
<!-- bilingual-en:end -->

*CPQR 主元回代与最小范数解*

<!-- bilingual-en:start -->
A [[线性最小二乘|least-squares problem]], which minimizes the length of the fitting residual, can have several coefficient vectors that give the same best fit. Column-pivoted QR, abbreviated **CPQR**, reorders the columns before a [[QR分解|QR factorization]] so that selected columns are handled first. Solving for those variables and setting the remaining variables to zero is a convenient way to choose a solution. It does not generally choose the solution with the smallest [[欧氏范数|Euclidean length]]. This distinction can remain even if the factorization and rank decision are exact.
<!-- bilingual-en:end -->

[[线性最小二乘]]最小化拟合残差的长度；其中多个系数向量可能给出同一个最佳拟合。**列主元 QR（CPQR）**先重排列，再作[[QR分解]]，使选定的列先被处理。只解这些变量、把其余变量设为零，是方便的选解方法，却通常不保证系数向量的[[欧氏范数|欧氏长度]]最小。即使分解与秩判断完全精确，这个区别仍可能存在。

<!-- bilingual-en:start -->
## The smallest example already separates the objectives
<!-- bilingual-en:end -->

## 最小反例

> [!example] Same fit, different lengths · 拟合相同，长度不同
> <!-- bilingual-en:start -->
> Let $A=[1\ 1]$ and $b=1$. The equation says only that $x_1+x_2=1$. Choosing the first variable as the pivot and setting the second to zero gives $x_{\mathrm{basic}}=(1,0)^T$, of length $1$. Every exact solution can be written
>
> $$x=(1/2,1/2)^T+t(1,-1)^T,\qquad
> \|x\|_2^2=1/2+2t^2\quad(t\in\mathbb R).$$
>
> The unique shortest solution is therefore $x_\star=(1/2,1/2)^T$, of length $1/\sqrt2$. Both solutions have zero residual. Pivoting to the other column instead returns $(0,1)^T$, which has the same problem. Notice that $A$ has full row rank; nonunique coefficients do not require rank below $\min(m,n)$.
> <!-- bilingual-en:end -->
>
> 取 $A=[1\ 1]$、$b=1$，方程只要求 $x_1+x_2=1$。把第一变量当主元、第二变量设为零，得到长度为 $1$ 的 $(1,0)^T$。全部实数解都可写成上述带参数 $t$ 的形式，长度平方为 $1/2+2t^2$，故唯一最短解为 $(1/2,1/2)^T$，长度 $1/\sqrt2$。两个解的残差都为零；换第二列做主元，只会得到同样不够短的 $(0,1)^T$。注意这个 $A$ 满行秩：系数不唯一，并不要求秩小于 $\min(m,n)$。

<!-- bilingual-en:start -->
## What setting free coordinates to zero accomplishes
<!-- bilingual-en:end -->

*主元回代解决了什么*

<!-- bilingual-en:start -->
For the exact argument, let $A$ be an $m\times n$ real or complex matrix of [[矩阵秩|rank]] $r$, meaning $r$ independent column directions, with $1\le r<n$. Suppose a column permutation and QR factorization give

$$AP=Q\begin{bmatrix}R_{11}&R_{12}\\0&0\end{bmatrix}.$$

Here $P$ is a [[置换矩阵|permutation matrix]] that reorders columns, $Q$ is an $m\times m$ [[正交矩阵|orthogonal matrix]] over the reals or [[Unitary 矩阵|unitary matrix]] over the complex numbers, and the $r\times r$ upper-triangular block $R_{11}$ is invertible. The block $R_{12}$ has $n-r$ columns. The lower zero block has $m-r$ rows and is absent if $r=m$. Write $^*$ for conjugate transpose, replacing $a+bi$ by $a-bi$ as well as transposing, where $i^2=-1$; over the reals it is $^T$.
<!-- bilingual-en:end -->

先讨论精确情形：$A$ 是[[矩阵秩|秩]]为 $r$ 的 $m\times n$ 实或复矩阵，即具有 $r$ 个独立列方向，$1\le r<n$。假设列置换与 QR 分解得到上式。$P$ 是重排列的[[置换矩阵]]；$Q$ 是 $m\times m$ 的实[[正交矩阵]]或复[[Unitary 矩阵|酉矩阵]]；$R_{11}$ 是可逆的 $r\times r$ 上三角块，$R_{12}$ 有 $n-r$ 列。下方零块有 $m-r$ 行，$r=m$ 时没有这些行。$^*$ 表示共轭转置，即转置并把 $a+bi$ 换成 $a-bi$，其中 $i^2=-1$；实数时就是 $^T$。

<!-- bilingual-en:start -->
Use reordered coordinates $z=P^*x$, so $x=Pz$, and transformed data $c=Q^*b$. Split $z=(z_1,z_2)$ after its first $r$ entries and $c=(c_1,c_2)$ after its first $r$ entries. Orthogonal/unitary transformations preserve length, so

$$\|Ax-b\|_2^2=\|R_{11}z_1+R_{12}z_2-c_1\|_2^2+\|c_2\|_2^2.$$

The last term cannot be changed by any coefficient. For any choice of $z_2$, solve $R_{11}z_1=c_1-R_{12}z_2$ to make the first term zero. **Back substitution** solves this triangular system from its last equation upward. Setting $z_2=0$ produces the basic solution, but the displayed objective shows why many other choices give the same minimum residual. The [[列置换后的解向量|permutation must be undone]] to report $x=Pz$; a permutation itself preserves the Euclidean norm.
<!-- bilingual-en:end -->

用重排坐标 $z=P^*x$，于是 $x=Pz$；再令 $c=Q^*b$。在各自前 $r$ 项后把它们分为 $z=(z_1,z_2)$、$c=(c_1,c_2)$。正交或酉变换保持长度，因此残差平方分解为上式。最后的 $\|c_2\|_2^2$ 无法通过任何系数改变；任取 $z_2$，都可解 $R_{11}z_1=c_1-R_{12}z_2$ 使第一项为零。**回代**就是从最后一条方程向上求解这个上三角系统。令 $z_2=0$ 选出 basic solution（基本解），但目标式表明其他选择也能给出同样小的残差。最后还须[[列置换后的解向量|还原置换]]得到 $x=Pz$；置换本身保持欧氏范数。

<!-- bilingual-en:start -->
The same distinction applies when the best residual is nonzero. For $A=\begin{bmatrix}1&1\\0&0\end{bmatrix}$ and $b=(1,2)^T$, the second observation cannot be fitted. Both $(1,0)^T$ and $(1/2,1/2)^T$ attain residual norm $2$, but only the latter has minimum coefficient norm.
<!-- bilingual-en:end -->

最佳残差非零时也一样。若 $A=\begin{bmatrix}1&1\\0&0\end{bmatrix}$、$b=(1,2)^T$，第二个观测无法拟合。$(1,0)^T$ 和 $(1/2,1/2)^T$ 都达到残差范数 $2$，但只有后者的系数范数最小。

<!-- bilingual-en:start -->
## Minimum norm removes a null-space component
<!-- bilingual-en:end -->

*最短解需要额外的几何条件*

<!-- bilingual-en:start -->
All least-squares solutions differ by an element of the [[零空间|null space]] $N(A)=\{v:Av=0\}$. The [[最小范数最小二乘解|unique minimum-norm solution]] is perpendicular to this space: otherwise one can remove a nonzero null-space component, leave the fit unchanged and shorten the vector. Setting selected coordinates to zero is not generally the same as imposing this perpendicularity. A basic solution can coincide with the minimum-norm solution for particular data, and if $R_{12}=0$ it does so for every right-hand side. The point is that there is no general guarantee from pivot-only back substitution.

A **complete orthogonal factorization** continues with right-side orthogonal/unitary transformations, rotating the variable space so that the free coordinates truly represent null-space directions. Setting those transformed coordinates to zero then enforces minimum norm before transforming back. LAPACK `xGELSY` performs this extra stage; SVD-based solvers or the [[Moore–Penrose 伪逆|Moore–Penrose pseudoinverse]] enforce the same minimum-norm geometry by another route.
<!-- bilingual-en:end -->

所有最小二乘解之差都属于[[零空间]] $N(A)=\{v:Av=0\}$。[[最小范数最小二乘解|唯一最小范数解]]必须垂直于该空间；否则去掉非零的零空间分量，就能在拟合不变时缩短向量。把选定坐标置零，通常不等于施加这种垂直条件。特定数据下基本解可能恰好最短，$R_{12}=0$ 时对所有右端也都最短；问题在于仅做主元回代没有一般保证。

**完整正交分解**继续从右侧施加正交或酉变换，旋转变量空间，使自由坐标真正对应零空间方向。在这些变换后的坐标中置零，再变换回来，才会满足最小范数条件。LAPACK 的 `xGELSY` 包含这个额外阶段；基于 SVD 的求解器或 [[Moore–Penrose 伪逆]]以另一条路线实现相同几何目标。

<!-- bilingual-en:start -->
## Choosing a rank is a separate decision
<!-- bilingual-en:end -->

## 与秩揭示的区别

<!-- bilingual-en:start -->
The derivation above assumes the exact rank and an invertible leading block. In software, CPQR may leave a small nonzero block $R_{22}$ that a tolerance treats as zero. The subsequent answer then refers to the chosen truncated rank model; it need not be the exact minimizer for the original untruncated matrix. The [[CPQR秩揭示边界|strength of ordinary CPQR as rank evidence]] is a separate question: column pivoting alone does not supply every guarantee of a strong rank-revealing method.

If $r=n$, the least-squares coefficient is unique, so there is no extra minimum-norm choice. If $A=0$, every coefficient has the same residual, and the unique minimum-norm choice is zero; no triangular inverse is required. State the rank model, norm and requested solution convention before choosing a [[秩亏最小二乘算法|solver]].
<!-- bilingual-en:end -->

上述推导假定精确秩已知，且领先块可逆。软件中的 CPQR 可能留下小而非零的 $R_{22}$，再按容差把它当作零；后续答案针对选定的截断秩模型，未必是原未截断矩阵的精确最优解。[[CPQR秩揭示边界|普通 CPQR 能提供多强的秩证据]]是另一问题，列主元化本身并不提供强秩揭示方法的全部保证。

若 $r=n$，最小二乘系数唯一，不再有额外最小范数选择；若 $A=0$，所有系数残差相同，唯一最小范数选择为零，无须求三角块的逆。选择[[秩亏最小二乘算法|求解器]]前，应说明采用的秩模型、范数及所需选解规则。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK DGELSY source and purpose](https://www.netlib.org/lapack/explore-html/d6/d4b/dgelsy_8f_source.html) and [complex ZGELSY source](https://netlib.org/lapack/lapack_routine/zgelsy.f): rank choice, right-side transformations, complete orthogonal factorization and mapping back to original coordinates. The real source explicitly calls `DTZRZF` and `DORMRZ` after pivoted QR.
- [LAPACK Users' Guide, Linear Least Squares Problems](https://netlib.org/lapack/lug/node27.html): residual minimization and the additional minimum-norm choice, including underdetermined problems. The block decomposition and both small counterexamples are direct calculations.
<!-- bilingual-en:end -->

- LAPACK DGELSY 与复数 ZGELSY 的原代码及 Purpose 段核对秩选择、右侧变换、完整正交分解和坐标还原；实数实现明确在列主元 QR 后调用 `DTZRZF`、`DORMRZ`。
- LAPACK 最小二乘章节核对残差最小化与额外最小范数选择，包括欠定问题；分块分解及两个小反例由正文直接计算。
