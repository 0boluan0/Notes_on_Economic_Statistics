---
aliases:
  - Gauss-Jordan inversion
  - Inverting a matrix by row reduction
student_os: knowledge-atom
atom_id: LA-SYS-041
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
requires:
  - "[[逆矩阵]]"
  - "[[高斯消元]]"
  - "[[初等矩阵]]"
  - "[[行最简形]]"
related:
  - "[[方阵单侧逆]]"
  - "[[主元与自由变量]]"
  - "[[可逆性与非零行列式]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# Gauss–Jordan inversion
<!-- bilingual-en:end -->

*Gauss–Jordan 求逆*

<!-- bilingual-en:start -->
Gauss–Jordan inversion finds a matrix that undoes a given square matrix. Attach an [[单位矩阵|identity matrix]] to its right, then perform the same row operations on both blocks. The left block is simplified to the identity; the right block records the operations that undo the original matrix. If the left block cannot become the identity, an ordinary [[逆矩阵|inverse]] does not exist.
<!-- bilingual-en:end -->

Gauss–Jordan 求逆用来寻找能撤销给定方阵作用的矩阵。在方阵右侧接上[[单位矩阵]]，再对左右两块执行完全相同的行操作。左块被化成单位矩阵时，右块就记录了撤销原矩阵的全部操作。若左块无法化成单位矩阵，普通[[逆矩阵]]就不存在。

<!-- bilingual-en:start -->
## Carry a record while reducing
<!-- bilingual-en:end -->

*一边化简，一边留下记录*

<!-- bilingual-en:start -->
Let $A$ be an $n\times n$ matrix over $\mathbb R$ or $\mathbb C$, with $n$ a positive integer. The identity $I_n$ has diagonal entries $1$ and other entries $0$. The combined matrix $[A\mid I_n]$ has $n$ rows and $2n$ columns; the bar separates the original matrix from the record.
<!-- bilingual-en:end -->

设 $A$ 是实数域或复数域上的 $n\times n$ 方阵，正整数 $n$ 为矩阵阶数。单位矩阵 $I_n$ 的对角元为 $1$，其余为零。拼接矩阵 $[A\mid I_n]$ 有 $n$ 行、$2n$ 列，竖线区分原矩阵和操作记录。

<!-- bilingual-en:start -->
Use [[高斯消元|Gaussian elimination]] to build pivots in the left block, swapping rows if needed. Apply each row change across all $2n$ entries. Normalize each nonzero pivot to $1$ and clear both below and above it, aiming for the [[行最简形|reduced row echelon form (RREF)]] of the left block. Pivots found in the right block cannot replace missing pivots in $A$.
<!-- bilingual-en:end -->

用[[高斯消元]]在左块建立主元，必要时换行。每次行操作都作用于整行 $2n$ 个元素。把非零主元归一为 $1$，并消清其下方和上方元素，目标是左块的[[行最简形|行最简形（RREF）]]。右块中的主元不能补充 $A$ 所缺少的主元。

> [!important] Read the result only when the left block is ready · 左块完成后再读取结果
> <!-- bilingual-en:start -->
> If the reduction reaches
>
> $$
> [A\mid I_n]\longrightarrow[I_n\mid B],
> $$
>
> then $B=A^{-1}$. If the completed left-block RREF is not $I_n$, then $A$ is singular. A single zero pivot candidate is not a failure certificate: a row exchange may supply a nonzero pivot.
> <!-- bilingual-en:end -->
>
> 若化简得到上述形式，则 $B=A^{-1}$。若左块完整的 RREF 不是 $I_n$，则 $A$ 奇异。仅看到某个候选主元为零还不能判定失败：换行可能找到可用的非零主元。

<!-- bilingual-en:start -->
## A complete two-by-two calculation
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
Keep the real matrix $A=\begin{pmatrix}1&2\\3&5\end{pmatrix}$. With $R_i$ denoting row $i$, first replace $R_2$ by $R_2-3R_1$:
<!-- bilingual-en:end -->

取实矩阵 $A=\begin{pmatrix}1&2\\3&5\end{pmatrix}$，用 $R_i$ 表示第 $i$ 行。先用 $R_2-3R_1$ 替换第二行：

$$
\left[\begin{array}{cc|cc}1&2&1&0\\3&5&0&1\end{array}\right]
\longrightarrow
\left[\begin{array}{cc|cc}1&2&1&0\\0&-1&-3&1\end{array}\right].
$$

<!-- bilingual-en:start -->
Next multiply row $2$ by $-1$, then subtract twice this new row $2$ from row $1$:
<!-- bilingual-en:end -->

再把第二行乘 $-1$，最后用第一行减去这个新第二行的两倍：

$$
\left[\begin{array}{cc|cc}1&2&1&0\\0&1&3&-1\end{array}\right]
\longrightarrow
\left[\begin{array}{cc|cc}1&0&-5&2\\0&1&3&-1\end{array}\right].
$$

<!-- bilingual-en:start -->
The left block is now $I_2$, so the right block is the inverse. Check both orders by ordinary matrix multiplication:
<!-- bilingual-en:end -->

左块已是 $I_2$，因此右块就是逆矩阵。直接按两个次序相乘核对：

$$
B=\begin{pmatrix}-5&2\\3&-1\end{pmatrix},
\qquad
AB=\begin{pmatrix}-5+6&2-2\\-15+15&6-5\end{pmatrix}=I_2,
\qquad
BA=\begin{pmatrix}-5+6&-10+10\\3-3&6-5\end{pmatrix}=I_2.
$$

<!-- bilingual-en:start -->
## Why the record is the inverse
<!-- bilingual-en:end -->

*为什么操作记录正好是逆矩阵*

<!-- bilingual-en:start -->
Each [[初等矩阵|elementary matrix]] records one reversible row operation. If their product is $E$, acting on the original two blocks gives $[EA\mid EI_n]=[EA\mid E]$. Therefore reaching $[I_n\mid B]$ gives $EA=I_n$ and $B=E$. Since $E$ is a product of invertible matrices, it is invertible. Multiplying $EA=I_n$ by $E^{-1}$ gives $A=E^{-1}$, hence $AE=I_n$ as well. Thus $B$ satisfies both inverse identities, without exchanging the order of arbitrary factors.
<!-- bilingual-en:end -->

每个[[初等矩阵]]记录一次可逆行操作。设它们的乘积为 $E$，作用在原来的两块上得到 $[EA\mid EI_n]=[EA\mid E]$。因此，达到 $[I_n\mid B]$ 就说明 $EA=I_n$ 且 $B=E$。$E$ 是可逆矩阵的乘积，所以它可逆；对 $EA=I_n$ 左乘 $E^{-1}$ 得 $A=E^{-1}$，因而 $AE=I_n$ 也成立。这就证明 $B$ 满足双侧逆恒等式，全程没有随意交换因子次序。

<!-- bilingual-en:start -->
There is also an equation-solving view. Write $B=[b_1\ \cdots\ b_n]$ and $I_n=[e_1\ \cdots\ e_n]$, where $e_j$ has a single $1$ in coordinate $j$. The equation $AB=I_n$ asks for all $n$ systems $Ab_j=e_j$ at once. Synchronized [[行操作的解集不变性|solution-preserving row operations]] solve these right-hand sides together. This explains why the identity is attached: its columns request every coordinate direction.
<!-- bilingual-en:end -->

也可以从解方程理解。写 $B=[b_1\ \cdots\ b_n]$、$I_n=[e_1\ \cdots\ e_n]$，其中 $e_j$ 只有第 $j$ 个坐标为 $1$。$AB=I_n$ 同时要求解出 $n$ 组方程 $Ab_j=e_j$。同步[[行操作的解集不变性|保持解集的行操作]]可以一起处理这些右端。接上单位矩阵，就是一次要求恢复所有坐标方向。

<!-- bilingual-en:start -->
## How failure is recognized
<!-- bilingual-en:end -->

## 执行边界

<!-- bilingual-en:start -->
Write $R$ for the completed left-block RREF. For an $n\times n$ left block, $R\ne I_n$ means it has fewer than $n$ pivots, hence a [[自由变量|free variable]] in its homogeneous system. Choose a nonzero free coordinate and solve for the pivot coordinates to obtain a nonzero vector $z$ with $Rz=0$. Because $R=EA$ and $E$ is invertible, $Az=0$ too. If an inverse existed, $z=A^{-1}Az=0$, a contradiction. This proves the failure criterion rather than treating an unfinished calculation as evidence.
<!-- bilingual-en:end -->

记左块完整的 RREF 为 $R$。对 $n\times n$ 左块，若 $R\ne I_n$，就少于 $n$ 个主元，所以齐次方程中有[[自由变量]]。令某个自由坐标非零，再求出主元坐标，便得到非零向量 $z$ 满足 $Rz=0$。由于 $R=EA$ 且 $E$ 可逆，也有 $Az=0$。若存在逆，就会推出 $z=A^{-1}Az=0$，矛盾。这才证明了失败判据，未完成的计算本身不算证据。

> [!example] A missing left pivot · 左块缺主元
> <!-- bilingual-en:start -->
> For $K=\begin{pmatrix}1&2\\2&4\end{pmatrix}$, subtracting twice row $1$ from row $2$ gives
>
> $$
> [K\mid I_2]\longrightarrow\left[\begin{array}{cc|cc}1&2&1&0\\0&0&-2&1\end{array}\right].
> $$
>
> The left block has a zero row. For instance, $z=(-2,1)^T$ is nonzero and $Kz=0$, where $T$ denotes [[矩阵转置|transpose]]. The nonzero right record does not repair the missing left pivot.
> <!-- bilingual-en:end -->
>
> 对上述 $K$，第二行减去第一行的两倍后，左块出现零行。例如 $z=(-2,1)^T\ne0$ 且 $Kz=0$，其中 $T$ 表示[[矩阵转置|转置]]。右侧非零的操作记录无法补回左块缺少的主元。

<!-- bilingual-en:start -->
The method computes the ordinary inverse of a square matrix. A rectangular matrix may instead have a [[左逆|left inverse]] or [[右逆|right inverse]], which requires a different size-specific question. In numerical work, if the goal is only to find $x$ for a given right side $b$ in $Ax=b$, a suitable factorization such as [[消元与LU分解|LU]] or [[QR分解|QR]] can be used directly; explicitly forming $A^{-1}$ is usually unnecessary. Exact-algebra success does not by itself certify floating-point accuracy.
<!-- bilingual-en:end -->

本方法计算方阵的普通逆矩阵。矩形矩阵可能有[[左逆]]或[[右逆]]，应另行说明具体尺寸与目标。数值计算中，若只是给定右端 $b$ 后求 $Ax=b$ 中的 $x$，可以直接使用合适的分解，例如[[消元与LU分解|LU]]或[[QR分解|QR]]，通常不必显式形成 $A^{-1}$。精确代数上的成功，也不自动保证浮点计算的准确度。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|MIT 18.06SC Session 1.3 summary]], pp. 2–3: simultaneous identity right-hand sides and the accumulated elementary-matrix proof.
- [Interactive Linear Algebra, §3.5.2](https://textbooks.math.gatech.edu/ila/matrix-inverses.html): the augmented inverse algorithm and the nonzero-null-vector failure criterion. The retained numerical example, intermediate operations and two-sided check are independently calculated.
- [LAPACK Users’ Guide, Linear Equations](https://www.netlib.org/lapack/lug/node26.html): factorization-based solution routines and the recommendation to avoid forming an explicit inverse merely to solve a system.
- [[01_Ax = b and the Four Subspaces#Session 1.4 Multiplication and inverse matrices|Course Session 1.4]]: retains the classroom context of Gauss–Jordan inversion.
<!-- bilingual-en:end -->

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|MIT 18.06SC Session 1.3 摘要]]，第 2–3 页：核对同时处理单位矩阵各右端与累积初等矩阵证明。
- [Interactive Linear Algebra，§3.5.2](https://textbooks.math.gatech.edu/ila/matrix-inverses.html)：核对增广求逆算法和非零零向量失败判据；保留的数值例、中间行操作与双侧乘积另行精确核算。
- [LAPACK Users’ Guide，Linear Equations](https://www.netlib.org/lapack/lug/node26.html)：核对通过分解求解以及无需仅为解方程而显式求逆的建议。
- [[01_Ax = b and the Four Subspaces#Session 1.4 Multiplication and inverse matrices|课程 Session 1.4]]：保留 Gauss–Jordan 求逆的课堂位置。
