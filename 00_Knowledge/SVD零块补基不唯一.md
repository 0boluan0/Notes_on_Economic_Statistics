---
aliases:
- 零奇异值补基不唯一
- Nullspace basis freedom in a full SVD
student_os: knowledge-atom
atom_id: LA-SVD-010
atom_set: singular-value-decomposition-low-rank
atom_type: invariance-boundary
status: source-checked
requires:
- '[[奇异向量]]'
- '[[SVD与四个基本子空间]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
related:
- '[[紧SVD]]'
- '[[奇异向量共同相位]]'
- '[[重奇异值下方向不唯一]]'
---

<!-- bilingual-en:start -->
# Independent nullspace bases in a full SVD
<!-- bilingual-en:end -->

*完整 SVD 的零空间独立补基*

<!-- bilingual-en:start -->
A zero stretch erases an input direction completely. Once the output is zero, it does not indicate which output basis vector should be paired with that input. This is why a [[奇异值分解|full SVD]] can choose its right and left zero-space bases independently. For a rectangular matrix, the two zero spaces can even have different dimensions.
<!-- bilingual-en:end -->

零伸缩会把一个输入方向完全消去。输出既然为零，就没有指明该把输入与哪条输出基向量配对。因此，[[奇异值分解|完整 SVD]] 的左右零空间基可以独立选择；长方矩阵两侧的零空间维数甚至可以不同。

> [!example] An input direction with no output partner / 有输入零方向，却没有输出搭档
> <!-- bilingual-en:start -->
> The row matrix $A=[1\ 0]$ takes $(x_1,x_2)^T$ to $x_1$. Its right [[零空间|nullspace]] is the line spanned by $(0,1)^T$, because $Ax=0$ exactly when $x_1=0$. The star is [[矩阵转置|conjugate transpose]], ordinary transpose for this real matrix. Its left nullspace $N(A^*)$ contains only zero: $A^*y=(y,0)^T=0$ forces $y=0$. A full SVD has two right basis vectors and just one left basis vector. The extra right vector has no extra left vector to match.
> <!-- bilingual-en:end -->
>
> 行矩阵 $A=[1\ 0]$ 把 $(x_1,x_2)^T$ 变为 $x_1$。这里星号表示[[矩阵转置|共轭转置]]，本实矩阵中就是普通转置。其右[[零空间]]是 $(0,1)^T$ 张成的直线；左零空间 $N(A^*)$ 却只有零，因为 $A^*y=(y,0)^T=0$ 迫使 $y=0$。完整 SVD 有两条右基向量，但只有一条左基向量，多出的右方向没有额外左方向可配对。

<!-- bilingual-en:start -->
## Which columns complete each basis?
<!-- bilingual-en:end -->

*哪些列补齐哪一侧的基*

<!-- bilingual-en:start -->
Let $A$ be a real or complex $m\times n$ matrix, $m,n\ge1$, of [[矩阵秩|rank]] $r$. In $A=U\Sigma V^*$, the full factors $U$ and $V$ have sizes $m\times m$ and $n\times n$. The star is [[矩阵转置|conjugate transpose]]. The first $r$ columns carry all positive singular values. Write the remaining columns as
<!-- bilingual-en:end -->

设 $A$ 为实或复 $m\times n$ 矩阵，$m,n\ge1$，[[矩阵秩|秩]]为 $r$。在 $A=U\Sigma V^*$ 中，完整因子 $U$、$V$ 的尺寸为 $m\times m$、$n\times n$；星号表示[[矩阵转置|共轭转置]]。前 $r$ 列承载全部正奇异值，将其余列记作

$$
V_0=[v_{r+1}\ \cdots\ v_n],\qquad U_0=[u_{r+1}\ \cdots\ u_m].
$$

<!-- bilingual-en:start -->
The columns of $V_0$ form an [[标准正交组|orthonormal basis]] of $N(A)=\{x:Ax=0\}$, so there are $n-r$ of them. The columns of $U_0$ form an orthonormal basis of $N(A^*)=\{y:A^*y=0\}$, so there are $m-r$ of them. These are input directions erased by $A$ and output directions perpendicular to its [[列空间|attainable outputs]], respectively; see [[SVD与四个基本子空间|the four-subspace interpretation]]. If a dimension is zero, that list of columns is empty.
<!-- bilingual-en:end -->

$V_0$ 的列构成 $N(A)=\{x:Ax=0\}$ 的[[标准正交组|标准正交基]]，共有 $n-r$ 条；$U_0$ 的列构成 $N(A^*)=\{y:A^*y=0\}$ 的标准正交基，共有 $m-r$ 条。前者是被 $A$ 消去的输入方向，后者是垂直于[[列空间|可达输出]]的输出方向，见 [[SVD与四个基本子空间]]。维数为零时，列的列表为空。

<!-- bilingual-en:start -->
For a positive [[奇异值|singular value]], the paired equation $Av_i=\sigma_i u_i$ determines $u_i=Av_i/\sigma_i$. At zero, $Av_i=0$ tells us only that the input belongs to $N(A)$; it supplies no nonzero output direction, and division by $\sigma_i$ is undefined. Independently, the left condition is $A^*u_j=0$.
<!-- bilingual-en:end -->

正[[奇异值]]的配对方程 $Av_i=\sigma_i u_i$ 可以给出 $u_i=Av_i/\sigma_i$。到零值时，$Av_i=0$ 只告诉我们输入属于 $N(A)$，没有提供非零输出方向，也不能再除以 $\sigma_i$；左侧则独立满足 $A^*u_j=0$。

<!-- bilingual-en:start -->
## Independent changes leave the product fixed
<!-- bilingual-en:end -->

*左右分别改基，乘积仍不变*

<!-- bilingual-en:start -->
Keep the positive columns $U_r,V_r$ fixed. Choose a $(m-r)\times(m-r)$ [[Unitary 矩阵|unitary matrix]] $Q_L$ for the left completion and a separate $(n-r)\times(n-r)$ unitary matrix $Q_R$ for the right completion. For real factors use [[正交矩阵|orthogonal matrices]]. Then
<!-- bilingual-en:end -->

固定正方向列 $U_r,V_r$，在左补基处任选 $(m-r)\times(m-r)$ 的 [[Unitary 矩阵|酉矩阵]] $Q_L$，右补基处另选 $(n-r)\times(n-r)$ 的酉矩阵 $Q_R$；实数因子使用[[正交矩阵]]。于是

$$
\widetilde U=[U_r\ U_0Q_L],\qquad \widetilde V=[V_r\ V_0Q_R],\qquad \widetilde U\Sigma\widetilde V^*=U\Sigma V^*=A.
$$

<!-- bilingual-en:start -->
Both new bases remain orthonormal, but no relation between $Q_L$ and $Q_R$ is required. Every term involving a zero-space column is multiplied by zero in $\Sigma$. Empty blocks require no choice. This differs from a [[重奇异值下方向不唯一|repeated positive block]], where the two changes must be the same to preserve the nonzero pairing.
<!-- bilingual-en:end -->

两组新基仍标准正交，但 $Q_L$ 与 $Q_R$ 无须有任何联系：$\Sigma$ 中涉及零空间列的项都乘零。空块不需要作任何选择。这与[[重奇异值下方向不唯一|重复正值块]]不同，后者为了保留非零配对，左右必须采用同一个变换。

> [!example] Only one zero column changes sign / 只改一侧零列的符号
> <!-- bilingual-en:start -->
> For the [[对角矩阵|diagonal matrix]] $A=\operatorname{diag}(2,0)$, take $\Sigma=A$ and initially $U=V=I_2$, the [[单位矩阵|identity]]. Replacing only $U$ by $\operatorname{diag}(1,-1)$ still gives $U\Sigma V^*=A$. The second output basis vector has changed sign while the second input vector has not. This is valid because that whole contribution is zero; making the same unpaired change in the first column would change the matrix.
> <!-- bilingual-en:end -->
>
> 对[[对角矩阵]] $A=\operatorname{diag}(2,0)$，先取 $\Sigma=A$、$U=V=I_2$，其中 $I_2$ 是[[单位矩阵]]。只把 $U$ 换为 $\operatorname{diag}(1,-1)$，仍有 $U\Sigma V^*=A$：第二输出基变号，第二输入基没变，分解仍正确，因为整个分量为零。若对第一列作同样的单侧改动，就会改变矩阵。

<!-- bilingual-en:start -->
## What compact output omits
<!-- bilingual-en:end -->

*紧分解省略了什么*

<!-- bilingual-en:start -->
A [[紧SVD|compact SVD]] keeps only the $r$ positive directions and omits both completions. The nullspaces still exist; the compact factors simply do not list their bases. For example, $A=[1\ 0]$ has one positive singular value and no zero on its rectangular diagonal, yet still has a one-dimensional right nullspace. Thus “no listed zero singular value” does not by itself mean that a rectangular matrix has no nullspace. At $A=0$, $r=0$ and every orthonormal input and output basis is allowed independently.
<!-- bilingual-en:end -->

[[紧SVD]]只保留 $r$ 个正方向，省去两侧补基。零空间仍然存在，只是紧因子没有列出其基。例如 $A=[1\ 0]$ 只有一个正奇异值，矩形对角线上没有零，却仍有一维右零空间。因此，长方矩阵“列出的奇异值没有零”本身不代表没有零空间。若 $A=0$，则 $r=0$，任意输入与输出标准正交基都可以独立选取。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 7.64 and 7.70–7.74, PDF pp.284, 287–288](https://linear.axler.net/LADR4e.pdf#page=284): nullspaces of the Gram map and extension of the positive singular lists to full input/output bases. Independent zero-block transformations are checked directly above.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT18.06SC Session3.5, final page]] and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S05_Lecture_Lecture_29_Singular_Value_Decomposition.pdf|Lecture29 transcript, final rank-one example and four-subspace discussion]]: real nullspace completions and dimensions. The wide, zero and independent-sign examples above are direct calculations.
<!-- bilingual-en:end -->

- Axler 第7.64及7.70–7.74条核对 Gram 映射零空间以及正方向向完整输入输出基的扩充；左右零块独立变换在正文直接验证。
- MIT Session3.5末页和 Lecture29 末尾秩一例子／四子空间讨论核对实数补基与维数；宽矩阵、零矩阵及单侧零列变号例子直接计算。
