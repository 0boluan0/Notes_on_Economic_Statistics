---
aliases:
  - "Jordan 标准形由各特征值的 Jordan 块尺寸唯一决定，至多相差块的排列"
  - Uniqueness of Jordan form
  - Jordan 形唯一性
student_os: knowledge-atom
atom_id: LA-EIG-034
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan标准形]]"
  - "[[相似矩阵]]"
  - "[[Jordan标准形的域条件]]"
related:
  - "[[Jordan链]]"
  - "[[相似保持特征空间维数]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 标准形由各特征值的 Jordan 块尺寸唯一决定，至多相差块的排列
<!-- bilingual-en:start -->
*The Jordan block sizes for each eigenvalue are unique up to reordering of the blocks*
<!-- bilingual-en:end -->

> [!summary] 唯一性定理
> 在特征多项式已经于底层域上分裂的前提下，相似矩阵具有完全相同的 Jordan 块资料：对每个特征值 $\lambda$，各块大小及其重数都唯一，唯一允许的变化只是交换块的排列顺序。
> <!-- bilingual-en:start -->
> Jordan form is canonical up to permutation of its blocks.
> <!-- bilingual-en:end -->

为什么块尺寸能被原矩阵本身确定？若 $B=M^{-1}AM$，则对每个 $j\ge1$，
$$
(B-\lambda I)^j=M^{-1}(A-\lambda I)^jM.
$$
所以 $M^{-1}$ 在两个核之间给出线性同构，以下各数都是相似不变量：
$$
d_j=\dim\ker\bigl((A-\lambda I)^j\bigr),\qquad d_0=0.
$$

对一个大小为 $s$ 的 Jordan 块，$\ker((J_s(\lambda)-\lambda I)^j)$ 的维数是 $\min(j,s)$。因此
$$
b_j:=d_j-d_{j-1}
$$
恰好等于大小至少为 $j$ 的 Jordan 块个数，而 $b_j-b_{j+1}$ 等于大小恰为 $j$ 的块个数。知道全部 $d_j$，就能逐层恢复每种块大小；换一组 Jordan 链不会改变最终的块尺寸资料。

这也说明“有多少个独立特征向量”仍可能不够。两个四阶幂零矩阵都可以有几何重数 $2$，却分别具有块尺寸 $3+1$ 与 $2+2$：前者的 $\dim\ker A^2=3$，后者则为 $4$，所以它们不相似。
<!-- bilingual-en:start -->
Successive nullities are similarity invariants. Their first differences count blocks of at least a given size, so their second differences recover the exact block sizes.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 两个矩阵拥有相同的特征值、代数重数和几何重数，能否据此断定它们相似？
>
> **答案：** 仍不能。较长 Jordan 链的分组可能不同，还要比较 $\dim\ker\bigl((A-\lambda I)^j\bigr)$ 等能恢复块尺寸的资料。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对不同块尺寸给出不同相似类。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S04_Recitation_Problem_Solving_Similar_Matrices.pdf|MIT Session 3.4 recitation]]：核对相似性判断与 Jordan 块资料。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.4.5 课件与 Recitation 例子|课程 3.4.5]]：核对连续零空间维数恢复块尺寸的思路。
