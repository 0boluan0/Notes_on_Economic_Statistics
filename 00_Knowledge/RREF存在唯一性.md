---
aliases:
  - 每个矩阵都行等价于唯一一个行最简形
  - Existence and uniqueness of reduced row echelon form
  - Uniqueness of RREF
student_os: knowledge-atom
atom_id: LA-SYS-047
atom_set: linear-systems-four-subspaces
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[行最简形]]"
  - "[[行等价]]"
  - "[[高斯消元]]"
  - "[[行空间]]"
  - "[[维数]]"
related:
  - "[[主元]]"
leads_to:
  - "[[RREF读取子空间基]]"
  - "[[Gauss-Jordan求逆]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# 每个矩阵都行等价于唯一一个行最简形
<!-- bilingual-en:start -->
*Every matrix is row equivalent to exactly one reduced row echelon form*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 对域 $\mathbb F$ 上任意有限的 $m\times n$ 矩阵 $A$，都存在唯一的行最简形 $R$ 与 $A$ 行等价。唯一的是最终矩阵 $R$；把 $A$ 化到 $R$ 的行操作步骤通常不唯一。
> <!-- bilingual-en:start -->
> Every matrix is row equivalent to a unique reduced row echelon matrix. The final RREF is unique, although the sequence of row operations used to reach it need not be.
> <!-- bilingual-en:end -->

存在性来自 Gauss–Jordan 消元：先逐列建立主元并消去主元下方，再把主元归一为 $1$，最后消去主元上方，有限步后便得到[[行最简形]]。

## 为什么最终形式唯一

行等价矩阵有同一个行空间 $W$。令 $\pi_j$ 只保留向量的前 $j$ 个坐标，并记
$$
d_j=\dim \pi_j(W).
$$
若 RREF 的主元列为 $p_1<\cdots<p_r$，那么
$$
d_j=\#\{i:p_i\le j\}.
$$
因此，第 $j$ 列是主元列，当且仅当 $d_j-d_{j-1}=1$。每个 $d_j$ 都只由 $W$ 决定，所以主元列位置也由 $W$ 唯一确定。

再定义主元坐标限制
$$
P:W\to\mathbb F^r,
\qquad
P(w)=(w_{p_1},\ldots,w_{p_r}).
$$
RREF 的非零行构成 $W$ 的一组基，而且它们在主元列组成 $I_r$，所以 $P$ 把这组基映到标准基，是一个同构。于是第 $i$ 个非零行只能是唯一的 $P^{-1}(e_i)$。每个非零行都被唯一确定；相同的 $m\times n$ 尺寸和秩又决定尾部恰有 $m-r$ 个零行，所以两个与 $A$ 行等价的 RREF 只能完全相同。秩为零时，这个结论给出的正是唯一的零矩阵。

## 边界

普通行阶梯形不唯一：主元可以缩放，主元上方也可以保留不同元素。只有同时要求主元归一并清空整个主元列，才得到唯一的 RREF。这个唯一性保证“从 RREF 读取结构”不依赖你选择了哪条合法消元路线。

> [!question]- 自检
> 两个人采用不同的换行与倍加顺序，为什么仍应得到同一个 RREF？
>
> **答案：** 两串操作都停在与原矩阵行等价的 RREF；定理说明这个行等价类中只有一个 RREF。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.7sum.pdf|MIT 18.06SC Session 1.7 summary]]：核对 RREF 的构造、主元列与自由列。
- [MIT 18.700 Lesson 1](https://samschiavone.github.io/courses/18-700/Lectures/LessonPlan1.pdf)：陈述“每个矩阵行等价于唯一的行最简矩阵”，并区分唯一的 RREF 与不唯一的普通阶梯形。
