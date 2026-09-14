---
aliases:
  - "元素非负且每行或每列之和为一的方阵称为行随机或列随机 Markov 矩阵"
  - Markov matrix
  - Stochastic matrix
student_os: knowledge-atom
atom_id: LA-EIG-039
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[Markov矩阵左右约定]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[Markov矩阵谱边界]]"
  - "[[DTMC时间齐次转移核]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# 元素非负且每行或每列之和为一的方阵称为行随机或列随机 Markov 矩阵
<!-- bilingual-en:start -->
*A nonnegative square matrix whose rows or columns sum to one is called a row- or column-stochastic Markov matrix*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对非负方阵 $P=(p_{ij})$：
> - 若每一行满足 $\sum_jp_{ij}=1$，则 $P$ 是**行随机矩阵**；
> - 若每一列满足 $\sum_ip_{ij}=1$，则 $P$ 是**列随机矩阵**。
>
> 两者都常称为 Markov matrix 或 stochastic matrix；使用时必须说明采用哪一种约定。
> <!-- bilingual-en:start -->
> A row-stochastic matrix has nonnegative entries and row sums equal to one. A column-stochastic matrix has nonnegative entries and column sums equal to one.
> <!-- bilingual-en:end -->

非负性让每个元素能够解释为转移概率，和为一保证从一个当前状态出发的全部下一步概率加总为一。行随机矩阵自然作用在行概率向量右侧；列随机矩阵是它的转置版本，自然作用在列概率向量左侧。具体的左右特征向量位置见[[Markov矩阵左右约定]]。

例如
$$
P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}
$$
每列之和为一且元素非负，所以它是列随机矩阵。它的两行之和分别为 $1.1$ 和 $0.9$，因此不是行随机矩阵。一个矩阵也可能同时满足两种约定，例如双随机矩阵。
<!-- bilingual-en:start -->
The displayed matrix is column-stochastic but not row-stochastic. A matrix may satisfy both conventions, in which case it is doubly stochastic.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 一个方阵每列之和为一，但含有负元素。它是 Markov 矩阵吗？
>
> **答案：** 不是。随机矩阵同时要求元素非负和相应方向的和为一。

## 来源与核验

- [Nick Higham, What Is a Stochastic Matrix?](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/)：核对行随机、列随机和双随机矩阵的定义。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]]：核对课程采用的列随机矩阵定义。
