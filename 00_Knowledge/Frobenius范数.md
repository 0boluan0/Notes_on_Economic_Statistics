---
aliases:
  - "矩阵的 Frobenius 范数是全部元素模平方和的平方根"
  - Frobenius norm
  - 矩阵元素平方和范数
student_os: knowledge-atom
atom_id: LA-SVD-017
atom_set: singular-value-decomposition-low-rank
atom_type: definition
status: source-checked
mastery_state: unassessed
requires: []
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[Frobenius范数与奇异值]]"
  - "[[奇异值能量比]]"
  - "[[截断SVD最佳低秩近似]]"
related:
  - "[[谱范数]]"
---

# 矩阵的 Frobenius 范数是全部元素模平方和的平方根
<!-- bilingual-en:start -->
*The Frobenius norm of a matrix is the square root of the sum of the squared magnitudes of all its entries*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对 $A=(a_{jk})\in\mathbb F^{m\times n}$，Frobenius 范数定义为
> $$
> \|A\|_F=\left(\sum_{j=1}^{m}\sum_{k=1}^{n}|a_{jk}|^2\right)^{1/2}.
> $$
> 把矩阵的所有元素依次排成一个长向量，$\|A\|_F$ 就是这个向量的 Euclidean 长度。
> <!-- bilingual-en:start -->
> The Frobenius norm is the Euclidean length of the vector formed by stacking all matrix entries.
> <!-- bilingual-en:end -->

它汇总全部元素的平方大小，因此适合测量整体重构误差。它与谱范数回答不同问题：谱范数寻找最强输入方向，Frobenius 范数则把矩阵所有元素的偏差一起累计。

例如
$$
A=\begin{bmatrix}1&2\\-2&0\end{bmatrix}
$$
满足
$$
\|A\|_F=\sqrt{1^2+2^2+(-2)^2+0^2}=3.
$$
在 SVD 坐标中，同一个量也等于全部奇异值平方和的平方根，见 [[Frobenius范数与奇异值]]。

> [!question]- 自检
> 若一个矩阵只有三个非零元素 $1,-2,2$，它的 Frobenius 范数是多少？
>
> **答案：** $\sqrt{1^2+(-2)^2+2^2}=3$；元素位于矩阵的什么位置不影响这个计算。

## 来源与核验

- [Cornell CS 6241, SVD and low-rank approximation](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-02-13.html)：核对 Frobenius 范数的元素定义及其与 SVD 的关系。
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, 4th ed., §2.3：核对 Frobenius 范数的标准定义。
