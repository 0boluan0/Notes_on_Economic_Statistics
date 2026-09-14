---
aliases:
  - 若增广矩阵 [A|I] 经同步行操作化为 [I|B] 则 B 等于 A 的逆矩阵
  - Gauss-Jordan inversion
  - Inverting a matrix by row reduction
student_os: knowledge-atom
atom_id: LA-SYS-041
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
mastery_state: unassessed
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

# 若增广矩阵 [A|I] 经同步行操作化为 [I|B] 则 B 等于 A 的逆矩阵
<!-- bilingual-en:start -->
*If synchronized row operations reduce $[A\mid I]$ to $[I\mid B]$, then $B=A^{-1}$*
<!-- bilingual-en:end -->

> [!summary] 方法
> 对 $n\times n$ 方阵 $A$，把单位矩阵接在右侧，并对整个增广矩阵同步做行操作：
> $$
> [A\mid I_n]\longrightarrow[I_n\mid B].
> $$
> 若左侧能够化为 $I_n$，则右侧的 $B$ 就是 $A^{-1}$；若左侧的 RREF 不是 $I_n$，则 $A$ 不可逆。
> <!-- bilingual-en:start -->
> Row-reduce the entire augmented matrix $[A\mid I_n]$. If the left block becomes $I_n$, the right block is $A^{-1}$; otherwise $A$ is not invertible.
> <!-- bilingual-en:end -->

设全部行操作合起来等于左乘可逆矩阵 $E$。同步操作把增广矩阵变为
$$
[EA\mid E].
$$
若左块成为 $I$，便有 $EA=I$，同时右块就是 $E=B$。由于 $E$ 是初等矩阵的乘积，所以它可逆；由 $EA=I$ 得 $A=E^{-1}$，进而 $AE=I$。因此 $B=E=A^{-1}$。这也解释了为什么必须对左右两块施行完全相同的行操作。

## 最小例子

$$
\left[\begin{array}{cc|cc}
1&2&1&0\\
3&5&0&1
\end{array}\right]
\longrightarrow
\left[\begin{array}{cc|cc}
1&0&-5&2\\
0&1&3&-1
\end{array}\right].
$$
因此
$$
\begin{bmatrix}1&2\\3&5\end{bmatrix}^{-1}
=\begin{bmatrix}-5&2\\3&-1\end{bmatrix}.
$$
把两矩阵相乘可验算结果确为 $I_2$。

## 执行边界

- 每一步必须同时作用于左右两块；只化简 $A$ 再另行猜测右块没有依据。
- 若左块缺少主元，RREF 不可能成为 $I_n$，算法在同一过程中同时证明逆不存在。
- 数值计算中，若目标只是解 $Ax=b$，通常直接用 LU、QR 等分解求解，不必显式形成 $A^{-1}$。

> [!question]- 自检
> 为什么左块化为 $I$ 时，右块恰好等于同一个消元矩阵乘积 $E$？
>
> **答案：** 因为右块起初是 $I$，同步行操作就是左乘 $E$，所以它变成 $EI=E$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.3sum.pdf|MIT 18.06SC Session 1.3 summary]]：核对 $[A\mid I]\to[I\mid A^{-1}]$ 的消元求逆方法。
- [[01_Math/02_linear algebra/01_Ax = b and the Four Subspaces.md#Session 1.4 Multiplication and inverse matrices|课程 Gauss–Jordan 部分]]：核对初等矩阵乘积解释与失败判据。
