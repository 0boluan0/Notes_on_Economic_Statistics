---
aliases:
  - "只在 CPQR 主元列上回代并把其余变量置零通常不保证原坐标中的最小范数解"
  - Pivoted QR back substitution need not give the minimum-norm solution
  - CPQR minimum-norm boundary
student_os: knowledge-atom
atom_id: LA-PROJ-029
atom_set: orthogonal-projection-least-squares
atom_type: numerical-boundary
status: source-checked
mastery_state: unassessed
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

# 只在 CPQR 主元列上回代并把其余变量置零通常不保证原坐标中的最小范数解
<!-- bilingual-en:start -->
*Back substitution on only the pivot columns of a column-pivoted QR factorisation, with the remaining variables set to zero, generally does not give the minimum-norm solution in the original coordinates*
<!-- bilingual-en:end -->

> [!summary] 数值边界
> 列主元 QR 能找出一组便于求解的独立列，但“在领先三角块上回代，再把非主元变量设为零”只选出一个最小残差解。这个坐标选择通常不与 Euclidean 最小范数条件对齐，因此不保证得到 $A^+b$。
> <!-- bilingual-en:start -->
> Column-pivoted QR can identify a convenient independent column set, but solving only for pivot variables and setting free variables to zero merely selects one minimum-residual solution. That coordinate choice need not be the Euclidean minimum-norm solution $A^+b$.
> <!-- bilingual-en:end -->

主元化通过置换把变量分成“先解的列”和“其余列”。将后一组变量机械置零是在当前坐标中选一个特解；最小范数解却要求从整个仿射解集里删除零空间分量。要保证这一点，需要完整正交分解从右侧继续处理变量空间，或直接使用 SVD/伪逆。

<!-- bilingual-en:start -->
Pivoting separates variables into a leading set and a remainder. Setting the latter to zero chooses a coordinate special solution, whereas minimum norm requires removing the null-space component from the entire affine solution set. A complete orthogonal factorisation adds right-side orthogonal transformations to enforce that geometry; the SVD or pseudoinverse does so directly.
<!-- bilingual-en:end -->

## 最小反例
<!-- bilingual-en:start -->
*Minimal counterexample*
<!-- bilingual-en:end -->

取 $A=[1\ 1]$、$b=1$。回代一个主元并把另一个变量置零可得 $x=(1,0)^T$，其范数为 $1$。但全部精确解满足 $x_1+x_2=1$，其中
$$
x_\star=(1/2,1/2)^T
$$
的范数为 $1/\sqrt2$，才是唯一最小范数解。两者残差都为零，区别只在系数范数。
<!-- bilingual-en:start -->
For $A=[1\ 1]$ and $b=1$, pivot back substitution with the other variable set to zero may return $(1,0)^T$, whose norm is $1$. The unique minimum-norm exact solution is $(1/2,1/2)^T$, whose norm is $1/\sqrt2$. Both have zero residual; only the latter minimizes the coefficient norm.
<!-- bilingual-en:end -->

## 与秩揭示的区别
<!-- bilingual-en:start -->
*Distinction from rank revelation*
<!-- bilingual-en:end -->

这里要区分两个问题：普通 CPQR 对有效秩能提供多强的证据，见 [[CPQR秩揭示边界]]；秩模型已经选定后，后一问题才是怎样从非唯一解中选到最小范数者。即使主元列正确识别了秩，只做主元回代也仍可能选错规范解。
<!-- bilingual-en:start -->
The CPQR rank-revealing boundary concerns how strongly ordinary CPQR supports an effective-rank decision. This note concerns the later choice of a minimum-norm member from a nonunique solution set. Even a correct rank decision does not make pivot-only back substitution a minimum-norm algorithm.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“残差已经最小”仍不足以证明 CPQR 主元回代得到的是最小范数解？
>
> **答案：** 秩亏时多个系数向量可产生同一个最小残差；还必须在这些向量中删除零空间分量，主元变量之外置零并不自动完成这一步。
>
> <!-- bilingual-en:start -->
> **Question:** Why does a minimum residual not prove that pivot-only CPQR back substitution gives the minimum-norm solution?
>
> **Answer:** Under rank deficiency, many coefficient vectors can produce the same minimum residual. Minimum norm additionally requires removing the null-space component, which setting nonpivot variables to zero does not generally accomplish.
> <!-- bilingual-en:end -->

## 来源与核验

- [LAPACK DGELSY](https://www.netlib.org/lapack/explore-html/d6/d4b/dgelsy_8f_source.html)：核对算法在 CPQR 与秩选择之后继续施加右侧正交变换，以形成完整正交分解并求最小范数解。
- [LAPACK Users' Guide: Linear Least Squares Problems](https://www.netlib.org/lapack/lug/node27.html)：核对可秩亏最小二乘同时最小化残差范数和解范数的目标。
<!-- bilingual-en:start -->
- LAPACK DGELSY supports the need for right-side orthogonal transformations after CPQR and rank selection to form a complete orthogonal factorisation and obtain a minimum-norm solution.
- The LAPACK Users' Guide supports the possibly rank-deficient least-squares objective that minimizes both residual norm and solution norm.
<!-- bilingual-en:end -->
