---
student_os: knowledge-atom
atom_id: LA-LU-001
aliases:
  - Nonzero LU pivots can be separated into a diagonal factor
  - LU分解的非零主元可单独提取为对角因子
  - LDU factorization
status: source-checked
---

<!-- bilingual-en:start -->
# Nonzero LU pivots can be separated into a diagonal factor
<!-- bilingual-en:end -->
*LU 分解的非零主元可单独提取为对角因子*

<!-- bilingual-en:start -->
Suppose $A=LU$ with $L$ unit lower triangular and $U$ upper triangular with nonzero diagonal. Define $D=\operatorname{diag}(u_{11},\ldots,u_{nn})$ and $\widehat U=D^{-1}U$. Then
<!-- bilingual-en:end -->

设 $A=LU$，其中 $L$ 为单位下三角，$U$ 为对角元非零的上三角。定义 $D=\operatorname{diag}(u_{11},\ldots,u_{nn})$ 及 $\widehat U=D^{-1}U$，则

$$
A=LD\widehat U,
$$

<!-- bilingual-en:start -->
and $\widehat U$ is unit upper triangular. Left multiplication by $D^{-1}$ divides row $i$ of $U$ by its pivot $u_{ii}$, so its diagonal becomes one. For example,
<!-- bilingual-en:end -->

且 $\widehat U$ 是单位上三角。左乘 $D^{-1}$ 把 $U$ 的第 $i$ 行除以主元 $u_{ii}$，因此对角元变为一。例如

$$
\begin{pmatrix}2&1\\0&3\end{pmatrix}
=\begin{pmatrix}2&0\\0&3\end{pmatrix}
\begin{pmatrix}1&1/2\\0&1\end{pmatrix}.
$$

<!-- bilingual-en:start -->
The original $U$ and normalized $\widehat U$ are different matrices. The construction requires nonzero pivots so that $D^{-1}$ exists. With a pivoted factorization $PA=LU$, the same normalization gives $PA=LD\widehat U$.
<!-- bilingual-en:end -->

原来的 $U$ 与归一后的 $\widehat U$ 是不同矩阵。此构造要求主元非零，保证 $D^{-1}$ 存在。若从带置换的 $PA=LU$ 出发，同样得到 $PA=LD\widehat U$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=1|MIT: Factorization into A = LU, p. 1]]
  <!-- bilingual-en:start -->
  Gives the diagonal-pivot factorization and the two-by-two example; the normalization is checked by multiplying $D\widehat U$.
  <!-- bilingual-en:end -->
  核对主元对角因子与二阶例子；由 $D\widehat U$ 回乘验证归一化。
