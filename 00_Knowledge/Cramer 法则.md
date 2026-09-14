---
aliases:
  - "Cramer 法则用换列后的行列式给出可逆方阵方程的解分量"
  - Cramer's rule
  - Determinant formula for a linear-system component
student_os: knowledge-atom
atom_id: LA-DET-011
atom_set: determinants
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[伴随矩阵求逆]]"
related:
  - "[[可逆性与非零行列式]]"
  - "[[行列式转置不变性]]"
part_of:
  - "[[行列式.canvas]]"
---

# Cramer 法则用换列后的行列式给出可逆方阵方程的解分量
<!-- bilingual-en:start -->
*Cramer's rule expresses each component of a uniquely solvable square system by replacing one column*
<!-- bilingual-en:end -->

> [!summary] 核心定理
> 设 $A=[a_1\ \cdots\ a_n]$ 为 $n\times n$ 方阵，$Ax=b$，且 $\det A\ne0$。令 $A_j(b)$ 表示把 $A$ 的第 $j$ 列换成 $b$ 后所得矩阵，则唯一解满足
> $$
> x_j=\frac{\det A_j(b)}{\det A},\qquad j=1,\ldots,n.
> $$
> <!-- bilingual-en:start -->
> Let $A=[a_1\ \cdots\ a_n]$ be an invertible square matrix and let $A_j(b)$ replace its $j$th column by $b$. The unique solution of $Ax=b$ satisfies $x_j=\det A_j(b)/\det A$.
> <!-- bilingual-en:end -->

把[[伴随矩阵求逆|伴随矩阵求逆公式]]代入 $x=A^{-1}b$，便可直接得到上式。若要看清分子为什么恰好是“换第 $j$ 列后的行列式”，还可以从行列式对单列的线性作第二条推导。这里的逐列线性由[[行列式转置不变性|逐行多线性转置到列]]得到。由
$$
b=Ax=x_1a_1+\cdots+x_na_n
$$
可得
$$
\det A_j(b)
=\sum_{k=1}^n x_k\det[a_1\ \cdots\ a_{j-1}\ a_k\ a_{j+1}\ \cdots\ a_n].
$$
当 $k\ne j$ 时，新矩阵有两列相同，行列式为零；只剩 $k=j$ 的 $x_j\det A$。因此 Cramer 法则不是一套脱离结构的换列口诀；伴随公式与逐列线性给出的是同一个结构。
<!-- bilingual-en:start -->
Substituting the adjugate inverse formula into $x=A^{-1}b$ gives Cramer's rule directly. A second derivation explains the replaced column: since $b=\sum_k x_ka_k$, column linearity expands $\det A_j(b)$ into a sum. Every term with $k\ne j$ has two equal columns and vanishes; the remaining term is $x_j\det A$. The adjugate formula and column linearity expose the same structure rather than an isolated replacement trick.
<!-- bilingual-en:end -->

## 二阶例子

对
$$
\begin{bmatrix}2&1\\1&3\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=
\begin{bmatrix}5\\7\end{bmatrix},
$$
$\det A=5$。换第一列得 $\det\begin{bmatrix}5&1\\7&3\end{bmatrix}=8$；换第二列得 $\det\begin{bmatrix}2&5\\1&7\end{bmatrix}=9$。所以 $x_1=8/5$、$x_2=9/5$，代回两条方程都成立。
<!-- bilingual-en:start -->
For the displayed $2\times2$ system, $\det A=5$. Replacing the first and second columns gives determinants $8$ and $9$, so $x_1=8/5$ and $x_2=9/5$; substitution verifies both equations.
<!-- bilingual-en:end -->

## 边界与用途

- 必须是方阵且 $\det A\ne0$。若 $\det A=0$，Cramer 的比值公式整体失效；即使某个分量形式上出现 $0/0$，也不能据此区分无解与无穷多解，应回到秩或消元。
- Cramer 法则适合证明、参数敏感性和很小的符号系统。对大型稠密数值系统，逐列计算 determinant 代价高且会重复工作，通常直接用 LU、QR 等分解。
- 公式一次给一个分量；如果只关心一个 $x_j$，它在理论推导中可能比写出整个逆矩阵更干净。
<!-- bilingual-en:start -->
- The rule requires a square matrix with nonzero determinant. When the determinant vanishes, Cramer's ratio formula is unavailable. Even if one component formally gives $0/0$, that expression does not distinguish inconsistency from infinitely many solutions; use elimination or rank instead.
- It is useful for proofs, parameter sensitivity, and very small symbolic systems. Large dense numerical systems should normally be solved by factorization.
- The formula isolates one component, which can be useful when the full inverse is unnecessary.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么把第 $j$ 列换成 $b=\sum_kx_ka_k$ 后，展开式中只有 $x_j$ 对应的项保留下来？
>
> **答案：** 其他项把某个已有列 $a_k$ 再放到第 $j$ 列，造成两列相同，行列式为零；$k=j$ 时恢复原矩阵。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|MIT 18.06SC Session 2.7 summary]]：核对 Cramer 法则与伴随矩阵公式的联系。
- [[01_Math/02_linear algebra/02_Least Squares, Determinants and Eigenvalues.md#Cramer 法则|课程 Cramer 法则推导]]：核对换列公式、适用条件与奇异系统边界。
<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|MIT 18.06SC Session 2.7 summary]] was checked for Cramer's rule and its relation to the adjugate formula.
- [[01_Math/02_linear algebra/02_Least Squares, Determinants and Eigenvalues.md#Cramer 法则|The course derivation of Cramer's rule]] was checked for the replacement formula, assumptions, and singular-system boundary.
<!-- bilingual-en:end -->
