---
aliases:
  - "Jacobian 行列式是在一点对方阵 Jacobian 矩阵取行列式所得的标量"
  - "Jacobian 行列式是方阵导数在一点的一阶有向体积缩放因子"
  - Jacobian determinant
  - Local signed-volume factor
student_os: knowledge-atom
atom_id: LA-DET-015
atom_set: determinants
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[行列式体积与取向]]"
  - "[[Jacobian矩阵]]"
leads_to:
  - "[[多元换元公式]]"
  - "[[逆函数定理]]"
related:
  - "[[Jacobian奇异不推局部非单射]]"
  - "[[矩形Jacobian体积因子]]"
part_of:
  - "[[行列式.canvas]]"
  - "[[多元微分.canvas|多元微分]]"
---

# Jacobian 行列式是在一点对方阵 Jacobian 矩阵取行列式所得的标量
<!-- bilingual-en:start -->
*The Jacobian determinant at a point is the determinant of the square Jacobian matrix there*
<!-- bilingual-en:end -->

> [!summary] 核心定义
> 设 $T:\mathbb R^n\to\mathbb R^n$ 在 $x$ 处可微，$J_T(x)=DT(x)$。**Jacobian 行列式**定义为
> $$
> \det J_T(x).
> $$
> 因为
> $$
> T(x+h)=T(x)+J_T(x)h+o(\|h\|),
> $$
> $\det J_T(x)$ 给出这个一阶线性近似对局部有向 $n$ 维体积的缩放，$|\det J_T(x)|$ 给出不计取向的普通体积缩放。
>
> <!-- bilingual-en:start -->
> If $T:\mathbb R^n\to\mathbb R^n$ is differentiable at $x$, its Jacobian determinant is $\det J_T(x)=\det DT(x)$. The derivative is the first-order linear approximation; its determinant scales local signed $n$-volume, while the absolute determinant scales ordinary unsigned volume.
> <!-- bilingual-en:end -->

负号只表示一阶近似翻转了取向，不表示普通体积变成负数。$\det J_T(x)=0$ 则表示导数在该点不满秩，所以一阶近似把 $n$ 维体积压成零；这只是对导数的局部一阶判断，不能单独推出原非线性映射的单射性。

## 尺寸与顺序边界

- 普通 Jacobian determinant 只对输入、输出维数相同的方阵导数有定义；低维参数化进入更高维空间时，应改用[[矩形Jacobian体积因子|矩形 Jacobian 的 Gram 体积因子]]。
- 调换输入变量顺序会交换 Jacobian 的列，调换输出坐标顺序会交换行；一次交换使 determinant 变号，但不改变绝对体积因子。
- 这里描述的是方阵导数的一阶体积因子；积分换元条件见[[多元换元公式]]，局部可逆性见[[逆函数定理]]。

> [!question]- 自检
> 为什么交换两个输入变量后 $\det J_T$ 会变号，但局部普通体积因子不变？
>
> **答案：** 交换输入变量就是交换 Jacobian 的两列，因此有向体积反号；普通体积使用绝对值，所以不变。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf|MIT 18.06SC Session 2.7 summary]]：核对线性映射中 determinant 对有向体积、绝对 determinant 对普通体积的缩放解释。
- [[Jacobian矩阵]] 的来源核验导数作为局部线性映射以及方阵 Jacobian 的记号；换元公式与局部可逆性的条件分别在对应页面核验。

<!-- bilingual-en:start -->
The MIT 18.06SC summary supports the signed- and unsigned-volume interpretation of a determinant, while the sources in [[Jacobian矩阵]] support the derivative as a local linear map and the square-Jacobian notation.
<!-- bilingual-en:end -->
