---
aliases:
  - "同一粗网格的梯形值与中点值按一比二加权等于二倍细分后的Simpson值"
  - Simpson's trapezoidal-midpoint identity
student_os: knowledge-atom
atom_id: CALC-DEFINT-035
atom_type: theorem
status: source-checked
requires:
  - "[[梯形法]]"
  - "[[中点法]]"
  - "[[Simpson法]]"
part_of:
  - "[[定积分与应用.canvas]]"
---

# 同一粗网格的梯形值与中点值按一比二加权等于二倍细分后的Simpson值
<!-- bilingual-en:start -->
*One third of a coarse-grid trapezoidal value plus two thirds of its midpoint value equals Simpson's value on the grid with twice as many subintervals*
<!-- bilingual-en:end -->

设粗网格将 $[a,b]$ 等分为 $n$ 段。$T_n$ 与 $M_n$ 分别是在该粗网格上计算的 [[梯形法]] 与 [[中点法]]；将每段二等分后，共有 $N=2n$ 个小区间，细步长 $h=(b-a)/(2n)$。在这个细网格上的 [[Simpson法]] 满足 $S_{2n}=(T_n+2M_n)/3$。恒等式只需相关节点处的函数值存在，不需要导数或误差近似。
<!-- bilingual-en:start -->
Split $[a,b]$ into $n$ equal coarse subintervals and compute the [[梯形法|trapezoidal value]] $T_n$ and [[中点法|midpoint value]] $M_n$ on that partition. Halving each interval creates $N=2n$ fine subintervals with $h=(b-a)/(2n)$. The [[Simpson法|Simpson value]] on this fine grid satisfies $S_{2n}=(T_n+2M_n)/3$. This algebraic identity requires only defined function values at the nodes, without derivatives or an error approximation.
<!-- bilingual-en:end -->

令细网格节点 $x_i=a+ih$、$f_i=f(x_i)$。粗网格端点是偶数节点，粗网格中点是奇数节点；粗步长是 $2h$，因而
<!-- bilingual-en:start -->
Let $x_i=a+ih$ and $f_i=f(x_i)$ on the fine grid. Coarse endpoints have even indices and coarse midpoints have odd indices. Since the coarse step is $2h$,
<!-- bilingual-en:end -->

$$
T_n=h\left[f_0+2\sum_{j=1}^{n-1}f_{2j}+f_{2n}\right],
\qquad M_n=2h\sum_{j=1}^{n}f_{2j-1}.
$$

代入并合并权重，便得到细网格的 Simpson 公式：
<!-- bilingual-en:start -->
Substituting and collecting the weights gives the fine-grid Simpson formula:
<!-- bilingual-en:end -->

$$
\frac{T_n+2M_n}{3}
=\frac h3\left[f_0+4\sum_{j=1}^{n}f_{2j-1}
+2\sum_{j=1}^{n-1}f_{2j}+f_{2n}\right]
=S_{2n}.
$$

例如 $f(x)=x^2$、$[a,b]=[0,1]$，用一段粗网格得 $T_1=1/2$、$M_1=1/4$，所以 $(T_1+2M_1)/3=1/3=S_2$。这个组合使用粗网格端点和中点的并集，共 $2n+1$ 个节点；不能把两边下标直接都写成 $n$。
<!-- bilingual-en:start -->
For $f(x)=x^2$ on $[0,1]$ with one coarse subinterval, $T_1=1/2$ and $M_1=1/4$, so $(T_1+2M_1)/3=1/3=S_2$. The combination uses the union of coarse endpoints and midpoints, containing $2n+1$ nodes. The two sides must not all be labelled with subscript $n$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若已经在四个等宽小区间上算出 $T_4$ 与 $M_4$，组合得到 $S_4$ 还是 $S_8$？总共使用多少个不同节点？
>
> **答案：** 得到 $S_8$；五个粗端点加四个粗中点，共九个不同节点。
> <!-- bilingual-en:start -->
> If $T_4$ and $M_4$ use four equal subintervals, does their weighted combination give $S_4$ or $S_8$, and how many distinct nodes are used? **Answer:** It gives $S_8$, using five coarse endpoints plus four coarse midpoints: nine nodes.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=5|MIT 18.01SC Exam 3 题目，PDF 第 5 页，第 5 题]]与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|解答，PDF 第 6–7 页]]：核对 $T_n,M_n,S_{2n}$ 的网格定义及系数证明。解答第 7 页首个 $S_{2n}$ 展开末项印为 $x_{2n}$，下一行正确写作 $f(x_{2n})$；本卡按下一行并逐项代数核验。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Problems.pdf#page=5|MIT 18.01SC Exam 3 problems, PDF p. 5, Problem 5]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/07_Exams/Exam3_Solutions.pdf#page=6|solutions, PDF pp. 6–7]] support the three grid definitions and coefficient proof. In the first $S_{2n}$ expansion on p. 7, the last term is printed as $x_{2n}$; the next line correctly has $f(x_{2n})$. This card follows that line and checks every coefficient algebraically.
<!-- bilingual-en:end -->
