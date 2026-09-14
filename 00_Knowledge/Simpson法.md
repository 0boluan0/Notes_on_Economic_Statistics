---
aliases:
  - "复合Simpson法每两个等宽小区间拟合二次函数并按一四二交替权重求积"
  - Composite Simpson's one-third rule
student_os: knowledge-atom
atom_id: CALC-DEFINT-034
atom_type: method
status: source-checked
requires:
  - "[[定积分]]"
related:
  - "[[Simpson组合恒等式]]"
  - "[[Simpson误差界]]"
part_of:
  - "[[定积分与应用.canvas]]"
---

# 复合Simpson法每两个等宽小区间拟合二次函数并按一四二交替权重求积
<!-- bilingual-en:start -->
*Composite Simpson's one-third rule integrates a quadratic fitted across each pair of equal-width subintervals*
<!-- bilingual-en:end -->

对有限区间 $a<b$，取偶数 $N\ge2$ 个等宽小区间，$h=(b-a)/N$、$x_i=a+ih$。这里的 Simpson 法指复合 $1/3$ 公式：每两个小区间用通过三个节点的二次插值多项式近似 $f$，再把各段多项式的积分相加。
<!-- bilingual-en:start -->
For a finite interval with $a<b$, use an even number $N\ge2$ of equal-width subintervals, $h=(b-a)/N$, and $x_i=a+ih$. This is the composite one-third formula: fit a quadratic through the three nodes of each pair of subintervals, integrate it, and sum the contributions.
<!-- bilingual-en:end -->

$$
S_N=\frac h3\left[f(x_0)+4\sum_{j=1}^{N/2}f(x_{2j-1})
+2\sum_{j=1}^{N/2-1}f(x_{2j})+f(x_N)\right].
$$

在一个宽度为 $2h$ 的片段上，令中点坐标为 $u=0$，二次函数写作 $q(u)=A+Bu+Cu^2$。直接积分给出 $2hA+2Ch^3/3$，与 $\tfrac h3[q(-h)+4q(0)+q(h)]$ 相同；相邻片段共享端点，便得到全局权重 $1,4,2,4,\ldots,2,4,1$。三次项在对称区间上也抵消，所以三次及以下多项式都精确。
<!-- bilingual-en:start -->
On a panel of width $2h$, centre the coordinate at its midpoint and write $q(u)=A+Bu+Cu^2$. Its integral is $2hA+2Ch^3/3$, equal to $\tfrac h3[q(-h)+4q(0)+q(h)]$. Shared panel endpoints give the global weights $1,4,2,4,\ldots,2,4,1$. Cubic terms also cancel by symmetry, making the rule exact for every polynomial of degree at most three.
<!-- bilingual-en:end -->

例如 $\int_1^2dx/x$ 取 $N=2$，$h=1/2$，得到 $S_2=\tfrac16(1+4\cdot\tfrac23+\tfrac12)=25/36\approx0.694444$。取样点共有 $N+1$ 个；$N$ 必须为偶数以便成对分组，且这里的固定权重要求等距节点。一般函数的精度保证还需要 [[Simpson误差界]] 的光滑性条件。
<!-- bilingual-en:start -->
For $\int_1^2dx/x$ with $N=2$ and $h=1/2$, $S_2=\tfrac16(1+4\cdot\tfrac23+\tfrac12)=25/36\approx0.694444$. There are $N+1$ sample points. Pairing requires even $N$, and these fixed weights require equally spaced nodes. For general functions, an accuracy guarantee also needs the smoothness assumptions in [[Simpson误差界|the Simpson error bound]].
<!-- bilingual-en:end -->

> [!question]- 自检
> $N=6$ 时权重是什么？对 $f(x)=1$，如何检查外面的 $h/3$？
>
> **答案：** 权重为 $1,4,2,4,2,4,1$，总和 $18=3N$，乘 $h/3$ 后得到 $Nh=b-a$。
> <!-- bilingual-en:start -->
> What are the weights when $N=6$, and how does $f(x)=1$ check the factor $h/3$? **Answer:** The weights are $1,4,2,4,2,4,1$, summing to $18=3N$. Multiplication by $h/3$ gives $Nh=b-a$.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Ses63d_Problems.pdf#page=1|MIT 18.01SC Session 63d，PDF 第 1–2 页]]：文件实际是 Simpson 讲义；核对偶数等分、二次拟合、每组 $1,4,1$ 与复合权重。上面的对称坐标推导已直接验算。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/06_Problem_Sets/Ses63d_Problems.pdf#page=1|MIT 18.01SC Session 63d, PDF pp. 1–2]] is the Simpson lecture despite its filename. It supports even subdivision, quadratic fitting, the local $1,4,1$ rule, and composite weights. The centred-coordinate derivation was checked directly.
<!-- bilingual-en:end -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64b_Lecture_Notes.pdf#page=1|Session 64b，PDF 第 1 页]]与 [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64c_Lecture_Notes.pdf#page=1|Session 64c，PDF 第 1 页]]：核对 $1/x$ 例子、三次多项式精确性和常数函数检查。
<!-- bilingual-en:start -->
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64b_Lecture_Notes.pdf#page=1|Session 64b, PDF p. 1]] and [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses64c_Lecture_Notes.pdf#page=1|Session 64c, PDF p. 1]] support the $1/x$ example, cubic exactness, and constant-function check.
<!-- bilingual-en:end -->
