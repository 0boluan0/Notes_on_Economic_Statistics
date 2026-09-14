---
aliases:
  - 连续函数的定积分等于其任一原函数在两端的值之差
  - Newton–Leibniz formula
  - Antiderivative evaluation of a definite integral
student_os: knowledge-atom
atom_id: CALC-DEFINT-013
atom_type: theorem
status: source-checked
part_of:
  - "[[定积分与应用.canvas]]"
requires:
  - "[[原函数]]"
  - "[[积分求导定理]]"
  - "[[原函数相差常数]]"
leads_to:
  - "[[定积分换元]]"
  - "[[分部积分]]"
---

# 连续函数的定积分等于其任一原函数在两端的值之差
<!-- bilingual-en:start -->
*The integral of a continuous function equals the endpoint difference of any antiderivative*
<!-- bilingual-en:end -->

若 $f$ 在 $[a,b]$ 连续，$F$ 在 $[a,b]$ 连续且在 $(a,b)$ 满足 $F'=f$，则
$$\int_a^bf(x)\,dx=F(b)-F(a)=[F(x)]_a^b.$$
MIT 18.01SC 把这一形式称为微积分第一基本定理（FTC I）。左边是由累积定义的数，右边给出计算它的捷径；不是把“求原函数”当作积分的定义。
<!-- bilingual-en:start -->
For a continuous integrand and an antiderivative continuous up to the endpoints, the definite integral is the endpoint difference. MIT 18.01SC calls this FTC I. It evaluates a number already defined by accumulation rather than defining integration as antidifferentiation.
<!-- bilingual-en:end -->

为什么任选一个原函数都行？令 $G(x)=\int_a^xf(t)dt$，由[[积分求导定理]]知 $G'=f=F'$。[[原函数相差常数]]于是给出 $F-G=C$；用 $G(a)=0$ 消去 $C$，就有 $G(b)=F(b)-F(a)$。这个证明也解释了为何不定积分里的 $+C$ 在端点评价中相消。
<!-- bilingual-en:start -->
The accumulation function has the same derivative as the chosen antiderivative, so their difference is constant. Its zero initial accumulation identifies the constant and gives the endpoint formula. This also explains why additive constants cancel in definite evaluation.
<!-- bilingual-en:end -->

例如 $\int_0^\pi\sin x\,dx=[-\cos x]_0^\pi=2$。每次应用都应核对原函数导数、完整积分区间和端点次序。$[-1/x]_{-1}^1=-2$ 并不是 $\int_{-1}^1x^{-2}dx$ 的合法答案：零点处分隔了定义域，该题属于[[反常积分]]且发散。
<!-- bilingual-en:start -->
The sine example evaluates to two. Always check the derivative, the entire interval, and endpoint order. Subtracting endpoint values of $-1/x$ across zero does not evaluate the integral of $x^{-2}$: the singularity requires [[反常积分|improper integration]], which diverges here.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses47b_Lecture_Notes.pdf#page=1|MIT Session 47b，第 1 页]]：FTC I 编号及用 $x^3/3$ 作原函数的端点评价；[[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses47b_Lecture_Notes.pdf#page=2|第 2 页]]给出一拱正弦积分为 $2$ 的算例。
- [[01_Math/01_calculus/MIT18.01SC_Single_Variable_Calculus/03_Definite_Integrals/Ses52b_Lecture_Notes.pdf#page=1|MIT Session 52b，第 1 页]]：在连续被积函数条件下，用 FTC II 构造原函数，再以导数相同、相差常数证明端点公式；跨奇点例子另按反常积分定义检查。
<!-- bilingual-en:start -->
MIT supplies the endpoint theorem, its course numbering, and its proof from accumulation differentiation. The singular example was checked against the improper-integral definition.
<!-- bilingual-en:end -->
