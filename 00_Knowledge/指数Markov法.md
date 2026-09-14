---
aliases:
  - "指数 Markov 法把尾事件转为矩母函数优化问题"
  - Exponential Markov method
  - Chernoff method
  - Exponential moment method
  - 指数矩方法
student_os: knowledge-atom
atom_id: PROB-CONC-007
atom_set: probability-concentration
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov不等式]]"
part_of:
  - "[[概率不等式与集中界.canvas]]"
related:
  - "[[相互独立]]"
  - "[[两两独立不推出相互独立]]"
  - "[[乘法Chernoff上界]]"
  - "[[Hoeffding不等式]]"
---

# 指数 Markov 法把尾事件转为矩母函数优化问题
<!-- bilingual-en:start -->
*The exponential Markov method turns a tail event into an MGF optimisation problem*
<!-- bilingual-en:end -->

> [!summary] 原子方法
> 对任意 $\lambda>0$，若矩母函数 $M_X(\lambda)=E[e^{\lambda X}]<\infty$，则
> $$
> P(X\ge a)\le e^{-\lambda a}M_X(\lambda).
> $$
> 记 $D_+=\{\lambda>0:M_X(\lambda)<\infty\}$。因为 $D_+$ 中每个 $\lambda$ 都给一个合法上界，可以再取
> $$
> P(X\ge a)\le\inf_{\lambda\in D_+}e^{-\lambda a}M_X(\lambda).
> $$
> 这是下确界；最优值未必由某个有限的 $\lambda$ 取到。若 $D_+$ 为空，指数 Markov 不产生有限上界。
> <!-- bilingual-en:start -->
> For any $\lambda>0$, if the moment-generating function $M_X(\lambda)=E[e^{\lambda X}]$ is finite, then
> $$P(X\ge a)\le e^{-\lambda a}M_X(\lambda).$$
> Let $D_+=\{\lambda>0:M_X(\lambda)<\infty\}$. Since every $\lambda\in D_+$ yields a valid bound, one may optimise:
> $$P(X\ge a)\le\inf_{\lambda\in D_+}e^{-\lambda a}M_X(\lambda).$$
> This is an infimum and need not be attained at any finite $\lambda$. If $D_+$ is empty, exponential Markov supplies no finite bound.
> <!-- bilingual-en:end -->

指数函数在 $\lambda>0$ 时单调递增，因此

$$
\{X\ge a\}=\{e^{\lambda X}\ge e^{\lambda a}\}.
$$

变量 $e^{\lambda X}$ 总是非负，Markov 便可应用。指数变换使较大的 $X$ 被更强地放大；优化 $\lambda$ 是在 threshold penalty $e^{-\lambda a}$ 与 MGF growth $M_X(\lambda)$ 之间找平衡。
<!-- bilingual-en:start -->
For $\lambda>0$, the exponential function is increasing, so the two displayed events are identical. The transformed variable $e^{\lambda X}$ is always non-negative and Markov applies. Exponentiation magnifies large values of $X$; optimising $\lambda$ balances the threshold penalty $e^{-\lambda a}$ against the MGF growth $M_X(\lambda)$.
<!-- bilingual-en:end -->

若 $S=\sum_iX_i$ 且 $X_1,\ldots,X_n$ mutually independent，则在各期望有限时

$$
E[e^{\lambda S}]
=E\!\left[\prod_i e^{\lambda X_i}\right]
=\prod_iE[e^{\lambda X_i}].
$$

这一步使用的是联合乘积分解；[[相互独立|mutual independence]] 是保证该分解成立的一条充分条件，不是“出现这种乘积等式当且仅当变量相互独立”的判据。[[两两独立不推出相互独立|Pairwise independence]] 一般不足以推出分解。控制左尾可对 $-X$ 使用同一路线，但需在相应的负参数方向存在有限指数矩。
<!-- bilingual-en:start -->
If $S=\sum_iX_i$ and the variables are mutually independent, the MGF factorises as displayed whenever the expectations are finite. Mutual independence is a sufficient condition for this factorisation, not an if-and-only-if test inferred from the appearance of a product equality. [[两两独立不推出相互独立|Pairwise independence]] generally does not imply the factorisation. A lower tail can be handled by applying the same method to $-X$, provided the corresponding negative-direction exponential moment is finite.
<!-- bilingual-en:end -->

> [!question]- 自检
> 指数 Markov 的哪一步需要 $\lambda>0$？标准独立和推导在哪一步把 mutual independence 当作充分条件？
> <!-- bilingual-en:start -->
> Which step of exponential Markov requires $\lambda>0$, and where does the standard independent-sum derivation use mutual independence as a sufficient condition?
> <!-- bilingual-en:end -->
>
> **答案：** $\lambda>0$ 保证指数变换保持右尾事件方向；标准推导用 mutual independence 充分保证和的 MGF 可分解为各项 MGF 的乘积，但单个乘积等式本身不能反推相互独立。
> <!-- bilingual-en:start -->
> **Answer:** Positivity of $\lambda$ makes exponentiation preserve the upper-tail direction. The standard derivation uses mutual independence as a sufficient condition for factorising the MGF of a sum; a product equality by itself does not imply mutual independence.
> <!-- bilingual-en:end -->

**继续：** 用 $[0,1]$ 上的凸性控制每项 MGF 得到 [[乘法Chernoff上界]]；只保留一般支持区间长度则得到 [[Hoeffding不等式]]。
<!-- bilingual-en:start -->
**Continue with:** Bounding each MGF by convexity on $[0,1]$ yields [[乘法Chernoff上界|the multiplicative Chernoff upper-tail bound]]. Retaining only general support-interval lengths yields [[Hoeffding不等式|Hoeffding's additive bound]].
<!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 第 20.5.6 节：核对 Chernoff 证明中的指数变换、Markov、由相互独立保证的矩母函数乘积分解，以及对参数的优化。
- [MIT OCW 18.S096，*Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science*，定理 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf)：独立核对指数矩母函数路线，并定位证明中实际使用独立性的步骤。
<!-- bilingual-en:start -->
- Section 20.5.6 of [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] verifies exponentiation, Markov, mutual-independence factorisation, and optimisation in the Chernoff proof.
- [MIT OCW 18.S096, *Ten Lectures and Forty-Two Open Problems in the Mathematics of Data Science*, Theorem 4.3](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-of-data-science-fall-2015/5f0f7205d1cf274e80d77345a7edbf2a_MIT18_S096F15_TenLec.pdf) independently verifies the exponential-MGF route and identifies the exact step where independence is used.
<!-- bilingual-en:end -->
