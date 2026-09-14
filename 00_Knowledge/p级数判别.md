---
aliases:
  - "p级数的倒数幂求和当且仅当指数大于一收敛"
  - p-series convergence criterion
student_os: knowledge-atom
atom_id: CALC-SER-013
atom_type: theorem
status: source-checked
requires:
  - "[[级数积分判别]]"
  - "[[无穷远 p 积分判别]]"
related:
  - "[[级数通项判别]]"
  - "[[调和数对数增长]]"
part_of:
  - "[[无穷级数与幂级数.canvas]]"
---

# p级数的倒数幂求和当且仅当指数大于一收敛
<!-- bilingual-en:start -->
*A p-series of reciprocal powers converges exactly when its exponent exceeds one*
<!-- bilingual-en:end -->

对实数 $p$，$p$ 级数是 $\sum_{n=1}^\infty n^{-p}$，其敛散判据为
<!-- bilingual-en:start -->
For real $p$, a p-series is $\sum_{n=1}^\infty n^{-p}$, with the criterion
<!-- bilingual-en:end -->

$$
\sum_{n=1}^{\infty}\frac1{n^p}\text{ 收敛}\iff p>1.
$$

当 $p>0$ 时，$f(x)=x^{-p}$ 在 $[1,\infty)$ 上连续、为正、递减，故 [[级数积分判别]] 将问题交给 [[无穷远 p 积分判别]]。当 $p\le0$ 时，通项不趋于零，直接由 [[级数通项判别]] 得到发散，不能硬套要求递减的积分判别。
<!-- bilingual-en:start -->
For $p>0$, $x^{-p}$ is continuous, positive, and decreasing on $[1,\infty)$, so the [[级数积分判别|integral test]] reduces the question to the [[无穷远 p 积分判别|p-integral at infinity]]. For $p\le0$, the terms do not approach zero, so the [[级数通项判别|term test]] proves divergence directly.
<!-- bilingual-en:end -->

临界点 $p=1$ 是调和级数，虽然通项趋于零，部分和仍以对数速度无界增长，见 [[调和数对数增长]]。例如 $\sum n^{-3/2}$ 收敛，而 $\sum n^{-1/2}$ 发散。改变有限起点不改变这条判据；收敛时的级数和一般不等于 $1/(p-1)$，后者是对应积分的值。
<!-- bilingual-en:start -->
The boundary $p=1$ is the harmonic series: its terms approach zero but its partial sums grow without bound, as shown in [[调和数对数增长|logarithmic growth of harmonic numbers]]. Thus $\sum n^{-3/2}$ converges and $\sum n^{-1/2}$ diverges. Changing a finite starting index does not change the criterion; the convergent series generally does not sum to the corresponding integral value $1/(p-1)$.
<!-- bilingual-en:end -->

## 来源与核验

- [OpenStax Calculus Volume 2 §5.3，The p-Series 与式 (5.9)](https://openstax.org/books/calculus-volume-2/pages/5-3-the-divergence-and-integral-tests)：支持全部实数 $p$ 的分类、$p\le0$ 的通项处理和 $p>0$ 的积分证明；本卡例子由该判据直接代入得到。
<!-- bilingual-en:start -->
- [OpenStax Calculus Volume 2 §5.3, The p-Series and equation (5.9)](https://openstax.org/books/calculus-volume-2/pages/5-3-the-divergence-and-integral-tests) supports the full real-parameter classification, the term argument for $p\le0$, and the integral proof for $p>0$. The examples follow by substitution.
<!-- bilingual-en:end -->
