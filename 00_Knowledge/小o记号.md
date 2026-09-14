---
aliases:
  - 小o记号表示一个函数相对于另一个函数的比值趋于零
  - Little o notation
student_os: knowledge-atom
atom_id: CS-ASYM-004
atom_type: definition
status: source-checked
part_of:
  - "[[渐近记号与算法复杂度.canvas]]"
requires:
  - "[[无穷极限与无穷远极限]]"
leads_to:
  - "[[对数慢于正幂]]"
  - "[[幂函数慢于指数函数]]"
related:
  - "[[严格大O不等于小o]]"
  - "[[渐近比较的符号与零点边界]]"
---

# 小o记号表示一个函数相对于另一个函数的比值趋于零
<!-- bilingual-en:start -->
*Little o notation means that one function's ratio to another tends to zero*
<!-- bilingual-en:end -->

对同一输入域上最终严格为正的函数 $f,g$，**小 o 记号** $f=o(g)$ 表示 $f(n)/g(n)\to0$（$n\to\infty$）。它说 $f$ 相对于 $g$ 可以忽略，而不只是被 $g$ 的某一个常数倍控制。
<!-- bilingual-en:start -->
For eventually strictly positive functions $f,g$ on the same input domain, **little o notation** $f=o(g)$ means $f(n)/g(n)\to0$ as $n\to\infty$. It says that $f$ becomes negligible relative to $g$, not merely bounded by one constant multiple of $g$.
<!-- bilingual-en:end -->

$$
f=o(g)
\quad\Longleftrightarrow\quad
\forall\varepsilon>0\;\exists N_\varepsilon\;\forall n\ge N_\varepsilon:
\quad f(n)\le\varepsilon g(n).
$$

阈值可以依赖你要求的精度 $\varepsilon$，但不能在固定 $\varepsilon$ 后随 $n$ 改变。例如 $n=o(n^2)$，因为比值 $1/n\to0$；而 $2n$ 不是 $o(n)$，因为比值恒为 2。取 $\varepsilon=1$ 即见 $o(g)$ 蕴含[[大O记号|大 O 上界]] $O(g)$。
<!-- bilingual-en:start -->
The threshold may depend on the requested precision $\varepsilon$, but cannot vary with $n$ once $\varepsilon$ is fixed. For example, $n=o(n^2)$ because $1/n\to0$, whereas $2n$ is not $o(n)$ because its ratio is always two. Taking $\varepsilon=1$ shows that little o implies [[大O记号|big O]].
<!-- bilingual-en:end -->

“严格更小”不能只解释为单向大 O：$f=O(g)$ 且 $g\notin O(f)$ 仍可能不满足 $f=o(g)$，见 [[严格大O不等于小o]]。带符号余项常用 $r=o(g)$ 表示 $|r|/g\to0$，其中 $g$ 最终正；这与 $r/g\to0$ 等价，详见 [[渐近比较的符号与零点边界]]。
<!-- bilingual-en:start -->
“Strictly smaller” cannot be replaced by one-way big O: $f=O(g)$ and $g\notin O(f)$ need not imply $f=o(g)$; see [[严格大O不等于小o|strict big O versus little o]]. For signed remainders, $r=o(g)$ commonly means $|r|/g\to0$ with eventually positive $g$, equivalently $r/g\to0$. See [[渐近比较的符号与零点边界|sign and zero boundaries]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|MIT Session 24，印刷页 528–529，PDF 页 1–2]]，Definition 13.7.1、Lemma 13.7.7：支持比值趋零、幂函数例及小 o 蕴含大 O。本文明确分母最终正，量词形式由零极限定义展开；有符号零极限与绝对值零极限等价。
<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|MIT Session 24, printed pp. 528–529, PDF pp. 1–2]], Definition 13.7.1 and Lemma 13.7.7, supports the zero-ratio definition, power example and implication to big O. The denominator is explicitly eventually positive. The quantifiers unpack convergence to zero, which is equivalent for a signed ratio and its absolute value.
<!-- bilingual-en:end -->
