---
aliases:
  - 同参数纯 AK 经济保持初始人均资本比率而不会因贫穷自动追赶
  - Convergence boundary of the pure AK model
student_os: knowledge-atom
atom_id: MACRO-ENDO-006
atom_type: boundary
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 同参数纯 AK 经济保持初始人均资本比率而不会因贫穷自动追赶
<!-- bilingual-en:start -->
*Pure AK economies with common parameters preserve their initial per-capita capital ratio rather than catching up automatically from poverty.*
<!-- bilingual-en:end -->

考虑两个外生常储蓄率的纯 [[AK模型|AK]] 经济，具有相同且不变的 $A,s,\delta,n$，只有正的初始人均资本不同。由 [[AK人均增长条件]]，二者的人均资本都按 $g=sA-\delta-n$ 增长，所以初始较穷者不会仅因资本较少而增长更快。
<!-- bilingual-en:start -->
Consider two pure [[AK模型|AK]] economies with the same fixed $A,s,\delta,n$ and an exogenous saving rate, differing only in positive initial capital per person. By [[AK人均增长条件|the AK growth law]], both grow at $g=sA-\delta-n$. Having less initial capital does not itself give the poorer economy faster growth.
<!-- bilingual-en:end -->

写出路径并相除，就得到准确的比较对象：
<!-- bilingual-en:start -->
Writing and dividing the two paths identifies the exact comparison:
<!-- bilingual-en:end -->

$$
k_i(t)=k_i(0)e^{gt},\quad i=1,2,
\qquad
\frac{k_1(t)}{k_2(t)}=\frac{k_1(0)}{k_2(0)}.
$$

相同 $A$ 下，产出比率 $y_1/y_2$ 也恒定。例如初值为 $5$ 和 $10$ 时，两国即使都按每年 $3\%$ 增长，较穷者的人均资本仍始终是另一国的一半。恒定的是比例差距；若 $g>0$，绝对资本差距反而按 $e^{gt}$ 扩大。
<!-- bilingual-en:start -->
With common $A$, the output ratio $y_1/y_2$ is constant too. If initial capital is $5$ and $10$ and both grow at $3\%$ annually, the poorer economy remains at half the other's capital per person. It is the proportional gap that stays fixed; for $g>0$, the absolute capital gap grows with $e^{gt}$.
<!-- bilingual-en:end -->

这解释了纯 AK 为何没有标准 Solow 中资本边际回报随资本增加而下降的追赶动力，见 [[Solow 收敛边界]]。在固定参数的纯 AK 中，增长率从初始时刻就恒定，也没有先快后慢地趋向同一人均水平的转型过程。
<!-- bilingual-en:start -->
Pure AK lacks the catch-up force created by diminishing marginal returns in standard Solow; see [[Solow 收敛边界|Solow's convergence boundary]]. With fixed parameters, its growth rate is constant from the outset, without a transition towards a common per-capita level.
<!-- bilingual-en:end -->

这一结论只针对这里的纯 AK 设定。若生产技术、储蓄行为、技术扩散或制度随状态变化，路径就可能不同；不同参数的两个经济也可能偶然缩小差距。因此不能把此处的比率结果推广为“所有内生增长模型都没有收敛”，也不能用任意两国的收入变化直接检验它。
<!-- bilingual-en:start -->
This conclusion is restricted to the pure AK specification used here. State-dependent technology, saving, diffusion or institutions can change the paths, and economies with different parameters may happen to narrow their gap. The ratio result is neither a claim about all endogenous-growth models nor a test applicable to arbitrary country pairs.
<!-- bilingual-en:end -->

## 来源与核验

- [Acemoglu，MIT 14.452，2016 Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf#page=57)，PDF第57–59页：支持线性技术、固定增长率、从初值出发的指数路径与无转型动态。比率恒定、绝对差距路径以及 $5$ 与 $10$ 的例子均由两条指数路径独立推出；未把结论推广至全部内生增长模型。
<!-- bilingual-en:start -->
The lecture supplies linear technology, constant growth and exponential paths without transition dynamics. Constant ratios, absolute-gap dynamics and the example are independently derived by comparing two such paths. The conclusion is not extended to all endogenous-growth models.
<!-- bilingual-en:end -->
