---
aliases:
  - 在线 Softmax 按共同最大值重标定各块的指数和与加权和以合并精确输出
  - Online Softmax merges exact outputs by rescaling block exponential sums and weighted sums to a common maximum
student_os: knowledge-atom
atom_id: LLM-INF-037
atom_type: mechanism
status: source-checked
part_of:
  - "[[LLM 推理效率.canvas]]"
---

# 在线 Softmax 按共同最大值重标定各块的指数和与加权和以合并精确输出
<!-- bilingual-en:start -->
*Online Softmax merges exact outputs by rescaling block exponential sums and weighted sums to a common maximum*
<!-- bilingual-en:end -->

一行 [[Softmax]] 的所有位置共用一个分母。要分块计算注意力而保持这个共同分母，可以为每块保存三个量：最大分数、减去该最大值后的指数和，以及用同样指数加权的 value 和。合并时先把两块换到同一个最大值，再相加，最后才归一化。

<!-- bilingual-en:start -->
Every position in one [[Softmax]] row shares a denominator. Blockwise attention preserves it by retaining each block's maximum score, shifted exponential sum, and identically weighted value sum. Rescale both blocks to a common maximum, add the sums, and normalize afterward.
<!-- bilingual-en:end -->

固定一个 query，其可见位置分数为 $s_j$，value 为向量 $v_j$。先考虑非空且分数有限的块 $A$，定义

<!-- bilingual-en:start -->
For one query, let visible positions have scores $s_j$ and value vectors $v_j$. For a nonempty block $A$ with finite scores, define
<!-- bilingual-en:end -->

$$
m_A=\max_{j\in A}s_j,\qquad
\ell_A=\sum_{j\in A}e^{s_j-m_A},\qquad
u_A=\sum_{j\in A}e^{s_j-m_A}v_j.
$$

对互不重叠的两块 $A,B$，令 $m=\max(m_A,m_B)$，则

<!-- bilingual-en:start -->
For disjoint blocks $A$ and $B$, use $m=\max(m_A,m_B)$ and combine them as
<!-- bilingual-en:end -->

$$
\ell=e^{m_A-m}\ell_A+e^{m_B-m}\ell_B,\qquad
u=e^{m_A-m}u_A+e^{m_B-m}u_B,\qquad o=\frac u\ell.
$$

原因只需展开第一项：$e^{m_A-m}e^{s_j-m_A}=e^{s_j-m}$。两块都变成以同一个 $m$ 为基准的和，因此 $u/\ell=\sum_j e^{s_j}v_j/\sum_j e^{s_j}$，正是整行注意力输出。继续把 $(m,\ell,u)$ 当作一个块，便能逐块处理全部位置。

<!-- bilingual-en:start -->
Expanding a term gives $e^{m_A-m}e^{s_j-m_A}=e^{s_j-m}$. Both blocks therefore use the same reference, and $u/\ell$ equals the full row's normalized weighted sum. Treating the merged triple as one block permits repeated merging.
<!-- bilingual-en:end -->

## 三个位置的手算
<!-- bilingual-en:start -->
*A three-position calculation*
<!-- bilingual-en:end -->

取分数 $(\log2,0,\log6)$，标量 values 为 $(1,4,8)$。前两个位置为块 A，最后一个为块 B：

<!-- bilingual-en:start -->
Take scores $(\log2,0,\log6)$ and scalar values $(1,4,8)$. Put the first two positions in block A and the last in block B:
<!-- bilingual-en:end -->

$$
(m_A,\ell_A,u_A)=(\log2,3/2,3),\qquad
(m_B,\ell_B,u_B)=(\log6,1,8).
$$

合并最大值是 $\log6$，块 A 的重标定系数为 $1/3$，于是 $\ell=(1/3)(3/2)+1=3/2$，$u=(1/3)3+8=9$，输出为 6。直接用全行算也得到 $(2\cdot1+1\cdot4+6\cdot8)/(2+1+6)=6$。若把两块各自输出 2 和 8 简单平均，却会得到 5；错在把两个块当成了等权重。

<!-- bilingual-en:start -->
The common maximum is $\log6$, so block A is rescaled by $1/3$. This gives $\ell=3/2$, $u=9$, and output 6, matching the full-row calculation. Averaging the separate block outputs 2 and 8 would give 5, incorrectly assigning equal mass to the two blocks.
<!-- bilingual-en:end -->

在 [[FlashAttention]] 中，分数块可处理完即释放，累计状态不需要保存全部权重。若某块全被掩码排除，应按零贡献跳过，不能计算 $-\infty-(-\infty)$；整行无可见位置时仍不存在可归一化的分布，见 [[注意力掩码归一化]]。上述等价在精确运算下成立，有限精度的合并顺序仍可能影响舍入。

<!-- bilingual-en:start -->
In [[FlashAttention]], processed score blocks can be discarded without retaining every weight. A fully masked block contributes zero and must not evaluate $-\infty-(-\infty)$. A fully masked row remains undefined, as explained by [[注意力掩码归一化|masked normalization]]. Exact arithmetic gives equality; floating-point merging order can affect rounding.
<!-- bilingual-en:end -->

## 来源与核验

- [Dao et al. (2022), *FlashAttention*](https://arxiv.org/html/2205.14135v2)，§3.1 的分块 Softmax 恒等式、Algorithm 1：支持最大值重标定与累计输出。这里使用未归一化 $u$ 表示相同加权和，合并证明及三位置例子为独立展开。
- [[Softmax]]、[[注意力掩码归一化]]提供函数及可见集合的准确共享定义。

<!-- bilingual-en:start -->
FlashAttention supplies the blockwise normalization identities. The unnormalized weighted-sum notation, expanded proof, and three-position calculation are worked derivations. The linked shared atoms define Softmax and masking.
<!-- bilingual-en:end -->
