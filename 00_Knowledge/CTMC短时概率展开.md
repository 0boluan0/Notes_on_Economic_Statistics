---
aliases:
  - "生成率只出现在短时概率的一阶项而不是一步概率"
  - Infinitesimal transition probabilities
  - Short-time CTMC expansion
  - 短时转移概率
student_os: knowledge-atom
atom_id: PROB-CTMC-004
atom_set: continuous-time-markov-chains
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[生成矩阵约束]]"
related:
  - "[[CTMC转移半群]]"
leads_to:
  - "[[Kolmogorov前后向方程]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 生成率只出现在短时概率的一阶项而不是一步概率
<!-- bilingual-en:start -->
*Generator rates appear in the first-order term of short-time probabilities, not as one-step probabilities*
<!-- bilingual-en:end -->

> [!summary] 一阶展开
> 对 $i\ne j$，
> $$
> q_{ij}=\lim_{h\downarrow0}\frac{p_{ij}(h)}h,
> $$
> 而对角元满足
> $$
> q_{ii}=\lim_{h\downarrow0}\frac{p_{ii}(h)-1}{h}.
> $$
> 因而逐项有
> $$
> p_{ij}(h)=q_{ij}h+o(h),\qquad
> p_{ii}(h)=1-q_ih+o(h).
> $$
> <!-- bilingual-en:start -->
> A generator entry is a derivative at time zero. Multiplying the rate by a short duration gives only the first-order transition probability.
> <!-- bilingual-en:end -->

有限状态下可压缩为矩阵式 $P(h)=I+hQ+o(h)$。这里的 $o(h)$ 负责两个及以上跳跃等更高阶事件；它不是说对任意有限 $h$ 都有 $P(h)=I+hQ$。当状态空间可数且速率无界时，也不能未经控制就把逐项余项升级为统一矩阵范数余项。

> [!example] 速率可以大于一
> 若设备年故障率为 $q_{01}=5$，数字 5 完全合法，因为它不是概率。对 $h=0.01$ 年，短时故障概率约为 $5\times0.01=0.05$；不能说“一步故障概率为 5”。

> [!question]- 自检
> 为什么 $I+hQ$ 只适合很短的 $h$，而且未必本身是任意 $h$ 下的转移矩阵？
>
> **答案：** 它只保留关于 $h$ 的一阶项；较大 $h$ 时高阶的多次跳跃概率不可忽略，线性近似还可能产生负概率。

## 来源与核验

- [Ward Whitt, Continuous-Time Markov Chains, equations (3.1)–(3.6)](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=7)：核对生成矩阵为 $P(t)$ 在零点的右导数。
- [MIT OCW 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf)：核对生成率与有限时长转移概率的区别。
