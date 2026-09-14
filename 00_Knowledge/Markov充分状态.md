---
aliases:
  - "Markov 充分状态保留预测未来所需的全部历史信息"
  - "状态是否充分决定一阶 Markov 表示能否成立"
  - Sufficient Markov state
  - Markov 状态充分性
  - Markov状态充分性
  - 充分状态
student_os: knowledge-atom
atom_id: PROB-DTMC-002
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Markov性]]"
  - "[[条件独立]]"
related:
  - "[[Markov switching模型]]"
  - "[[CTMC完整路径估计]]"
leads_to:
  - "[[状态扩充]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# Markov 充分状态保留预测未来所需的全部历史信息
<!-- bilingual-en:start -->
*A sufficient Markov state retains all information from the history that is needed to predict the future*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $\mathcal H_n$ 表示时刻 $n$ 已观察到的完整历史，状态变量 $S_n$ 由这段历史构造。若对每个未来事件 $A$ 都有
> $$
> \Pr(A\mid\mathcal H_n)=\Pr(A\mid S_n),
> $$
> 就称 $S_n$ 对未来预测是 **Markov 充分状态**。它不是“记录得最多”的状态，而是恰好保留了所有会改变未来条件分布的信息。
> <!-- bilingual-en:start -->
> A state is Markov sufficient when conditioning on it gives the same future law as conditioning on the entire observed history.
> <!-- bilingual-en:end -->

充分性永远相对于所预测的系统和所选状态表示而言。同一组原始观测既可能在一个状态定义下满足 Markov 性，也可能因遗漏年龄、库存、持续时间或隐含制度而失败。因此“这个系统是 Markov 的”必须同时说明什么被当作状态。

> [!example] 随机游走的位置
> 若 $Z_{n+1}=Z_n+\varepsilon_{n+1}$，且新增量与过去独立，那么当前位置 $Z_n$ 已足以给出下一步及以后的位置分布。走到当前位置所经过的整条路线不会再改变预测，所以 $Z_n$ 是充分状态。

> [!question]- 自检
> 已知状态 $S_n$ 后，再加入上一期观测会改变下一步预测。$S_n$ 是否充分？
>
> **答案：** 不充分。上一期仍提供额外预测信息，说明当前状态遗漏了相关历史。

## 来源与核验

- [MIT OCW 6.262, Chapter 3, Definition 3.1.1 and Example 3.1.1](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=2)：核对“状态汇总与未来有关的过去信息”以及状态选择对 Markov 表示的作用。
- [[随机过程基础.canvas|随机过程基础]]：用于区分过程本身、观察历史和由历史构造的状态表示。
