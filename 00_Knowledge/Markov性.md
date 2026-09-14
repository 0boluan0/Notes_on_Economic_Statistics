---
aliases:
  - "马尔可夫性是给定当前状态后未来与完整过去条件独立"
  - Markov property
  - Markov 性
  - 无后效性
student_os: knowledge-atom
atom_id: PROB-DTMC-001
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件独立]]"
  - "[[随机过程]]"
related:
  - "[[Markov充分状态]]"
  - "[[DTMC时间齐次转移核]]"
  - "[[CTMC时间齐次转移函数]]"
leads_to:
  - "[[离散时间Markov链]]"
  - "[[CTMC定义]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 马尔可夫性是给定当前状态后未来与完整过去条件独立
<!-- bilingual-en:start -->
*The Markov property is conditional independence of the future and the full past given the present state*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对随机过程 $\{X_t\}$，把时刻 $t$ 以前的完整历史记为 $\mathcal F_t^-=\sigma(X_s:s<t)$，把时刻 $t$ 以后的完整未来记为 $\mathcal F_t^+=\sigma(X_u:u>t)$。Markov 性要求
> $$
> \mathcal F_t^+\ \perp\!\!\!\perp\ \mathcal F_t^-\mid X_t.
> $$
> 也就是说，一旦知道当前状态，过去不再为未来的条件分布补充预测信息。
> <!-- bilingual-en:start -->
> The Markov property says that the future and the earlier history are conditionally independent once the present state is known.
> <!-- bilingual-en:end -->

对离散时间、离散状态过程，最常用的一步写法是：只要所条件化的历史有正概率，
$$
\Pr(X_{n+1}=j\mid X_n=i,X_{n-1}=i_{n-1},\ldots,X_0=i_0)
=\Pr(X_{n+1}=j\mid X_n=i).
$$
一步性质可迭代到任意有限未来，因此它是上面“完整未来只看现在”在 DTMC 中的工作形式。

条件独立不等于相邻状态独立。例如确定性链 $X_{n+1}=X_n$ 中，相邻状态完全相同；但给定 $X_n$ 后，更早历史仍不会改变未来，所以它满足 Markov 性。“无后效性”也不是说历史从未发生，而是说当前状态已经汇总了与未来有关的历史信息。

> [!question]- 自检
> 若已知 $X_n$ 后，再告诉你 $X_{n-1}$ 会改变 $X_{n+1}$ 的条件分布，这个状态表示满足 Markov 性吗？
>
> **答案：** 不满足。更早历史仍含有当前状态没有保留的预测信息。

## 来源与核验

- [MIT OCW 6.262, Chapter 3, Definition 3.1.1](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf)：核对 DTMC 的一步条件概率表述及“状态汇总相关过去”的解释。
- [[条件独立]]：复用一般条件独立定义，把“过去”和“未来”视为给定当前状态后的两个随机对象。
- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx]]：仅用于确认课程采用“无后效性”术语。
