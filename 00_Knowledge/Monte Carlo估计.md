---
aliases:
  - "Monte Carlo 估计通过从指定概率模型反复抽样，用样本平均近似难以直接计算的期望"
  - "Monte Carlo estimation approximates a difficult expectation by repeatedly sampling from a specified probability model and averaging"
  - "Monte Carlo 模拟"
  - "Simple Monte Carlo"
student_os: knowledge-atom
atom_id: PROB-MC-004
atom_set: monte-carlo-methods
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[随机变量]]"
  - "[[期望线性性]]"
leads_to:
  - "[[Monte Carlo均值标准误]]"
  - "[[Monte Carlo误差分层]]"
  - "[[Monte Carlo实现验证]]"
  - "[[方差缩减]]"
related:
  - "[[伪随机数生成器]]"
  - "[[风险蒙特卡洛]]"
---

# Monte Carlo 估计通过从指定概率模型反复抽样，用样本平均近似难以直接计算的期望
<!-- bilingual-en:start -->
*Monte Carlo estimation approximates a difficult expectation by repeatedly sampling from a specified probability model and averaging*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若目标量能写成 $\mu=E_P[g(X)]$，Monte Carlo 估计从指定分布 $P$ 生成重复样本，计算每个样本的 $g(X)$，再用样本平均近似 $\mu$。
> <!-- bilingual-en:start -->
> If a target can be written as $\mu=E_P[g(X)]$, Monte Carlo estimation draws repeatedly from the specified law $P$, evaluates $g(X)$, and uses the sample average to approximate $\mu$.
> <!-- bilingual-en:end -->

对独立样本 $X_1,\ldots,X_N\sim P$，最基本的估计量是

$$
\widehat\mu_N=\frac1N\sum_{i=1}^N g(X_i).
$$

它把一个概率问题拆成三个可检查的对象：抽样分布 $P$、单次输出规则 $g$、以及汇总估计量。概率也属于这个框架：若 $g(X)=\mathbf 1\{X\in A\}$，那么 $\mu=P(X\in A)$，样本平均就是事件 $A$ 的模拟频率。
<!-- bilingual-en:start -->
The method separates a probability problem into a sampling law $P$, a per-draw output rule $g$, and an aggregation rule. Probabilities are a special case: with $g(X)=\mathbf 1\{X\in A\}$, the sample average is the simulated frequency of event $A$.
<!-- bilingual-en:end -->

Monte Carlo 的优势不是自动“更准确”，而是能把解析上困难的积分、概率或期望转成可重复计算。精度仍取决于抽样设计和输出分布；基准标准误见 [[Monte Carlo均值标准误]]，模型与实现边界见 [[Monte Carlo误差分层]]。
<!-- bilingual-en:start -->
Monte Carlo turns analytically difficult integrals, probabilities, or expectations into repeatable computation. It does not guarantee accuracy by itself; precision depends on the sampling design and output distribution.
<!-- bilingual-en:end -->

这张卡只定义独立重复抽样的基本入口。MCMC、quasi-Monte Carlo、序贯 Monte Carlo 或嵌套模拟会改变抽样机制或估计量，不能仅凭名称沿用这里的误差公式。
<!-- bilingual-en:start -->
This atom defines the simple repeated-sampling entry point. MCMC, quasi-Monte Carlo, sequential Monte Carlo, and nested simulation alter the sampling mechanism or estimator and require their own error analysis.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 若要用模拟估计 $P(X>c)$，应选择什么 $g(X)$？
>
> **答案：** $g(X)=\mathbf 1\{X>c\}$；样本平均就是超过阈值的模拟比例。
> <!-- bilingual-en:start -->
> Use $g(X)=\mathbf 1\{X>c\}$; its sample average is the simulated exceedance proportion.
> <!-- bilingual-en:end -->

## 来源与核验

- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, “Simple Monte Carlo”](https://artowen.su.domains/mc/Ch-intro.pdf)：核对以样本期望估计总体期望的基本定义与指示变量概率估计。
- MIT 6.100L, [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec26.pdf|Lecture 26 slides, pp. 21–28]]：核对“定义一次实验—重复—记录—汇总”的入门模拟框架。
