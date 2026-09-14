---
aliases:
  - "Bayes 法则用似然与先验反转条件方向"
  - Bayes' rule
  - Bayes theorem
  - 贝叶斯法则
student_os: knowledge-atom
atom_id: PROB-BAYES-001
atom_set: probability-foundations
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件概率]]"
  - "[[全概率公式]]"
part_of:
  - "[[概率空间、条件概率与 Bayes 法则.canvas]]"
---

# Bayes 法则用似然与先验反转条件方向
<!-- bilingual-en:start -->
*Bayes' rule uses likelihood and prior probability to reverse the conditioning direction*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 对完备互斥假设 $H_1,\ldots,H_k$，若每个 $P(H_\ell)>0$ 且证据满足 $P(B)>0$，
> $$P(H_j\mid B)=\frac{P(B\mid H_j)P(H_j)}{\sum_{\ell=1}^{k}P(B\mid H_\ell)P(H_\ell)}.$$
> 分子是“似然 $\times$ 先验”，分母是证据的总概率，负责把所有假设的后验重新归一化到总和为 $1$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> For mutually exclusive and exhaustive hypotheses $H_1,\ldots,H_k$, if every $P(H_\ell)>0$ and the evidence satisfies $P(B)>0$,
> $$P(H_j\mid B)=\frac{P(B\mid H_j)P(H_j)}{\sum_{\ell=1}^{k}P(B\mid H_\ell)P(H_\ell)}.$$
> The numerator is “likelihood $\times$ prior”; the denominator is the total probability of the evidence and renormalises all posterior probabilities to sum to one.
> <!-- bilingual-en:end -->

## 最容易混淆的方向
<!-- bilingual-en:start -->
*The most easily confused direction*
<!-- bilingual-en:end -->

$P(B\mid H_j)$ 回答“假设成立时看到证据的概率”，而 $P(H_j\mid B)$ 回答“看到证据后假设成立的概率”。两者通常不相等；反转方向必须纳入先验和其他可能产生同一证据的假设。
<!-- bilingual-en:start -->
$P(B\mid H_j)$ asks how likely the evidence is if the hypothesis is true, whereas $P(H_j\mid B)$ asks how likely the hypothesis is after seeing the evidence. They are generally unequal; reversing direction requires the prior and the other hypotheses that could produce the same evidence.
<!-- bilingual-en:end -->

零先验概率的分区项不能直接写 $P(B\mid H_\ell)$；在有限初等版本中可将该零概率项从分区中删去，或改用联合概率 $P(B\cap H_\ell)$ 的写法。
<!-- bilingual-en:start -->
A zero-prior partition element does not support the elementary conditional likelihood $P(B\mid H_\ell)$. In the finite elementary setting, omit that null element or write the formula using the joint probability $P(B\cap H_\ell)$ instead.
<!-- bilingual-en:end -->

> [!question]- 自检
> Bayes 法则的分母为什么不能只写 $P(B\mid H_j)$？
> <!-- bilingual-en:start -->
> Why can the denominator in Bayes' rule not be just $P(B\mid H_j)$?
> <!-- bilingual-en:end -->
>
> **答案：** 分母必须包含证据 $B$ 在所有完备假设下出现的总概率，才能把后验归一化。
> <!-- bilingual-en:start -->
> **Answer:** The denominator must include the total probability of evidence $B$ across all exhaustive hypotheses so that the posterior is normalised.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf|MIT 6.042J Bayes' Theorem slides]]：核对 Bayes 法则、全概率分母以及 odds 解释。
- [MIT 18.05 Class 3 preparation](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf)：核对 prior、likelihood 与 posterior 的方向。
<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf|MIT 6.042J Bayes' Theorem slides]] were checked for Bayes' rule, the total-probability denominator, and the odds interpretation.
- [MIT 18.05 Class 3 preparation](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf) was checked for the direction of the prior, likelihood, and posterior.
<!-- bilingual-en:end -->
