---
aliases:
  - "Probability Spaces and Conditional Probability"
  - "条件概率与贝叶斯法则"
  - "Probability Foundations"
  - "概率基础"
status: source-checked
---

# 概率空间、条件概率与 Bayes 法则
<!-- bilingual-en:start -->
*Probability spaces, conditional probability, and Bayes' rule*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用样本空间、随机变量和分布描述不确定性，并把条件信息、平均结果和波动写成可运算对象。
> **具体锚点：** 医疗检测阳性后的患病概率不是检测灵敏度；它还取决于患病率和假阳性率，需要 Bayes 法则。
> **核心难点：** 事件独立、随机变量不相关和条件独立是不同层次；期望是分布加权平均，不是最可能值。
> **为什么重要：** 统计推断、随机过程、计量经济学和风险管理都以此为语法。
> **继续：** 时间结构进入 [[随机过程基础|随机过程、平稳性与遍历性]]；有限样本空间与组合进入 [[离散概率方法：计数、指示变量与集中界|离散概率]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** It uses sample spaces, random variables, and distributions to describe uncertainty, turning conditional information, averages, and variability into objects that can be calculated.
> **Concrete anchor:** The probability of disease after a positive test is not the sensitivity of the test. It also depends on prevalence and the false-positive rate, which is why Bayes' rule is needed.
> **Central difficulty:** Independence of events, lack of correlation between random variables, and conditional independence are different concepts. An expectation is a distribution-weighted average, not the most likely value.
> **Why it matters:** Statistical inference, stochastic processes, econometrics, and risk management all use this language.
> **Continue with:** For time-indexed uncertainty, go to [[随机过程基础|stochastic-process foundations]]. For finite sample spaces and combinatorial methods, go to [[离散概率方法：计数、指示变量与集中界|discrete probability methods]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/05_随机过程/02_随机过程的概念和分类.docx]] 与 [[01_Math/05_随机过程/随机过程信息提取.docx]]：支持课程范围与记号。
> - [MIT OCW 6.262 Discrete Stochastic Processes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/)：核验 Poisson、renewal、Markov 与 martingale 的定义和长期结论。
> - [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] 与本地 MIT 6.042J OCW 材料：支持证明、离散结构、计数和概率。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|course notes on stochastic-process concepts and classification]] and [[01_Math/05_随机过程/随机过程信息提取.docx|extracted stochastic-process course material]] support the course scope and notation.
> - [MIT OCW 6.262 Discrete Stochastic Processes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/) was used to verify definitions and long-run results for Poisson and renewal processes, Markov chains, and martingales.
> - [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] and the local MIT 6.042J OCW materials support proofs, discrete structures, counting, and probability.
<!-- bilingual-en:end -->

## 样本空间、事件与概率
<!-- bilingual-en:start -->
*Sample spaces, events, and probability*
<!-- bilingual-en:end -->

样本空间 $\Omega$ 列出可能结果，事件是其子集，概率测度满足非负、总概率为 1 和可列可加。建模首先要说明一次随机试验的单位；把多次试验混进同一基本结果会导致条件概率分母错误。
<!-- bilingual-en:start -->
The sample space $\Omega$ lists the possible outcomes, and an event is a subset of that space. A probability measure is non-negative, assigns probability one to the whole space, and is countably additive. Modelling should begin by specifying what constitutes one trial; mixing several trials into a single elementary outcome often produces the wrong denominator in conditional probabilities.
<!-- bilingual-en:end -->

三个公理不是计算口诀，而是检查模型是否自洽的底线。若事件 $A_1,A_2,\ldots$ 两两互斥，则 $P(\cup_i A_i)=\sum_iP(A_i)$；若不是互斥事件，直接相加会重复计算交集。补事件公式 $P(A^c)=1-P(A)$ 常比枚举目标事件更省力。
<!-- bilingual-en:start -->
The three axioms are not merely computational rules; they are the minimum consistency requirements for a probability model. For pairwise disjoint events $A_1,A_2,\ldots$, $P(\cup_i A_i)=\sum_iP(A_i)$. Without disjointness, direct addition double-counts intersections. The complement rule $P(A^c)=1-P(A)$ is often easier than enumerating the target event directly.
<!-- bilingual-en:end -->

## 条件概率与 Bayes 法则
<!-- bilingual-en:start -->
*Conditional probability and Bayes' rule*
<!-- bilingual-en:end -->

$P(A\mid B)=P(A\cap B)/P(B)$ 表示在 B 已知发生后重新归一化。Bayes 法则把似然 $P(B\mid A)$ 与先验 $P(A)$ 合成后验。基率低时，即使检测灵敏，阳性中也可能有大量假阳性。
<!-- bilingual-en:start -->
$P(A\mid B)=P(A\cap B)/P(B)$ renormalises probabilities after learning that $B$ occurred. Bayes' rule combines the likelihood $P(B\mid A)$ with the prior $P(A)$ to obtain a posterior probability. When the base rate is low, even a sensitive test may produce many false positives among all positive results.
<!-- bilingual-en:end -->

全概率公式先按互斥且完备的情形 $H_1,\ldots,H_k$ 分层，再把条件概率加权：$P(B)=\sum_jP(B\mid H_j)P(H_j)$。Bayes 法则随后反转条件方向：
<!-- bilingual-en:start -->
The law of total probability first partitions the problem into mutually exclusive and exhaustive cases $H_1,\ldots,H_k$, then averages the conditional probabilities: $P(B)=\sum_jP(B\mid H_j)P(H_j)$. Bayes' rule then reverses the conditioning direction:
<!-- bilingual-en:end -->

$$
P(H_j\mid B)=\frac{P(B\mid H_j)P(H_j)}{\sum_\ell P(B\mid H_\ell)P(H_\ell)}.
$$

## Worked example：医疗检测
<!-- bilingual-en:start -->
*Worked example: medical testing*
<!-- bilingual-en:end -->

设患病率为 1%，灵敏度为 90%，特异度为 95%。在 10,000 人的直观频数表中，约 100 人患病，其中 90 人阳性；9,900 人未患病，其中约 495 人假阳性。因此阳性后的患病概率约为 $90/(90+495)=15.4\%$，远低于 90%。
<!-- bilingual-en:start -->
Suppose prevalence is 1%, sensitivity is 90%, and specificity is 95%. In a natural-frequency table for 10,000 people, about 100 have the disease and 90 of them test positive. Among the 9,900 without the disease, about 495 test positive falsely. The probability of disease after a positive result is therefore about $90/(90+495)=15.4\%$, far below 90%.
<!-- bilingual-en:end -->

这一步最容易错在把 $P(阳性\mid 患病)$ 直接写成 $P(患病\mid 阳性)$。诊断方法是先画二叉树或写频数表，并检查分母是否确实包含“所有阳性者”。
<!-- bilingual-en:start -->
The most common error is to replace $P(\text{positive}\mid\text{disease})$ with $P(\text{disease}\mid\text{positive})$. A reliable diagnostic is to draw a probability tree or a natural-frequency table and verify that the denominator truly contains everyone who tests positive.
<!-- bilingual-en:end -->

## 条件概率、独立与因果提醒
<!-- bilingual-en:start -->
*Conditional probability, independence, and a causal caution*
<!-- bilingual-en:end -->

独立要求 $P(A\cap B)=P(A)P(B)$。两两独立不必共同独立；条件独立也不等于无条件独立。概率依赖本身不说明因果方向。
<!-- bilingual-en:start -->
Independence requires $P(A\cap B)=P(A)P(B)$. Pairwise independence need not imply mutual independence, and conditional independence is not the same as unconditional independence. Probabilistic dependence alone does not establish a causal direction.
<!-- bilingual-en:end -->

若 $P(B)>0$，独立也等价于 $P(A\mid B)=P(A)$：得知 B 不改变 A 的概率。互斥事件若都有正概率则不可能独立，因为一个发生会使另一个概率降为零。判断独立必须依据生成机制或联合概率，而不能因为事件名字“看起来无关”。
<!-- bilingual-en:start -->
When $P(B)>0$, independence is equivalently $P(A\mid B)=P(A)$: learning $B$ does not change the probability of $A$. Two mutually exclusive events with positive probabilities cannot be independent, because observing one reduces the probability of the other to zero. Independence must be justified from the generating mechanism or the joint probabilities, not from event labels that merely sound unrelated.
<!-- bilingual-en:end -->

## 建模步骤与失败诊断
<!-- bilingual-en:start -->
*Modelling workflow and failure diagnostics*
<!-- bilingual-en:end -->

先写基本结果，再写目标事件；确认基本结果是否等可能；若获得新信息，重写条件下的样本空间；最后才选择计数、全概率或 Bayes 法则。算完后检查概率是否落在 $[0,1]$、分区权重是否和为 1、以及后验是否随证据方向合理变化。
<!-- bilingual-en:start -->
First define the elementary outcomes, then the target event. Check whether the elementary outcomes are genuinely equiprobable. When new information arrives, rewrite the conditional sample space before choosing counting, total probability, or Bayes' rule. Finally, check that the result lies in $[0,1]$, that partition weights sum to one, and that the posterior moves in a direction consistent with the evidence.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 为什么 $P(患病\mid 阳性)$ 不等于检测灵敏度？
<!-- bilingual-en:start -->
*Why is $P(\text{disease}\mid\text{positive})$ not the test sensitivity?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 灵敏度是 $P(阳性\mid 患病)$；反向条件还要结合患病率和假阳性率按 Bayes 法则归一化。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Sensitivity is $P(\text{positive}\mid\text{disease})$. Reversing the conditioning direction also requires prevalence and the false-positive rate, combined and renormalised through Bayes' rule.
<!-- bilingual-en:end -->

### 两个正概率事件互斥时，为什么它们不独立？
<!-- bilingual-en:start -->
*Why are two mutually exclusive events with positive probability not independent?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 互斥给 $P(A\cap B)=0$，而正概率给 $P(A)P(B)>0$，不满足独立所需的乘法关系。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Mutual exclusivity gives $P(A\cap B)=0$, while positive marginal probabilities imply $P(A)P(B)>0$, so the multiplicative condition for independence fails.
<!-- bilingual-en:end -->

### 一个后验概率极高，是否自动说明证据具有因果作用？
<!-- bilingual-en:start -->
*Does a very high posterior probability automatically show that the evidence has a causal effect?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能。Bayes 法则更新概率关系，不识别干预后的反事实；因果结论还需要研究设计和因果假设。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. Bayes' rule updates probabilistic relationships; it does not identify intervention counterfactuals. A causal conclusion requires a research design and causal assumptions.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx]] 与 [[01_Math/05_随机过程/随机过程信息提取.docx]]：支持课程范围与记号。
- [MIT OCW 6.262 Discrete Stochastic Processes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/)：核验 Poisson、renewal、Markov 与 martingale 的定义和长期结论。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] 与本地 MIT 6.042J OCW 材料：支持证明、离散结构、计数和概率。
<!-- bilingual-en:start -->
- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|course notes on stochastic-process concepts and classification]] and [[01_Math/05_随机过程/随机过程信息提取.docx|extracted stochastic-process course material]] support the course scope and notation.
- [MIT OCW 6.262 Discrete Stochastic Processes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/) was checked for the course's probability foundations and stochastic-process terminology.
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf]] and the local MIT 6.042J OCW materials support finite probability spaces and discrete probability reasoning.
<!-- bilingual-en:end -->
- [MIT 18.05 Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf)：逐项核验概率公理、条件概率、全概率公式、Bayes 法则和医疗检测型例题。
<!-- bilingual-en:start -->
- [MIT 18.05 Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) was used to verify the probability axioms, conditional probability, the law of total probability, Bayes' rule, and the medical-testing style example.
<!-- bilingual-en:end -->
