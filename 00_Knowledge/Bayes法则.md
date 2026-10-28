---
aliases:
  - Bayes' rule
  - Bayes theorem
  - 贝叶斯法则
student_os: knowledge-atom
atom_id: PROB-BAYES-001
atom_set: probability-foundations
atom_type: theorem
status: source-checked
requires:
  - "[[条件概率]]"
  - "[[全概率公式]]"
part_of:
  - "[[概率空间、条件概率与 Bayes 法则.canvas]]"
---

<!-- bilingual-en:start -->
# Bayes' rule
<!-- bilingual-en:end -->
*Bayes 法则（贝叶斯法则）*

<!-- bilingual-en:start -->
**Bayes' rule** calculates how the probability of a hypothesis changes after we observe some evidence. Suppose we know the defect rates of two factories. Those rates answer “How likely is a defect, given the factory?” If a randomly chosen part turns out to be defective, Bayes' rule answers the reverse question: “Which factory is it likely to have come from?” The factories' supply shares matter as well as their defect rates.
<!-- bilingual-en:end -->

**Bayes 法则**计算看到证据之后，一个假设的概率如何变化。例如，我们知道两家工厂各自的缺陷率，这回答的是“已知来自这家工厂，有缺陷的概率是多少”。现在随机抽到一个零件，发现它有缺陷，Bayes 法则则回答反方向的问题：“它可能来自哪家工厂？”除了缺陷率，还必须考虑各厂原本的供货比例。

<!-- bilingual-en:start -->
## Follow the defective parts back to their source
<!-- bilingual-en:end -->
*沿着缺陷零件，反查来源*

<!-- bilingual-en:start -->
Factory A supplies 60% of a shipment with a 2% defect rate; B supplies the remaining 40% with a 5% defect rate. Imagine 10,000 parts matching those proportions exactly. Let $D$ mean “defective”. The [[概率乘法法则|multiplication rule]] gives the number of defective parts in each group, and the [[全概率公式|law of total probability]] combines the groups:
<!-- bilingual-en:end -->

A 厂供应一批货物的 60%，缺陷率为 2%；B 厂供应其余 40%，缺陷率为 5%。想象 10,000 个恰好符合这些比例的零件，用 $D$ 表示“有缺陷”。[[概率乘法法则]]算出各组中的缺陷零件，[[全概率公式]]把它们合起来：

| Factory / 工厂 | All parts / 全部零件 | Defective / 缺陷零件 |
|---|---:|---:|
| A | 6,000 | $6{,}000\times0.02=120$ |
| B | 4,000 | $4{,}000\times0.05=200$ |
| Total / 合计 | 10,000 | 320 |

> [!example] Among the defects, how many come from B? · 缺陷零件中有多少来自 B？
> <!-- bilingual-en:start -->
> Once we know the selected part is defective, the relevant group contains 320 parts. Of these, 200 came from B. Therefore
> <!-- bilingual-en:end -->
>
> 一旦知道抽中的零件有缺陷，考虑范围就缩小到 320 个缺陷零件，其中 200 个来自 B 厂。因此
>
> $$P(B\mid D)=\frac{200}{320}=0.625.$$
>
> <!-- bilingual-en:start -->
> Before inspecting the part, its probability of coming from B was 40%; after observing a defect, it is 62.5%. This is not B's 5% defect rate: $P(D\mid B)$ and $P(B\mid D)$ ask different questions.
> <!-- bilingual-en:end -->
>
> 检查之前，零件来自 B 厂的概率是 40%；看到缺陷后，这个概率变为 62.5%。它不是 B 厂的 5% 缺陷率：$P(D\mid B)$ 与 $P(B\mid D)$ 问的是不同问题。

<!-- bilingual-en:start -->
The vertical bar means “given”: [[条件概率|conditional probability]] restricts attention to the event after the bar. In the first direction we look within B's parts; in the second we look within defective parts. The numerator counts the same overlap, while the denominator changes.
<!-- bilingual-en:end -->

竖线表示“给定”：[[条件概率]]把范围限制在竖线后面的事件。前一个方向只看 B 厂零件，后一个方向只看缺陷零件。两个分子计算的是同一块重叠部分，改变的是分母所用的范围。

<!-- bilingual-en:start -->
## Reading prior, likelihood, and posterior
<!-- bilingual-en:end -->
*先验、似然、后验各表示什么*

> [!abstract] Bayes' rule for two events · 两个事件的 Bayes 法则
> <!-- bilingual-en:start -->
> For a hypothesis $H$ with $P(H)>0$ and evidence $E$ with $P(E)>0$,
> <!-- bilingual-en:end -->
>
> 假设事件 $H$ 满足 $P(H)>0$，证据事件 $E$ 满足 $P(E)>0$ 时，
>
> $$P(H\mid E)=\frac{P(E\mid H)P(H)}{P(E)}.$$
>
> <!-- bilingual-en:start -->
> $P(H)$ is the **prior**, before using this evidence; $P(E\mid H)$ is the **likelihood**, how probable the evidence is under the hypothesis; $P(H\mid E)$ is the **posterior**, after incorporating the evidence. Dividing by $P(E)$ rescales the relevant joint probability into a share of all cases where the evidence occurs.
> <!-- bilingual-en:end -->
>
> $P(H)$ 是使用这条证据之前的**先验**；$P(E\mid H)$ 是**似然**，即假设成立时证据出现的概率；$P(H\mid E)$ 是纳入证据之后的**后验**。除以 $P(E)$，是把相应的联合概率重新缩放成所有出现该证据的情况中的比例。

<!-- bilingual-en:start -->
The rule follows directly from the two descriptions of the overlap: $P(H\cap E)=P(E\mid H)P(H)$ and $P(H\cap E)=P(H\mid E)P(E)$. Equate them and divide by $P(E)$. For factory B, the numerator is $0.05\times0.40=0.020$ and the denominator is $0.032$, recovering $0.625$.
<!-- bilingual-en:end -->

它直接来自同一个交集的两种写法：$P(H\cap E)=P(E\mid H)P(H)$，以及 $P(H\cap E)=P(H\mid E)P(E)$。令两式相等，再除以 $P(E)$ 即可。对 B 厂来说，分子是 $0.05\times0.40=0.020$，分母为 $0.032$，仍得到 $0.625$。

<!-- bilingual-en:start -->
When $H_1,\ldots,H_k$ are mutually exclusive and exhaustive possibilities, every possible outcome belongs to exactly one of them. If their prior probabilities are positive, total probability supplies the denominator:
<!-- bilingual-en:end -->

若 $H_1,\ldots,H_k$ 是互斥而完备的各种可能性，每个可能结果都恰好属于其中一种。它们的先验概率均为正时，可以用全概率公式展开分母：

$$P(H_j\mid E)=\frac{P(E\mid H_j)P(H_j)}{\sum_{\ell=1}^{k}P(E\mid H_\ell)P(H_\ell)}.$$

<!-- bilingual-en:start -->
The index $j$ identifies the hypothesis being evaluated; $\ell$ runs through all hypotheses in the denominator. The sum combines every way the evidence can occur, including alternatives to $H_j$. This makes the posterior probabilities over all hypotheses add to one.
<!-- bilingual-en:end -->

下标 $j$ 指定正在考察的假设；分母的 $\ell$ 则遍历全部假设。这个求和包括证据出现的所有途径，也包括 $H_j$ 之外的解释，因此所有假设的后验概率才会加起来等于一。

> [!warning] Evidence needs a nonzero denominator · 证据的总概率必须为正
> <!-- bilingual-en:start -->
> If $P(E)=0$, this elementary event formula cannot update on $E$. A zero-prior hypothesis has $P(H\cap E)=0$ and hence posterior zero when $P(E)>0$, but $P(E\mid H)$ is then undefined. Use the joint-probability form $P(H\mid E)=P(H\cap E)/P(E)$, or omit null terms when summing, without multiplying an undefined conditional probability by zero.
> <!-- bilingual-en:end -->
>
> 若 $P(E)=0$，这个初等事件公式不能据此更新。若某假设的先验为零，则 $P(H\cap E)=0$，在 $P(E)>0$ 时后验也为零，但这时 $P(E\mid H)$ 没有定义。应使用联合概率形式 $P(H\mid E)=P(H\cap E)/P(E)$，或在求和中略去零概率项，不能写成未定义的条件概率乘零。

<!-- bilingual-en:start -->
## From a few hypotheses to statistical inference
<!-- bilingual-en:end -->
*从几个假设到统计推断*

<!-- bilingual-en:start -->
In statistical inference, a [[似然函数|likelihood function]] fixes the observed data and compares candidate parameters. It is not generally a probability distribution over those parameters. A [[先验分布|prior distribution]] supplies their probability weights; multiplying and normalising gives a [[后验分布|posterior distribution]]. Continuous parameters use density and integral versions under their own conditions, rather than dividing by the probability of an exact continuous observation.
<!-- bilingual-en:end -->

统计推断中的[[似然函数]]把已观察到的数据固定，比较不同候选参数；它通常不是参数的概率分布。[[先验分布]]提供参数的概率权重，相乘并归一化后得到[[后验分布]]。连续参数需要使用满足相应条件的密度与积分版本，不能直接除以某个精确连续观测的点概率。

<!-- bilingual-en:start -->
For example, two candidate models may assign the observed data likelihoods $0.729$ and $0.001$. Equal priors give the first model posterior $0.729/(0.729+0.001)=729/730$. Priors $0.01$ and $0.99$ instead give $(0.729\times0.01)/(0.729\times0.01+0.001\times0.99)=81/92$. The likelihoods are unchanged; changing the prior changes the posterior. Comparing likelihoods alone therefore does not determine a posterior probability.
<!-- bilingual-en:end -->

例如，两个候选模型对同一数据给出似然 $0.729$ 和 $0.001$。相等先验下，第一个模型的后验为 $0.729/(0.729+0.001)=729/730$；先验换成 $0.01$ 和 $0.99$，则后验变为 $(0.729\times0.01)/(0.729\times0.01+0.001\times0.99)=81/92$。似然没有变，先验改变却使后验改变，因此只比较似然不能确定后验概率。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/gmm|EM 怎样做软分配]]
- [[07_Programme/可视化实验室/实验/fairness|相同错误率为何仍有不同 PPV]]
<!-- visual-lab:links:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf|MIT 6.042J Bayes' Theorem, slides 9–16 and 25]]: supports reversing the conditional direction and including all sources of evidence in the denominator. The factory calculation applies those identities to an explicitly specified model.
- [MIT 18.05 Class 3, §7, pp. 8–9](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf#page=8): supports the derivation from the symmetric intersection and the distinction between opposite conditioning directions.
- [CMU 36-705 Lecture 24, §2, pp. 1–2](https://stat.cmu.edu/~larry/=stat705/Lecture24.pdf): supports the likelihood–prior–posterior roles and normalisation over parameters. Both two-model posterior fractions above are recalculated directly.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf|MIT 6.042J Bayes' Theorem，第 9–16、25 张幻灯片]]：支持条件方向反转，以及分母必须包含证据的全部来源；工厂例子把这些恒等式用于明确给定的模型。
- [MIT 18.05 Class 3，第 7 节，第 8–9 页](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class03-prep.pdf#page=8)：支持从同一交集的两种写法推导法则，以及两个条件方向的区别。
- [CMU 36-705 Lecture 24，第 2 节，第 1–2 页](https://stat.cmu.edu/~larry/=stat705/Lecture24.pdf)：支持似然、先验、后验的角色以及对参数的归一化；本文两个候选模型的后验分数均直接复算。
