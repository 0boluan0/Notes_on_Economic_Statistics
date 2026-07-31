---
aliases:
  - MIT 6.042J Unit 4 Probability
  - MIT 6.042J 概率论
  - Probability
tags:
  - course/MIT-6.042J
  - math/discrete-mathematics
  - topic/probability
course: MIT 6.042J Mathematics for Computer Science
unit: 4
sessions: 28-35
source: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/
status: complete
---

# Unit 4 — Probability

> [!abstract] 本章要解决什么
> 概率不是“凭直觉猜比例”，而是先定义实验、样本空间与概率测度，再在这个模型内做可检查的代数。本章从离散概率树出发，依次建立条件概率、Bayes、独立性、随机变量、期望、方差、集中界、抽样与随机游走。最终目标是能区分三类问题：**事件有多可能、数值平均会在哪里、随机过程长期会去哪里**。
> <!-- bilingual-en:start -->
> Probability is not a matter of guessing proportions by intuition. It begins by defining an experiment, a sample space, and a probability measure, and then performs checkable algebra within that model. Starting from discrete probability trees, this chapter develops conditional probability, Bayes' rule, independence, random variables, expectation, variance, concentration bounds, sampling, and random walks. Its ultimate aim is to distinguish three questions: **How likely is an event? Where is a numerical quantity centred on average? Where does a random process go in the long run?**
> <!-- bilingual-en:end -->

本笔记严格依照 MIT OCW Spring 2015 Unit 4 的官方 block 顺序整理：Session 28 → Session 29 → Problem Set 11 → Session 30 → Session 31 → Session 32 → Problem Set 12 → Session 33 → Session 34 → Session 35。课程入口见 [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)；本地材料总索引见 [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]]；前置章节见 [[03_Counting|Unit 3 — Counting]]。
<!-- bilingual-en:start -->
This notebook is organized in strict accordance with the official block order of MIT OCW Spring 2015 Unit 4: Session 28 → Session 29 → Problem Set 11 → Session 30 → Session 31 → Session 32 → Problem Set 12 → Session 33 → Session 34 → Session 35.  For the course entry, see [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/); for the general index of local materials, see [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]]; and for the preceding section, see [[03_Counting|Unit 3 — Counting]].
<!-- bilingual-en:end -->

> [!info] 题解来源与编号
> - `O28-01` 表示 Session 28 的第 1 个 online feedback prompt；96 个 prompt 均逐个中文转述，并保留可核验计算。
> - `C28-1` 表示 `cp28.pdf` 的 Problem 1。CP 与 Problem Set 的课程包只含题目，所以下文标为**非官方独立题解**。
> - 本地 `cp32.pdf` 的实际内容是图论复习题，与 Session 32 的 expectation 主题不一致。本笔记保留链接并逐题解答，同时明确标注这项来源异常。
> <!-- bilingual-en:start -->
> - `O28-01` denotes the first online-feedback prompt in Session 28. All 96 prompts are restated individually in Chinese, with verifiable calculations preserved.
> - `C28-1` denotes Problem 1 in `cp28.pdf`. The course package provides only the questions for the Classroom Problems and Problem Sets, so the solutions below are labelled **unofficial independent solutions**.
> - The local `cp32.pdf` actually contains graph-theory review questions rather than material on expectation. This notebook preserves the link and answers every question while marking the source mismatch explicitly.
> <!-- bilingual-en:end -->

## 课程导航与覆盖

| 顺序 | 内容 | 官方 blocks | Online prompts | CP 编号题 | 评量 |
|---:|---|---:|---:|---:|---|
| 1 | Session 28：Intro to Discrete Probability | 8 | 11 | 6 | — |
| 2 | Session 29：Conditional Probability | 10 | 6 | 4 | — |
| 3 | Problem Set 11 | — | — | — | 3 题 |
| 4 | Session 30：Independence & Causality | 7 | 5 | 4 | — |
| 5 | Session 31：Random Variables & Density Functions | 8 | 21 | 5 | — |
| 6 | Session 32：Expectation | 13 | 12 | 5 | — |
| 7 | Problem Set 12 | — | — | — | 3 题 |
| 8 | Session 33：Markov & Chebyshev Bounds | 11 | 21 | 5 | — |
| 9 | Session 34：Sampling & Confidence | 10 | 13 | 6 | — |
| 10 | Session 35：Random Walks & PageRank | 5 | 7 | 3 | — |
| **合计** |  | **72** | **96** | **38** | **6 题** |

## 官方 block 顺序速查

| Session | 官方顺序 |
|---:|---|
| 28 | Tree Model → Socks and Shoes → Simplified Monty Hall Tree → Simplify Prize Tree → Sample Spaces → Sum Rule Practice → Addition Rule → Fun With Coins |
| 29 | Conditional Definitions → Dicey Sum → Total Probability → Cavities and Candy → Bayes → Two Boys → Monty Hall → Conditional Probability → Dicey Game → Crocodiles |
| 30 | Independence → Independent Dice → Mutual Independence → Mutual Dice → Independent vs Disjoint → Labeled Balls → Paradox |
| 31 | Bigger Number → RV Independence → Odd Heads → Uniform & Binomial → Late Date → Random Number → PDF to CDF → Dice/Coin |
| 32 | Expectation → Uneven Dice → Expected Heads → Heads exercise → Total Expectation → Dice/Coin → Mean Failure → Machines → Linearity → Fair/Biased → Boards → Great Expectations → Uniform Expectation |
| 33 | Deviation → Don't Expect → Markov → Markov exercise → Chebyshev → TA Brain → Variance → Variance practice → Coins → Bounds → Implications |
| 34 | LLN → Not So Strong → Independent Sampling → Coin Sampling → Birthday → Naboo → Sampling & Confidence → Confidence → Random Sampling → Fingers |
| 35 | Random Walks → Stationary Distributions → PageRank → Random Walks exercise → Random Walks continued |

> [!tip] 四个反复出现的检查问题
> 1. 概率是对哪个样本空间定义的？
> 2. 条件事件的概率是否大于零？
> 3. 使用乘法、方差可加或抽样定理时，独立性到底是哪一种？
> 4. 结论是关于未知常数，还是关于抽样程序产生的随机变量？
> <!-- bilingual-en:start -->
> 1. On which sample space is the probability defined?
> 2. Does the conditioning event have positive probability?
> 3. Exactly which form of independence is required by the multiplication rule, variance additivity, or a sampling theorem?
> 4. Is the conclusion about an unknown fixed constant or about a random variable generated by the sampling procedure?
> <!-- bilingual-en:end -->

---

## Session 28 — Intro to Discrete Probability

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：怎样把随机实验变成数学对象？树上路径概率为什么相乘、互斥叶的概率为什么相加？事件不互斥时如何避免重复计算？
<!-- bilingual-en:start -->
**Learning questions**: How can a random experiment be represented as a mathematical object? Why are probabilities multiplied along a tree path and added across mutually exclusive leaves? How can double counting be avoided when events overlap?
<!-- bilingual-en:end -->

**前置知识**：集合、可数求和、乘法法则与容斥。核心卡片：[[概率空间、条件概率与 Bayes 法则#样本空间、事件与概率|离散概率空间]]、[[概率空间、条件概率与 Bayes 法则#建模步骤与失败诊断|概率四步法]]。
<!-- bilingual-en:start -->
**Prerequisites**: sets, countable sums, the multiplication rule, and inclusion–exclusion. Core notes: [[概率空间、条件概率与 Bayes 法则#样本空间、事件与概率|discrete probability spaces]] and the [[概率空间、条件概率与 Bayes 法则#建模步骤与失败诊断|four-step probability workflow]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session28.pdf#page=1|Session 28 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp28.pdf#page=1|cp28, pp. 1–4]]

### 4.1.1 Tree Model — 从实验到概率树
<!-- bilingual-en:start -->
*4.1.1 Tree Model—from lab to probabilistic tree*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_tree_model.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/dEsFEK4vnV4.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_tree_model.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/dEsFEK4vnV4.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

一个离散概率空间是二元组 $(\Omega,\Pr)$：
<!-- bilingual-en:start -->
A discrete probability space is a pair $(\Omega,\Pr)$:
<!-- bilingual-en:end -->

- $\Omega$ 是互斥且穷尽的 outcomes（结果）集合；
- 每个 $\omega\in\Omega$ 有 $\Pr(\omega)\ge0$；
- $\sum_{\omega\in\Omega}\Pr(\omega)=1$。
<!-- bilingual-en:start -->
- $\Omega$ is a mutually exclusive and exhaustive collection of outcomes;
- $\Pr(\omega)\ge0$ per $\omega\in\Omega$;
- $\sum_{\omega\in\Omega}\Pr(\omega)=1$.
<!-- bilingual-en:end -->

事件 $A\subseteq\Omega$ 的概率定义为
<!-- bilingual-en:start -->
The probability of an event $A\subseteq\Omega$ is defined as
<!-- bilingual-en:end -->

$$
\Pr(A)=\sum_{\omega\in A}\Pr(\omega).
$$

在概率树中，一条根到叶的路径依次发生若干选择。边标签是“已到达当前节点”条件下下一步的概率，故路径概率是边概率之积；一个事件通常含若干互斥叶，故事件概率是相应叶概率之和。这就是四步法：
<!-- bilingual-en:start -->
In a probability tree, a root-to-leaf path records a sequence of choices. Each edge is labelled by the probability of the next step conditional on reaching the current node, so the probability of a path is the product of its edge probabilities. An event usually consists of several mutually exclusive leaves, so its probability is the sum of their probabilities. This yields the four-step workflow:
<!-- bilingual-en:end -->

1. 定义 sample space；
2. 指出目标事件；
3. 给每个 outcome 合法概率；
4. 对目标 outcomes 求和。
<!-- bilingual-en:start -->
1. Define sample space;
2. Indicate the target event;
3. Assign a valid probability to every outcome;
4. Sum the probabilities of the target outcomes.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-sample-space-events.png|900]]

读图：全集框是 $\Omega$，内部区域是事件；先确认基本结果互斥且穷尽，再用“叶上相乘、事件内相加”读取概率。
<!-- bilingual-en:start -->
Read: The full set box is $\Omega$, and the inner area is the event; first confirm that the basic results are mutually exclusive and exhaustive, and then read the probability using "multiply on leaves, add within events."
<!-- bilingual-en:end -->

> [!warning] 结果不等概率时不能用“有利个数/总个数”
> $\Pr(A)=|A|/|\Omega|$ 只在有限且每个 outcome 等概率时成立。概率树的叶深不同、边概率不同，都可能使叶不等概率。
> <!-- bilingual-en:start -->
> $\Pr(A)=|A|/|\Omega|$ is only valid if there is a finite number of equal probabilities per outcome.  Different leaf depths and different edge probabilities of probabilistic trees may lead to unequal probabilities of leaves.
> <!-- bilingual-en:end -->

### 4.1.2 Socks and Shoes — Online O28-01–O28-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.2_socks-and-shoes|4.1.2]]。鞋为 2 黑、3 棕；袜为 3 红、4 棕、6 黑，分别独立均匀抽一双。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.2_socks-and-shoes|4.1.2]].  The shoes are 2 black and 3 brown; the socks are 3 red, 4 brown and 6 black, and one pair is smoked independently and evenly.
<!-- bilingual-en:end -->

> [!question]- O28-01–O28-06 逐题解答
> **O28-01** outcome 应表示什么？答：一对颜色 $(\text{shoe color},\text{sock color})$，因为题目只观察颜色。
>
> **O28-02** outcome 有多少个？鞋色 2 种、袜色 3 种，故 $2\cdot3=6$。
>
> **O28-03** 同色事件含多少个 outcome？只有 $(B,B)$ 与 $(Br,Br)$，共 2 个。
>
> **O28-04** 黑鞋概率：$2/(2+3)=2/5=0.4$。
>
> **O28-05** 红袜概率：$3/(3+4+6)=3/13\approx0.2308$。
>
> **O28-06** 同色概率：
> $$
> \Pr(B,B)+\Pr(Br,Br)
> =\frac25\frac6{13}+\frac35\frac4{13}
> =\frac{24}{65}\approx0.3692.
> $$
> 两项对应互斥颜色结果，所以可相加。
> <!-- bilingual-en:start -->
> What should**O28-01**outcome mean?  A: A pair of colors, $(\text{shoe color},\text{sock color})$, because the topic only looks at colors.
> How many**O28-02**outcome?  Shoe 2, sock 3, so $2\cdot3=6$.
> How many outcome are included in the**O28-03**events of the same color?  Only $(B,B)$ and $(Br,Br)$, 2 total.
> **O28-04**Black Shoe Probability: $2/(2+3)=2/5=0.4$.
> **O28-05**Red Sox probability: $3/(3+4+6)=3/13\approx0.2308$.
> **O28-06**Same Color Probability:
> The two items correspond to mutually exclusive color results, so they can be added.
> <!-- bilingual-en:end -->

### 4.1.3 Simplified Monty Hall Tree — 压缩对称分支
<!-- bilingual-en:start -->
*4.1.3 Simplified Monty Hall Tree—Compressed Symmetric Branching*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplifiedMonty.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L30HPgryd6I.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplifiedMonty.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L30HPgryd6I.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

Monty Hall 的完整树可按“奖品位置—初选—主持人开门—是否换门”展开。若策略只关心输赢，可以把初选状态压缩成：
<!-- bilingual-en:start -->
The full Monty Hall tree can be expanded as “prize location → contestant's initial choice → door opened by the host → stay or switch.” If the strategy cares only about winning or losing, the initial-choice state can be compressed to:
<!-- bilingual-en:end -->

- 初选中奖，概率 $1/3$；换门必输；
- 初选未中，概率 $2/3$；主持人排除另一扇羊门后，换门必胜。
<!-- bilingual-en:start -->
- the initial choice is correct, with probability $1/3$; switching loses;
- the initial choice is wrong, with probability $2/3$; after the host eliminates another goat door, switching wins.
<!-- bilingual-en:end -->

所以 stay 胜率 $1/3$，switch 胜率 $2/3$。压缩树合法的条件是：合并分支在后续行为和目标事件上完全等价，并把被合并概率相加；不能只因为图形大小相似就合并。
<!-- bilingual-en:start -->
stay $1/3$, switch $2/3$.  The condition of the validity of the compressed tree is that the merging branches are completely equivalent to the subsequent behavior and the target event, and the merging probabilities are summed up, and the merging branches cannot be merged just because the graphs are similar in size.
<!-- bilingual-en:end -->

### 4.1.4 Simplify Prize Tree — Online O28-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.4_simplify-prize-tree|4.1.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.4_simplify-prize-tree|4.1.4]].
<!-- bilingual-en:end -->

> [!question]- O28-07
> 哪些理由足以压缩 switch 策略的树？
>
> **答案**：奖品位置后的分支对称；“奖品门与初选门”可压缩为“初选是否命中”；初选命中时 switch 与主持人具体开哪门无关且必输；初选羊门的两个对称分支有同样概率和结局，可合并。错误理由是“所有 outcome 等概率”或“分支画得一样大”，二者都不是概率等价条件。
> <!-- bilingual-en:start -->
> What are the reasons to compress the tree of switch policies?
> **Answer:** The branches following the prize location are symmetric, so “prize door and initial choice” can be reduced to “whether the initial choice is correct.” If it is correct, switching loses regardless of which eligible door the host opens. The two symmetric branches in which the initial choice is wrong have the same probability and outcome, so they may be merged. Neither “all outcomes are equally likely” nor “the branches look equally large” is a valid reason for probabilistic equivalence.
> <!-- bilingual-en:end -->

### 4.1.5 Sample Spaces — 公理与推导规则
<!-- bilingual-en:start -->
*4.1.5 Sample Spaces — Axioms and Derivation Rules*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SampleSpaces.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Amd_bNYzgUw.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SampleSpaces.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Amd_bNYzgUw.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

由定义可以逐步推出常用规则。若 $A\cap B=\varnothing$，则
<!-- bilingual-en:start -->
The standard rules follow directly from the axioms. If $A\cap B=\varnothing$, then
<!-- bilingual-en:end -->

$$
\Pr(A\cup B)=\Pr(A)+\Pr(B).
$$

对一般事件，把 $B$ 拆成 $(B-A)\mathbin{\dot\cup}(A\cap B)$：
<!-- bilingual-en:start -->
For general events, break $B$ into $(B-A)\mathbin{\dot\cup}(A\cap B)$:
<!-- bilingual-en:end -->

$$
\Pr(A\cup B)=\Pr(A)+\Pr(B)-\Pr(A\cap B).
$$

取 $B=\overline A$ 得 complement rule：$\Pr(\overline A)=1-\Pr(A)$。若 $A\subseteq B$，则 $B=A\mathbin{\dot\cup}(B-A)$，故单调性 $\Pr(A)\le\Pr(B)$。
<!-- bilingual-en:start -->
Take $B=\overline A$ and get complement rule:$\Pr(\overline A)=1-\Pr(A)$.  If $A\subseteq B$, then $B=A\mathbin{\dot\cup}(B-A)$, so monotonicity $\Pr(A)\le\Pr(B)$.
<!-- bilingual-en:end -->

并集界（union bound）不要求独立：
<!-- bilingual-en:start -->
Union bound does not require independence:
<!-- bilingual-en:end -->

$$
\Pr\!\left(\bigcup_{i\ge1}A_i\right)\le\sum_{i\ge1}\Pr(A_i).
$$

证明时令 $B_1=A_1$，$B_i=A_i-\bigcup_{j<i}A_j$。则 $B_i$ 两两不交、$B_i\subseteq A_i$，且两边并集相同。因此
<!-- bilingual-en:start -->
$B_1=A_1$, $B_i=A_i-\bigcup_{j<i}A_j$.  then $B_i$ has no intersection, $B_i\subseteq A_i$, and the union of the two sides is the same.  therefore
<!-- bilingual-en:end -->

$$
\Pr\!\left(\bigcup_iA_i\right)=\sum_i\Pr(B_i)\le\sum_i\Pr(A_i).
$$

### 4.1.6 Sum Rule Practice — Online O28-08

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.6_sum-rule-practice|4.1.6]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.6_sum-rule-practice|4.1.6]].
<!-- bilingual-en:end -->

> [!question]- O28-08
> 三个两两互斥事件各有概率 $1/4$，至少一个发生的概率是多少？
>
> **解**：互斥使交集项全为零，故 $\Pr(E_0\cup E_1\cup E_2)=3/4=0.75$。
> <!-- bilingual-en:start -->
> Three pairwise mutually exclusive events each have a probability $1/4$, what is the probability that at least one of them will occur?
> **Solution**: Mutual exclusion makes the intersection terms all zero, so $\Pr(E_0\cup E_1\cup E_2)=3/4=0.75$.
> <!-- bilingual-en:end -->

### 4.1.7 Addition Rule — Online O28-09

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.7_addition-rule|4.1.7]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.7_addition-rule|4.1.7]].
<!-- bilingual-en:end -->

> [!question]- O28-09
> 两颗公平骰子点数和为 7 或 11 的概率？
>
> **解**：36 个等概率有序结果中，和 7 有 6 个、和 11 有 2 个，且二事件互斥：
> $$
> \Pr(S=7\text{ or }11)=\frac6{36}+\frac2{36}=\frac29\approx0.22.
> $$
> <!-- bilingual-en:start -->
> Two fair dice points and a probability of 7 or 11?
> **Solution**: Among the 36 equiprobability ordered results, there are 6 out of the sum of 7, and 2 out of the sum of 11, and the two events are mutually exclusive:
> <!-- bilingual-en:end -->

### 4.1.8 Fun With Coins — Online O28-10–O28-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.8_fun-with-coins|4.1.8]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.8_fun-with-coins|4.1.8]].
<!-- bilingual-en:end -->

> [!question]- O28-10–O28-11
> **O28-10** 公平硬币永远只出反面的概率？前 $n$ 次全反为 $2^{-n}$；“永远全反”包含在每个此前缀事件中，故概率至多 $2^{-n}$ 对所有 $n$ 成立，只能为 $0$。
>
> **O28-11** 无限次投掷至少一次正面的概率？它是上一事件的补集，故 $1-0=1$。概率 1 不表示逻辑上不存在全反序列，而表示该集合测度为 0。
> <!-- bilingual-en:start -->
> Fair Coin Always Has a Bad Chance?  The former $n$ is $2^{-n}$, and the "forever" is contained in each prefix event, so the probability at most $2^{-n}$ is true for all $n$, and can only be $0$.
> **O28-11**Probability of throwing at least one positive pitch indefinitely?  It is a complement to the previous event, so $1-0=1$.  Probability 1 does not mean that there is no inverse sequence logically, but that the set measure is 0.
> <!-- bilingual-en:end -->

### CP28 — 非官方独立题解（6 道）
<!-- bilingual-en:start -->
*CP28 — Unofficial Independent Question (6)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp28.pdf#page=1|cp28]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp28.pdf#page=1|cp28]].
<!-- bilingual-en:end -->

> [!example]- C28-1–C28-6 完整解答
> **C28-1 四门 Monty Hall。** 初选命中奖品概率 $1/4$，所以 stay 胜率 $1/4$。Zelda 初选错误概率 $3/4$；主持人开掉一个空门后，另有两扇未选门且其中恰一扇有奖，均匀选一扇，故
> $$
> \Pr(\text{switch win})=\frac34\cdot\frac12=\frac38.
> $$
> 树的 outcome 可记为 $(P,C,H,F)$；事件分别是 $F=C=P$ 与 $F=P\ne C$。
>
> **C28-2 系统失效界。** 设 $F_i$ 为第 $i$ 个元件失效，$F=\bigcup_iF_i$，且每个 $\Pr(F_i)=p$。
> - 最小值可达：以概率 $p$ 令所有元件同时失效，以 $1-p$ 无元件失效，则 $\Pr(F)=p$。
> - 若 $np\le1$，最大值可达：每个单点 outcome $\{i\}$ 赋概率 $p$，空集赋 $1-np$，则 $\Pr(F)=np$。
> - 任取 $i$，$F_i\subseteq F$，故 $p\le\Pr(F)$；并集界给 $\Pr(F)\le\sum_i p=np$。一般上界应写 $\min(1,np)$。
>
> **C28-3 重启硬币赛。** 记 $q=1-p$，第一人一轮以 $HT$ 获胜概率 $pq$，第二人以 $TH$ 获胜也是 $pq$，重启概率 $p^2+q^2$。令第一人最终胜率为 $s$：
> $$s=pq+(p^2+q^2)s,\qquad s=\frac{pq}{1-p^2-q^2}=\frac12$$
> 对 $0<p<1$。永不决出概率为 $\lim_n(p^2+q^2)^n=0$。边界 $p\in\{0,1\}$ 时每轮必重启，永不决出概率为 1。
>
> **C28-4 可数并集界。** 定义 $B_n=A_n-\bigcup_{i<n}A_i$；$B_n$ 两两互斥、$B_n\subseteq A_n$，且 $\bigcup B_n=\bigcup A_n$。于是
> $$\Pr(\bigcup A_n)=\sum\Pr(B_n)\le\sum\Pr(A_n).$$
>
> **C28-5 五条概率规则。** 由 $A=(A-B)\mathbin{\dot\cup}(A\cap B)$ 得 difference rule；由 $\Omega=A\mathbin{\dot\cup}\bar A$ 得 complement；将 $A\cup B$ 拆为 $A\mathbin{\dot\cup}(B-A)$ 得二项容斥；舍去非负交集项得二事件 union bound；由 $B=A\mathbin{\dot\cup}(B-A)$ 得 $A\subseteq B\Rightarrow\Pr(A)\le\Pr(B)$。
>
> **C28-6 三局两胜。** 令红袜队单局胜率 $p=3/5$、对手 $q=2/5$。
> - 打满三局要求前两局各胜一次：$2pq=12/25$。
> - 系列冠军输掉首局的序列为 $LRR$ 或 $RLL$：$qp^2+pq^2=pq=6/25$。
> - 较强的红袜队赢系列：$p^2+2p^2q=81/125$。
> <!-- bilingual-en:start -->
> **C28-1 Four-door Monty Hall.** The initial choice is correct with probability $1/4$, so staying wins with probability $1/4$. It is wrong with probability $3/4$; after the host opens an empty door, the prize is among the two remaining unchosen doors.
> The outcome of the tree can be recorded as $(P,C,H,F)$; the events are $F=C=P$ and $F=P\ne C$, respectively.
> **C28-2 system failure bound.**Let $F_i$ be the $i$ component that fails, $F=\bigcup_iF_i$, and each $\Pr(F_i)=p$.
> - Minimum up to: $\Pr(F)=p$ if all components fail at the same time with probability $p$, or if no components fail with $1-p$.
> - If $np\le1$, the maximum value can be: each single point outcome $\{i\}$ assigns a probability of $p$, empty set assigns $1-np$, then $\Pr(F)=np$.
> - $i$, $F_i\subseteq F$, so $p\le\Pr(F)$; and bound to $\Pr(F)\le\sum_i p=np$.  The general upper bound should be $\min(1,np)$.
> **C28-3 restarts the coin competition.**$q=1-p$, $pq$ for the first man in a round with a $HT$ win, $pq$ for the second man with a $TH$ win, $p^2+q^2$ for the restart.  The first man's final win was $s$:
> $$s=pq+(p^2+q^2)s,\qquad s=\frac{pq}{1-p^2-q^2}=\frac12$$
> Yes, $0<p<1$.  $\lim_n(p^2+q^2)^n=0$.  Boundary $p\in\{0,1\}$ must be restarted in each round, never determining the probability of 1.
> **C28-4 countable union bound.**Defines $B_n=A_n-\bigcup_{i<n}A_i$; $B_n$ is mutually exclusive, $B_n\subseteq A_n$, and $\bigcup B_n=\bigcup A_n$.  therefore
> $$\Pr(\bigcup A_n)=\sum\Pr(B_n)\le\sum\Pr(A_n).$$
> **C28-5 Five rules of probability.**difference rule from $A=(A-B)\mathbin{\dot\cup}(A\cap B)$; complement from $\Omega=A\mathbin{\dot\cup}\bar A$; binomial exclusion from splitting $A\cup B$ into $A\mathbin{\dot\cup}(B-A)$; binomial event union bound from discarding nonnegative intersection terms; $A\subseteq B\Rightarrow\Pr(A)\le\Pr(B)$ from $B=A\mathbin{\dot\cup}(B-A)$.
> **C28-6 Best-of-three series.** Let the Red Sox win one game with probability $p=3/5$, and let the opponent's win probability be $q=2/5$.
> - Three full innings, one for each of the first two: $2pq=12/25$.
> - Series champions who lose their first game are $LRR$ or $RLL$:$qp^2+pq^2=pq=6/25$.
> - The stronger Red Sox win series: $p^2+2p^2q=81/125$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 两个不互斥事件能否直接相加概率？缺少什么修正项？
> 2. 并集界为什么不要求独立性？
> 3. 概率为 0 是否等于逻辑上不可能？
>
> <!-- bilingual-en:start -->
> 1. Can the probabilities of two non-disjoint events simply be added? What correction term is needed?
> 2. Why does the union bound not require independence?
> 3. Does probability zero mean logical impossibility?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 不能；减去 $\Pr(A\cap B)$。2. 证明只用不交化与单调性。3. 不等于；无限样本空间中可有非空零测事件。
> <!-- bilingual-en:start -->
> 1. No; subtract $\Pr(A\cap B)$. 2. Its proof uses only disjointification and monotonicity. 3. No; an infinite sample space can contain nonempty events of probability zero.
> <!-- bilingual-en:end -->

**知识链**：[[概率空间、条件概率与 Bayes 法则#样本空间、事件与概率|离散概率空间]] → [[概率空间、条件概率与 Bayes 法则#建模步骤与失败诊断|概率四步法]] → sum/addition/complement rules → union bound → 条件概率。
<!-- bilingual-en:start -->
**Knowledge Chain**:[[概率空间、条件概率与 Bayes 法则#样本空间、事件与概率|discrete probability space]] → [[概率空间、条件概率与 Bayes 法则#建模步骤与失败诊断|probabilistic four-step method]] → sum/addition/complement rules → union bound→Conditional Probability.
<!-- bilingual-en:end -->

---

## Session 29 — Conditional Probability

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：得知新信息后为何要缩小样本空间？怎样从正向诊断率反推出患病后验？Monty Hall 与“两孩问题”为什么必须说明信息产生机制？
<!-- bilingual-en:start -->
**Learning Problem**: Why shrink the sample space when new information is known?  How to derive the post-mortem of illness from the positive diagnostic rate?  Why does Monty Hall and the two-child problem have to explain the information generation mechanism?
<!-- bilingual-en:end -->

**前置知识**：事件交并、概率树与 partition。核心卡片：[[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|条件概率与 Bayes 定理]]。
<!-- bilingual-en:start -->
**Prerequisites**: event intersection, probability tree, and partition.  Core card: [[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|Conditional Probability and Bayes Theorem]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session29.pdf#page=1|Session 29 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp29.pdf#page=1|cp29, pp. 1–3]]

### 4.2.1 Conditional Probability Definitions

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ConditProbability.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Cu9_LaaWgHo.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ConditProbability.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Cu9_LaaWgHo.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

若 $\Pr(B)>0$，在已知 $B$ 发生后，$A$ 的条件概率定义为
<!-- bilingual-en:start -->
If $\Pr(B)>0$, after a known $B$ occurs, the conditional probability of $A$ is defined as
<!-- bilingual-en:end -->

$$
\Pr(A\mid B)=\frac{\Pr(A\cap B)}{\Pr(B)}.
$$

分母把原概率空间重新归一化到 $B$；若 $\Pr(B)=0$，这个初等定义没有意义。移项得到 multiplication rule：
<!-- bilingual-en:start -->
The denominator renormalizes the original probability space to $B$; if $\Pr(B)=0$, this elementary definition is meaningless.  Shift term to get multiplication rule:
<!-- bilingual-en:end -->

$$
\Pr(A\cap B)=\Pr(A\mid B)\Pr(B)
=\Pr(B\mid A)\Pr(A).
$$

连续多步树则有 chain rule：
<!-- bilingual-en:start -->
For a sequence of events, the chain rule gives
<!-- bilingual-en:end -->

$$
\Pr(A_1\cap\cdots\cap A_n)
=\Pr(A_1)\prod_{i=2}^{n}\Pr(A_i\mid A_1\cap\cdots\cap A_{i-1}),
$$

前提是出现的条件事件概率非零。
<!-- bilingual-en:start -->
provided that every conditioning event appearing in the formula has positive probability.
<!-- bilingual-en:end -->

### 4.2.2 Dicey Sum — Online O29-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.2_dicey-sum|4.2.2]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.2_dicey-sum|4.2.2]].
<!-- bilingual-en:end -->

> [!question]- O29-01
> 已知两骰点数和为 4，至少一颗为 3 的概率？条件空间只有 $(1,3),(2,2),(3,1)$ 三个等概率结果，其中两个含 3，故答案 $2/3\approx0.67$。
> <!-- bilingual-en:start -->
> Do you know the probability that two dice points are 4 and at least one is 3?  Conditional space has only $(1,3),(2,2),(3,1)$ three equal probability results, two of which contain 3, so the answer is $2/3\approx0.67$.
> <!-- bilingual-en:end -->

### 4.2.3 Law of Total Probability

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawTotalProbab.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/F3y8qupFfUs.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawTotalProbab.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/F3y8qupFfUs.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

若 $B_1,\dots,B_m$ 两两互斥、并为 $\Omega$，且所用 $\Pr(B_i)>0$，则
<!-- bilingual-en:start -->
If $B_1,\dots,B_m$ is mutually exclusive and $\Omega$ and $\Pr(B_i)>0$ is used, then
<!-- bilingual-en:end -->

$$
\Pr(A)=\sum_{i=1}^{m}\Pr(A\cap B_i)
=\sum_{i=1}^{m}\Pr(A\mid B_i)\Pr(B_i).
$$

这是“按原因分支，再把到达同一结果的路径相加”。若某 $B_i$ 的概率为 0，可直接省略该项，而不是定义 $\Pr(A\mid B_i)$。
<!-- bilingual-en:start -->
This is "branch by cause and add the paths to the same result."  If the probability of a $B_i$ is 0, you can omit it it directly instead of defining the $\Pr(A\mid B_i)$.
<!-- bilingual-en:end -->

### 4.2.4 Cavities and Candy — Online O29-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.4_cavities-and-candy|4.2.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.4_cavities-and-candy|4.2.4]].
<!-- bilingual-en:end -->

> [!question]- O29-02
> $\Pr(C)=1/4$，$\Pr(E\mid C)=4/5$，$\Pr(E\mid\bar C)=1/3$。求吃糖概率。
>
> $$
> \Pr(E)=\frac45\frac14+\frac13\frac34=\frac9{20}=0.45.
> $$
> 第二项必须乘 $\Pr(\bar C)=3/4$；官方反馈文字有一处把它误写成 $\Pr(C)$，但数值计算使用的是补事件。
> <!-- bilingual-en:start -->
> $\Pr(C)=1/4$, $\Pr(E\mid C)=4/5$, and $\Pr(E\mid\bar C)=1/3$. By the law of total probability,
> The second term must be multiplied by $\Pr(\bar C)=3/4$; the official feedback text incorrectly states it as $\Pr(C)$ in one place, but the numerical calculation uses supplementary events.
> <!-- bilingual-en:end -->

### 4.2.5 Bayes' Theorem

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/e-yQFC6dACA.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/e-yQFC6dACA.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

由交集概率的两种分解得到
<!-- bilingual-en:start -->
It is obtained from two factorizations of intersection probability
<!-- bilingual-en:end -->

$$
\Pr(B_j\mid A)
=\frac{\Pr(A\mid B_j)\Pr(B_j)}{\sum_i\Pr(A\mid B_i)\Pr(B_i)},
$$

其中 $\{B_i\}$ 是 partition，且 $\Pr(A)>0$。分子是“likelihood × prior”，分母是所有原因对证据的总贡献。罕见病检测中，即使 sensitivity 很高，若先验患病率很低，false positives 仍可能主导分母。
<!-- bilingual-en:start -->
where $\{B_i\}$ is partition and $\Pr(A)>0$.  The numerator is "likelihood × prior" and the denominator is the total contribution of all causes to the evidence.  In rare disease testing, even if sensitivity is high, false positives may dominate the denominator if a priori prevalence is low.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-bayes-tree.png|900]]

读图：从每个原因分支乘到证据叶得到联合概率，再用所有证据叶之和归一化；后验不是把条件符号简单倒过来。
<!-- bilingual-en:start -->
Read: Multiply each cause branch by the evidence leaf to obtain a joint probability, and then normalize with the sum of all the evidence leaves; the posterior is not simply inverting the conditional symbol.
<!-- bilingual-en:end -->

### 4.2.6 Two Boys — Online O29-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.6_two-boys|4.2.6]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.6_two-boys|4.2.6]].
<!-- bilingual-en:end -->

> [!question]- O29-03
> 两个孩子中“至少一个是男孩”，两男概率？在出生次序等概率且信息就是事件 $B=\{BB,BG,GB\}$ 时，条件空间三项等概率，故 $\Pr(BB\mid B)=1/3$。
>
> 边界：若信息是“较大的孩子是男孩”，条件空间只有 $BB,BG$，答案为 $1/2$；所以必须说明观察协议。
> <!-- bilingual-en:start -->
> "At least one of the two kids is a boy." Two boys?  When the birth order is equal to the probability and the information is the event $B=\{BB,BG,GB\}$, the three terms in conditional space are equal to the probability, so $\Pr(BB\mid B)=1/3$.
> Boundary: If the information is "the older child is a boy", the conditional space is only $BB,BG$ and the answer is $1/2$; therefore, the observation protocol must be specified.
> <!-- bilingual-en:end -->

### 4.2.7 Monty Hall Problem — 信息协议决定后验
<!-- bilingual-en:start -->
*4.2.7 Monty Hall Problem—Informational Protocol Decision Post*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MontyHallConfus.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BEAv82FinM0.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MontyHallConfus.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BEAv82FinM0.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

经典的 **switch 策略在游戏开始前的胜率**只依赖主持人知道奖品位置、一定打开未选羊门、一定给换门机会：初选错误的概率是 $2/3$，而初选一旦错误，主持人只能排除另一扇羊门，换门必胜，所以 switch 胜率为 $2/3$。
<!-- bilingual-en:start -->
The classical **pre-game winning probability of the switching strategy** relies only on the host knowing the prize location, always opening an unchosen goat door, and always offering the opportunity to switch. The initial choice is wrong with probability $2/3$; whenever it is wrong, the host can only eliminate the other goat door, so switching must win. Therefore the switching strategy wins with probability $2/3$.
<!-- bilingual-en:end -->

但若还要在看见“主持人具体打开了哪一扇门”后计算后验，就必须知道他的开门协议。若他在两扇可开羊门中均匀随机选择，则观察任一具体开门结果后，初选门仍有后验 $1/3$，唯一未开的门有 $2/3$；若他按不对称的确定规则选门，这两个条件后验可能改变，尽管事前 switch 胜率仍是 $2/3$。若主持人可能随机开到奖品、可能不提供换门，或其行为还依赖其他信息，就必须重建概率树，不能沿用经典结论。
<!-- bilingual-en:start -->
To compute a posterior probability after observing exactly which door the host opened, however, we must know the host's door-opening protocol. If the host chooses uniformly between two eligible goat doors, then after either possible observation the initially chosen door still has posterior probability $1/3$ and the only other unopened door has probability $2/3$. Under an asymmetric deterministic rule, those conditional probabilities can change even though the ex ante switching win rate remains $2/3$. If the host might reveal the prize, might not offer a switch, or uses additional information, the probability tree must be rebuilt; the classical conclusion cannot simply be reused.
<!-- bilingual-en:end -->

### 4.2.8 Conditional Probability — Online O29-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.8_conditional-probability|4.2.8]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.8_conditional-probability|4.2.8]].
<!-- bilingual-en:end -->

> [!question]- O29-04
> $\Pr(A)=0.8$、$\Pr(L)=0.4$、$\Pr(L\mid A)=0.3$，求 $\Pr(A\mid L)$。
>
> $$
> \Pr(A\mid L)=\frac{\Pr(L\mid A)\Pr(A)}{\Pr(L)}
> =\frac{0.3\cdot0.8}{0.4}=0.6.
> $$
> <!-- bilingual-en:start -->
> $\Pr(A)=0.8$, $\Pr(L)=0.4$, $\Pr(L\mid A)=0.3$, $\Pr(A\mid L)$.
> <!-- bilingual-en:end -->

### 4.2.9 Dicey Game — Online O29-05

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.9_dicey-game|4.2.9]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.9_dicey-game|4.2.9]].
<!-- bilingual-en:end -->

> [!question]- O29-05
> 两骰和 7 则赢，double 则重投，其他输，最多三投；已知首投 double，最终赢率？条件信息等于已经消耗一次，剩两投。每投赢 $1/6$、double $1/6$，故
> $$
> \Pr(W\mid\text{first pair})=\frac16+\frac16\frac16=\frac7{36}.
> $$
> <!-- bilingual-en:start -->
> Two dice and seven wins, double restarts, the other loses, up to three; double is known to be the first to win?  The condition information is equal to one time already consumed, with two shots remaining.  $1/6$, double $1/6$ per win, so
> <!-- bilingual-en:end -->

### 4.2.10 Watch Out for Crocodiles — Online O29-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.10_watch-out-for-crocodiles|4.2.10]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.10_watch-out-for-crocodiles|4.2.10]].
<!-- bilingual-en:end -->

> [!question]- O29-06
> 雨、晴、冰雹概率分别 $1/4,1/4,1/2$；见鳄鱼条件概率分别 $1/2,5/8,1$。由全概率公式
> $$
> \Pr(C)=\frac14\frac12+\frac14\frac58+\frac12=\frac{25}{32}.
> $$
> <!-- bilingual-en:start -->
> $1/4,1/4,1/2$ for rain, $1/2,5/8,1$ for sunny hail and for crocodile.  By the total probability formula
> <!-- bilingual-en:end -->

### CP29 — 非官方独立题解（4 道）
<!-- bilingual-en:start -->
*CP29 — Unofficial Independent Questions (4)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp29.pdf#page=1|cp29]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp29.pdf#page=1|cp29]].
<!-- bilingual-en:end -->

> [!example]- C29-1–C29-4 完整解答
> **C29-1 Beaver Fever。** 已知 $\Pr(B)=0.001$、sensitivity $\Pr(Y\mid B)=0.99$、specificity $\Pr(\bar Y\mid\bar B)=0.97$，故 false-positive rate 为 $0.03$。
> $$
> \Pr(Y)=0.99(0.001)+0.03(0.999)=0.03096,
> $$
> $$
> \Pr(B\mid Y)=\frac{0.99(0.001)}{0.03096}=\frac{11}{344}\approx0.03198.
> $$
> 若疫苗只够 2% 人口，而阳性约占 3.096%，就在所有阳性者中随机抽取比例 $0.02/0.03096$ 接种。真患者先以概率 $0.99$ 被检出，因此被覆盖的期望比例是
> $$
> \Pr(\text{vaccinated}\mid B)
> =0.99\frac{0.02}{0.03096}
> \approx0.6395,
> $$
> 即约 $64.0\%$，远高于不检测而随机接种时的 2%。不能漏掉 sensitivity $\Pr(Y\mid B)=0.99$；$64.6\%$ 只是“阳性者中被抽中接种”的比例。
>
> **C29-2 三囚犯。** 三个获释二人组各概率 $1/3$。守卫说 Foo-Foo 的概率贡献为：$\{S,F\}$ 分支 $1/3$，$\{F,V\}$ 分支因二选一贡献 $1/6$。故
> $$
> \Pr(S\mid\text{guard says }F)=\frac{1/3}{1/3+1/6}=\frac23.
> $$
> Sauron 把两个后验候选误当成等概率，忽略了守卫发言机制。
>
> **C29-3 两副牌。** 完整牌与缺黑桃 A 的牌先验各 $1/2$；抽到红心 8 的 likelihood 分别 $1/52,1/51$：
> $$
> \Pr(\text{complete}\mid8\heartsuit)
> =\frac{(1/2)(1/52)}{(1/2)(1/52)+(1/2)(1/51)}=\frac{51}{103}.
> $$
>
> **C29-4 HTT 与 HHT 竞赛。** 令 $a,b,c$ 分别为当前最长有效后缀为 $H,HT,HH$ 时 HTT 先出现的概率。则
> $$
> b=\tfrac12(1)+\tfrac12a,\qquad c=\tfrac12c+\tfrac12(0)=0,\qquad a=\tfrac12b+\tfrac12c.
> $$
> 解得 $a=1/3$。开始前的 $T$ 不改变状态，第一次 $H$ 后进入 $a$，所以 HTT 先出现概率为 $1/3$，HHT 为 $2/3$。
> <!-- bilingual-en:start -->
> **C29-1 Beaver Fever.**$\Pr(B)=0.001$, sensitivity $\Pr(Y\mid B)=0.99$, specificity $\Pr(\bar Y\mid\bar B)=0.97$ are known, so false-positive rate is $0.03$.
> If only 2% of the population has been vaccinated, and the positive rate is about 3.096%, a random sample of all positive people will be vaccinated with $0.02/0.03096$.  The true patient is first detected with a probability $0.99$, so the expected proportion to be covered is
> That is about $64.0\%$, much higher than the 2% rate when the vaccine was randomly administered without testing.  sensitivity $\Pr(Y\mid B)=0.99$ should not be missed; $64.6\%$ is only a proportion of "those who are positive have been vaccinated".
> **C29-2, prisoner three.**$1/3$ for each of the three released two.  The guard said that the probability contribution of Foo-Foo is $1/3$ for the $\{S,F\}$ branch and $1/6$ for the $\{F,V\}$ branch.
> Sauron mistook the two posterior candidates for equal probability and ignored the guarded speech mechanism.
> Two decks of**C29-3.**Full and missing ace of spades $1/2$ apiece; drawn to 8 of hearts likelihood $1/52,1/51$:
> **C29-4 HTT and HHT competition.**The probability that HTT first occurs when $a,b,c$ is respectively the current longest valid suffix of $H,HT,HH$.
> $a=1/3$.  $T$ does not change the state before the start, after the first $H$ into $a$, so the probability of HTT first appeared is $1/3$, HHT is $2/3$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 为什么 $\Pr(A\mid B)$ 要求 $\Pr(B)>0$？
> 2. sensitivity 为 99% 是否足以推出阳性者 99% 患病？
> 3. Monty Hall 的主持人若在剩余门中随机开一扇、并可能开到奖品，原结论还能直接用吗？
>
> <!-- bilingual-en:start -->
> 1. Why does $\Pr(A\mid B)$ require $\Pr(B)>0$?
> 2. Is a 99% sensitivity sufficient to cause 99% of those who are positive to fall ill?
> 3. If the host of Monty Hall opened a random door in the remaining doors and could have given the prize, could the original conclusion be used directly?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 要除以 $\Pr(B)$ 并重新归一化。2. 不足，还需患病先验与 false-positive rate。3. 不能；“主持人必定避开奖品”这一 likelihood 已被改变，必须重建概率树。
> <!-- bilingual-en:start -->
> 1. Divide by $\Pr(B)$ and re-normalize.  2. Insufficient, illness prior and false-positive rate are also required.  3. No; the likelihood "Moderators must avoid prizes" has been changed and the probability tree must be rebuilt.
> <!-- bilingual-en:end -->

**知识链**：条件空间 → multiplication rule → total probability → [[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|Bayes]] → 信息协议 → independence。
<!-- bilingual-en:start -->
**Knowledge Chain**: Conditional Space → multiplication rule → total probability → [[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|Bayes]] → Information Protocol → independence.
<!-- bilingual-en:end -->

---

## Problem Set 11 — Discrete & Conditional Probability

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps11.pdf#page=1|Problem Set 11, pp. 1–3]]。以下为非官方独立题解。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps11.pdf#page=1|Problem Set 11, pp. 1–3]].  The following are unofficial independent questions.
<!-- bilingual-en:end -->

> [!example]- P11-1 三色矩阵中的单色矩形
> **(a)** 长度 4 的一行共有 $3^4=81$ 种颜色 pattern；82 行由 pigeonhole principle 至少两行完全相同。
>
> **(b)** 在这两条相同行中，4 个列位置只用 3 色，至少两列同色；这两列与两行的四个交点同色，形成矩形。
>
> **(c)** 每一行中至少有一对位置同色。用“颜色 + 两列位置”标记一行，可选标签数为
> $$3\binom42=18.$$
> 19 行中至少两行共享同一标签，即同一对列上出现同一种颜色，仍形成单色矩形。若一行有多个标签，任选一个固定规则产生的标签即可应用 pigeonhole。
> <!-- bilingual-en:start -->
> **(a)**A line of length 4 has a total of $3^4=81$ colors pattern; 82 lines are identical by pigeonhole principle at least two lines.
> **(b)**In these two identical rows, the four column positions are three colors only, and at least two columns are the same color; the two columns are the same color as the four intersections of the two rows, forming a rectangle.
> **(c)**At least one pair of position identically in each row.  Mark a row with Color + Two Column Position, with the number of optional labels
> $$3\binom42=18.$$
> At least two of the 19 rows share the same label, meaning that the same color appears on the same pair of columns, still forming a monochrome rectangle.  If there is more than one label in a row, the pigeonhole can be applied to any label generated by a fixed rule.
> <!-- bilingual-en:end -->

> [!example]- P11-2 红黑牌的最优停时
> 对含 $r$ 张红牌、$b$ 张黑牌的随机牌堆，证明任意策略胜率至多 $b/(r+b)$。对 $n=r+b$ 归纳。$n=1$ 时唯一牌若黑则 $b/n=1$ 且必胜，若红则 $b/n=0$ 且必败，基例成立。立即取牌的胜率正是 $b/n$；若跳过首张，再按任意最优后续策略，归纳假设给
> $$
> \Pr(\text{win after skip})
> \le\frac rn\frac b{n-1}+\frac bn\frac{b-1}{n-1}
> =\frac bn.
> $$
> 第一项是先揭红牌，第二项是先揭黑牌。任何随机化策略只是“取/跳”两者的凸组合，也不能超过 $b/n$。因此 26 红 26 黑时最优胜率 $1/2$，立即取顶牌已经达到。
> <!-- bilingual-en:start -->
> For random stacks of $r$ red cards and $b$ black cards, it is proved that the winning probability of any strategy is up to $b/(r+b)$.  Summarize $n=r+b$.  When $n=1$, the only card is $b/n=1$ if it is black and $b/n=0$ if it is red, and the base case is true.  The winning probability of winning the card immediately is exactly $b/n$. If the first card is skipped, and then any optimal follow-up strategy is applied, the hypotheses given in the following paragraph are summarized.
> The first is to unveil the red card. The second is to unveil the black card.  Any randomization strategy is only a convex combination of "pick/hop" and cannot exceed $b/n$.  So 26 red 26 black best chance of winning $1/2$, immediately top has reached.
> <!-- bilingual-en:end -->

> [!example]- P11-3 两张手牌与条件信息
> 记两张 A 为 $A_1,A_2$，第三张为 $J$。把 outcome 写成“手牌 + 被揭牌”，共有 6 个等概率结果：$A_1A_2$ 各揭一张、$A_1J$ 各揭一张、$A_2J$ 各揭一张。
>
> **(a–b)** $K\ge1$ 包含全部 6 项，条件下 $K=2$ 占 2 项，概率 $1/3$；$A_1$ 在手中含 4 项，其中 2 项 $K=2$，概率 $1/2$；揭出 $A_1$ 含 2 项，其中 1 项 $K=2$，概率 $1/2$；揭出任一 A 含 4 项，其中 2 项 $K=2$，概率 $1/2$。
>
> **(c)** 一副 $d$ 张不同牌随机取 $h$ 张，固定 $A_1$ 入手概率
> $$
> \frac{\binom{d-1}{h-1}}{\binom dh}=\frac hd.
> $$
>
> **(d)** 若共有 $a$ 种 A，则 $\Pr(A_1\text{ in hand}\mid K=2)=2/a$。Bayes 给
> $$
> \Pr(K=2\mid A_1\text{ in hand})
> =\Pr(K=2)\frac{2/a}{h/d}
> =\Pr(K=2)\frac{2d}{ah}.
> $$
>
> **(e)** $\Pr(\text{revealed Ace}\mid K=2)=2/h$，而揭出的牌边际上是全牌堆均匀一张，故 $\Pr(\text{revealed Ace})=a/d$。再次用 Bayes 得同一表达式 $\Pr(K=2)2d/(ah)$，所以两个条件概率相等。
> <!-- bilingual-en:start -->
> Two A's for $A_1,A_2$, and the third for $J$.  The outcome is described as "hand and unveiled" with six equal probability outcomes: one for $A_1A_2$, one for $A_1J$, and one for $A_2J$.
> **(a-b)**$K\ge1$ contains all six terms, under the condition that $K=2$ contains two terms with a probability of $1/3$; $A_1$ contains four terms, two of which are $K=2$ with a probability of $1/2$; revealed $A_1$ contains two terms, one of which is $K=2$ with a probability of $1/2$; revealed A contains four terms, two of which are $K=2$ with a probability of $1/2$.
> **(c)**One pair of $d$ cards with different cards randomly selected $h$ cards, fixed $A_1$ starting probability
> **(d)**$\Pr(A_1\text{ in hand}\mid K=2)=2/a$ if there are $a$ of A.  Bayes, here
> **(e)** $\Pr(\text{revealed Ace}\mid K=2)=2/h$. Marginally, the revealed card is uniformly distributed over the deck, so $\Pr(\text{revealed Ace})=a/d$. Bayes' rule again gives $\Pr(K=2)2d/(ah)$, proving that the two conditional probabilities are equal.
> <!-- bilingual-en:end -->

---

## Session 30 — Independence & Causality

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：两个事件“不互相提供信息”如何写成公式？pairwise independence 为什么不够？统计相关为什么不自动等于因果关系？
<!-- bilingual-en:start -->
**Learning Questions**: How do two events "don't inform each other" be formulated?  Why isn't pairwise independence enough?  Why is statistical correlation not automatically equal to causation?
<!-- bilingual-en:end -->

**前置知识**：条件概率、Bayes、乘法法则。核心卡片：[[概率空间、条件概率与 Bayes 法则#条件概率、独立与因果提醒|事件独立性]]、[[条件期望与独立性#独立、不相关与条件期望|相互独立]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: conditional probability, Bayes, multiplication rule.  Core cards: [[概率空间、条件概率与 Bayes 法则#条件概率、独立与因果提醒|event independence]], [[条件期望与独立性#独立、不相关与条件期望|be independent of each other]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session30.pdf#page=1|Session 30 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp30.pdf#page=1|cp30, pp. 1–4]]

### 4.3.1 Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Independence.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/1vQ2x5O_xqk.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Independence.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/1vQ2x5O_xqk.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

事件 $A,B$ 独立，定义为
<!-- bilingual-en:start -->
events $A,B$ independent, defined as
<!-- bilingual-en:end -->

$$
\Pr(A\cap B)=\Pr(A)\Pr(B).
$$

若 $\Pr(B)>0$，它等价于 $\Pr(A\mid B)=\Pr(A)$；乘积定义更基本，因为即使 $\Pr(B)=0$ 也合法。独立性会传给补事件，例如
<!-- bilingual-en:start -->
If $\Pr(B)>0$, it is equivalent to $\Pr(A\mid B)=\Pr(A)$; the product definition is more basic because even $\Pr(B)=0$ is legal.  Independence is passed to the replenishment event, such as
<!-- bilingual-en:end -->

$$
\Pr(A\cap\bar B)=\Pr(A)-\Pr(A\cap B)
=\Pr(A)[1-\Pr(B)]=\Pr(A)\Pr(\bar B).
$$

互斥与独立方向相反：若 $A,B$ 互斥且都具有正概率，则 $\Pr(A\cap B)=0$，不可能等于正数 $\Pr(A)\Pr(B)$。
<!-- bilingual-en:start -->
Mutex is the opposite of independence: if $A,B$ is mutex and all have positive probabilities, then $\Pr(A\cap B)=0$, cannot be equal to positive $\Pr(A)\Pr(B)$.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-independence-grid.png|900]]

读图：独立时每个交叉格的面积等于对应行边际与列边际之积；仅仅“看起来没有重叠”表示互斥，不表示独立。
<!-- bilingual-en:start -->
Reading: The area of each intersection cell when independent is equal to the product of the corresponding row margin and column margin; only "does not appear to overlap" indicates mutex, not independence.
<!-- bilingual-en:end -->

### 4.3.2 Independent Dice Rolls — Online O30-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.2_independent-dice-rolls|4.3.2]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.2_independent-dice-rolls|4.3.2]].
<!-- bilingual-en:end -->

> [!question]- O30-01
> 两颗骰子中哪些事件对独立？正确项为 $(A_1,B_1)$、$(A_1,B_6)$、$(A_1,S_7)$。前两项来自两骰独立；最后一项满足
> $$
> \Pr(A_1\cap S_7)=\frac1{36}=\frac16\frac6{36}.
> $$
> 反例：$A_3\cap S_2=\varnothing$；而 $A_2\cap S_6$ 概率 $1/36\neq(1/6)(5/36)$。
> <!-- bilingual-en:start -->
> Which of the two dice events are independent?  Correct entries are $(A_1,B_1)$, $(A_1,B_6)$, $(A_1,S_7)$.  The first two are from two separate dice; the last is satisfied
> Counterexample: $A_3\cap S_2=\varnothing$; and $A_2\cap S_6$ probability $1/36\neq(1/6)(5/36)$.
> <!-- bilingual-en:end -->

### 4.3.3 Mutual Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MutualIndepend.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/wJzBU7Do1ls.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MutualIndepend.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/wJzBU7Do1ls.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

$A_1,\dots,A_n$ mutually independent，要求对每个非空索引集 $J\subseteq\{1,\dots,n\}$ 都有
<!-- bilingual-en:start -->
$A_1,\dots,A_n$ mutually independent, required for each non-empty index set $J\subseteq\{1,\dots,n\}$
<!-- bilingual-en:end -->

$$
\Pr\!\left(\bigcap_{j\in J}A_j\right)=\prod_{j\in J}\Pr(A_j).
$$

只检查所有 pair 得到 pairwise independence；只检查全部 $n$ 个的交集也不够，因为中间大小的子集仍可能失败。相互独立蕴含 pairwise，反向不成立。
<!-- bilingual-en:start -->
Only checking all pair yields pairwise independence; checking only all $n$ intersections is not sufficient because a subset of the intermediate size may still fail.  pairwise is contained independently of each other, and the reverse is not true.
<!-- bilingual-en:end -->

### 4.3.4 Mutually Independent Dice Rolls — Online O30-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.4_mutually-independent-dice-rolls|4.3.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.4_mutually-independent-dice-rolls|4.3.4]].
<!-- bilingual-en:end -->

> [!question]- O30-02
> 三骰事件组中 mutually independent 的只有 $(A_1,B_1,C_1)$ 与 $(A_1,B_6,C_3)$，因为它们分别只限制三次独立投掷。含总和事件的选项会产生约束；例如
> $$
> \Pr(A_1\cap B_1\cap S_7)=\frac1{216}\neq\frac16\frac16\frac{15}{216}.
> $$
> <!-- bilingual-en:start -->
> The only mutually independent in the tri-dice event group are $(A_1,B_1,C_1)$ and $(A_1,B_6,C_3)$, because they are limited to three separate throws, respectively.  Options with sum events result in constraints; for example,
> <!-- bilingual-en:end -->

### 4.3.5 Independent vs Disjoint — Online O30-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.5_independent-vs-disjoint|4.3.5]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.5_independent-vs-disjoint|4.3.5]].
<!-- bilingual-en:end -->

> [!question]- O30-03
> 两事件何时能同时互斥和独立？互斥给 $\Pr(A\cap B)=0$，独立要求它等于 $\Pr(A)\Pr(B)$，所以至少一个事件概率必须为 0。官方选项写“至少一个为空”；在一般概率空间中更精确的条件是“至少一个为零概率事件”，它未必是空集。
> <!-- bilingual-en:start -->
> When can the two events be mutually exclusive and independent?  Mutually exclusive to $\Pr(A\cap B)=0$, independently requires it equal to $\Pr(A)\Pr(B)$, so at least one event probability must be 0.  The official option writes "at least one empty"; a more precise condition in general probability space is "at least one zero probability event", which is not necessarily an empty set.
> <!-- bilingual-en:end -->

### 4.3.6 Labeled Balls — Online O30-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.6_labeled-balls|4.3.6]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.6_labeled-balls|4.3.6]].
<!-- bilingual-en:end -->

> [!question]- O30-04
> 从 $110,101,011,000$ 等概率抽一串，$A_i$ 表示第 $i$ 位为 1。三事件均概率 $1/2$，任意两者交集恰含一串、概率 $1/4$，故 pairwise independent；但三者交集为空，而边际乘积为 $1/8$，故不 mutually independent，也不互斥。
> <!-- bilingual-en:start -->
> Extract a string from the $110,101,011,000$ equal probability where $A_i$ indicates that the $i$ bit is 1.  The three events have the same probability $1/2$, and any intersection of the two events contains a string, probability $1/4$, so pairwise independent; but the intersection of the three events is empty, and the marginal product is $1/8$, so it is not mutually independent and mutually exclusive.
> <!-- bilingual-en:end -->

### 4.3.7 Paradox — Online O30-05 与因果边界
<!-- bilingual-en:start -->
*4.3.7 Paradox — Online O30-05 and the Causal Boundary*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.7_paradox|4.3.7]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.7_paradox|4.3.7]].
<!-- bilingual-en:end -->

> [!question]- O30-05
> 若 $A,B$ 各自使 $\Pr(H\mid E)>\Pr(H)$，$A\cup B$ 是否必为 $H$ 的证据？不必。公平骰令 $H=\{3,4\}$、$A=\{1,2,3,4\}$、$B=\{3,4,5,6\}$。则 $\Pr(H)=1/3$，$\Pr(H\mid A)=\Pr(H\mid B)=1/2$；但 $A\cup B=\Omega$，条件概率回到 $1/3$。
> <!-- bilingual-en:start -->
> If $A,B$ each makes $\Pr(H\mid E)>\Pr(H)$, does $A\cup B$ have to be evidence of $H$?  No.  $H=\{3,4\}$, $A=\{1,2,3,4\}$, $B=\{3,4,5,6\}$.  $\Pr(H)=1/3$, $\Pr(H\mid A)=\Pr(H\mid B)=1/2$; but $A\cup B=\Omega$, the conditional probability returns to $1/3$.
> <!-- bilingual-en:end -->

独立性和条件概率描述联合分布，不自动给出 causal direction。共同原因、选择偏差或对 collider 条件化都能制造或消除相关。要主张因果，还需干预设计或额外结构假设。
<!-- bilingual-en:start -->
Independence and conditional probability describe a joint distribution; they do not by themselves establish a causal direction. Common causes, selection bias, and conditioning on a collider can all create or remove associations. A causal claim additionally requires an intervention-based design or structural assumptions.
<!-- bilingual-en:end -->

### CP30 — 非官方独立题解（4 道）
<!-- bilingual-en:start -->
*CP30 — Unofficial Independent Questions (4)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp30.pdf#page=1|cp30]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp30.pdf#page=1|cp30]].
<!-- bilingual-en:end -->

> [!example]- C30-1–C30-4 完整解答
> **C30-1 Sally 的大学选择。** 六个叶概率（分母统一为 144）依次为 Yale happy/unhappy：$16,32$；MIT：$35,25$；Little Hoop：$33,3$，总和 144。因此
> $$
> \Pr(H)=\frac{16+35+33}{144}=\frac7{12},\qquad
> \Pr(Y\mid H)=\frac{16}{84}=\frac4{21}.
> $$
> Yale 与快乐不独立，因为 $\Pr(Y\cap H)=16/144\neq(4/12)(7/12)$；MIT 与快乐独立，因为 $35/144=(5/12)(7/12)$。数值独立不表示 MIT “造成”快乐概率不变。
>
> **C30-2 三枚硬币与 parity。** $\Omega=\{H,T\}^3$，各 outcome 概率 $1/8$；$A,B,C$ 分别为对应位是 H，$D$ 为 H 数为偶数，四事件边际均 $1/2$。任意至多三个事件的交集都限制三个独立二进制条件，概率等于边际乘积，所以它们 3-wise independent；但 $A\cap B\cap C\cap D=\varnothing$，而四个边际乘积为 $1/16$，故不 mutually independent。
>
> **C30-3 随机图。**
> (a) $P(x,y)\equiv\exists v\,[E(x,v)\land E(v,y)]$。
> (b) 独立的是 1、2、3；第 4、5 对共享底层边变量，通常相关。
> (c) 对每个 $v\notin\{x,y\}$，经 $v$ 的二步路概率为 $p^2$，不同 $v$ 使用互不相同的边，故
> $$r:=\Pr(\neg P(x,y))=(1-p^2)^{n-2}.$$
> (d) $x,y$ 同处三角形当且仅当 $E(x,y)\land P(x,y)$；两事件独立，故概率 $p(1-r)$。
>
> **C30-4 独立性命题。**
> (a) 真：$\Pr(A\cap\bar B)=\Pr(A)-\Pr(A\cap B)=\Pr(A)\Pr(\bar B)$。
> (b)、(c) 假：在等概率空间 $\{000,011,101,110\}$ 中，令 $A,B,C$ 为对应位等于 1。三者 pairwise independent，但 $A$ 不独立于 $B\cap C$，也不独立于 $B\cup C$。
> (d) 真：用容斥和三个已知独立关系，
> $$
> \Pr[A\cap(B\cup C)]
> =\Pr(A)[\Pr(B)+\Pr(C)-\Pr(B\cap C)]
> =\Pr(A)\Pr(B\cup C).
> $$
> <!-- bilingual-en:start -->
> **C30-1 Sally's college choice.**Six leaf probabilities (denominator 144) were Yale happy/unhappy:$16,32$, MIT:$35,25$, and Little Hoop:$33,3$, totaling 144.  therefore
> Yale is not independent of happiness because $\Pr(Y\cap H)=16/144\neq(4/12)(7/12)$; MIT is independent of happiness because $35/144=(5/12)(7/12)$.  Numerical independence does not mean that the MIT "causes" the same probability of happiness.
> **C30-2 Three Coins and parity.**$\Omega=\{H,T\}^3$, each outcome probability $1/8$; $A,B,C$ is the corresponding bit H, $D$ is the H number is even, the marginal of all four events is $1/2$.  Any intersection of up to three events constrains three independent binary conditions, with the probability equal to the marginal product, so they are 3-wise independent; but $A\cap B\cap C\cap D=\varnothing$, and the four marginal products are $1/16$, so they are not mutually independent.
> **C30-3 random graph.**
> (a)$P(x,y)\equiv\exists v\,[E(x,v)\land E(v,y)]$.
> (b) Separately 1, 2, 3; pairs 4 and 5 share the underlying edge variable and are usually related.
> (c) For each $v\notin\{x,y\}$, the two-step probability through $v$ is $p^2$, and different $v$ use different edges.
> $$r:=\Pr(\neg P(x,y))=(1-p^2)^{n-2}.$$
> (d) $x,y$ is triangle if and only if $E(x,y)\land P(x,y)$; two events are independent, so the probability $p(1-r)$.
> **C30-4 independence proposition.**
> (a) True: $\Pr(A\cap\bar B)=\Pr(A)-\Pr(A\cap B)=\Pr(A)\Pr(\bar B)$.
> (b), (c) False: In the equal probability space $\{000,011,101,110\}$, let $A,B,C$ be the corresponding bit equal to 1.  The three are pairwise independent, but $A$ is not independent of $B\cap C$ or $B\cup C$.
> (d) True: with inclusion and three known independent relationships,
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. pairwise independent 是否足以推出 mutually independent？
> 2. 两个正概率互斥事件能独立吗？
> 3. 相关性为何不能单独证明因果？
>
> <!-- bilingual-en:start -->
> 1. Is pairwise independent sufficient to launch mutually independent?
> 2. Can two positive probability mutually exclusive events be independent?
> 3. Why can't relevance prove cause and effect alone?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 不足，需检查所有非空子集。2. 不能。3. 共同原因、选择机制和反向因果都可产生同一联合分布。
> <!-- bilingual-en:start -->
> 1. Insufficient to check all non-empty subsets.  2. No.  3. Common causes, selection mechanisms, and reverse causality all produce the same joint distribution.
> <!-- bilingual-en:end -->

**知识链**：[[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|条件概率]] → [[概率空间、条件概率与 Bayes 法则#条件概率、独立与因果提醒|两两独立]] → [[条件期望与独立性#独立、不相关与条件期望|相互独立]] → 随机变量独立 → 方差可加。
<!-- bilingual-en:start -->
**Knowledge chain**:[[概率空间、条件概率与 Bayes 法则#条件概率与 Bayes 法则|conditional probability]] → [[概率空间、条件概率与 Bayes 法则#条件概率、独立与因果提醒|pairwise independence]] → [[条件期望与独立性#独立、不相关与条件期望|be independent of each other]] → Independent of random variables → Additive of variance.
<!-- bilingual-en:end -->

---

## Session 31 — Random Variables & Density Functions

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：怎样把复杂 outcome 压缩成一个数值？离散 PMF、CDF 与事件概率如何互换？“两个随机变量独立”究竟需要检查什么？
<!-- bilingual-en:start -->
**Learning questions**: How can a complex outcome be compressed into a single numerical value? How are a discrete PMF, a CDF, and event probabilities converted into one another? What exactly must be checked to establish independence of two random variables?
<!-- bilingual-en:end -->

**前置知识**：函数、条件概率、独立性、二项式系数。核心卡片：[[随机变量、分布与矩#随机变量与分布|随机变量]]、[[随机变量、分布与矩#随机变量与分布|概率质量函数]]、[[随机变量、分布与矩#随机变量与分布|概率密度函数]]。
<!-- bilingual-en:start -->
**Prerequisites**: functions, conditional probability, independence, and binomial coefficients. Core notes: [[随机变量、分布与矩#随机变量与分布|random variables]], [[随机变量、分布与矩#随机变量与分布|probability mass functions]], and [[随机变量、分布与矩#随机变量与分布|probability density functions]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session31.pdf#page=1|Session 31 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp31.pdf#page=1|cp31, pp. 1–3]]

### 4.4.1 Bigger Number Game

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BigerNmberGme.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BH4qlkYCLW0.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BigerNmberGme.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BH4qlkYCLW0.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

对手写两个不同数，只随机展示一个。看似未见数比已见数大的概率是 $1/2$，但可以先独立抽一个 threshold $T$：看到 $x$ 后，若 $x<T$ 就换，否则保留。若 $T$ 落在两个数之间，无论展示哪个都必胜；否则胜率为 $1/2$。因此
<!-- bilingual-en:start -->
Write two different numbers by hand and show only one at random.  The probability that the seemingly invisible number is greater than the visible number is $1/2$, but you can first draw a threshold $T$ independently: if you see a $x$, change the $x<T$, or keep it.  If the $T$ falls between two numbers, it is a winner no matter which is shown; otherwise, the win is $1/2$.  therefore
<!-- bilingual-en:end -->

$$
\Pr(\text{win})=\frac12+\frac12\Pr(T\text{ lies between the numbers})>\frac12
$$

只要 threshold 分布给任意非空区间正概率。优势来自额外随机化，不是预测对手；若数值范围有限，还可选择专门的离散 threshold。
<!-- bilingual-en:start -->
As long as the threshold distribution is given to any non-empty interval with positive probability.  Advantages are derived from additional randomization, not the forecasted adversary; if the range of values is limited, a specialized discrete threshold can be selected.
<!-- bilingual-en:end -->

### 4.4.2 Random Variables & Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomVaribles.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VJzv6WJTtNc.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomVaribles.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VJzv6WJTtNc.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

随机变量 $R$ 是从样本空间到数值集合的函数 $R:\Omega\to\mathbb R$。随机的是输入 outcome，不是函数规则。事件
<!-- bilingual-en:start -->
The random variable $R$ is a function $R:\Omega\to\mathbb R$ from the sample space to the numerical set.  Random is an input outcome, not a function rule.  event
<!-- bilingual-en:end -->

$$
[R=r]:=\{\omega\in\Omega:R(\omega)=r\}.
$$

离散随机变量的 PMF 为
<!-- bilingual-en:start -->
The PMF of discrete random variables is
<!-- bilingual-en:end -->

$$
p_R(r)=\Pr(R=r),\qquad p_R(r)\ge0,\qquad \sum_rp_R(r)=1.
$$

MIT 材料把这个离散质量函数也记作 $\operatorname{PDF}_R$；现代常用语中应叫 PMF，而连续型变量的 PDF $f_R$ 满足 $\Pr(a\le R\le b)=\int_a^b f_R(x)\,dx$，单点概率通常为 0。
<!-- bilingual-en:start -->
The discrete mass function is also called $\operatorname{PDF}_R$ in MIT. The modern common term is called PMF, and the PDF $f_R$ of continuous variables satisfies $\Pr(a\le R\le b)=\int_a^b f_R(x)\,dx$, and the single point probability is usually zero.
<!-- bilingual-en:end -->

对**离散**随机变量 $R,S$，独立等价于对所有值 $r,s$ 都有：
<!-- bilingual-en:start -->
For the**discrete**random variable $R,S$, independently equivalent to all values $r,s$:
<!-- bilingual-en:end -->

$$
\Pr(R=r,S=s)=\Pr(R=r)\Pr(S=s).
$$

等价地，由 $R$ 决定的任一事件与由 $S$ 决定的任一事件独立。只验证协方差为 0 通常不够。
<!-- bilingual-en:start -->
Equivalently, any event determined by $R$ is independent of any event determined by $S$.  It is usually not sufficient to verify only that covariance is 0.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-random-variable-pmf.png|900]]

读图：多个 sample outcomes 可映到同一随机变量值；某根柱子的高度是该逆像事件的总概率，而 CDF 是从左向右累加柱高。
<!-- bilingual-en:start -->
Reading: Multiple sample outcomes can be mapped to the same random variable value; the height of a column is the total probability of the inverse event, and the CDF is the cumulative column height from left to right.
<!-- bilingual-en:end -->

### 4.4.3 Odd Heads and Matches — Online O31-01–O31-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.3_odd-heads-and-matches|4.4.3]]。三枚公平硬币，$I_O$ 指示正面数为奇，$M$ 指示三次全相同。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.3_odd-heads-and-matches|4.4.3]].  Three fair coins, $I_O$ indicates odd number on the front, and $M$ indicates identical three times.
<!-- bilingual-en:end -->

> [!question]- O31-01–O31-07
> **O31-01** $\Pr(I_O=1)=[\binom31+\binom33]/8=1/2$。
>
> **O31-02** $\Pr(I_O=0)=1-1/2=1/2$。
>
> **O31-03** $M=1$ 对应 HHH、TTT，故 $2/8=1/4$。
>
> **O31-04** $M=1,I_O=1$ 仅 HHH，概率 $1/8$。
>
> **O31-05** $M=0,I_O=1$ 等价于恰一正面，概率 $3/8$。
>
> **O31-06** $M=1,I_O=0$ 仅 TTT，概率 $1/8$。
>
> **O31-07** $M=0,I_O=0$ 等价于恰两正面，概率 $3/8$。
>
> 四格分别等于对应边际之积，例如 $(1/4)(1/2)=1/8$，所以 $M$ 与 $I_O$ 独立。
> <!-- bilingual-en:start -->
> $\Pr(I_O=1)=[\binom31+\binom33]/8=1/2$.
> **O31-02**$\Pr(I_O=0)=1-1/2=1/2$.
> **O31-03**$M=1$ corresponds to HHH, TTT, so $2/8=1/4$.
> **O31-04**$M=1,I_O=1$ HHH only with a probability of $1/8$.
> **O31-05**$M=0,I_O=1$ is equivalent to just one positive, and the probability is $3/8$.
> **O31-06**$M=1,I_O=0$ TTT only with $1/8$ probability.
> **O31-07**$M=0,I_O=0$ is equivalent to exactly two positive faces with a probability $3/8$.
> Each of the four cells equals the product of its corresponding marginal probabilities—for example, $(1/4)(1/2)=1/8$—so $M$ and $I_O$ are independent.
> <!-- bilingual-en:end -->

### 4.4.4 Uniform & Binomial Random Variables

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_UniformBinomial.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L2yOSFsMvnc.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_UniformBinomial.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L2yOSFsMvnc.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

在含 $n$ 个值的有限集合上 uniform distribution 给每值概率 $1/n$。若做 $n$ 次 mutually independent Bernoulli trials，每次成功概率 $p$，成功数 $X$ 服从 binomial：
<!-- bilingual-en:start -->
uniform distribution gives the probability $1/n$ per value on a finite set of $n$ values.  If mutually independent Bernoulli trials is perform $n$ times, that probability of success is $p$ for each time, and the success number $X$ is subject to binomial:
<!-- bilingual-en:end -->

$$
\Pr(X=k)=\binom nkp^k(1-p)^{n-k},\qquad k=0,\dots,n.
$$

$\binom nk$ 选择哪些 trial 成功；每个固定成功位置 pattern 的概率由独立性给出 $p^k(1-p)^{n-k}$。若 trials 不独立或成功率不相同，不能直接套此公式。
<!-- bilingual-en:start -->
$\binom nk$ chooses which trial succeeds; the probability of each fixed success position pattern is given by independence $p^k(1-p)^{n-k}$.  If trials is not independent or the success rate is not the same, you cannot directly apply this formula.
<!-- bilingual-en:end -->

### 4.4.5 Late for a Date — Online O31-08–O31-10

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.5_late-for-a-date|4.4.5]]。Jess 在 $-10,\dots,10$ 分钟均匀到达；Sean 的延误数 $T=S+5\sim\operatorname{Bin}(10,1/3)$，两者独立。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.5_late-for-a-date|4.4.5]].  Jess arrives evenly in $-10,\dots,10$ minutes, and Sean delays $T=S+5\sim\operatorname{Bin}(10,1/3)$, both independent.
<!-- bilingual-en:end -->

> [!question]- O31-08–O31-10
> **O31-08** $\Pr(J=0)=1/21$。
>
> **O31-09** Sean 准时等于逛 5 店：
> $$\Pr(S=0)=\binom{10}{5}(1/3)^5(2/3)^5=\frac{896}{6561}.$$
>
> **O31-10**
> $$\Pr(J=S)=\sum_{k=-5}^{5}\Pr(J=k)\Pr(S=k)
> =\frac1{21}\sum_{k=-5}^{5}\Pr(S=k)=\frac1{21}.$$
> 官方反馈最后一句把它口误成 $1/20$；正确值与官方答案栏一致，为 $1/21$。
> <!-- bilingual-en:start -->
> $\Pr(J=0)=1/21$.
> **O31-09**Sean browsed 5 stores on time:
> $$\Pr(S=0)=\binom{10}{5}(1/3)^5(2/3)^5=\frac{896}{6561}.$$
> **O31-10**
> $$\Pr(J=S)=\sum_{k=-5}^{5}\Pr(J=k)\Pr(S=k)
> =\frac1{21}\sum_{k=-5}^{5}\Pr(S=k)=\frac1{21}.$$
> The final sentence of the official feedback misstates it as $1/20$; the correct value corresponds to the official answer column, which is $1/21$.
> <!-- bilingual-en:end -->

### 4.4.6 A Random Number — Online O31-11–O31-18

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.6_a-random-number|4.4.6]]。首枚偏币以 $3/5$ 出正面；正面则掷公平骰返回点数，反面则投三枚公平币并返回正面数的两倍。$F$ 为首投正面 indicator。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.6_a-random-number|4.4.6]]. The first biased coin lands heads with probability $3/5$. On heads, roll a fair die and return its value; on tails, toss three fair coins and return twice the number of heads. Let $F$ be the indicator that the first toss is heads.
<!-- bilingual-en:end -->

> [!question]- O31-11–O31-18
> **O31-11** $N=0$ 只可能 $F=0$ 且后三投全反：$(2/5)(1/8)=1/20=0.05$。
>
> **O31-12** $N=3$ 只可能 $F=1$ 且骰出 3：$(3/5)(1/6)=1/10$。
>
> **O31-13** $N=6$ 可由骰出 6 或后三投全正：$1/10+1/20=3/20$。
>
> **O31-14** 无路径返回 7，概率 0。
>
> **O31-15** 条件 $F=0$ 后，$N=6$ 要三正面：$1/8$。
>
> **O31-16** Bayes：$\Pr(F=0\mid N=6)=(1/20)/(3/20)=1/3$。
>
> **O31-17** $N+F=5$ 只能 $F=1,N=4$：$(3/5)(1/6)=1/10$。
>
> **O31-18** $N+F=6$ 来自 $(F,N)=(1,5)$ 或 $(0,6)$，概率 $1/10+1/20=3/20$。
> <!-- bilingual-en:start -->
> $N=0$ is possible only when $F=0$ and all three subsequent coin tosses are tails: $(2/5)(1/8)=1/20=0.05$.
> **O31-12** $N=3$ is possible only when $F=1$ and the die shows 3: $(3/5)(1/6)=1/10$.
> **O31-13** $N=6$ can arise either from rolling a 6 or from getting heads on all three subsequent tosses: $1/10+1/20=3/20$.
> **O31-14** No outcome gives $N=7$, so the probability is $0$.
> **O31-15** Conditional on $F=0$, $N=6$ requires three heads, which has probability $1/8$.
> **O31-16** By Bayes' rule, $\Pr(F=0\mid N=6)=(1/20)/(3/20)=1/3$.
> **O31-17** $N+F=5$ requires $F=1$ and $N=4$: $(3/5)(1/6)=1/10$.
> **O31-18** $N+F=6$ arises from $(F,N)=(1,5)$ or $(0,6)$, so its probability is $1/10+1/20=3/20$.
> <!-- bilingual-en:end -->

### 4.4.7 PDF to CDF — Online O31-19

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.7_pdf-to-cdf|4.4.7]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.7_pdf-to-cdf|4.4.7]].
<!-- bilingual-en:end -->

CDF 定义为 $F_X(x)=\Pr(X\le x)$，必定单调不减、右连续，且两端极限为 0 与 1。若 $X$ 在整数 $1,\dots,12$ 上均匀，
<!-- bilingual-en:start -->
CDF is defined as $F_X(x)=\Pr(X\le x)$, which must be monotonic, non-subtractive, right-continuous, and the two end limits are 0 and 1.  if that $X$ is uniform ov the integer $1,\dots,12$,
<!-- bilingual-en:end -->

> [!question]- O31-19
> $$F_X(8)=\sum_{k=1}^{8}\frac1{12}=\frac23\approx0.67.$$
> 这里是离散 uniform；区间 $[1,12]$ 上的连续 uniform 会得到 $(8-1)/(12-1)=7/11$，二者不可混用。
> <!-- bilingual-en:start -->
> $$F_X(8)=\sum_{k=1}^{8}\frac1{12}=\frac23\approx0.67.$$
> Here is the discrete uniform; the continuous uniform on the interval $[1,12]$ will get $(8-1)/(12-1)=7/11$, and the two can not be mixed.
> <!-- bilingual-en:end -->

### 4.4.8 Dice and Coin Game — Online O31-20–O31-21

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.8_dice-and-coin-game|4.4.8]]。先掷骰得 $X$，再投 $X$ 枚公平币，$Y$ 为正面数。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.8_dice-and-coin-game|4.4.8]].  Roll the dice first, $X$, then cast the fair coin $X$, $Y$ for the front.
<!-- bilingual-en:end -->

> [!question]- O31-20–O31-21
> **O31-20** 对 $X$ 分层；只有 $X=4,5,6$ 可得 4 个正面：
> $$\Pr(Y=4)=\frac16\left(\frac1{16}+\frac5{32}+\frac{15}{64}\right)=\frac{29}{384}.$$
>
> **O31-21**
> $$\Pr(X=5\mid Y=4)
> =\frac{(1/6)\binom54/2^5}{29/384}=\frac{10}{29}.$$
> <!-- bilingual-en:start -->
> Tiering for $X$; only $X=4,5,6$ has 4 front faces:
> $$\Pr(Y=4)=\frac16\left(\frac1{16}+\frac5{32}+\frac{15}{64}\right)=\frac{29}{384}.$$
> **O31-21**
> $$\Pr(X=5\mid Y=4)
> =\frac{(1/6)\binom54/2^5}{29/384}=\frac{10}{29}.$$
> <!-- bilingual-en:end -->

### CP31 — 非官方独立题解（5 道）
<!-- bilingual-en:start -->
*CP31 — Unofficial Independent Questions (5)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp31.pdf#page=1|cp31]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp31.pdf#page=1|cp31]].
<!-- bilingual-en:end -->

> [!example]- C31-1–C31-5 完整解答
> **C31-1 Bigger Number 的 minimax 上界。** Team 1 在七个相邻 pair $\{0,1\},\dots,\{6,7\}$ 中均匀选一个，再随机摆放。Team 2 若看到端点 0 或 7 可必胜，此情形总概率 $1/7$；看到内点 $1,\dots,6$ 时，它等可能是所选 pair 的较大或较小数，任何决策至多胜 $1/2$。故 Team 2 胜率至多 $1/7+(6/7)(1/2)=4/7$，Team 1 至少胜 $3/7$。
>
> **C31-2 indicators 与事件独立。** 若 $I_A,I_B$ 独立，取值 $(1,1)$ 的等式直接给 $\Pr(A\cap B)=\Pr(A)\Pr(B)$。反之若 $A,B$ 独立，补事件规则给 $A,\bar B$、$\bar A,B$、$\bar A,\bar B$ 也分别独立；这正覆盖 indicators 的四组取值，所以两个随机变量独立。
>
> **C31-3 最大值分布。** $R_i$ 独立且在 $1,\dots,n$ 均匀，$M=\max_iR_i$：
> $$
> \Pr(M=1)=n^{-m},\qquad
> \Pr(M\le k)=\left(\frac kn\right)^m,
> $$
> $$
> p_M(k)=\Pr(M\le k)-\Pr(M\le k-1)
> =\left(\frac kn\right)^m-\left(\frac{k-1}{n}\right)^m.
> $$
>
> **C31-4 binomial 的 mode。** 令 $q=1-p$，相邻 PMF 比为
> $$\frac{p_J(k)}{p_J(k-1)}=\frac{n-k+1}{k}\frac pq.$$
> 它大于 1 当且仅当 $k<np+p$，小于 1 当且仅当 $k>np+p$，所以 PMF 先增后减。若 $np$ 为整数，mode 在 $k=np$；用 Stirling 公式代入 $\binom n{np}p^{np}q^{nq}$，指数项相消，得到
> $$\max_kp_J(k)\sim\frac1{\sqrt{2\pi npq}}.$$
> 边界 $p=0$ 或 $1$ 时分布退化，不能使用含 $pq$ 的渐近式。
>
> **C31-5 生到第一个女孩。** $B$ 为此前男孩数，独立且男女各半：
> $$p_B(i)=2^{-(i+1)},\qquad
> F_B(i)=\Pr(B\le i)=1-2^{-(i+1)},\quad i\ge0.$$
> 第二式的补事件是前 $i+1$ 个孩子全为男孩。
> <!-- bilingual-en:start -->
> **C31-1: A minimax upper bound for Bigger Number.** Team 1 chooses uniformly from the seven adjacent pairs $\{0,1\},\dots,\{6,7\}$ and then randomly assigns the two numbers. If Team 2 sees an endpoint, 0 or 7, it can win with certainty; this case has total probability $1/7$. If it sees one of the interior values $1,\dots,6$, that value is equally likely to be the larger or the smaller member of the chosen pair, so no decision rule can win with probability greater than $1/2$. Hence Team 2's winning probability is at most $1/7+(6/7)(1/2)=4/7$, and Team 1 wins with probability at least $3/7$.
> **C31-2: Independence of indicators and events.** If $I_A$ and $I_B$ are independent, their joint probability at $(1,1)$ gives $\Pr(A\cap B)=\Pr(A)\Pr(B)$ directly. Conversely, if $A$ and $B$ are independent, the complement rules show that $A$ and $\bar B$, $\bar A$ and $B$, and $\bar A$ and $\bar B$ are also independent. These four cases cover every pair of values of the two indicators, so $I_A$ and $I_B$ are independent random variables.
> **C31-3: Distribution of the maximum.** Let the $R_i$ be independent and uniform on $1,\dots,n$, and let $M=\max_iR_i$:
> **C31-4: The mode of a binomial distribution.** Let $q=1-p$. The ratio of adjacent PMF values is
> $$\frac{p_J(k)}{p_J(k-1)}=\frac{n-k+1}{k}\frac pq.$$
> This ratio is greater than 1 exactly when $k<np+p$ and less than 1 exactly when $k>np+p$, so the PMF first increases and then decreases. If $np$ is an integer, a mode occurs at $k=np$. Applying Stirling's formula to $\binom n{np}p^{np}q^{nq}$ cancels the exponential terms and gives
> $$\max_kp_J(k)\sim\frac1{\sqrt{2\pi npq}}.$$
> At the boundary values $p=0$ or $p=1$, the distribution is degenerate and the asymptotic expression containing $pq$ does not apply.
> **C31-5: Continue until the first girl.** Let $B$ be the number of boys born before the first girl, assuming independent births with equal probabilities for boys and girls:
> $$p_B(i)=2^{-(i+1)},\qquad
> F_B(i)=\Pr(B\le i)=1-2^{-(i+1)},\quad i\ge0.$$
> The complement event in the second formula is that the first $i+1$ children are all boys.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 随机变量与它的取值有什么区别？
> 2. 离散 PMF 和连续 PDF 的单点概率有何不同？
> 3. 怎样由 CDF 恢复离散 PMF？
>
> <!-- bilingual-en:start -->
> 1. What is the difference between a random variable and its value?
> 2. What is the difference between discrete PMF and continuous PDF?
> 3. How to recover discrete PMF from CDF?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 随机变量是 outcome 到数值的函数；取值是函数输出。2. PMF 柱高就是单点概率；连续 PDF 的点高不是点概率。3. $p_X(k)=F_X(k)-F_X(k^-)$，整数支撑时为 $F_X(k)-F_X(k-1)$。
> <!-- bilingual-en:start -->
> 1. Random variables are outcome to the value of the function output.  2.  The PMF column height is just a single point probability, and the continuous PDF column height is not a point probability.  3.  $p_X(k)=F_X(k)-F_X(k^-)$, $F_X(k)-F_X(k-1)$ for integer support.
> <!-- bilingual-en:end -->

**知识链**：[[随机变量、分布与矩#随机变量与分布|随机变量]] → [[随机变量、分布与矩#随机变量与分布|PMF]] / CDF → uniform 与 binomial → 随机变量独立 → [[随机变量、分布与矩#期望、方差与协方差|期望]]。
<!-- bilingual-en:start -->
**Knowledge Chain**: [[随机变量、分布与矩#随机变量与分布|random variable]] → [[随机变量、分布与矩#随机变量与分布|PMF]] / CDF → uniform Independent of binomial → Random Variables → [[随机变量、分布与矩#期望、方差与协方差|anticipate]].
<!-- bilingual-en:end -->

---

## Session 32 — Expectation

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：期望为什么是概率加权平均而非“最可能值”？为什么不独立也能拆开和的期望？怎样用 indicator 避免求完整分布？
<!-- bilingual-en:start -->
**Learning problem**: Why are expectations weighted averages of probability rather than "most likely"?  Why can we disconnect and hope without independence?  How to Avoid Finding Complete Distribution with indicator?
<!-- bilingual-en:end -->

**前置知识**：离散随机变量、PMF、条件概率、无穷级数。核心卡片：[[随机变量、分布与矩#期望、方差与协方差|期望]]、[[指示变量与随机计数#从事件到 0/1 随机变量|指示随机变量]]、[[指示变量与随机计数#把“数量”写成 indicators 的和|期望线性性]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: Discrete random variables, PMF, conditional probability, infinite series.  Core cards: [[随机变量、分布与矩#期望、方差与协方差|anticipate]], [[指示变量与随机计数#从事件到 0/1 随机变量|Indicates a random variable]], [[指示变量与随机计数#把“数量”写成 indicators 的和|expected linearity]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session32.pdf#page=1|Session 32 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp32.pdf#page=1|local cp32, pp. 1–3]]

### 4.5.1 Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Expectation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/YVQdVzSkcmQ.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Expectation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/YVQdVzSkcmQ.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

离散随机变量 $R$ 的期望为
<!-- bilingual-en:start -->
The expectation of discrete random variable $R$ is
<!-- bilingual-en:end -->

$$
\mathbb E[R]=\sum_r r\,\Pr(R=r),
$$

前提是该和绝对收敛，或至少正负部分的处理使期望有定义。对任意函数 $g$，LOTUS 给
<!-- bilingual-en:start -->
This requires absolute convergence, or at least well-defined positive and negative parts. For any function $g$, the law of the unconscious statistician (LOTUS) gives
<!-- bilingual-en:end -->

$$
\mathbb E[g(R)]=\sum_r g(r)\Pr(R=r),
$$

无需先求 $g(R)$ 的完整 PMF。期望可以不在支撑集内，例如 51 次公平投币的正面数期望 25.5，却绝不取 25.5。
<!-- bilingual-en:start -->
There is no need to derive the full PMF of $g(R)$ first. An expectation need not belong to the support: for example, the expected number of heads in 51 fair coin tosses is 25.5, although the random variable can never equal 25.5.
<!-- bilingual-en:end -->

### 4.5.2 Uneven Dice — Online O32-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.2_uneven-dice|4.5.2]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.2_uneven-dice|4.5.2]].
<!-- bilingual-en:end -->

> [!question]- O32-01
> 骰子以 $1/4$ 概率出偶数、$3/4$ 出奇数，同类内部均匀。偶数各概率 $1/12$，奇数各概率 $1/4$，故
> $$
> \mathbb E[R]=\frac{1+3+5}{4}+\frac{2+4+6}{12}=\frac{13}{4}=3.25.
> $$
> 官方反馈首个求和公式漏写了取值系数，但后续展开与答案正确。
> <!-- bilingual-en:start -->
> Dice is even with $1/4$ probability and odd with $3/4$ probability, the same kind of internal uniformity.  even probability $1/12$, odd probability $1/4$, therefore
> The first sum formula in official feedback omits the value-taking coefficient, but subsequent expansions and answers are correct.
> <!-- bilingual-en:end -->

### 4.5.3 Expected Number of Heads

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ExpectNumber.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/D9l-pIg1Ayo.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ExpectNumber.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/D9l-pIg1Ayo.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

若 $I_i$ 指示第 $i$ 次成功，则总成功数 $X=\sum_iI_i$。由于 $\mathbb E[I_i]=\Pr(I_i=1)=p_i$，线性性给
<!-- bilingual-en:start -->
If $I_i$ indicates the $i$, the total number of successes is $X=\sum_iI_i$.  Because of $\mathbb E[I_i]=\Pr(I_i=1)=p_i$, linearity is given to
<!-- bilingual-en:end -->

$$
\mathbb E[X]=\sum_i p_i.
$$

这一步不需要独立；独立性只在推导 binomial PMF 或方差相加时需要。
<!-- bilingual-en:start -->
This step does not require independence; independence is required only when deriving binomial PMF or variance additions.
<!-- bilingual-en:end -->

### 4.5.4 Expected Number of Heads — Online O32-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.4_expected-number-of-heads|4.5.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.4_expected-number-of-heads|4.5.4]].
<!-- bilingual-en:end -->

> [!question]- O32-02
> 200 枚公平硬币正面数期望：$\sum_{i=1}^{200}\mathbb E[I_i]=200/2=100$。
> <!-- bilingual-en:start -->
> 200 Fair Coins Face Number Expectation: $\sum_{i=1}^{200}\mathbb E[I_i]=200/2=100$.
> <!-- bilingual-en:end -->

### 4.5.5 Total Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_TotalExpectatn.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/mqoDXWrSais.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_TotalExpectatn.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/mqoDXWrSais.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

若 $B_1,\dots,B_m$ 构成 partition，且相关条件期望有定义，则
<!-- bilingual-en:start -->
If $B_1,\dots,B_m$ form a partition and the relevant conditional expectations exist, then
<!-- bilingual-en:end -->

$$
\mathbb E[R]=\sum_i\mathbb E[R\mid B_i]\Pr(B_i).
$$

证明从定义展开：
<!-- bilingual-en:start -->
Expanding the definition gives:
<!-- bilingual-en:end -->

$$
\sum_r r\Pr(R=r)
=\sum_r r\sum_i\Pr(R=r\mid B_i)\Pr(B_i)
=\sum_i\Pr(B_i)\mathbb E[R\mid B_i].
$$

交换求和需要有限情形，或无穷情形下满足非负性/绝对收敛等条件。
<!-- bilingual-en:start -->
Interchanging the sums is automatic in the finite case; in the infinite case it requires conditions such as nonnegativity or absolute convergence.
<!-- bilingual-en:end -->

### 4.5.6 Another Dice and Coin Game — Online O32-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.6_another-dice-and-coin-game|4.5.6]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.6_another-dice-and-coin-game|4.5.6]].
<!-- bilingual-en:end -->

> [!question]- O32-03
> 公平币正面则反复掷骰直到奇数，反面则直到偶数。条件分布分别在 $\{1,3,5\}$ 与 $\{2,4,6\}$ 均匀，条件均值 3 与 4：
> $$\mathbb E[R]=\tfrac12(3)+\tfrac12(4)=3.5.$$
> <!-- bilingual-en:start -->
> Faircoin rolls the dice again and again to odd numbers on the front and even numbers on the back.  The conditional distributions are uniform at $\{1,3,5\}$ and $\{2,4,6\}$, respectively. Conditional means 3 and 4:
> $$\mathbb E[R]=\tfrac12(3)+\tfrac12(4)=3.5.$$
> <!-- bilingual-en:end -->

### 4.5.7 Mean Time to Failure

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MeanTimeFailure.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Dqx56lZ_icg.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MeanTimeFailure.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Dqx56lZ_icg.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

每轮独立地以概率 $p>0$ 首次成功，等待轮数 $T\in\{1,2,\dots\}$ 服从 geometric：
<!-- bilingual-en:start -->
Each round succeeds for the first time independently with the probability $p>0$, and the waiting rounds $T\in\{1,2,\dots\}$ obey the geometric:
<!-- bilingual-en:end -->

$$
\Pr(T=k)=(1-p)^{k-1}p.
$$

用 first-step analysis，第一轮必消耗 1；若失败（概率 $1-p$），剩余等待与原问题同分布：
<!-- bilingual-en:start -->
With first-step analysis, the first round must consume 1. If it fails (probability $1-p$), the remaining waiting is the same as the original problem:
<!-- bilingual-en:end -->

$$
\mathbb E[T]=1+(1-p)\mathbb E[T],
\qquad
\mathbb E[T]=\frac1p.
$$

若各轮不独立或成功率随时间变化，这个结论不能直接使用。
<!-- bilingual-en:start -->
This conclusion cannot be used directly if the rounds are not independent or if the success rate varies over time.
<!-- bilingual-en:end -->

### 4.5.8 Three Machines Failing — Online O32-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.8_three-machines-failing|4.5.8]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.8_three-machines-failing|4.5.8]].
<!-- bilingual-en:end -->

> [!question]- O32-04
> 三台机器每轮各以 $1/3$ 失败且 mutually independent，同时失败概率 $(1/3)^3=1/27$。轮次独立时首次同时失败的期望等待为 $1/(1/27)=27$。
> <!-- bilingual-en:start -->
> Each of the three machines failed at $1/3$ and mutually independent with a $(1/3)^3=1/27$ probability of failure.  The expected wait for the first simultaneous failure in a round of independence is $1/(1/27)=27$.
> <!-- bilingual-en:end -->

### 4.5.9 Linearity of Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LinearityExpect.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/KFcodn4qfrQ.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LinearityExpect.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/KFcodn4qfrQ.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

对有限个有期望的随机变量，
<!-- bilingual-en:start -->
For a finite number of random variables,
<!-- bilingual-en:end -->

$$
\mathbb E\!\left[\sum_{i=1}^{n}a_iR_i+c\right]
=\sum_{i=1}^{n}a_i\mathbb E[R_i]+c.
$$

证明可直接在共同样本空间求和：
<!-- bilingual-en:start -->
The result follows by summing directly over the common sample space:
<!-- bilingual-en:end -->

$$
\sum_{\omega}\left(\sum_i a_iR_i(\omega)+c\right)\Pr(\omega)
=\sum_i a_i\sum_{\omega}R_i(\omega)\Pr(\omega)+c.
$$

这里从未分解联合概率，所以不要求 $R_i$ 独立。相反，$\mathbb E[RS]=\mathbb E[R]\mathbb E[S]$ 通常要求独立；偶然相等不能反推出独立。
<!-- bilingual-en:start -->
No joint probability is factorised in this argument, so independence of the $R_i$ is unnecessary. By contrast, the identity $\mathbb E[RS]=\mathbb E[R]\mathbb E[S]$ is generally justified by independence; an accidental equality does not imply independence in reverse.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-expectation-variance.png|900]]

读图：期望定位概率质量的“平衡点”，方差则用到该点的平方距离加权；同均值的两个分布可以有完全不同的离散程度。
<!-- bilingual-en:start -->
Reading the figure: expectation locates the balance point of the probability mass, while variance weights squared distance from that point. Two distributions with the same mean can therefore have completely different spreads.
<!-- bilingual-en:end -->

### 4.5.10 Fair and Biased Coins — Online O32-05

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.10_fair-and-biased-coins|4.5.10]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.10_fair-and-biased-coins|4.5.10]].
<!-- bilingual-en:end -->

> [!question]- O32-05
> 100 枚公平币与 100 枚正面率 $1/4$ 的偏币，总正面数期望
> $$100(1/2)+100(1/4)=75.$$
> 不需要 200 次投掷相互独立。
> <!-- bilingual-en:start -->
> With 100 fair coins and 100 biased coins whose head probability is $1/4$, the expected total number of heads is
> $$100(1/2)+100(1/4)=75.$$
> The 200 tosses need not be mutually independent for this expectation calculation.
> <!-- bilingual-en:end -->

### 4.5.11 Binomial Board Breaking — Online O32-06–O32-08

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.11_binomial-board-breaking|4.5.11]]。$X\sim\operatorname{Bin}(5,0.8)$。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.11_binomial-board-breaking|4.5.11]].  $X\sim\operatorname{Bin}(5,0.8)$.
<!-- bilingual-en:end -->

> [!question]- O32-06–O32-08
> **O32-06** $\Pr(X=2)=\binom52(0.8)^2(0.2)^3=0.0512$。
>
> **O32-07** $\Pr(X\le3)=1-\binom54(0.8)^4(0.2)-(0.8)^5=0.26272$。
>
> **O32-08** $\mathbb E[X]=np=5(0.8)=4$。

### 4.5.12 Great Expectations — Online O32-09–O32-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.12_great-expectations|4.5.12]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.12_great-expectations|4.5.12]].
<!-- bilingual-en:end -->

> [!question]- O32-09–O32-11
> **O32-09** 公平 6 面骰和 12 面骰之和期望：$(6+1)/2+(12+1)/2=10$。
>
> **O32-10** 以 $1/6$ 选均匀 $1,\dots,99$（均值 50），否则选均匀 $1,\dots,999$（均值 500）：
> $$\mathbb E[G]=\frac16(50)+\frac56(500)=425.$$
>
> **O32-11** 两电脑独立时，输出乘积期望 $50\cdot500=25000$。若不独立，边际均值不足以确定乘积期望。
> <!-- bilingual-en:start -->
> Fair 6 dice and 12 dice combined Expectations: $(6+1)/2+(12+1)/2=10$.
> **O32-10**Select uniform $1,\dots,99$ (mean 50) at $1/6$, otherwise select uniform $1,\dots,999$ (mean 500):
> $$\mathbb E[G]=\frac16(50)+\frac56(500)=425.$$
> **O32-11**When two computers are independent, the output product expects $50\cdot500=25000$.  Without independence, the marginal mean is insufficient to determine the product expectation.
> <!-- bilingual-en:end -->

### 4.5.13 Expectation of a Uniform Distribution — Online O32-12

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.13_expectation-of-a-uniform-distribution|4.5.13]]。$X$ 在 $-n,\dots,n$ 均匀，$Y=X^2$。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.13_expectation-of-a-uniform-distribution|4.5.13]].  $X$ is uniform at $-n,\dots,n$, $Y=X^2$.
<!-- bilingual-en:end -->

> [!question]- O32-12
> 正确项：$\mathbb E[X]=0$；$\mathbb E[Y]>\mathbb E[X]$（$n>0$）；线性性给 $\mathbb E[X+Y]=\mathbb E[X]+\mathbb E[Y]$；且本例中 $\mathbb E[XY]=\mathbb E[X^3]=0=\mathbb E[X]\mathbb E[Y]$。最后一个等式只是对称导致的巧合；$Y$ 完全由 $X$ 决定，所以二者不独立。边界 $n=0$ 时 $\mathbb E[Y]>\mathbb E[X]$ 不成立，官方题隐含 $n>0$。
> <!-- bilingual-en:start -->
> Correct entries: $\mathbb E[X]=0$; $\mathbb E[Y]>\mathbb E[X]$ ($n>0$); linearity given to $\mathbb E[X+Y]=\mathbb E[X]+\mathbb E[Y]$; and in this example $\mathbb E[XY]=\mathbb E[X^3]=0=\mathbb E[X]\mathbb E[Y]$.  The last equation is simply a coincidence caused by symmetry; $Y$ is completely determined by $X$, so they are not independent.  The $\mathbb E[Y]>\mathbb E[X]$ is not valid when the boundary is $n=0$, and the official question implies $n>0$.
> <!-- bilingual-en:end -->

### CP32 — 本地来源异常与非官方独立题解（5 道）
<!-- bilingual-en:start -->
*CP32—Local Source Exceptions and Unofficial Independent Questions (5)*
<!-- bilingual-en:end -->

> [!warning] 本地文件与 Session 主题不匹配
> [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp32.pdf#page=1|cp32]] 的页眉是 Week 8, Wed.，内容为图论而非 expectation。为保证 cp28–cp35 每个编号题都有归属，以下按本地 PDF 原题解答；它不代表官方 Session 32 block 内容。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp32.pdf#page=1|cp32]]'s header is Week 8, Wed., which is for graph theory, not expectation.  To ensure that each cp28-cp35 numbered question is attributed, the following is answered as a local PDF original; it does not represent official Session 32 block content.
> <!-- bilingual-en:end -->

> [!example]- C32-1–C32-5 完整解答
> **C32-1 男女平均伴侣数。** 把人作为二分图顶点、伴侣关系作为边，边数记 $E$。男性平均为 $E/m$、女性平均为 $E/f$，不同是因分母不同，不足以推出夸大。已知 $E/m=1.1E/f$，所以 $m=(10/11)f$，即题中 $c=10/11$。排除无伴侣者后，非处男男性数 $0.95m$、非处女女性数 $0.8f$，平均之比
> $$\frac{E/(0.95m)}{E/(0.8f)}=\frac{16}{19}\frac fm,$$
> 故 $x=16/19$。又因 $f=1.1m>m$，不可能把每位女性单射匹配到不同男性。
>
> **C32-2 奇度顶点。** Handshaking lemma 给 $\sum_v\deg(v)=2|E|$ 为偶数。偶度项之和为偶数，所以奇度项个数必须为偶数，否则奇数个奇数之和为奇。握手图立即给 (b)。对 (c)，看 George 所在 connected component；该 component 的奇度顶点数也为偶数，已有 George 一个，必有另一个，并由连通性存在从 George 到他的 handshake sequence。
>
> **C32-3 全部同构。** 两图的两个三度点必须满足 $\{3,4\}\mapsto\{c,d\}$。若 $3\mapsto c,4\mapsto d$，则 $5\mapsto e,6\mapsto f$；若端点交换，则 $5\mapsto f,6\mapsto e$。顶点 $1,2$ 都与两个三度点相邻，可独立交换到 $a,b$。故共四个同构：
> $$
> \begin{aligned}
> &(1,2,3,4,5,6)\mapsto(a,b,c,d,e,f),\\
> &(1,2,3,4,5,6)\mapsto(b,a,c,d,e,f),\\
> &(1,2,3,4,5,6)\mapsto(a,b,d,c,f,e),\\
> &(1,2,3,4,5,6)\mapsto(b,a,d,c,f,e).
> \end{aligned}
> $$
> 度数与三条 internally disjoint 路径的长度结构排除了其他映射。
>
> **C32-4 同构不变量。** 保持的是 (a) 顶点数为 7、(b) 有 Hamilton cycle、(c) 有两个 8 度点、(e) 删任一边仍连通、(f) 有两条 vertex-disjoint cycles、(h) 存在全等边长绘制、(i) 两个不变量的 OR、(j) 不变量的否定。不保持的是 (d) 当前图画中两边等长和 (g) 顶点标签间的集合包含；它们依赖表示而非抽象邻接关系。
>
> **C32-5 邻域在同构下保持。** 对任意 $h\in V(H)$：
> $$
> \begin{aligned}
> h\in H(f(v))
> &\Longleftrightarrow (f(v),h)\in E(H)\\
> &\Longleftrightarrow (v,f^{-1}(h))\in E(G)\\
> &\Longleftrightarrow f^{-1}(h)\in G(v)\\
> &\Longleftrightarrow h\in f(G(v)).
> \end{aligned}
> $$
> 外延性给 $H(f(v))=f(G(v))$。因 $f$ 为 bijection，邻域大小相等，所以 $\operatorname{outdeg}_H(f(v))=\operatorname{outdeg}_G(v)$；$f$ 进一步在所有出度为 $k$ 的顶点间给出 bijection，计数相同。
> <!-- bilingual-en:start -->
> **C32-1 Average numbers of partners for men and women.** Model people as vertices of a bipartite graph and partnerships as edges; let the number of edges be $E$. The male and female averages are respectively $E/m$ and $E/f$. Their difference comes from different denominators and does not by itself imply exaggeration. Given $E/m=1.1E/f$, we obtain $m=(10/11)f$, so $c=10/11$. Excluding people with no partners leaves $0.95m$ men and $0.8f$ women, giving the ratio of conditional averages
> $$\frac{E/(0.95m)}{E/(0.8f)}=\frac{16}{19}\frac fm,$$
> Thus $x=16/19$. Since $f=1.1m>m$, it is impossible to match every woman with a distinct man.
> **C32-2 Odd-degree vertices.** The handshaking lemma gives $\sum_v\deg(v)=2|E|$, which is even. The even-degree terms contribute an even total, so the number of odd-degree terms must be even; otherwise their sum would be odd. This proves part (b) immediately. For part (c), consider George's connected component. That component also has an even number of odd-degree vertices. Since George is one, there must be another, and a path of handshakes connects George to that person.
> **C32-3 All graph isomorphisms.** The two degree-3 vertices must satisfy $\{3,4\}\mapsto\{c,d\}$. If $3\mapsto c$ and $4\mapsto d$, then $5\mapsto e$ and $6\mapsto f$; swapping the endpoints gives $5\mapsto f$ and $6\mapsto e$. Vertices $1$ and $2$ are adjacent to both degree-3 vertices and may be independently exchanged with $a$ and $b$. Hence there are four isomorphisms:
> Degree and the length structure of the three internally disjoint paths rule out other mappings.
> **C32-4 Isomorphism invariants.** The following are preserved: (a) having 7 vertices; (b) having a Hamilton cycle; (c) having two vertices of degree 8; (e) remaining connected after any one edge is deleted; (f) having two vertex-disjoint cycles; (h) admitting a drawing with all edges of equal length; (i) the OR of two invariant properties; and (j) the negation of an invariant property. The following are not preserved: (d) two edges having equal length in the current drawing and (g) set inclusion between vertex labels. Those depend on the representation rather than the abstract adjacency relation.
> **C32-5 neighborhood is preserved under isomorphism.**For any $h\in V(H)$:
> By extensionality, $H(f(v))=f(G(v))$. Because $f$ is a bijection, the two neighbourhoods have the same size, so $\operatorname{outdeg}_H(f(v))=\operatorname{outdeg}_G(v)$. Moreover, $f$ restricts to a bijection between the vertices of outdegree $k$ in the two graphs, so their counts agree.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 线性性为何不要求独立？
> 2. $\mathbb E[X]=a$ 是否推出 $\Pr(X=a)>0$？
> 3. geometric 等待时间均值 $1/p$ 需要哪些重复试验条件？
>
> <!-- bilingual-en:start -->
> 1. Why does linearity not require independence?
> 2. Is $\Pr(X=a)>0$ available in $\mathbb E[X]=a$?
> 3. What are the repeated trial conditions required for the average geometric latency of $1/p$?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 它来自同一样本空间上的有限求和交换。2. 不推出。3. 每轮成功率固定为 $p>0$，且失败后剩余过程与原过程同分布，通常由轮次独立保证。
> <!-- bilingual-en:start -->
> 1. It comes from a finite sum exchange over the same sample space.  2. Not available.  3. The success rate of each round is fixed at $p>0$, and the remaining process after failure is the same distribution as the original process, which is usually guaranteed by the round independently.
> <!-- bilingual-en:end -->

**知识链**：[[随机变量、分布与矩#随机变量与分布|PMF]] → [[随机变量、分布与矩#期望、方差与协方差|期望]] → [[指示变量与随机计数#从事件到 0/1 随机变量|indicator]] → [[指示变量与随机计数#把“数量”写成 indicators 的和|线性性]] → total expectation → [[随机变量、分布与矩#期望、方差与协方差|方差]]。
<!-- bilingual-en:start -->
**Knowledge Chain**: [[随机变量、分布与矩#随机变量与分布|PMF]] → [[随机变量、分布与矩#期望、方差与协方差|anticipate]] → [[指示变量与随机计数#从事件到 0/1 随机变量|indicator]] → [[指示变量与随机计数#把“数量”写成 indicators 的和|linearity]] → total expectation → [[随机变量、分布与矩#期望、方差与协方差|variance]].
<!-- bilingual-en:end -->

---

## Problem Set 12 — Independence, Random Variables & Expectation

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps12.pdf#page=1|Problem Set 12, pp. 1–4]]。以下为非官方独立题解。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps12.pdf#page=1|Problem Set 12, pp. 1–4]].  The following are unofficial independent questions.
<!-- bilingual-en:end -->

> [!example]- P12-1 随机变量相等事件与独立性
> **(a)** $R$ 在有限集合 $V$ 上 uniform 且与 $S$ 独立。事件 $[R=S]$ 是互斥并：
> $$
> \Pr(R=S)=\sum_{b\in V}\Pr(R=b,S=b)
> =\sum_b\frac1{|V|}\Pr(S=b)=\frac1{|V|}.
> $$
> 这给出了原课文直觉句的严格版本。
>
> **(b)** 设 $A=[R=S]$、$B=[S=T]$，且 $R$ 与 pair $(S,T)$ 独立。则
> $$
> \Pr(A\cap B)
> =\sum_b\Pr(R=b,S=b,T=b)
> =\frac1{|V|}\sum_b\Pr(S=b,T=b)
> =\frac1{|V|}\Pr(B).
> $$
> 又由 (a) $\Pr(A)=1/|V|$，所以 $\Pr(A\cap B)=\Pr(A)\Pr(B)$。
>
> **(c)** 六个等概率 triples 中，每个 $(S,T)\in\{(1,1),(2,3),(3,2)\}$ 都分别与 $R=1,2$ 搭配一次，故 $R$ 与 $(S,T)$ 独立。$A=[R=S]$ 出现在第 1、4 项，$\Pr(A)=1/3$；$B=[S=T]$ 出现在第 1、2 项，$\Pr(B)=1/3$；交集只第 1 项，概率 $1/6\neq1/9$。同时 $S,T$ 的每个值 $1,2,3$ 都各出现两次，所以各自 uniform。
> <!-- bilingual-en:start -->
> **(a)**The $R$ is uniform on a finite set $V$ and is independent of $S$.  The event $[R=S]$ is mutually exclusive and:
> This gives a rigorous version of the intuitive sentence in the original text.
> **(b)**Suppose $A=[R=S]$, $B=[S=T]$, and $R$ is independent of pair $(S,T)$.
> Also by (a) $\Pr(A)=1/|V|$, so $\Pr(A\cap B)=\Pr(A)\Pr(B)$.
> **(c)**In the six equal probability triples, each $(S,T)\in\{(1,1),(2,3),(3,2)\}$ is paired with $R=1,2$ once, so $R$ and $(S,T)$ are independent.  $A=[R=S]$ appears in items 1 and 4, $\Pr(A)=1/3$; $B=[S=T]$ appears in items 1 and 2, $\Pr(B)=1/3$; intersection is only item 1, probability $1/6\neq1/9$.  Each value of $S,T$, $1,2,3$, appears twice, so each uniform.
> <!-- bilingual-en:end -->

> [!example]- P12-2 二战 pooled testing
> 每组先做 1 次混合检测；至少一人患病的概率是 $1-(1-p)^k$，此时再做 $k$ 次。共有 $g=n/k$ 组，所以
> $$
> \mathbb E[T]=\frac nk+n[1-(1-p)^k].
> $$
> 当 $p$ 小且 $kp$ 小，$1-(1-p)^k\approx kp$，故
> $$
> \frac{\mathbb E[T]}n\approx\frac1k+kp.
> $$
> 对 $k$ 求最小值：$-1/k^2+p=0$，取 $k\approx p^{-1/2}$，得到 $\mathbb E[T]\approx2n\sqrt p$。原 PDF 的 “approximately $n\sqrt p$” 忽略了常数 2；若按渐近阶理解则为 $\Theta(n\sqrt p)$。
>
> 当 $n=10^6,p=0.01,k\approx10$，精确期望比例为
> $$0.1+1-0.99^{10}\approx0.1956,$$
> 即约做 195,618 次检测，节省约 $80.44\%$。更深层方案可先测大池，阳性池递归二分；稀疏患病时会把逐人复检改成定位所需的对数层数，进一步减少期望检测数。
> <!-- bilingual-en:start -->
> Each group is tested once; at least one person has a $1-(1-p)^k$ chance of getting sick, and then $k$.  $g=n/k$ group, so...
> when $p$ is small and $kp$ is small, $1-(1-p)^k\approx kp$, so
> The minimum value of $k$ is $-1/k^2+p=0$ and $k\approx p^{-1/2}$ to get $\mathbb E[T]\approx2n\sqrt p$.  The "approximately $n\sqrt p$" of the original PDF ignores the constant 2; $\Theta(n\sqrt p)$ if understood at asymptotic order.
> When $n=10^6,p=0.01,k\approx10$, the exact expected ratio is
> $$0.1+1-0.99^{10}\approx0.1956,$$
> That's about 195,618 tests, saving about $80.44\%$.  The deeper scheme can test large pool first, and the positive pool recursively divides into two parts; when the sparse disease occurs, the person-by-person re-inspection is changed into the logarithmic layer number needed by the location, and the expected number of tests is further reduced.
> <!-- bilingual-en:end -->

> [!example]- P12-3 随机赋值与 $k$-SAT
> 一个 $k$-clause 仅在全部 $k$ 个 literals 为假时失败；独立公平赋值下
> $$
> \Pr(\text{clause true})=1-2^{-k}.
> $$
> 令 $I_i$ 指示第 $i$ 个 clause 为真，$X=\sum_{i=1}^nI_i$。clauses 即使共享变量，线性性仍给
> $$
> \mathbb E[X]=n(1-2^{-k}).
> $$
> 若 $n<2^k$，则 $\mathbb E[X]>n-1$。若没有 assignment 同时满足全部 clauses，则整数 $X$ 对每个 outcome 都至多 $n-1$，从而期望也至多 $n-1$，矛盾。因此至少存在一个 assignment 使 $X=n$，即序列 satisfiable。这是 probabilistic method：用正期望证明对象存在。
> <!-- bilingual-en:start -->
> A $k$-clause fails only if all $k$ literals are false; under independent fair assignment
> Let $I_i$ indicate that the $i$ clause is true, $X=\sum_{i=1}^nI_i$.  clauses Even if a variable is shared, linearity still gives
> If $n<2^k$, $\mathbb E[X]>n-1$.  If there is no assignment that satisfies all clauses at the same time, then the integer $X$ is at most $n-1$ for each outcome, and thus the expectation is at most $n-1$, contradictory.  There is therefore at least one assignment that makes $X=n$, the sequence satisfiable.  This is probabilistic method: proving the existence of an object with positive expectations.
> <!-- bilingual-en:end -->

---

## Session 33 — Deviation: Markov & Chebyshev Bounds

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：只知道均值能限制多大的尾概率？再知道方差能改善多少？方差相加究竟要求 independent 还是 pairwise independent？
<!-- bilingual-en:start -->
**Learning Problem**:How much tail probability can we limit by knowing only the mean?  How much better can variance be?  independent or pairwise independent is required for variance addition?
<!-- bilingual-en:end -->

**前置知识**：期望、indicator、独立性、平方展开。核心卡片：[[随机变量、分布与矩#期望、方差与协方差|方差]]、[[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov 不等式]]、[[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev 不等式]]。
<!-- bilingual-en:start -->
**Prerequisites**: Expectations, indicator, Independence, Squared Expansion.  Core cards: [[随机变量、分布与矩#期望、方差与协方差|variance]], [[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov inequality]], [[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev inequality]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf#page=1|Session 33 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp33.pdf#page=1|cp33, pp. 1–4]]

### 4.6.1 Deviation from the Mean

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_DeviatTheMean.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ALn1McUXg-c.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_DeviatTheMean.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ALn1McUXg-c.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

均值是重心，不是“典型 outcome”的保证。要描述离散程度，定义
<!-- bilingual-en:start -->
Mean is the center of gravity, not the guarantee of "typical outcome".  To describe the degree of dispersion, define
<!-- bilingual-en:end -->

$$
\operatorname{Var}(R)=\mathbb E[(R-\mu)^2],
\qquad \mu=\mathbb E[R],
$$

标准差 $\sigma_R=\sqrt{\operatorname{Var}(R)}$ 与 $R$ 同单位。平方使正负偏差不相消，并更重罚远端值。
<!-- bilingual-en:start -->
The standard deviation $\sigma_R=\sqrt{\operatorname{Var}(R)}$ has the same units as $R$. Squaring prevents positive and negative deviations from cancelling and gives greater weight to distant values.
<!-- bilingual-en:end -->

### 4.6.2 Don't Expect the Expectation — Online O33-01–O33-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.2_don-t-expect-the-expectation|4.6.2]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.2_don-t-expect-the-expectation|4.6.2]].
<!-- bilingual-en:end -->

> [!question]- O33-01–O33-02
> **O33-01** 51 枚公平币正面数 $X$ 的期望 $51/2=25.5$。
>
> **O33-02** 期望不可取，但 $X=25$ 可以取：
> $$\Pr(X=25)=\binom{51}{25}2^{-51}\approx0.1101.$$
> <!-- bilingual-en:start -->
> 51 Fairmoney Fronts$X$Expected $51/2=25.5$.
> **O33-02**Expectations are not desirable, but $X=25$ is:
> $$\Pr(X=25)=\binom{51}{25}2^{-51}\approx0.1101.$$
> <!-- bilingual-en:end -->

### 4.6.3 Markov Bounds

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/m07lrb7m0D0.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/m07lrb7m0D0.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

[[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov 不等式证明]]：若 $R\ge0$ 且 $a>0$，则
<!-- bilingual-en:start -->
[[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov inequality proof]]: If $R\ge0$ and $a>0$
<!-- bilingual-en:end -->

$$
\boxed{\Pr(R\ge a)\le\frac{\mathbb E[R]}a}.
$$

逐点有 $R\ge aI_{\{R\ge a\}}$。两边取期望：
<!-- bilingual-en:start -->
$R\ge aI_{\{R\ge a\}}$.  Expect on both sides:
<!-- bilingual-en:end -->

$$
\mathbb E[R]\ge a\,\mathbb E[I_{\{R\ge a\}}]
=a\Pr(R\ge a),
$$

再除以正数 $a$。非负性不可删：负值可抵消均值，却不减少右尾。若已知 $R\ge b$，可对 $R-b\ge0$ 应用 Markov：
<!-- bilingual-en:start -->
Divide by a positive $a$.  Nonnegativity cannot be deleted: Negative values cancel out the mean without reducing the right-hand tail.  If $R\ge b$ is known, apply Markov to $R-b\ge0$:
<!-- bilingual-en:end -->

$$
\Pr(R\ge a)\le\frac{\mathbb E[R]-b}{a-b},\qquad a>b.
$$

### 4.6.4 Markov Bound — Online O33-03–O33-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.4_markov-bound|4.6.4]]。已知 $\mathbb E[R]=50$，求保证 $\Pr(R\ge x)\le1/2$ 的最小 $x$。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.4_markov-bound|4.6.4]].  Known $\mathbb E[R]=50$, minimum $x$ for guaranteed $\Pr(R\ge x)\le1/2$.
<!-- bilingual-en:end -->

> [!question]- O33-03–O33-06
> **O33-03** $R\ge0$：$50/x\le1/2$，故 $x=100$。
>
> **O33-04** $R\ge-50$：对 $R+50$ 用 Markov，$100/(x+50)\le1/2$，故 $x=150$。
>
> **O33-05** $R\ge30$：对 $R-30$ 用 Markov，$20/(x-30)\le1/2$，故 $x=70$。
>
> **O33-06** 仅知 $\Pr(R\le50)=0.7$ 不能保证非负下界，也不能用 Markov 得所求阈值；按题目约定答 6042。
> <!-- bilingual-en:start -->
> $R\ge0$:$50/x\le1/2$, so $x=100$.
> **O33-04**$R\ge-50$: Markov for $R+50$, $100/(x+50)\le1/2$, so $x=150$.
> **O33-05**$R\ge30$: Markov for $R-30$, $20/(x-30)\le1/2$, so $x=70$.
> **O33-06**Knowing only that $\Pr(R\le50)=0.7$ does not guarantee a non-negative lower bound, nor can Markov be used to obtain the desired threshold; answer 6042 as per the subject convention.
> <!-- bilingual-en:end -->

### 4.6.5 Chebyshev Bounds

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/uaa4P-kkLrA.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/uaa4P-kkLrA.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

[[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev 不等式证明]]：若 $R$ 有有限均值 $\mu$ 与方差 $\sigma^2$，则对 $a>0$，
<!-- bilingual-en:start -->
[[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev inequality proof]]: If $R$ has a finite mean $\mu$ and variance $\sigma^2$, then for $a>0$,
<!-- bilingual-en:end -->

$$
\boxed{\Pr(|R-\mu|\ge a)\le\frac{\sigma^2}{a^2}}.
$$

令非负变量 $Y=(R-\mu)^2$，事件 $|R-\mu|\ge a$ 等价于 $Y\ge a^2$。对 $Y$ 应用 Markov：
<!-- bilingual-en:start -->
Make the non-negative variable $Y=(R-\mu)^2$, the event $|R-\mu|\ge a$ equivalent to $Y\ge a^2$.  Apply Markov to $Y$:
<!-- bilingual-en:end -->

$$
\Pr(Y\ge a^2)\le\frac{\mathbb E[Y]}{a^2}
=\frac{\sigma^2}{a^2}.
$$

写成标准差倍数是 $\Pr(|R-\mu|\ge k\sigma)\le1/k^2$。它只给双侧上界，通常不紧，但不要求特定分布形状。
<!-- bilingual-en:start -->
It is written as a multiple of standard deviation $\Pr(|R-\mu|\ge k\sigma)\le1/k^2$.  It only gives a two-sided upper bound, which is usually not tight, but does not require a specific distribution shape.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-concentration-bounds.png|900]]

读图：Markov 只利用非负变量的均值控制单侧尾；Chebyshev 先把离均值的距离平方，再同时覆盖左右两端，阈值翻倍会让上界缩小四倍。
<!-- bilingual-en:start -->
Read: Markov only uses the mean of non-negative variables to control one-sided tail; Chebyshev first squared the distance from the mean and then covered the left and right ends at the same time, doubling the threshold will reduce the upper bound by four times.
<!-- bilingual-en:end -->

### 4.6.6 Inside the TA's Brain — Online O33-07–O33-09

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.6_inside-the-ta-s-brain|4.6.6]]。一百万神经元，放电数 $R$ 的均值 550,000；放电比不放电多 200,000 才识别坏答案。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.6_inside-the-ta-s-brain|4.6.6]]. There are one million neurons. The number $R$ that fire has mean 550,000; the system identifies a bad answer when at least 200,000 more neurons fire than do not fire.
<!-- bilingual-en:end -->

> [!question]- O33-07–O33-09
> **O33-07** $R-(1{,}000{,}000-R)\ge200{,}000$，故 $R\ge600{,}000$。
>
> **O33-08** Markov：$\Pr(R\ge600{,}000)\le550/600=11/12$。
>
> **O33-09** 若 $\sigma=25{,}000$，则目标单侧事件包含于 $|R-550{,}000|\ge50{,}000$，Chebyshev 给上界 $(25/50)^2=1/4$。这是对更大双侧事件的界，故合法但可能松。
> <!-- bilingual-en:start -->
> $R-(1{,}000{,}000-R)\ge200{,}000$, so $R\ge600{,}000$.
> **O33-08**Markov:$\Pr(R\ge600{,}000)\le550/600=11/12$.
> **O33-09** If $\sigma=25{,}000$, the target one-sided event is contained in $|R-550{,}000|\ge50{,}000$, so Chebyshev's inequality gives the upper bound $(25/50)^2=1/4$. This bounds the target event by a larger two-sided event, so the bound is valid but may be loose.
> <!-- bilingual-en:end -->

### 4.6.7 Variance

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Variance.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/o57CTwt1-ck.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Variance.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/o57CTwt1-ck.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

展开平方并用 $\mathbb E[R]=\mu$：
<!-- bilingual-en:start -->
Expand square and use $\mathbb E[R]=\mu$:
<!-- bilingual-en:end -->

$$
\operatorname{Var}(R)=\mathbb E[R^2]-\mu^2.
$$

常数平移不改变方差，缩放平方进入：
<!-- bilingual-en:start -->
Constant translation does not change the variance, zoom square into:
<!-- bilingual-en:end -->

$$
\operatorname{Var}(aR+b)=a^2\operatorname{Var}(R).
$$

一般地
<!-- bilingual-en:start -->
generally
<!-- bilingual-en:end -->

$$
\operatorname{Var}(R+S)
=\operatorname{Var}(R)+\operatorname{Var}(S)+2\operatorname{Cov}(R,S).
$$

若 $R,S$ independent，则 covariance 为 0。对一组变量，pairwise independence 已足以使所有交叉 covariance 为 0，从而方差相加；不需要 mutual independence。indicator $I\sim\operatorname{Bernoulli}(p)$ 的方差为 $p(1-p)$。
<!-- bilingual-en:start -->
If $R,S$ independent, covariance is 0.  For a set of variables, pairwise independence is sufficient so that all intersections covariance are 0, so that the variance is summed; mutual independence is not required.  The variance of indicator $I\sim\operatorname{Bernoulli}(p)$ is $p(1-p)$.
<!-- bilingual-en:end -->

### 4.6.8 Practice with Variance — Online O33-10–O33-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.8_practice-with-variance|4.6.8]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.8_practice-with-variance|4.6.8]].
<!-- bilingual-en:end -->

> [!question]- O33-10–O33-11
> **O33-10** $p(1-p)=1/4-(p-1/2)^2$，最大值 $1/4$，在 $p=1/2$ 达到。
>
> **O33-11** 独立 $X,Y$ 时
> $$\operatorname{Var}(aX+bY+c)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y).$$
> <!-- bilingual-en:start -->
> $p(1-p)=1/4-(p-1/2)^2$, the maximum value of $1/4$, reached at $p=1/2$.
> When**O33-11**Standalone $X,Y$
> $$\operatorname{Var}(aX+bY+c)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y).$$
> <!-- bilingual-en:end -->

### 4.6.9 Flipping Coins — Online O33-12–O33-15

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.9_flipping-coins|4.6.9]]。$X\sim\operatorname{Bin}(100,1/2)$。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.9_flipping-coins|4.6.9]].  $X\sim\operatorname{Bin}(100,1/2)$.
<!-- bilingual-en:end -->

> [!question]- O33-12–O33-15
> **O33-12** $\mathbb E[X]=50$。
>
> **O33-13** Markov：$\Pr(X\ge70)\le50/70=5/7$。
>
> **O33-14** 独立 indicators 方差相加：$\operatorname{Var}(X)=100(1/4)=25$。
>
> **O33-15** $X<30$ 或 $X>70$ 蕴含 $|X-50|\ge20$；Chebyshev 给 $25/20^2=1/16$。
> <!-- bilingual-en:start -->
> $\mathbb E[X]=50$.
> **O33-13**Markov:$\Pr(X\ge70)\le50/70=5/7$.
> **O33-14**The sum of independent indicators variances: $\operatorname{Var}(X)=100(1/4)=25$.
> **O33-15**$X<30$ or $X>70$ implies $|X-50|\ge20$; Chebyshev to $25/20^2=1/16$.
> <!-- bilingual-en:end -->

### 4.6.10 Practice with Bounds — Online O33-16–O33-17

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.10_practice-with-bounds|4.6.10]]。120 人均分 90。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.10_practice-with-bounds|4.6.10]].  120. 90 points per person.
<!-- bilingual-en:end -->

> [!question]- O33-16–O33-17
> **O33-16** 若分数非负，Markov 给至少 180 分的比例至多 $90/180=1/2$，人数至多 60。
>
> **O33-17** 若最低 30，对 $R-30$ 用 Markov：比例至多 $(90-30)/(180-30)=2/5$，人数至多 48。两个界都可由两点分布达到，故在给定信息下最优。
> <!-- bilingual-en:start -->
> If the score is non-negative, Markov gives at least 180 points a proportion of up to $90/180=1/2$, with a maximum of 60 people.
> **O33-17**If minimum 30, Markov for $R-30$: up to $(90-30)/(180-30)=2/5$, with up to 48 persons.  Both bounds can be reached by two-point distribution, so they are optimal under given information.
> <!-- bilingual-en:end -->

### 4.6.11 Implications of Expectation — Online O33-18–O33-21

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.11_implications-of-expectation|4.6.11]]。$X$ 为非负整数且 $\mathbb E[X]=5$。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.11_implications-of-expectation|4.6.11]].  $X$ is a non-negative integer and $\mathbb E[X]=5$.
<!-- bilingual-en:end -->

> [!question]- O33-18–O33-21
> **O33-18** $\Pr(X=5)$ 可以为 0，例如 $X=0,10$ 各半；也可为 1，例如常数 5，所以唯一保证项是“could be 0”。
>
> **O33-19** 必有正概率使 $X\le5$，否则 $X>5$ 几乎处处会令均值大于 5。
>
> **O33-20** $\mathbb E[X^2]$ 可以为 100：令 $X=20$ 概率 $1/4$，否则 0。且 $\mathbb E[X^2]\ge\mathbb E[X]^2=25$；官方反馈把这一式误写成自反不等式。
>
> **O33-21** Markov 给 $\Pr(X\ge1000)\le5/1000=1/200$；令 $X=1000$ 概率 $1/200$、否则 0 可达到，故不能保证更小通用界。
> <!-- bilingual-en:start -->
> $\Pr(X=5)$ can be 0, for example, half of $X=0,10$, or 1, for example, the constant 5, so the only guarantee is "could be 0".
> **O33-19**There must be a positive probability for $X\le5$, otherwise $X>5$ will almost everywhere make the mean greater than 5.
> **O33-20**$\mathbb E[X^2]$ can be 100: Let the $X=20$ probability be $1/4$, otherwise 0.  And $\mathbb E[X^2]\ge\mathbb E[X]^2=25$; official feedback miswrites this equation as a reflexive inequality.
> **O33-21**Markov to $\Pr(X\ge1000)\le5/1000=1/200$; let the $X=1000$ probability $1/200$, otherwise 0 can be reached, so that a smaller universal bound cannot be guaranteed.
> <!-- bilingual-en:end -->

### CP33 — 非官方独立题解（5 道）
<!-- bilingual-en:start -->
*CP33 — Unofficial Independent Questions (5)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp33.pdf#page=1|cp33]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp33.pdf#page=1|cp33]].
<!-- bilingual-en:end -->

> [!example]- C33-1–C33-5 完整解答
> **C33-1 冷牛病。** 令随机抽一头牛的体温为 $T$，每头牛等概率；$T\ge70$、$\mathbb E[T]=85$。对 $R=T-70\ge0$ 用 Markov，存活 $T\ge90$ 即 $R\ge20$：
> $$\Pr(\text{survive})\le\frac{15}{20}=\frac34.$$
> 400 头时可令 300 头为 $90^\circ$、100 头为 $70^\circ$，均温 $(300\cdot90+100\cdot70)/400=85$ 且恰 $3/4$ 存活，说明界紧。概率模型的 outcomes 是 400 头牛，均各概率 $1/400$；期望正是算术平均。
>
> **C33-2 赌博者。** 日胜局数 $W$ 的期望
> $$120(1/6)+60(1/2)+20(1/5)=54.$$
> Markov 给 $\Pr(W\ge108)\le1/2$。pairwise independence 足以使方差相加：
> $$\operatorname{Var}(W)=120\frac16\frac56+60\frac12\frac12+20\frac15\frac45=\frac{523}{15}.$$
> Chebyshev：
> $$\Pr(W\ge108)\le\Pr(|W-54|\ge54)\le\frac{523/15}{54^2}=\frac{523}{43740}.$$
>
> **C33-3 帽子错配。** $S_n=\sum_iX_i$，故 $\mathbb E[S_n]=n(1/n)=1$。对 $i\ne j$，
> $$\mathbb E[X_iX_j]=\Pr(X_i=X_j=1)=\frac1n\frac1{n-1}.$$
> 它不等于 $1/n^2$，所以不能直接用独立变量的方差和公式。但
> $$\mathbb E[S_n^2]=\sum_i\mathbb E[X_i]+2\sum_{i<j}\mathbb E[X_iX_j]=1+1=2,$$
> 从而 $\operatorname{Var}(S_n)=2-1^2=1$。若多于 10 人取回自己的帽子，则 $S_n\ge11$、$|S_n-1|\ge10$，Chebyshev 给概率至多 $1/100=1\%$。
>
> **C33-4 单色三角形。** 单个 triangle 单色概率 $m=r^3+b^3+g^3$；indicator 的均值 $m$、方差 $m(1-m)$。两三角形不共边时联合概率 $m^2$；共一边时为 $r^5+b^5+g^5$。当三色各 $1/3$，后者 $=1/81=m^2$，所以任意两个 triangle indicators 独立。令 $N=\binom n3$、$M$ 为单色三角形数：
> $$\mathbb E[M]=Nm=:\mu,\qquad\operatorname{Var}(M)=Nm(1-m)\le\mu.$$
> 因而
> $$\Pr(|M-\mu|>\sqrt{\mu\log\mu})\le\frac{\mu}{\mu\log\mu}=\frac1{\log\mu},$$
> 当 $n\to\infty$ 且固定 $m>0$ 时右边趋 0。
>
> **C33-5 正整数变量。** 保持 $\mathbb E[R]=2$，令 $R=N$ 的概率 $1/(N-1)$、$R=1$ 的概率 $1-1/(N-1)$，则方差随 $N$ 无界。因 $1/R\le1$，$\mathbb E[1/R]\le1$；上述构造使它任意接近 1，但等号会要求 $R=1$ 几乎处处并违背均值 2，所以 1 是 supremum、不取到。若再有 $R\le2$，正整数性和均值 2 强迫 $R=2$ 几乎处处，方差为 0。
> <!-- bilingual-en:start -->
> **C33-1 Cold cow disease.** Let $T$ be the temperature of a uniformly chosen cow. Then $T\ge70$ and $\mathbb E[T]=85$. Set $R=T-70\ge0$. Survival means $T\ge90$, equivalently $R\ge20$, so Markov's inequality gives
> $$\Pr(\text{survive})\le\frac{15}{20}=\frac34.$$
> For 400 cows, assign temperature $90^\circ$ to 300 and $70^\circ$ to 100. The average is $(300\cdot90+100\cdot70)/400=85$ and exactly $3/4$ survive, showing that the bound is tight. In this model the outcomes are the 400 cows, each with probability $1/400$, so expectation is the ordinary arithmetic mean.
> **C33-2 Gambler.**Expectations of $W$ in daily victories
> $$120(1/6)+60(1/2)+20(1/5)=54.$$
> Markov to $\Pr(W\ge108)\le1/2$.  pairwise independence is sufficient to add the variance:
> $$\operatorname{Var}(W)=120\frac16\frac56+60\frac12\frac12+20\frac15\frac45=\frac{523}{15}.$$
> Chebyshev:
> $$\Pr(W\ge108)\le\Pr(|W-54|\ge54)\le\frac{523/15}{54^2}=\frac{523}{43740}.$$
> **C33-3 Hat mismatch.** $S_n=\sum_iX_i$, so $\mathbb E[S_n]=n(1/n)=1$. For $i\ne j$,
> $$\mathbb E[X_iX_j]=\Pr(X_i=X_j=1)=\frac1n\frac1{n-1}.$$
> It is not equal to $1/n^2$, so you cannot use the variance and formulas of independent variables directly. However,
> $$\mathbb E[S_n^2]=\sum_i\mathbb E[X_i]+2\sum_{i<j}\mathbb E[X_iX_j]=1+1=2,$$
> So $\operatorname{Var}(S_n)=2-1^2=1$.  If more than 10 people retrieve their hats, $S_n\ge11$, $|S_n-1|\ge10$, Chebyshev give a probability of up to $1/100=1\%$.
> **C33-4 Monochromatic triangles.** A single triangle is monochromatic with probability $m=r^3+b^3+g^3$; its indicator has mean $m$ and variance $m(1-m)$. For two triangles that do not share an edge, the joint probability is $m^2$; if they share one edge, it is $r^5+b^5+g^5$. When all three colours have probability $1/3$, the latter equals $1/81=m^2$, so every pair of triangle indicators is independent. Let $N=\binom n3$ and let $M$ be the number of monochromatic triangles:
> $$\mathbb E[M]=Nm=:\mu,\qquad\operatorname{Var}(M)=Nm(1-m)\le\mu.$$
> thus
> $$\Pr(|M-\mu|>\sqrt{\mu\log\mu})\le\frac{\mu}{\mu\log\mu}=\frac1{\log\mu},$$
> whose right-hand side tends to $0$ as $n\to\infty$ with fixed $m>0$.
> **C33-5 A positive integer variable.**Keep $\mathbb E[R]=2$, let the probability of $R=N$ $1/(N-1)$, the probability of $R=1$ $1-1/(N-1)$, then the variance with $N$ unbounded.  Because $1/R\le1$,$\mathbb E[1/R]\le1$;the above construction makes it arbitrarily close to 1, but the equal sign would require $R=1$ almost everywhere and violate the mean value 2, so 1 is supremum, not taken.  If there is another $R\le2$, positive integers and mean 2 force $R=2$ almost everywhere, the variance is 0.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. Markov 的非负条件为何不可省？
> 2. Chebyshev 的 $a$ 是方差单位还是随机变量单位？
> 3. 方差和何时只需 pairwise independence？
>
> <!-- bilingual-en:start -->
> 1. Why is the non-negative condition of Markov indisputable?
> 2. Is Chebyshev's $a$ a unit of variance or a unit of random variables?
> 3. Variance and when only pairwise independence?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 负值可抵消均值却不减少正尾。2. 是随机变量单位；分母为 $a^2$。3. 有限和且各变量二阶矩存在时，pairwise independence 使每个 covariance 为 0。
> <!-- bilingual-en:start -->
> 1. Negative values offset the mean without reducing the positive tail.  2. Unit of random variable; denominator is $a^2$.  3. When the sum is finite and the second moment of each variable exists, pairwise independence makes each covariance equal to zero.
> <!-- bilingual-en:end -->

**知识链**：[[随机变量、分布与矩#期望、方差与协方差|均值]] → [[随机变量、分布与矩#期望、方差与协方差|方差]] → [[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov 证明]] → [[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev 证明]] → sampling bounds。
<!-- bilingual-en:start -->
**Knowledge Chain**: [[随机变量、分布与矩#期望、方差与协方差|mean]] → [[随机变量、分布与矩#期望、方差与协方差|variance]] → [[概率不等式与集中界#Markov：把均值当作尾部质量预算|Markov proof]] → [[概率不等式与集中界#Chebyshev：用 variance 控制双侧偏离|Chebyshev proof]] → sampling bounds.
<!-- bilingual-en:end -->

---

## Session 34 — Sampling & Confidence

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：样本均值为何随 $n$ 集中到总体均值？confidence 是关于参数还是关于程序？生日碰撞为什么可用 pairwise independence 计算方差？
<!-- bilingual-en:start -->
**Learning Problem**: Why does the Sample Mean converge to the Overall Mean with the $n$?  Is confidence about the parameter or about the program?  Why can I use pairwise independence to calculate variance for birthday collisions?
<!-- bilingual-en:end -->

**前置知识**：期望、方差、Chebyshev、pairwise independence。核心卡片：[[大数定律与中心极限定理#大数定律：稳定到什么|大数定律]]、[[大数定律与中心极限定理#大数定律：稳定到什么|弱大数定律证明]]、[[04_Probability#Session 34 — Sampling & Confidence|置信水平]]、[[04_Probability#Session 34 — Sampling & Confidence|置信区间]]。
<!-- bilingual-en:start -->
**Prerequisites**: Expectations, Variance, Chebyshev, pairwise independence.  Core cards: [[大数定律与中心极限定理#大数定律：稳定到什么|law of large numbers]], [[大数定律与中心极限定理#大数定律：稳定到什么|law of weak large numbers]], [[04_Probability#Session 34 — Sampling & Confidence|confidence level]], [[04_Probability#Session 34 — Sampling & Confidence|confidence interval]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session34.pdf#page=1|Session 34 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp34.pdf#page=1|cp34, pp. 1–4]]

### 4.7.1 Law of Large Numbers

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawLrgeNumbr.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-yo3YxfY47g.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawLrgeNumbr.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-yo3YxfY47g.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

设 $X_1,X_2,\dots$ pairwise independent，且共同均值 $\mu$、共同有限方差 $\sigma^2$。样本均值
<!-- bilingual-en:start -->
Suppose $X_1,X_2,\dots$ pairwise independent, and a common mean $\mu$ and a common finite variance $\sigma^2$.  sample mean
<!-- bilingual-en:end -->

$$
A_n=\frac1n\sum_{i=1}^{n}X_i
$$

满足 $\mathbb E[A_n]=\mu$，并因 pairwise independence，
<!-- bilingual-en:start -->
Meet $\mathbb E[A_n]=\mu$, and because of pairwise independence,
<!-- bilingual-en:end -->

$$
\operatorname{Var}(A_n)
=\frac1{n^2}\sum_{i=1}^{n}\operatorname{Var}(X_i)
=\frac{\sigma^2}{n}.
$$

Chebyshev 给
<!-- bilingual-en:start -->
Chebyshev, here
<!-- bilingual-en:end -->

$$
\Pr(|A_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0.
$$

这证明 $A_n$ [[大数定律与中心极限定理#极限定理入口|依概率收敛]]到 $\mu$，即 weak law；它不等同于“每条无限样本路径最终都收敛”的 strong law。
<!-- bilingual-en:start -->
This proves that $A_n$ [[大数定律与中心极限定理#极限定理入口|converges in probability]] to $\mu$, which is the weak law. It is not the strong-law claim that convergence holds almost surely along the infinite sample path.
<!-- bilingual-en:end -->

### 4.7.2 Not So Strong — Online O34-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.2_not-so-strong|4.7.2]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.2_not-so-strong|4.7.2]].
<!-- bilingual-en:end -->

> [!question]- O34-01
> $X_i$ 以相同概率 $f(i)$ 取 $\pm i2^i$，其余取 0。正负项对消，$\mathbb E[X_i]=0$；有限线性性给 $\mathbb E[S_n]=0$。这并不推出 $S_n$ 集中在 0，因为方差可能极大。
> <!-- bilingual-en:start -->
> Let $X_i$ take the values $+i2^i$ and $-i2^i$ with equal probability $f(i)$, and take $0$ otherwise. The positive and negative contributions cancel, so $\mathbb E[X_i]=0$; finite linearity then gives $\mathbb E[S_n]=0$. This does not imply that $S_n$ is concentrated near zero, because its variance may be extremely large.
> <!-- bilingual-en:end -->

### 4.7.3 Independent Sampling Theorem

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_IndependSmpling.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/MMn7q1M7pGI.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_IndependSmpling.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/MMn7q1M7pGI.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

对 Bernoulli samples $X_i\in\{0,1\}$，$\mathbb E[X_i]=p$、$\operatorname{Var}(X_i)=p(1-p)\le1/4$。若 pairwise independent，则样本比例 $A_n$ 满足
<!-- bilingual-en:start -->
For Bernoulli samples $X_i\in\{0,1\}$, $\mathbb E[X_i]=p$, $\operatorname{Var}(X_i)=p(1-p)\le1/4$.  If pairwise independent, the sample ratio $A_n$ satisfies
<!-- bilingual-en:end -->

$$
\Pr(|A_n-p|\ge\varepsilon)
\le\frac{1}{4n\varepsilon^2}.
$$

要把 failure probability 控制在 $\delta$ 以下，一个分布无关的充分条件是
<!-- bilingual-en:start -->
A sufficient condition for failure probability to be below $\delta$ is that
<!-- bilingual-en:end -->

$$
n\ge\frac1{4\delta\varepsilon^2}.
$$

所需样本量主要取决于误差容忍度与置信失败率，不直接随总体人数增长；这依赖均匀、独立或合适的无放回抽样设计。
<!-- bilingual-en:start -->
The required sample size is mainly determined by error tolerance and confidence failure rate, and does not directly increase with the population size; it depends on uniform, independent or suitable non-return sampling design.
<!-- bilingual-en:end -->

### 4.7.4 Sampling Coin Tosses — Online O34-02–O34-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.4_sampling-coin-tosses|4.7.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.4_sampling-coin-tosses|4.7.4]].
<!-- bilingual-en:end -->

> [!question]- O34-02–O34-04
> **O34-02** 公平币样本比例 $\mathbb E[A_n]=1/2$。
>
> **O34-03** $\operatorname{Var}(A_n)=1/(4n)$；要求 $\Pr(|A_n-1/2|>1/10)\le1/25$，Chebyshev 上界 $25/n$，故 $n\ge625$。
>
> **O34-04** $n=100$ 时令 $[1/(4n)]/\delta^2\le1/4$，得到 $\delta\ge1/10$；题问最小保证值为 $1/10$。
> <!-- bilingual-en:start -->
> FairCurrency Sample Ratio $\mathbb E[A_n]=1/2$.
> **O34-03**$\operatorname{Var}(A_n)=1/(4n)$; requires $\Pr(|A_n-1/2|>1/10)\le1/25$, Chebyshev upper bound $25/n$, so $n\ge625$.
> **O34-04**$n=100$ is $[1/(4n)]/\delta^2\le1/4$, get $\delta\ge1/10$, and the minimum guaranteed question value is $1/10$.
> <!-- bilingual-en:end -->

### 4.7.5 Birthday Matching

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Birthday.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TWVntUfXsKs.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Birthday.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TWVntUfXsKs.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

若 $n$ 人生日独立均匀落在 $d$ 天，令 $I_{ij}$ 指示第 $i,j$ 人同生日，匹配 pair 数
<!-- bilingual-en:start -->
If $n$'s birthday falls independently and evenly on $d$ days, $I_{ij}$ indicates that $i,j$'s birthday is the same, matching the pair number
<!-- bilingual-en:end -->

$$
M=\sum_{i<j}I_{ij}.
$$

每个 $\mathbb E[I_{ij}]=1/d$，故
<!-- bilingual-en:start -->
Per $\mathbb E[I_{ij}]=1/d$, therefore
<!-- bilingual-en:end -->

$$
\mathbb E[M]=\binom n2\frac1d.
$$

不同 pair indicators 即使共享一个人也 pairwise independent：例如给定第 $i$ 人生日后，第 $j,k$ 人各自匹配的概率仍为 $1/d$，联合为 $1/d^2$。因此
<!-- bilingual-en:start -->
Different pair indicators pairwise independent even if a person is shared: for example, given the $i$'s life date, the probability of the $j,k$'s matching each other is still $1/d$, and the union is $1/d^2$.  therefore
<!-- bilingual-en:end -->

$$
\operatorname{Var}(M)=\binom n2\frac1d\left(1-\frac1d\right).
$$

但整组 indicators 不 mutually independent，例如三个人的三对匹配事件存在传递约束。
<!-- bilingual-en:start -->
But the whole set of indicators is not mutually independent. For example, there are transmission constraints for three people's three pairs of matching events.
<!-- bilingual-en:end -->

### 4.7.6 Birthdays on Naboo — Online O34-05–O34-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.6_birthdays-on-naboo|4.7.6]]。200 个孩子、199 天。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.6_birthdays-on-naboo|4.7.6]].  200 children, 199 days.
<!-- bilingual-en:end -->

> [!question]- O34-05–O34-07
> **O34-05**
> $$\mathbb E[M]=\binom{200}{2}\frac1{199}=100.$$
>
> **O34-06**
> $$\operatorname{Var}(M)=\binom{200}{2}\frac1{199}\frac{198}{199}
> =\frac{19800}{199}.$$
> 官方反馈推导中误写 $\binom{100}{2}$，但答案栏和正确计算均对应 $\binom{200}{2}$。
>
> **O34-07** 区间 80 到 120 的补事件包含于 $|M-100|>20$；Chebyshev：
> $$\Pr(80\le M\le120)\ge1-\frac{19800/199}{20^2}
> =1-\frac{99}{398}.$$
> <!-- bilingual-en:start -->
> $$\mathbb E[M]=\binom{200}{2}\frac1{199}=100.$$
> **O34-06**
> $$\operatorname{Var}(M)=\binom{200}{2}\frac1{199}\frac{198}{199}
> =\frac{19800}{199}.$$
> The official feedback derivation incorrectly wrote $\binom{100}{2}$, but both the answer bar and the correct calculation correspond to $\binom{200}{2}$.
> **O34-07**Supplemental events in the range 80 to 120 are contained in $|M-100|>20$;Chebyshev:
> $$\Pr(80\le M\le120)\ge1-\frac{19800/199}{20^2}
> =1-\frac{99}{398}.$$
> <!-- bilingual-en:end -->

### 4.7.7 Sampling & Confidence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplingConfid.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Q-6Cw8tYVeY.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplingConfid.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Q-6Cw8tYVeY.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

未知总体比例 $p$ 是固定常数，抽样比例 $\hat p$ 才是随机变量。若抽样前证明
<!-- bilingual-en:start -->
The unknown population proportion $p$ is a fixed constant, whereas the sample proportion $\hat p$ is a random variable. If, before sampling, we establish that
<!-- bilingual-en:end -->

$$
\Pr_p(|\hat p-p|\le\varepsilon)\ge1-\delta,
$$

则随机区间 $[\hat p-\varepsilon,\hat p+\varepsilon]$ 在重复执行程序时至少以 $1-\delta$ 的比例覆盖 $p$。观察数据后，不能在频率学派解释下说“固定的 $p$ 有 $1-\delta$ 概率落在这一个已实现区间”；应说该构造方法有相应 coverage/confidence。
<!-- bilingual-en:start -->
then the random interval $[\hat p-\varepsilon,\hat p+\varepsilon]$ covers $p$ in at least a proportion $1-\delta$ of repeated applications. Under the frequentist interpretation, once the data have been observed we should not say that the fixed parameter $p$ has probability $1-\delta$ of lying in this realised interval. Instead, the interval-construction procedure has the stated coverage, or confidence level.
<!-- bilingual-en:end -->

### 4.7.8 Confidence — Online O34-08–O34-10

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.8_confidence|4.7.8]]。100 个样本给 $\pm1$、95% confidence，其他条件固定。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.8_confidence|4.7.8]].  100 samples were given to $\pm1$, 95% confidence, other conditions were fixed.
<!-- bilingual-en:end -->

> [!question]- O34-08–O34-10
> **O34-08** 总体从 5000 增至 10000：独立均匀抽样的分布无关界不变，仍 100。
>
> **O34-09** tolerance 从 1 加倍到 2，样本量按 $1/\varepsilon^2$ 缩为四分之一，得 25。
>
> **O34-10** failure probability 从 0.05 降为 0.01，Chebyshev 型样本量按 $1/\delta$ 放大 5 倍，得 500。
> <!-- bilingual-en:start -->
> from 5000 to 10000: The independent uniform sampling distribution has no change, but remains 100.
> **O34-09**tolerance doubled from 1 to 2, and the sample size was reduced by $1/\varepsilon^2$ to a quarter, yielding 25.
> **O34-10**failure probability was reduced from 0.05 to 0.01, and the sample size of Chebyshev was enlarged 5 times to 500 at $1/\delta$.
> <!-- bilingual-en:end -->

### 4.7.9 Random Sampling — Online O34-11–O34-12

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.9_random-sampling|4.7.9]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.9_random-sampling|4.7.9]].
<!-- bilingual-en:end -->

> [!question]- O34-11–O34-12
> **O34-11** 正确事实：某个固定选民的偏好已确定，概率是 0 或 1；随机抽法使每位选民成为第 3 个样本的概率相同；若第 2 人与第 1 人来自同州，其支持概率可能不等于全国 $p$。错误项把固定对象与随机抽中者混淆，或忽略重复抽样与条件选择。
>
> **O34-12** 观察 $\hat p=0.53$ 且程序保证 $\Pr(|\hat p-p|\le0.04)\ge0.95$ 后，正确表述是：“$p$ 在 $0.53\pm0.04$ 内，或者发生了至多 5% 的异常抽样结果”；也可说“我们对该区间有 95% confidence”。不能说 $p$ 本身现在以 95% 概率落入区间。
> <!-- bilingual-en:start -->
> A fixed voter's preference is already determined, so its probability is 0 or 1. The random sampling scheme gives every voter the same probability of becoming the third sampled person. If the second person comes from the same state as the first, their support probability need not equal the national proportion $p$. The incorrect options confuse a fixed individual with a randomly selected sample member, or ignore repeated sampling and conditional selection.
> **O34-12** After observing $\hat p=0.53$, given that the procedure guarantees $\Pr(|\hat p-p|\le0.04)\ge0.95$, the correct statement is: “Either $p$ lies within $0.53\pm0.04$, or an outcome from the procedure's exceptional 5% has occurred.” Equivalently, “We have 95% confidence in this interval.” Under a frequentist interpretation, we cannot say that the fixed parameter $p$ now has a 95% probability of lying in this realized interval.
> <!-- bilingual-en:end -->

### 4.7.10 Above Average Number of Fingers — Online O34-13

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.10_above-average-number-of-fingers|4.7.10]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.10_above-average-number-of-fingers|4.7.10]].
<!-- bilingual-en:end -->

> [!question]- O34-13
> 多数人手指数高于平均数，是因为缺失手指的总量多于额外手指的总量；“缺失远多于额外”与较弱的“至少略多”都足以解释。分布偏斜时，大多数观测高于均值并不矛盾；均值不是中位数。
> <!-- bilingual-en:start -->
> Most HMIs are above average because more fingers are missing than extra, and both "much more than extra" and weak "at least a little more" are sufficient explanations.  When the distribution is skewed, it is not contradictory that most observations are higher than the mean; the mean is not the median.
> <!-- bilingual-en:end -->

### CP34 — 非官方独立题解（6 道）
<!-- bilingual-en:start -->
*CP34 — Unofficial Independent Question (6)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp34.pdf#page=1|cp34]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp34.pdf#page=1|cp34]].
<!-- bilingual-en:end -->

> [!example]- C34-1–C34-6 完整解答
> **C34-1 Gallup poll。** indicator 方差 $p(1-p)\le1/4$。$n=1928$、误差 $\varepsilon=0.03$ 时，
> $$\Pr(|\hat p-p|\ge0.03)\le\frac1{4(1928)(0.03)^2}\approx0.1441,$$
> 所以 Chebyshev 只保证约 $85.59\%$ confidence。大于 99% 可用已知 Bernoulli/binomial 分布计算精确尾概率，或用更强的 Chernoff/normal approximation。即使接受全部计算，也不能给固定 $p$ 赋“高概率”；能评价的是区间构造程序的长期覆盖率。
>
> **C34-2 同生日 pair。** $E_{ij}$ 为匹配 indicator：
> $$\mathbb E[E_{ij}]=1/d,\quad\operatorname{Var}(E_{ij})=\frac1d(1-\frac1d),$$
> $$
> \mathbb E[D]=\binom n2/d,\quad
> \operatorname{Var}(D)=\binom n2\frac1d(1-\frac1d).
> $$
> 对 $n=500,d=7305$，$\mu\approx17.077$、$\sigma^2\approx17.075$。若 $|D-\mu|<6$，整数 $D\in[12,23]$；Chebyshev 给该事件概率至少 $1-17.075/36\approx0.5257>1/2$。
>
> **C34-3 弱大数定律的量词。**
> $$\forall\varepsilon>0\ \forall\delta>0\ \exists n_0\ \forall n\ge n_0:
> f(n,\varepsilon)\ge1-\delta.$$
> 顺序很重要：$n_0$ 可依赖 $\varepsilon,\delta$，但一旦选定，所有更大 $n$ 都满足。
>
> **C34-4 出版偏差。** “单次检验在零效应下有 5% false-positive rate”不等于“被选择发表的结果只有 5% 错”。若大量团队检验无效药物，期刊只看显著结果，非显著试验被隐藏；条件在“已显著且已投稿/发表”后，错误率可很高，甚至所有发表项都来自 false positives。试验彼此独立不能消除这种共同选择规则造成的 selection bias。
>
> **C34-5 总体大小为何不进入样本量主项。** 每次均匀抽一趟车，超速 indicator 的均值都是总体比例 $p$、方差至多 $1/4$；$n$ 次平均的方差按 $1/n$ 缩小。无论总体有一百万还是一亿，只要抽样近似独立且均匀，同一 $n$ 就给同一误差界。总体很小时无放回抽样反而产生有限总体修正，使方差更小。
>
> **C34-6 广义 pairwise sampling。** $A_n=n^{-1}\sum_iX_i$，且 $\operatorname{Var}(X_i)\le b$。pairwise independence 给
> $$\operatorname{Var}(A_n)=\frac1{n^2}\sum_i\operatorname{Var}(X_i)\le\frac bn.$$
> Chebyshev 即得
> $$\Pr(|A_n-\mu_n|\ge\varepsilon)\le\frac{b}{\varepsilon^2n}.$$
> 因右边趋 0，补事件概率趋 1，得到 generalized weak law。
> <!-- bilingual-en:start -->
> **C34-1 Gallup poll.**indicator variance $p(1-p)\le1/4$.  $n=1928$, error $\varepsilon=0.03$,
> $$\Pr(|\hat p-p|\ge0.03)\le\frac1{4(1928)(0.03)^2}\approx0.1441,$$
> So Chebyshev only guarantees about $85.59\%$ confidence.  More than 99% of that exact tail probability can be calculate using known Bernoulli/binomial distributions, or with a stronger Chernoff/normal approximation.  Even if all the calculations are accepted, the fixed $p$ cannot be given a "high probability"; what can be evaluated is the long-term coverage of the interval constructor.
> **C34-2 pair.**$E_{ij}$ matches indicator:
> $$\mathbb E[E_{ij}]=1/d,\quad\operatorname{Var}(E_{ij})=\frac1d(1-\frac1d),$$
> For $n=500,d=7305$, $\mu\approx17.077$, $\sigma^2\approx17.075$.  If $|D-\mu|<6$, the integer is $D\in[12,23]$; Chebyshev gives the event a probability of at least $1-17.075/36\approx0.5257>1/2$.
> **C34-3's quantifier for the law of weak large numbers.**
> $$\forall\varepsilon>0\ \forall\delta>0\ \exists n_0\ \forall n\ge n_0:
> f(n,\varepsilon)\ge1-\delta.$$
> Order is important: $n_0$ can rely on $\varepsilon,\delta$, but once selected, all larger $n$ are satisfied.
> **C34-4 publication bias.**"5% false-positive rate for a single test with zero effect" is not equal to "only 5% incorrect results are selected for publication".  If a large number of teams tested ineffective drugs, journals only looked at significant results, and non-significant trials were hidden; the condition was that after "significant and published/published," the error rate could be high, and even all published items were from false positives.  The independence of trials does not eliminate the selection bias caused by this common selection rule.
> **Why population size does not determine the sample-size requirement in C34-5.** The mean of each speeding indicator is the population proportion $p$, and its variance is at most $1/4$. Averaging $n$ approximately independent, uniformly sampled observations reduces the variance by a factor of $n$. Thus the same $n$ gives the same error bound whether the population contains one million or one hundred million members. If the population is small and sampling is without replacement, the finite-population correction makes the variance smaller still.
> **C34-6 Generalized pairwise sampling.**$A_n=n^{-1}\sum_iX_i$ and $\operatorname{Var}(X_i)\le b$.  pairwise independence, here
> $$\operatorname{Var}(A_n)=\frac1{n^2}\sum_i\operatorname{Var}(X_i)\le\frac bn.$$
> Chebyshev's inequality then gives
> $$\Pr(|A_n-\mu_n|\ge\varepsilon)\le\frac{b}{\varepsilon^2n}.$$
> As the right-hand side tends to zero, the probability of the complementary event tends to one, yielding the generalised weak law.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. weak law 的收敛对象是什么？
> 2. 95% confidence 是给参数赋 95% 后验概率吗？
> 3. birthday pair indicators 为何可 pairwise independent 却不 mutually independent？
>
> <!-- bilingual-en:start -->
> 1. What is the convergence object for weak law?
> 2. Does 95% confidence assign a 95% posterior probability to the parameter?
> 3. Why does birthday pair indicators pairwise independent not mutually independent?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 对每个固定 $\varepsilon>0$，偏差概率趋 0。2. 不是；它描述随机区间程序的覆盖率。3. 任意两对的联合仍分解，但三对之间有“前两对匹配则第三对必匹配”的约束。
> <!-- bilingual-en:start -->
> 1. For each fixed $\varepsilon>0$, the deviation probability tends to zero. 2. No; it describes the coverage of the random interval procedure. 3. Every pair of indicators has a factorising joint distribution, but three indicators can be constrained: if the first two matching events occur, the third must occur as well.
> <!-- bilingual-en:end -->

**知识链**：[[随机变量、分布与矩#期望、方差与协方差|样本方差缩放]] → [[大数定律与中心极限定理#大数定律：稳定到什么|弱大数定律]] → independent sampling → [[04_Probability#Session 34 — Sampling & Confidence|置信水平]] / [[04_Probability#Session 34 — Sampling & Confidence|置信区间]] → 选择偏差。
<!-- bilingual-en:start -->
**Knowledge chain:** [[随机变量、分布与矩#期望、方差与协方差|scaling of the sample-mean variance]] → [[大数定律与中心极限定理#大数定律：稳定到什么|weak law of large numbers]] → independent sampling → [[04_Probability#Session 34 — Sampling & Confidence|confidence level]] / [[04_Probability#Session 34 — Sampling & Confidence|confidence interval]] → selection bias.
<!-- bilingual-en:end -->

---

## Session 35 — Random Walks & PageRank

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：图上的随机移动怎样写成矩阵？stationary distribution 存在、唯一与收敛是同一件事吗？PageRank 为什么是一个长期访问概率？
<!-- bilingual-en:start -->
**Learning Problem**: How to Write Random Movements on Graph into Matrix?  Is stationary distribution the same thing as existence, uniqueness and convergence?  Why is PageRank a long-term access probability?
<!-- bilingual-en:end -->

**前置知识**：有向图、矩阵乘法、条件概率、递推。核心卡片：[[离散时间马尔可夫链#Markov 性与转移矩阵|图上随机游走]]、[[离散时间马尔可夫链#Markov 性与转移矩阵|Markov 链]]、[[离散时间马尔可夫链#平稳分布与长期收敛|平稳分布]]、[[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge**:Diagram, Matrix Multiplication, Conditional Probability, Recursion.  Core cards: [[离散时间马尔可夫链#Markov 性与转移矩阵|random walk on graph]], [[离散时间马尔可夫链#Markov 性与转移矩阵|Markov chain]], [[离散时间马尔可夫链#平稳分布与长期收敛|stationary distribution]], [[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session35.pdf#page=1|Session 35 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp35.pdf#page=1|cp35, pp. 1–3]]

### 4.8.1 Random Walks

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomWalks.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-j7MoM3P_J8.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomWalks.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-j7MoM3P_J8.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

有限状态 random walk 的 transition matrix $P$ 满足
<!-- bilingual-en:start -->
transition matrix $P$ satisfaction for finite state random walk
<!-- bilingual-en:end -->

$$
P_{uv}=\Pr(X_{t+1}=v\mid X_t=u),\qquad
P_{uv}\ge0,\qquad \sum_vP_{uv}=1.
$$

Markov property 是：给定当前状态后，下一步与更早历史无关。若行向量 $d_t$ 表示时刻 $t$ 的状态分布，则 total probability 给
<!-- bilingual-en:start -->
Markov property is: Given the current state, the next step is independent of the earlier history.  If the row vector $d_t$ represents the state distribution of the moment $t$, then total probability gives
<!-- bilingual-en:end -->

$$
d_{t+1}=d_tP,\qquad d_t=d_0P^t.
$$

路径 $v_0\to\cdots\to v_t$ 的概率是 $d_0(v_0)\prod_{i=0}^{t-1}P_{v_iv_{i+1}}$。
<!-- bilingual-en:start -->
The probability of the path $v_0\to\cdots\to v_t$ is $d_0(v_0)\prod_{i=0}^{t-1}P_{v_iv_{i+1}}$.
<!-- bilingual-en:end -->

### 4.8.2 Stationary Distributions

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_StatinaryDist.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iZX8WEGZTVw.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_StatinaryDist.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iZX8WEGZTVw.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

分布 $\pi$ stationary，当且仅当
<!-- bilingual-en:start -->
$\pi$ stationary if and only if
<!-- bilingual-en:end -->

$$
\pi=\pi P,\qquad \pi_v\ge0,\qquad\sum_v\pi_v=1.
$$

它表达每个顶点的流入概率等于其质量。必须区分：
<!-- bilingual-en:start -->
It expresses that the inflow probability of each vertex is equal to its mass.  It must be distinguished from:
<!-- bilingual-en:end -->

- **存在**：有限 Markov chain 至少有一个 stationary distribution；
- **唯一**：有限 irreducible chain 有唯一 stationary distribution；
- **从任意初值收敛**：还需排除周期振荡，常用充分条件是 irreducible 且 aperiodic。
<!-- bilingual-en:start -->
-**Present**: At least one stationary distribution for a finite Markov chain;
-**Unique*: Limited irreducible chain has Unique stationary distribution;
-**Converge from any initial value**: Periodic oscillations must also be excluded, and the usual sufficient conditions are irreducible and aperiodic.
<!-- bilingual-en:end -->

两节点确定性交替链有 stationary $(1/2,1/2)$，但从单点出发的分布永远在两个单点间跳动，不收敛。这直接否定“有 stationary 就会趋近 stationary”。
<!-- bilingual-en:start -->
The two-node deterministic alternating chain has stationary $(1/2,1/2)$, but the distribution from a single point always jumps between the two points and does not converge.  This simply denies that "if there is an stationary, it will approach an stationary".
<!-- bilingual-en:end -->

### 4.8.3 PageRank

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pagerank.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QKO_2WQkZ0k.pdf#page=1|transcript]]
<!-- bilingual-en:start -->
Resource: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pagerank.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QKO_2WQkZ0k.pdf#page=1|transcript]]
<!-- bilingual-en:end -->

在网页 digraph 上，基础 Google walk 从页面 $u$ 均匀选择一个 outgoing link，$P_{uv}=1/\operatorname{outdeg}(u)$。[[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]] 把 stationary mass 解释为长期访问频率：被许多高质量页面指向的页面得到较大流入。
<!-- bilingual-en:start -->
On the page digraph, the base Google walk evenly selects a outgoing link,$P_{uv}=1/\operatorname{outdeg}(u)$ from the page $u$.  [[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]] interprets stationary mass as a long-term frequency of visits: pages directed by many high quality pages get a large inflow.
<!-- bilingual-en:end -->

实际模型加入 teleportation：
<!-- bilingual-en:start -->
Add teleportation to actual model:
<!-- bilingual-en:end -->

$$
P_\alpha=\alpha P+(1-\alpha)\frac1n\mathbf1\mathbf1^\mathsf T,\qquad0<\alpha<1,
$$

并为 dangling pages 指定合法转移。正的 teleportation 使有限链 irreducible 且 aperiodic，因此 stationary distribution 唯一并从任意初值收敛。排名仍取决于链接图与 damping/teleportation 选择，不是网页“客观价值”的无假设度量。
<!-- bilingual-en:start -->
and specify legal transfer for dangling pages.  A positive teleportation makes the finite chain irreducible and aperiodic, so that stationary distribution is unique and converges from any initial value.  Ranking still depends on the link graph and damping/teleportation selection and is not a hypothetical measure of the "objective value" of a webpage.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-random-walk-pagerank.png|900]]

读图：左侧沿边传播当前概率，右侧 stationary 向量满足“传播一次保持不变”；PageRank 再加入少量全局跳转，打破封闭类与周期。
<!-- bilingual-en:start -->
Read: Current probability of left-hand side propagation along edges, right-hand side stationary vector satisfies "one time propagation remains unchanged"; PageRank adds a small number of global jumps to break the closed class and cycle.
<!-- bilingual-en:end -->

### 4.8.4 Random Walks — Online O35-01–O35-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.4_random-walks|4.8.4]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.4_random-walks|4.8.4]].
<!-- bilingual-en:end -->

> [!question]- O35-01–O35-04
> **O35-01** 图 1 中 $x\leftrightarrow y$ 确定性交替，stationary 的 $d(x)=1/2$。
>
> **O35-02** 归一化给 $d(y)=1/2$。
>
> **O35-03** 图 2 中 $w\to z$ 概率 1，$z\to w,z$ 概率 $0.9,0.1$。平衡式 $\pi_w=0.9\pi_z$ 与归一化给 $\pi_z=10/19$。
>
> **O35-04** 从 $x$ 出发不会接近 $(1/2,1/2)$；偶数时刻全在 $x$，奇数时刻全在 $y$，周期为 2。
> <!-- bilingual-en:start -->
> Figure 1 $x\leftrightarrow y$ deterministic alternation, stationary $d(x)=1/2$.
> **O35-02**normalized to $d(y)=1/2$.
> **O35-03**Figure 2 shows $w\to z$ probability 1 and $z\to w,z$ probability $0.9,0.1$.  Balanced $\pi_w=0.9\pi_z$ and normalized to $\pi_z=10/19$.
> **O35-04**from $x$ will not approach $(1/2,1/2)$, even moment all at $x$, odd moment all at $y$, period 2.
> <!-- bilingual-en:end -->

### 4.8.5 Random Walks Continued — Online O35-05–O35-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.5_random-walks-cont|4.8.5]]。
<!-- bilingual-en:start -->
Original and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.5_random-walks-cont|4.8.5]].
<!-- bilingual-en:end -->

> [!question]- O35-05–O35-07
> **O35-05** 图 2 从 $w$ 出发会趋近 stationary。前几步 $(1,0)\to(0,1)\to(0.9,0.1)\to(0.09,0.91)\to(0.819,0.181)$；振荡幅度按 $0.9^t$ 衰减。
>
> **O35-06** 图 3 有两个 absorbing states $a,d$；任意 $\lambda\delta_a+(1-\lambda)\delta_d$ 都 stationary，$\lambda\in[0,1]$，故有不可数无限多个。
>
> **O35-07** 令 $h_b,h_c$ 为从 $b,c$ 最终吸收到 $d$ 的概率。由图
> $$h_b=\tfrac12h_c,\qquad h_c=\tfrac12h_b+\tfrac12,$$
> 解得 $h_b=1/3$。所以从 $b$ 长期在 $d$ 的概率趋近 $1/3$。
> <!-- bilingual-en:start -->
> Figure 2 Approaching stationary from $w$.  In the first few steps, $(1,0)\to(0,1)\to(0.9,0.1)\to(0.09,0.91)\to(0.819,0.181)$, the oscillation amplitude decays by $0.9^t$.
> **O35-06**Figure 3 has two absorbing states $a,d$; any $\lambda\delta_a+(1-\lambda)\delta_d$ has stationary, $\lambda\in[0,1]$, so there are infinitely many.
> **O35-07**Let $h_b,h_c$ be the probability that $b,c$ will eventually absorb to $d$.  By graph
> $$h_b=\tfrac12h_c,\qquad h_c=\tfrac12h_b+\tfrac12,$$
> $h_b=1/3$.  So the long-term probability from $b$ to $d$ approaches $1/3$.
> <!-- bilingual-en:end -->

### CP35 — 非官方独立题解（3 道）
<!-- bilingual-en:start -->
*CP35 — Unofficial Independent Question (3)*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp35.pdf#page=1|cp35]]。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp35.pdf#page=1|cp35]].
<!-- bilingual-en:end -->

> [!example]- C35-1–C35-3 完整解答
> **C35-1 三个 random-walk graphs。**
> (a) 图 1 stationary 为 $(1/2,1/2)$。
> (b) 从 $x$ 出发在 $\delta_x,\delta_y$ 间交替；初始分布只有恰为 $(1/2,1/2)$ 时各步都 stationary，因此才“收敛”到它。
> (c) 图 2 stationary 为 $(\pi_w,\pi_z)=(9/19,10/19)$。
> (d) 从 $w$ 的分布如 O35-05，振荡衰减并趋近该分布。
> (e) 图 3 的全部 convex mixtures $\lambda\delta_a+(1-\lambda)\delta_d$ stationary，故不可数多个。
> (f) 从 $b$ 最终到 $d$ 概率为 $1/3$。
> (g) 不 strongly connected 但 stationary 唯一的例子：$x\to y$ 概率 1、$y\to y$ 概率 1；唯一 stationary 是 $\delta_y$。
>
> **C35-2 uniform stationary 的充要条件。** 有 $n$ 个顶点，uniform $u(v)=1/n$。传播后顶点 $v$ 的质量
> $$ (uP)(v)=\sum_u\frac1nP_{uv}=\frac1n\sum_{u\in\operatorname{into}(v)}p(u,v). $$
> 它对每个 $v$ 等于 $u(v)=1/n$，当且仅当每个顶点的 incoming edge probabilities 之和为 1。
>
> **C35-3 symmetric Google graph。** 若错误地用 $d(v)=\operatorname{outdeg}(v)/e$ 排名，页面可添加许多 outgoing links 人为抬高自己的“分数”；真实 PageRank 看 stationary inflow，增加外链通常分散自身流出概率，不能这样直接获益。证明 $d$ stationary：对每条 $u\to v$，
> $$d(u)p(u,v)=\frac{\operatorname{outdeg}(u)}e\frac1{\operatorname{outdeg}(u)}=\frac1e.$$
> symmetric 使 $\operatorname{indeg}(v)=\operatorname{outdeg}(v)$，所以流入总量为 $\operatorname{outdeg}(v)/e=d(v)$；又 $\sum_vd(v)=1$，故确为分布。
> <!-- bilingual-en:start -->
> **C35-1 three random-walk graphs.**
> (a) Figure 1 stationary is $(1/2,1/2)$.
> (b) Alternate between $\delta_x,\delta_y$ starting from $x$; the initial distribution is only $(1/2,1/2)$ when the steps are stationary, and thus "converge" to it.
> (c) Figure 2 stationary is $(\pi_w,\pi_z)=(9/19,10/19)$.
> (d) From a $w$ distribution such as O35-05, the oscillation decays and approaches this distribution.
> (e) Every convex mixture $\lambda\delta_a+(1-\lambda)\delta_d$ is stationary for Figure 3, so there are uncountably many stationary distributions.
> (f) The probability from $b$ to $d$ is $1/3$.
> (g) Not strongly connected but the only example of stationary: $x\to y$ probability 1, $y\to y$ probability 1; the only stationary is $\delta_y$.
> A necessary and sufficient condition for**C35-2 uniform stationary.**has $n$ vertices, uniform $u(v)=1/n$.  The mass of the post-propagation vertex $v$
> $$ (uP)(v)=\sum_u\frac1nP_{uv}=\frac1n\sum_{u\in\operatorname{into}(v)}p(u,v). $$
> It is equal to $u(v)=1/n$ for each $v$ if and only if the sum of incoming edge probabilities for each vertex is 1.
> **C35-3 Symmetric Google graph.** If one incorrectly ranked pages by $d(v)=\operatorname{outdeg}(v)/e$, a page could inflate its score by adding outgoing links. Actual PageRank depends on stationary inflow; adding outgoing links normally spreads a page's outgoing probability more thinly and therefore does not provide this direct advantage. To prove that $d$ is stationary, observe that for every edge $u\to v$,
> $$d(u)p(u,v)=\frac{\operatorname{outdeg}(u)}e\frac1{\operatorname{outdeg}(u)}=\frac1e.$$
> symmetric makes $\operatorname{indeg}(v)=\operatorname{outdeg}(v)$, so the total inflow is $\operatorname{outdeg}(v)/e=d(v)$; and $\sum_vd(v)=1$, so it is indeed distributed.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. stationary distribution 存在是否保证从单点出发收敛？
> 2. irreducible 与 aperiodic 分别解决什么问题？
> 3. PageRank 中 teleportation 的数学作用是什么？
>
> <!-- bilingual-en:start -->
> 1. Does the existence of stationary distribution guarantee convergence from a single point?
> 2. What issues does irreducible and aperiodic address separately?
> 3. What is the mathematical role of teleportation in PageRank?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 自检答案
> 1. 不保证，周期链可永久振荡。2. irreducible 排除多个封闭类并给唯一性；aperiodic 排除周期振荡。3. 使有限链连通且非周期，从而得到唯一、可收敛的 stationary distribution。
> <!-- bilingual-en:start -->
> 1. No. A periodic chain may oscillate forever. 2. Irreducibility rules out multiple closed communicating classes and gives uniqueness; aperiodicity rules out periodic oscillation. 3. Teleportation makes the finite chain irreducible and aperiodic, yielding a unique stationary distribution to which the chain converges.
> <!-- bilingual-en:end -->

**知识链**：[[离散时间马尔可夫链#Markov 性与转移矩阵|图上随机游走]] → [[离散时间马尔可夫链#Markov 性与转移矩阵|转移矩阵]] → [[离散时间马尔可夫链#平稳分布与长期收敛|平稳分布]] → irreducibility/aperiodicity → [[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]]。
<!-- bilingual-en:start -->
**Knowledge Chain**: [[离散时间马尔可夫链#Markov 性与转移矩阵|random walk on graph]] → [[离散时间马尔可夫链#Markov 性与转移矩阵|transfer matrix]] → [[离散时间马尔可夫链#平稳分布与长期收敛|stationary distribution]] → irreducibility/aperiodicity → [[离散时间马尔可夫链#平稳分布与长期收敛|PageRank]].
<!-- bilingual-en:end -->

---

## 全章方法地图
<!-- bilingual-en:start -->
*full chapter method map*
<!-- bilingual-en:end -->

| 问题特征 | 首选工具 | 必做检查 |
|---|---|---|
| 多阶段随机实验 | 概率树、四步法 | outcome 是否互斥穷尽、边为条件概率 |
| 已知新信息 | 条件概率、Bayes | 条件事件概率与信息协议 |
| “互不影响” | independence 乘积式 | pairwise 还是 mutual |
| 只关心某个数值 | 随机变量、PMF/CDF | 离散还是连续 |
| 求平均贡献 | indicator、线性期望 | 期望是否存在；通常不需独立 |
| 求离散程度 | variance/covariance | 二阶矩与独立条件 |
| 仅知均值的正尾 | Markov | 变量非负、阈值为正 |
| 知均值和方差的偏离 | Chebyshev | 双侧事件与平方阈值 |
| 样本比例误差 | WLLN/sampling bound | 均匀抽样、pairwise independence |
| 解读 poll 区间 | confidence/coverage | 参数固定、区间随机 |
| 图上长期访问 | stationary equation | 唯一性与收敛性分开 |
<!-- bilingual-en:start -->
|Problem Characteristics|Preferences|Required Checks|
|—|—|—|
|Multi-stage Random Experiment|Probability Tree, Four-Step Method|Whether outcome is Mutually Exclusive Exhaustion, and Edges are Conditional Probabilities|
| Known New | Conditional Probability, Bayes | Conditional Event Probability and Information Protocol |
| "Do Not Affect Each Other" | independence Product | pairwise or mutual |
| Only care about a number | Random variables, PMF/CDF | Discrete or Continuous |
| Average Contribution | indicator, Linear Expectation | Whether Expectation is Existence; Usually no independence is needed |
|Dispersion Degree|variance/covariance|Second Moment and Independent Condition|
| Only the positive tail of the mean is known | Markov | The variable is non-negative and the threshold is positive |
|Knowing Deviation of Mean and Variance|Chebyshev |Two-sided Event and Squared Threshold|
|Sample Ratio Error|WLLN/sampling bound|Uniform Sampling, pairwise independence |
|Interpreting poll Interval|confidence/coverage |Arguments Fixed, Interval Random|
| Long-term access on the graph | stationary equation | Uniqueness is separate from convergence |
<!-- bilingual-en:end -->

## 覆盖与资源核对
<!-- bilingual-en:start -->
*Coverage and Resource Reconciliation*
<!-- bilingual-en:end -->

- 官方在线题：O28-01–11（11）、O29-01–06（6）、O30-01–05（5）、O31-01–21（21）、O32-01–12（12）、O33-01–21（21）、O34-01–13（13）、O35-01–07（7），合计 **96**。
- Classroom Problems：C28-1–6、C29-1–4、C30-1–4、C31-1–5、C32-1–5、C33-1–5、C34-1–6、C35-1–3，合计 **38**；本地 cp32 的主题错配已经显式标注。
- 作业：Problem Set 11 的 P11-1–3 与 Problem Set 12 的 P12-1–3，合计 **6** 个编号主问题，全部子问均解答。
- 官方 blocks：Session 28–35 依次为 8、10、7、8、13、11、10、5 个，共 **72**；每个 video/exercise 按原顺序出现。
- 教学图：7 张 PNG 均在首次展开相关概念处以宽度 900 嵌入，并附中文读图说明。
<!-- bilingual-en:start -->
- Official online questions: O28-01-11(11), O29-01-06(6), O30-01-05(5), O31-01-21(21), O32-01-12(12), O33-01-21(21), O34-01-13(13), O35-01-07(7), total**96**.
- Classroom Problems:C28-1-6, C29-1-4, C30-1-4, C31-1-5, C32-1-5, C33-1-5, C34-1-6, C35-1-3, total**38**; topic mismatches for local cp32 are explicitly noted.
- Job: P11-1-3 for Problem Set 11 and P12-1-3 for Problem Set 12, total**6**encoded main questions, all subquestions answered.
- Official blocks:Session 28-35 are 8, 10, 7, 8, 13, 11, 10, 5 in order of**72**; each video/exercise appears in the original order.
- Instructional drawings: 7 PNGs are embedded at 900 Width at the first expansion of the relevant concepts, with a Chinese readout description.
<!-- bilingual-en:end -->

> [!summary] 一句话收束
> **概率空间**定义可能性，**条件化**更新信息，**独立性**允许乘法，**随机变量**把结果数值化，**期望与方差**描述中心和离散，**集中界与抽样**把局部随机变成总体保证，**stationary distribution**则把同一思想延伸到长期随机过程。
> <!-- bilingual-en:start -->
> a **probability space** defines what can happen, **conditioning** updates information, **independence** justifies factorisation, a **random variable** assigns numbers to outcomes, **expectation and variance** describe centre and dispersion, **concentration bounds and sampling** turn local randomness into population-level guarantees, and a **stationary distribution** extends the same perspective to long-run random processes.
> <!-- bilingual-en:end -->
