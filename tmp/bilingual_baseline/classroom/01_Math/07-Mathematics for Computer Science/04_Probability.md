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

本笔记严格依照 MIT OCW Spring 2015 Unit 4 的官方 block 顺序整理：Session 28 → Session 29 → Problem Set 11 → Session 30 → Session 31 → Session 32 → Problem Set 12 → Session 33 → Session 34 → Session 35。课程入口见 [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)；本地材料总索引见 [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]]；前置章节见 [[03_Counting|Unit 3 — Counting]]。

> [!info] 题解来源与编号
> - `O28-01` 表示 Session 28 的第 1 个 online feedback prompt；96 个 prompt 均逐个中文转述，并保留可核验计算。
> - `C28-1` 表示 `cp28.pdf` 的 Problem 1。CP 与 Problem Set 的课程包只含题目，所以下文标为**非官方独立题解**。
> - 本地 `cp32.pdf` 的实际内容是图论复习题，与 Session 32 的 expectation 主题不一致。本笔记保留链接并逐题解答，同时明确标注这项来源异常。

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

---

## Session 28 — Intro to Discrete Probability

### 本节问题、前置知识与资源

**学习问题**：怎样把随机实验变成数学对象？树上路径概率为什么相乘、互斥叶的概率为什么相加？事件不互斥时如何避免重复计算？

**前置知识**：集合、可数求和、乘法法则与容斥。核心卡片：[[Discrete Probability Space|离散概率空间]]、[[Four-Step Probability Method|概率四步法]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session28.pdf#page=1|Session 28 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp28.pdf#page=1|cp28, pp. 1–4]]

### 4.1.1 Tree Model — 从实验到概率树

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_tree_model.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/dEsFEK4vnV4.pdf#page=1|transcript]]

一个离散概率空间是二元组 $(\Omega,\Pr)$：

- $\Omega$ 是互斥且穷尽的 outcomes（结果）集合；
- 每个 $\omega\in\Omega$ 有 $\Pr(\omega)\ge0$；
- $\sum_{\omega\in\Omega}\Pr(\omega)=1$。

事件 $A\subseteq\Omega$ 的概率定义为

$$
\Pr(A)=\sum_{\omega\in A}\Pr(\omega).
$$

在概率树中，一条根到叶的路径依次发生若干选择。边标签是“已到达当前节点”条件下下一步的概率，故路径概率是边概率之积；一个事件通常含若干互斥叶，故事件概率是相应叶概率之和。这就是四步法：

1. 定义 sample space；
2. 指出目标事件；
3. 给每个 outcome 合法概率；
4. 对目标 outcomes 求和。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-sample-space-events.png|900]]

读图：全集框是 $\Omega$，内部区域是事件；先确认基本结果互斥且穷尽，再用“叶上相乘、事件内相加”读取概率。

> [!warning] 结果不等概率时不能用“有利个数/总个数”
> $\Pr(A)=|A|/|\Omega|$ 只在有限且每个 outcome 等概率时成立。概率树的叶深不同、边概率不同，都可能使叶不等概率。

### 4.1.2 Socks and Shoes — Online O28-01–O28-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.2_socks-and-shoes|4.1.2]]。鞋为 2 黑、3 棕；袜为 3 红、4 棕、6 黑，分别独立均匀抽一双。

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

### 4.1.3 Simplified Monty Hall Tree — 压缩对称分支

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplifiedMonty.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L30HPgryd6I.pdf#page=1|transcript]]

Monty Hall 的完整树可按“奖品位置—初选—主持人开门—是否换门”展开。若策略只关心输赢，可以把初选状态压缩成：

- 初选中奖，概率 $1/3$；换门必输；
- 初选未中，概率 $2/3$；主持人排除另一扇羊门后，换门必胜。

所以 stay 胜率 $1/3$，switch 胜率 $2/3$。压缩树合法的条件是：合并分支在后续行为和目标事件上完全等价，并把被合并概率相加；不能只因为图形大小相似就合并。

### 4.1.4 Simplify Prize Tree — Online O28-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.4_simplify-prize-tree|4.1.4]]。

> [!question]- O28-07
> 哪些理由足以压缩 switch 策略的树？
>
> **答案**：奖品位置后的分支对称；“奖品门与初选门”可压缩为“初选是否命中”；初选命中时 switch 与主持人具体开哪门无关且必输；初选羊门的两个对称分支有同样概率和结局，可合并。错误理由是“所有 outcome 等概率”或“分支画得一样大”，二者都不是概率等价条件。

### 4.1.5 Sample Spaces — 公理与推导规则

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SampleSpaces.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Amd_bNYzgUw.pdf#page=1|transcript]]

由定义可以逐步推出常用规则。若 $A\cap B=\varnothing$，则

$$
\Pr(A\cup B)=\Pr(A)+\Pr(B).
$$

对一般事件，把 $B$ 拆成 $(B-A)\mathbin{\dot\cup}(A\cap B)$：

$$
\Pr(A\cup B)=\Pr(A)+\Pr(B)-\Pr(A\cap B).
$$

取 $B=\overline A$ 得 complement rule：$\Pr(\overline A)=1-\Pr(A)$。若 $A\subseteq B$，则 $B=A\mathbin{\dot\cup}(B-A)$，故单调性 $\Pr(A)\le\Pr(B)$。

并集界（union bound）不要求独立：

$$
\Pr\!\left(\bigcup_{i\ge1}A_i\right)\le\sum_{i\ge1}\Pr(A_i).
$$

证明时令 $B_1=A_1$，$B_i=A_i-\bigcup_{j<i}A_j$。则 $B_i$ 两两不交、$B_i\subseteq A_i$，且两边并集相同。因此

$$
\Pr\!\left(\bigcup_iA_i\right)=\sum_i\Pr(B_i)\le\sum_i\Pr(A_i).
$$

### 4.1.6 Sum Rule Practice — Online O28-08

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.6_sum-rule-practice|4.1.6]]。

> [!question]- O28-08
> 三个两两互斥事件各有概率 $1/4$，至少一个发生的概率是多少？
>
> **解**：互斥使交集项全为零，故 $\Pr(E_0\cup E_1\cup E_2)=3/4=0.75$。

### 4.1.7 Addition Rule — Online O28-09

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.7_addition-rule|4.1.7]]。

> [!question]- O28-09
> 两颗公平骰子点数和为 7 或 11 的概率？
>
> **解**：36 个等概率有序结果中，和 7 有 6 个、和 11 有 2 个，且二事件互斥：
> $$
> \Pr(S=7\text{ or }11)=\frac6{36}+\frac2{36}=\frac29\approx0.22.
> $$

### 4.1.8 Fun With Coins — Online O28-10–O28-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S28_4.1.8_fun-with-coins|4.1.8]]。

> [!question]- O28-10–O28-11
> **O28-10** 公平硬币永远只出反面的概率？前 $n$ 次全反为 $2^{-n}$；“永远全反”包含在每个此前缀事件中，故概率至多 $2^{-n}$ 对所有 $n$ 成立，只能为 $0$。
>
> **O28-11** 无限次投掷至少一次正面的概率？它是上一事件的补集，故 $1-0=1$。概率 1 不表示逻辑上不存在全反序列，而表示该集合测度为 0。

### CP28 — 非官方独立题解（6 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp28.pdf#page=1|cp28]]。

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

> [!question]- 三道自检
> 1. 两个不互斥事件能否直接相加概率？缺少什么修正项？
> 2. 并集界为什么不要求独立性？
> 3. 概率为 0 是否等于逻辑上不可能？
>
> [!success]- 自检答案
> 1. 不能；减去 $\Pr(A\cap B)$。2. 证明只用不交化与单调性。3. 不等于；无限样本空间中可有非空零测事件。

**知识链**：[[Discrete Probability Space|离散概率空间]] → [[Four-Step Probability Method|概率四步法]] → sum/addition/complement rules → union bound → 条件概率。

---

## Session 29 — Conditional Probability

### 本节问题、前置知识与资源

**学习问题**：得知新信息后为何要缩小样本空间？怎样从正向诊断率反推出患病后验？Monty Hall 与“两孩问题”为什么必须说明信息产生机制？

**前置知识**：事件交并、概率树与 partition。核心卡片：[[Conditional Probability and Bayes Theorem|条件概率与 Bayes 定理]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session29.pdf#page=1|Session 29 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp29.pdf#page=1|cp29, pp. 1–3]]

### 4.2.1 Conditional Probability Definitions

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ConditProbability.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Cu9_LaaWgHo.pdf#page=1|transcript]]

若 $\Pr(B)>0$，在已知 $B$ 发生后，$A$ 的条件概率定义为

$$
\Pr(A\mid B)=\frac{\Pr(A\cap B)}{\Pr(B)}.
$$

分母把原概率空间重新归一化到 $B$；若 $\Pr(B)=0$，这个初等定义没有意义。移项得到 multiplication rule：

$$
\Pr(A\cap B)=\Pr(A\mid B)\Pr(B)
=\Pr(B\mid A)\Pr(A).
$$

连续多步树则有 chain rule：

$$
\Pr(A_1\cap\cdots\cap A_n)
=\Pr(A_1)\prod_{i=2}^{n}\Pr(A_i\mid A_1\cap\cdots\cap A_{i-1}),
$$

前提是出现的条件事件概率非零。

### 4.2.2 Dicey Sum — Online O29-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.2_dicey-sum|4.2.2]]。

> [!question]- O29-01
> 已知两骰点数和为 4，至少一颗为 3 的概率？条件空间只有 $(1,3),(2,2),(3,1)$ 三个等概率结果，其中两个含 3，故答案 $2/3\approx0.67$。

### 4.2.3 Law of Total Probability

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawTotalProbab.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/F3y8qupFfUs.pdf#page=1|transcript]]

若 $B_1,\dots,B_m$ 两两互斥、并为 $\Omega$，且所用 $\Pr(B_i)>0$，则

$$
\Pr(A)=\sum_{i=1}^{m}\Pr(A\cap B_i)
=\sum_{i=1}^{m}\Pr(A\mid B_i)\Pr(B_i).
$$

这是“按原因分支，再把到达同一结果的路径相加”。若某 $B_i$ 的概率为 0，可直接省略该项，而不是定义 $\Pr(A\mid B_i)$。

### 4.2.4 Cavities and Candy — Online O29-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.4_cavities-and-candy|4.2.4]]。

> [!question]- O29-02
> $\Pr(C)=1/4$，$\Pr(E\mid C)=4/5$，$\Pr(E\mid\bar C)=1/3$。求吃糖概率。
>
> $$
> \Pr(E)=\frac45\frac14+\frac13\frac34=\frac9{20}=0.45.
> $$
> 第二项必须乘 $\Pr(\bar C)=3/4$；官方反馈文字有一处把它误写成 $\Pr(C)$，但数值计算使用的是补事件。

### 4.2.5 Bayes' Theorem

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BayesTheorm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/e-yQFC6dACA.pdf#page=1|transcript]]

由交集概率的两种分解得到

$$
\Pr(B_j\mid A)
=\frac{\Pr(A\mid B_j)\Pr(B_j)}{\sum_i\Pr(A\mid B_i)\Pr(B_i)},
$$

其中 $\{B_i\}$ 是 partition，且 $\Pr(A)>0$。分子是“likelihood × prior”，分母是所有原因对证据的总贡献。罕见病检测中，即使 sensitivity 很高，若先验患病率很低，false positives 仍可能主导分母。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-bayes-tree.png|900]]

读图：从每个原因分支乘到证据叶得到联合概率，再用所有证据叶之和归一化；后验不是把条件符号简单倒过来。

### 4.2.6 Two Boys — Online O29-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.6_two-boys|4.2.6]]。

> [!question]- O29-03
> 两个孩子中“至少一个是男孩”，两男概率？在出生次序等概率且信息就是事件 $B=\{BB,BG,GB\}$ 时，条件空间三项等概率，故 $\Pr(BB\mid B)=1/3$。
>
> 边界：若信息是“较大的孩子是男孩”，条件空间只有 $BB,BG$，答案为 $1/2$；所以必须说明观察协议。

### 4.2.7 Monty Hall Problem — 信息协议决定后验

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MontyHallConfus.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BEAv82FinM0.pdf#page=1|transcript]]

经典的 **switch 策略在游戏开始前的胜率**只依赖主持人知道奖品位置、一定打开未选羊门、一定给换门机会：初选错误的概率是 $2/3$，而初选一旦错误，主持人只能排除另一扇羊门，换门必胜，所以 switch 胜率为 $2/3$。

但若还要在看见“主持人具体打开了哪一扇门”后计算后验，就必须知道他的开门协议。若他在两扇可开羊门中均匀随机选择，则观察任一具体开门结果后，初选门仍有后验 $1/3$，唯一未开的门有 $2/3$；若他按不对称的确定规则选门，这两个条件后验可能改变，尽管事前 switch 胜率仍是 $2/3$。若主持人可能随机开到奖品、可能不提供换门，或其行为还依赖其他信息，就必须重建概率树，不能沿用经典结论。

### 4.2.8 Conditional Probability — Online O29-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.8_conditional-probability|4.2.8]]。

> [!question]- O29-04
> $\Pr(A)=0.8$、$\Pr(L)=0.4$、$\Pr(L\mid A)=0.3$，求 $\Pr(A\mid L)$。
>
> $$
> \Pr(A\mid L)=\frac{\Pr(L\mid A)\Pr(A)}{\Pr(L)}
> =\frac{0.3\cdot0.8}{0.4}=0.6.
> $$

### 4.2.9 Dicey Game — Online O29-05

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.9_dicey-game|4.2.9]]。

> [!question]- O29-05
> 两骰和 7 则赢，double 则重投，其他输，最多三投；已知首投 double，最终赢率？条件信息等于已经消耗一次，剩两投。每投赢 $1/6$、double $1/6$，故
> $$
> \Pr(W\mid\text{first pair})=\frac16+\frac16\frac16=\frac7{36}.
> $$

### 4.2.10 Watch Out for Crocodiles — Online O29-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S29_4.2.10_watch-out-for-crocodiles|4.2.10]]。

> [!question]- O29-06
> 雨、晴、冰雹概率分别 $1/4,1/4,1/2$；见鳄鱼条件概率分别 $1/2,5/8,1$。由全概率公式
> $$
> \Pr(C)=\frac14\frac12+\frac14\frac58+\frac12=\frac{25}{32}.
> $$

### CP29 — 非官方独立题解（4 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp29.pdf#page=1|cp29]]。

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

> [!question]- 三道自检
> 1. 为什么 $\Pr(A\mid B)$ 要求 $\Pr(B)>0$？
> 2. sensitivity 为 99% 是否足以推出阳性者 99% 患病？
> 3. Monty Hall 的主持人若在剩余门中随机开一扇、并可能开到奖品，原结论还能直接用吗？
>
> [!success]- 自检答案
> 1. 要除以 $\Pr(B)$ 并重新归一化。2. 不足，还需患病先验与 false-positive rate。3. 不能；“主持人必定避开奖品”这一 likelihood 已被改变，必须重建概率树。

**知识链**：条件空间 → multiplication rule → total probability → [[Conditional Probability and Bayes Theorem|Bayes]] → 信息协议 → independence。

---

## Problem Set 11 — Discrete & Conditional Probability

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps11.pdf#page=1|Problem Set 11, pp. 1–3]]。以下为非官方独立题解。

> [!example]- P11-1 三色矩阵中的单色矩形
> **(a)** 长度 4 的一行共有 $3^4=81$ 种颜色 pattern；82 行由 pigeonhole principle 至少两行完全相同。
>
> **(b)** 在这两条相同行中，4 个列位置只用 3 色，至少两列同色；这两列与两行的四个交点同色，形成矩形。
>
> **(c)** 每一行中至少有一对位置同色。用“颜色 + 两列位置”标记一行，可选标签数为
> $$3\binom42=18.$$
> 19 行中至少两行共享同一标签，即同一对列上出现同一种颜色，仍形成单色矩形。若一行有多个标签，任选一个固定规则产生的标签即可应用 pigeonhole。

> [!example]- P11-2 红黑牌的最优停时
> 对含 $r$ 张红牌、$b$ 张黑牌的随机牌堆，证明任意策略胜率至多 $b/(r+b)$。对 $n=r+b$ 归纳。$n=1$ 时唯一牌若黑则 $b/n=1$ 且必胜，若红则 $b/n=0$ 且必败，基例成立。立即取牌的胜率正是 $b/n$；若跳过首张，再按任意最优后续策略，归纳假设给
> $$
> \Pr(\text{win after skip})
> \le\frac rn\frac b{n-1}+\frac bn\frac{b-1}{n-1}
> =\frac bn.
> $$
> 第一项是先揭红牌，第二项是先揭黑牌。任何随机化策略只是“取/跳”两者的凸组合，也不能超过 $b/n$。因此 26 红 26 黑时最优胜率 $1/2$，立即取顶牌已经达到。

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

---

## Session 30 — Independence & Causality

### 本节问题、前置知识与资源

**学习问题**：两个事件“不互相提供信息”如何写成公式？pairwise independence 为什么不够？统计相关为什么不自动等于因果关系？

**前置知识**：条件概率、Bayes、乘法法则。核心卡片：[[Independence of Events|事件独立性]]、[[Mutual Independence|相互独立]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session30.pdf#page=1|Session 30 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp30.pdf#page=1|cp30, pp. 1–4]]

### 4.3.1 Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Independence.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/1vQ2x5O_xqk.pdf#page=1|transcript]]

事件 $A,B$ 独立，定义为

$$
\Pr(A\cap B)=\Pr(A)\Pr(B).
$$

若 $\Pr(B)>0$，它等价于 $\Pr(A\mid B)=\Pr(A)$；乘积定义更基本，因为即使 $\Pr(B)=0$ 也合法。独立性会传给补事件，例如

$$
\Pr(A\cap\bar B)=\Pr(A)-\Pr(A\cap B)
=\Pr(A)[1-\Pr(B)]=\Pr(A)\Pr(\bar B).
$$

互斥与独立方向相反：若 $A,B$ 互斥且都具有正概率，则 $\Pr(A\cap B)=0$，不可能等于正数 $\Pr(A)\Pr(B)$。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-independence-grid.png|900]]

读图：独立时每个交叉格的面积等于对应行边际与列边际之积；仅仅“看起来没有重叠”表示互斥，不表示独立。

### 4.3.2 Independent Dice Rolls — Online O30-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.2_independent-dice-rolls|4.3.2]]。

> [!question]- O30-01
> 两颗骰子中哪些事件对独立？正确项为 $(A_1,B_1)$、$(A_1,B_6)$、$(A_1,S_7)$。前两项来自两骰独立；最后一项满足
> $$
> \Pr(A_1\cap S_7)=\frac1{36}=\frac16\frac6{36}.
> $$
> 反例：$A_3\cap S_2=\varnothing$；而 $A_2\cap S_6$ 概率 $1/36\neq(1/6)(5/36)$。

### 4.3.3 Mutual Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MutualIndepend.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/wJzBU7Do1ls.pdf#page=1|transcript]]

$A_1,\dots,A_n$ mutually independent，要求对每个非空索引集 $J\subseteq\{1,\dots,n\}$ 都有

$$
\Pr\!\left(\bigcap_{j\in J}A_j\right)=\prod_{j\in J}\Pr(A_j).
$$

只检查所有 pair 得到 pairwise independence；只检查全部 $n$ 个的交集也不够，因为中间大小的子集仍可能失败。相互独立蕴含 pairwise，反向不成立。

### 4.3.4 Mutually Independent Dice Rolls — Online O30-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.4_mutually-independent-dice-rolls|4.3.4]]。

> [!question]- O30-02
> 三骰事件组中 mutually independent 的只有 $(A_1,B_1,C_1)$ 与 $(A_1,B_6,C_3)$，因为它们分别只限制三次独立投掷。含总和事件的选项会产生约束；例如
> $$
> \Pr(A_1\cap B_1\cap S_7)=\frac1{216}\neq\frac16\frac16\frac{15}{216}.
> $$

### 4.3.5 Independent vs Disjoint — Online O30-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.5_independent-vs-disjoint|4.3.5]]。

> [!question]- O30-03
> 两事件何时能同时互斥和独立？互斥给 $\Pr(A\cap B)=0$，独立要求它等于 $\Pr(A)\Pr(B)$，所以至少一个事件概率必须为 0。官方选项写“至少一个为空”；在一般概率空间中更精确的条件是“至少一个为零概率事件”，它未必是空集。

### 4.3.6 Labeled Balls — Online O30-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.6_labeled-balls|4.3.6]]。

> [!question]- O30-04
> 从 $110,101,011,000$ 等概率抽一串，$A_i$ 表示第 $i$ 位为 1。三事件均概率 $1/2$，任意两者交集恰含一串、概率 $1/4$，故 pairwise independent；但三者交集为空，而边际乘积为 $1/8$，故不 mutually independent，也不互斥。

### 4.3.7 Paradox — Online O30-05 与因果边界

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S30_4.3.7_paradox|4.3.7]]。

> [!question]- O30-05
> 若 $A,B$ 各自使 $\Pr(H\mid E)>\Pr(H)$，$A\cup B$ 是否必为 $H$ 的证据？不必。公平骰令 $H=\{3,4\}$、$A=\{1,2,3,4\}$、$B=\{3,4,5,6\}$。则 $\Pr(H)=1/3$，$\Pr(H\mid A)=\Pr(H\mid B)=1/2$；但 $A\cup B=\Omega$，条件概率回到 $1/3$。

独立性和条件概率描述联合分布，不自动给出 causal direction。共同原因、选择偏差或对 collider 条件化都能制造或消除相关。要主张因果，还需干预设计或额外结构假设。

### CP30 — 非官方独立题解（4 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp30.pdf#page=1|cp30]]。

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

> [!question]- 三道自检
> 1. pairwise independent 是否足以推出 mutually independent？
> 2. 两个正概率互斥事件能独立吗？
> 3. 相关性为何不能单独证明因果？
>
> [!success]- 自检答案
> 1. 不足，需检查所有非空子集。2. 不能。3. 共同原因、选择机制和反向因果都可产生同一联合分布。

**知识链**：[[Conditional Probability and Bayes Theorem|条件概率]] → [[Independence of Events|两两独立]] → [[Mutual Independence|相互独立]] → 随机变量独立 → 方差可加。

---

## Session 31 — Random Variables & Density Functions

### 本节问题、前置知识与资源

**学习问题**：怎样把复杂 outcome 压缩成一个数值？离散 PMF、CDF 与事件概率如何互换？“两个随机变量独立”究竟需要检查什么？

**前置知识**：函数、条件概率、独立性、二项式系数。核心卡片：[[Random Variable|随机变量]]、[[Probability Mass Function|概率质量函数]]、[[Probability Density Function|概率密度函数]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session31.pdf#page=1|Session 31 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp31.pdf#page=1|cp31, pp. 1–3]]

### 4.4.1 Bigger Number Game

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_BigerNmberGme.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/BH4qlkYCLW0.pdf#page=1|transcript]]

对手写两个不同数，只随机展示一个。看似未见数比已见数大的概率是 $1/2$，但可以先独立抽一个 threshold $T$：看到 $x$ 后，若 $x<T$ 就换，否则保留。若 $T$ 落在两个数之间，无论展示哪个都必胜；否则胜率为 $1/2$。因此

$$
\Pr(\text{win})=\frac12+\frac12\Pr(T\text{ lies between the numbers})>\frac12
$$

只要 threshold 分布给任意非空区间正概率。优势来自额外随机化，不是预测对手；若数值范围有限，还可选择专门的离散 threshold。

### 4.4.2 Random Variables & Independence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomVaribles.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VJzv6WJTtNc.pdf#page=1|transcript]]

随机变量 $R$ 是从样本空间到数值集合的函数 $R:\Omega\to\mathbb R$。随机的是输入 outcome，不是函数规则。事件

$$
[R=r]:=\{\omega\in\Omega:R(\omega)=r\}.
$$

离散随机变量的 PMF 为

$$
p_R(r)=\Pr(R=r),\qquad p_R(r)\ge0,\qquad \sum_rp_R(r)=1.
$$

MIT 材料把这个离散质量函数也记作 $\operatorname{PDF}_R$；现代常用语中应叫 PMF，而连续型变量的 PDF $f_R$ 满足 $\Pr(a\le R\le b)=\int_a^b f_R(x)\,dx$，单点概率通常为 0。

对**离散**随机变量 $R,S$，独立等价于对所有值 $r,s$ 都有：

$$
\Pr(R=r,S=s)=\Pr(R=r)\Pr(S=s).
$$

等价地，由 $R$ 决定的任一事件与由 $S$ 决定的任一事件独立。只验证协方差为 0 通常不够。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-random-variable-pmf.png|900]]

读图：多个 sample outcomes 可映到同一随机变量值；某根柱子的高度是该逆像事件的总概率，而 CDF 是从左向右累加柱高。

### 4.4.3 Odd Heads and Matches — Online O31-01–O31-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.3_odd-heads-and-matches|4.4.3]]。三枚公平硬币，$I_O$ 指示正面数为奇，$M$ 指示三次全相同。

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

### 4.4.4 Uniform & Binomial Random Variables

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_UniformBinomial.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L2yOSFsMvnc.pdf#page=1|transcript]]

在含 $n$ 个值的有限集合上 uniform distribution 给每值概率 $1/n$。若做 $n$ 次 mutually independent Bernoulli trials，每次成功概率 $p$，成功数 $X$ 服从 binomial：

$$
\Pr(X=k)=\binom nkp^k(1-p)^{n-k},\qquad k=0,\dots,n.
$$

$\binom nk$ 选择哪些 trial 成功；每个固定成功位置 pattern 的概率由独立性给出 $p^k(1-p)^{n-k}$。若 trials 不独立或成功率不相同，不能直接套此公式。

### 4.4.5 Late for a Date — Online O31-08–O31-10

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.5_late-for-a-date|4.4.5]]。Jess 在 $-10,\dots,10$ 分钟均匀到达；Sean 的延误数 $T=S+5\sim\operatorname{Bin}(10,1/3)$，两者独立。

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

### 4.4.6 A Random Number — Online O31-11–O31-18

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.6_a-random-number|4.4.6]]。首枚偏币以 $3/5$ 出正面；正面则掷公平骰返回点数，反面则投三枚公平币并返回正面数的两倍。$F$ 为首投正面 indicator。

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

### 4.4.7 PDF to CDF — Online O31-19

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.7_pdf-to-cdf|4.4.7]]。

CDF 定义为 $F_X(x)=\Pr(X\le x)$，必定单调不减、右连续，且两端极限为 0 与 1。若 $X$ 在整数 $1,\dots,12$ 上均匀，

> [!question]- O31-19
> $$F_X(8)=\sum_{k=1}^{8}\frac1{12}=\frac23\approx0.67.$$
> 这里是离散 uniform；区间 $[1,12]$ 上的连续 uniform 会得到 $(8-1)/(12-1)=7/11$，二者不可混用。

### 4.4.8 Dice and Coin Game — Online O31-20–O31-21

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S31_4.4.8_dice-and-coin-game|4.4.8]]。先掷骰得 $X$，再投 $X$ 枚公平币，$Y$ 为正面数。

> [!question]- O31-20–O31-21
> **O31-20** 对 $X$ 分层；只有 $X=4,5,6$ 可得 4 个正面：
> $$\Pr(Y=4)=\frac16\left(\frac1{16}+\frac5{32}+\frac{15}{64}\right)=\frac{29}{384}.$$
>
> **O31-21**
> $$\Pr(X=5\mid Y=4)
> =\frac{(1/6)\binom54/2^5}{29/384}=\frac{10}{29}.$$

### CP31 — 非官方独立题解（5 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp31.pdf#page=1|cp31]]。

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

> [!question]- 三道自检
> 1. 随机变量与它的取值有什么区别？
> 2. 离散 PMF 和连续 PDF 的单点概率有何不同？
> 3. 怎样由 CDF 恢复离散 PMF？
>
> [!success]- 自检答案
> 1. 随机变量是 outcome 到数值的函数；取值是函数输出。2. PMF 柱高就是单点概率；连续 PDF 的点高不是点概率。3. $p_X(k)=F_X(k)-F_X(k^-)$，整数支撑时为 $F_X(k)-F_X(k-1)$。

**知识链**：[[Random Variable|随机变量]] → [[Probability Mass Function|PMF]] / CDF → uniform 与 binomial → 随机变量独立 → [[Expected Value|期望]]。

---

## Session 32 — Expectation

### 本节问题、前置知识与资源

**学习问题**：期望为什么是概率加权平均而非“最可能值”？为什么不独立也能拆开和的期望？怎样用 indicator 避免求完整分布？

**前置知识**：离散随机变量、PMF、条件概率、无穷级数。核心卡片：[[Expected Value|期望]]、[[Indicator Random Variable|指示随机变量]]、[[Linearity of Expectation|期望线性性]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session32.pdf#page=1|Session 32 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp32.pdf#page=1|local cp32, pp. 1–3]]

### 4.5.1 Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Expectation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/YVQdVzSkcmQ.pdf#page=1|transcript]]

离散随机变量 $R$ 的期望为

$$
\mathbb E[R]=\sum_r r\,\Pr(R=r),
$$

前提是该和绝对收敛，或至少正负部分的处理使期望有定义。对任意函数 $g$，LOTUS 给

$$
\mathbb E[g(R)]=\sum_r g(r)\Pr(R=r),
$$

无需先求 $g(R)$ 的完整 PMF。期望可以不在支撑集内，例如 51 次公平投币的正面数期望 25.5，却绝不取 25.5。

### 4.5.2 Uneven Dice — Online O32-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.2_uneven-dice|4.5.2]]。

> [!question]- O32-01
> 骰子以 $1/4$ 概率出偶数、$3/4$ 出奇数，同类内部均匀。偶数各概率 $1/12$，奇数各概率 $1/4$，故
> $$
> \mathbb E[R]=\frac{1+3+5}{4}+\frac{2+4+6}{12}=\frac{13}{4}=3.25.
> $$
> 官方反馈首个求和公式漏写了取值系数，但后续展开与答案正确。

### 4.5.3 Expected Number of Heads

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ExpectNumber.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/D9l-pIg1Ayo.pdf#page=1|transcript]]

若 $I_i$ 指示第 $i$ 次成功，则总成功数 $X=\sum_iI_i$。由于 $\mathbb E[I_i]=\Pr(I_i=1)=p_i$，线性性给

$$
\mathbb E[X]=\sum_i p_i.
$$

这一步不需要独立；独立性只在推导 binomial PMF 或方差相加时需要。

### 4.5.4 Expected Number of Heads — Online O32-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.4_expected-number-of-heads|4.5.4]]。

> [!question]- O32-02
> 200 枚公平硬币正面数期望：$\sum_{i=1}^{200}\mathbb E[I_i]=200/2=100$。

### 4.5.5 Total Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_TotalExpectatn.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/mqoDXWrSais.pdf#page=1|transcript]]

若 $B_1,\dots,B_m$ 构成 partition，且相关条件期望有定义，则

$$
\mathbb E[R]=\sum_i\mathbb E[R\mid B_i]\Pr(B_i).
$$

证明从定义展开：

$$
\sum_r r\Pr(R=r)
=\sum_r r\sum_i\Pr(R=r\mid B_i)\Pr(B_i)
=\sum_i\Pr(B_i)\mathbb E[R\mid B_i].
$$

交换求和需要有限情形，或无穷情形下满足非负性/绝对收敛等条件。

### 4.5.6 Another Dice and Coin Game — Online O32-03

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.6_another-dice-and-coin-game|4.5.6]]。

> [!question]- O32-03
> 公平币正面则反复掷骰直到奇数，反面则直到偶数。条件分布分别在 $\{1,3,5\}$ 与 $\{2,4,6\}$ 均匀，条件均值 3 与 4：
> $$\mathbb E[R]=\tfrac12(3)+\tfrac12(4)=3.5.$$

### 4.5.7 Mean Time to Failure

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MeanTimeFailure.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Dqx56lZ_icg.pdf#page=1|transcript]]

每轮独立地以概率 $p>0$ 首次成功，等待轮数 $T\in\{1,2,\dots\}$ 服从 geometric：

$$
\Pr(T=k)=(1-p)^{k-1}p.
$$

用 first-step analysis，第一轮必消耗 1；若失败（概率 $1-p$），剩余等待与原问题同分布：

$$
\mathbb E[T]=1+(1-p)\mathbb E[T],
\qquad
\mathbb E[T]=\frac1p.
$$

若各轮不独立或成功率随时间变化，这个结论不能直接使用。

### 4.5.8 Three Machines Failing — Online O32-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.8_three-machines-failing|4.5.8]]。

> [!question]- O32-04
> 三台机器每轮各以 $1/3$ 失败且 mutually independent，同时失败概率 $(1/3)^3=1/27$。轮次独立时首次同时失败的期望等待为 $1/(1/27)=27$。

### 4.5.9 Linearity of Expectation

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LinearityExpect.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/KFcodn4qfrQ.pdf#page=1|transcript]]

对有限个有期望的随机变量，

$$
\mathbb E\!\left[\sum_{i=1}^{n}a_iR_i+c\right]
=\sum_{i=1}^{n}a_i\mathbb E[R_i]+c.
$$

证明可直接在共同样本空间求和：

$$
\sum_{\omega}\left(\sum_i a_iR_i(\omega)+c\right)\Pr(\omega)
=\sum_i a_i\sum_{\omega}R_i(\omega)\Pr(\omega)+c.
$$

这里从未分解联合概率，所以不要求 $R_i$ 独立。相反，$\mathbb E[RS]=\mathbb E[R]\mathbb E[S]$ 通常要求独立；偶然相等不能反推出独立。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-expectation-variance.png|900]]

读图：期望定位概率质量的“平衡点”，方差则用到该点的平方距离加权；同均值的两个分布可以有完全不同的离散程度。

### 4.5.10 Fair and Biased Coins — Online O32-05

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.10_fair-and-biased-coins|4.5.10]]。

> [!question]- O32-05
> 100 枚公平币与 100 枚正面率 $1/4$ 的偏币，总正面数期望
> $$100(1/2)+100(1/4)=75.$$
> 不需要 200 次投掷相互独立。

### 4.5.11 Binomial Board Breaking — Online O32-06–O32-08

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.11_binomial-board-breaking|4.5.11]]。$X\sim\operatorname{Bin}(5,0.8)$。

> [!question]- O32-06–O32-08
> **O32-06** $\Pr(X=2)=\binom52(0.8)^2(0.2)^3=0.0512$。
>
> **O32-07** $\Pr(X\le3)=1-\binom54(0.8)^4(0.2)-(0.8)^5=0.26272$。
>
> **O32-08** $\mathbb E[X]=np=5(0.8)=4$。

### 4.5.12 Great Expectations — Online O32-09–O32-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.12_great-expectations|4.5.12]]。

> [!question]- O32-09–O32-11
> **O32-09** 公平 6 面骰和 12 面骰之和期望：$(6+1)/2+(12+1)/2=10$。
>
> **O32-10** 以 $1/6$ 选均匀 $1,\dots,99$（均值 50），否则选均匀 $1,\dots,999$（均值 500）：
> $$\mathbb E[G]=\frac16(50)+\frac56(500)=425.$$
>
> **O32-11** 两电脑独立时，输出乘积期望 $50\cdot500=25000$。若不独立，边际均值不足以确定乘积期望。

### 4.5.13 Expectation of a Uniform Distribution — Online O32-12

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S32_4.5.13_expectation-of-a-uniform-distribution|4.5.13]]。$X$ 在 $-n,\dots,n$ 均匀，$Y=X^2$。

> [!question]- O32-12
> 正确项：$\mathbb E[X]=0$；$\mathbb E[Y]>\mathbb E[X]$（$n>0$）；线性性给 $\mathbb E[X+Y]=\mathbb E[X]+\mathbb E[Y]$；且本例中 $\mathbb E[XY]=\mathbb E[X^3]=0=\mathbb E[X]\mathbb E[Y]$。最后一个等式只是对称导致的巧合；$Y$ 完全由 $X$ 决定，所以二者不独立。边界 $n=0$ 时 $\mathbb E[Y]>\mathbb E[X]$ 不成立，官方题隐含 $n>0$。

### CP32 — 本地来源异常与非官方独立题解（5 道）

> [!warning] 本地文件与 Session 主题不匹配
> [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp32.pdf#page=1|cp32]] 的页眉是 Week 8, Wed.，内容为图论而非 expectation。为保证 cp28–cp35 每个编号题都有归属，以下按本地 PDF 原题解答；它不代表官方 Session 32 block 内容。

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

> [!question]- 三道自检
> 1. 线性性为何不要求独立？
> 2. $\mathbb E[X]=a$ 是否推出 $\Pr(X=a)>0$？
> 3. geometric 等待时间均值 $1/p$ 需要哪些重复试验条件？
>
> [!success]- 自检答案
> 1. 它来自同一样本空间上的有限求和交换。2. 不推出。3. 每轮成功率固定为 $p>0$，且失败后剩余过程与原过程同分布，通常由轮次独立保证。

**知识链**：[[Probability Mass Function|PMF]] → [[Expected Value|期望]] → [[Indicator Random Variable|indicator]] → [[Linearity of Expectation|线性性]] → total expectation → [[Variance|方差]]。

---

## Problem Set 12 — Independence, Random Variables & Expectation

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps12.pdf#page=1|Problem Set 12, pp. 1–4]]。以下为非官方独立题解。

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

---

## Session 33 — Deviation: Markov & Chebyshev Bounds

### 本节问题、前置知识与资源

**学习问题**：只知道均值能限制多大的尾概率？再知道方差能改善多少？方差相加究竟要求 independent 还是 pairwise independent？

**前置知识**：期望、indicator、独立性、平方展开。核心卡片：[[Variance|方差]]、[[Markov Inequality|Markov 不等式]]、[[Chebyshev Inequality|Chebyshev 不等式]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session33.pdf#page=1|Session 33 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp33.pdf#page=1|cp33, pp. 1–4]]

### 4.6.1 Deviation from the Mean

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_DeviatTheMean.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ALn1McUXg-c.pdf#page=1|transcript]]

均值是重心，不是“典型 outcome”的保证。要描述离散程度，定义

$$
\operatorname{Var}(R)=\mathbb E[(R-\mu)^2],
\qquad \mu=\mathbb E[R],
$$

标准差 $\sigma_R=\sqrt{\operatorname{Var}(R)}$ 与 $R$ 同单位。平方使正负偏差不相消，并更重罚远端值。

### 4.6.2 Don't Expect the Expectation — Online O33-01–O33-02

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.2_don-t-expect-the-expectation|4.6.2]]。

> [!question]- O33-01–O33-02
> **O33-01** 51 枚公平币正面数 $X$ 的期望 $51/2=25.5$。
>
> **O33-02** 期望不可取，但 $X=25$ 可以取：
> $$\Pr(X=25)=\binom{51}{25}2^{-51}\approx0.1101.$$

### 4.6.3 Markov Bounds

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_MarkovBounds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/m07lrb7m0D0.pdf#page=1|transcript]]

[[Markov Inequality Proof|Markov 不等式证明]]：若 $R\ge0$ 且 $a>0$，则

$$
\boxed{\Pr(R\ge a)\le\frac{\mathbb E[R]}a}.
$$

逐点有 $R\ge aI_{\{R\ge a\}}$。两边取期望：

$$
\mathbb E[R]\ge a\,\mathbb E[I_{\{R\ge a\}}]
=a\Pr(R\ge a),
$$

再除以正数 $a$。非负性不可删：负值可抵消均值，却不减少右尾。若已知 $R\ge b$，可对 $R-b\ge0$ 应用 Markov：

$$
\Pr(R\ge a)\le\frac{\mathbb E[R]-b}{a-b},\qquad a>b.
$$

### 4.6.4 Markov Bound — Online O33-03–O33-06

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.4_markov-bound|4.6.4]]。已知 $\mathbb E[R]=50$，求保证 $\Pr(R\ge x)\le1/2$ 的最小 $x$。

> [!question]- O33-03–O33-06
> **O33-03** $R\ge0$：$50/x\le1/2$，故 $x=100$。
>
> **O33-04** $R\ge-50$：对 $R+50$ 用 Markov，$100/(x+50)\le1/2$，故 $x=150$。
>
> **O33-05** $R\ge30$：对 $R-30$ 用 Markov，$20/(x-30)\le1/2$，故 $x=70$。
>
> **O33-06** 仅知 $\Pr(R\le50)=0.7$ 不能保证非负下界，也不能用 Markov 得所求阈值；按题目约定答 6042。

### 4.6.5 Chebyshev Bounds

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ChebyhevBouds.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/uaa4P-kkLrA.pdf#page=1|transcript]]

[[Chebyshev Inequality Proof|Chebyshev 不等式证明]]：若 $R$ 有有限均值 $\mu$ 与方差 $\sigma^2$，则对 $a>0$，

$$
\boxed{\Pr(|R-\mu|\ge a)\le\frac{\sigma^2}{a^2}}.
$$

令非负变量 $Y=(R-\mu)^2$，事件 $|R-\mu|\ge a$ 等价于 $Y\ge a^2$。对 $Y$ 应用 Markov：

$$
\Pr(Y\ge a^2)\le\frac{\mathbb E[Y]}{a^2}
=\frac{\sigma^2}{a^2}.
$$

写成标准差倍数是 $\Pr(|R-\mu|\ge k\sigma)\le1/k^2$。它只给双侧上界，通常不紧，但不要求特定分布形状。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-concentration-bounds.png|900]]

读图：Markov 只利用非负变量的均值控制单侧尾；Chebyshev 先把离均值的距离平方，再同时覆盖左右两端，阈值翻倍会让上界缩小四倍。

### 4.6.6 Inside the TA's Brain — Online O33-07–O33-09

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.6_inside-the-ta-s-brain|4.6.6]]。一百万神经元，放电数 $R$ 的均值 550,000；放电比不放电多 200,000 才识别坏答案。

> [!question]- O33-07–O33-09
> **O33-07** $R-(1{,}000{,}000-R)\ge200{,}000$，故 $R\ge600{,}000$。
>
> **O33-08** Markov：$\Pr(R\ge600{,}000)\le550/600=11/12$。
>
> **O33-09** 若 $\sigma=25{,}000$，则目标单侧事件包含于 $|R-550{,}000|\ge50{,}000$，Chebyshev 给上界 $(25/50)^2=1/4$。这是对更大双侧事件的界，故合法但可能松。

### 4.6.7 Variance

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Variance.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/o57CTwt1-ck.pdf#page=1|transcript]]

展开平方并用 $\mathbb E[R]=\mu$：

$$
\operatorname{Var}(R)=\mathbb E[R^2]-\mu^2.
$$

常数平移不改变方差，缩放平方进入：

$$
\operatorname{Var}(aR+b)=a^2\operatorname{Var}(R).
$$

一般地

$$
\operatorname{Var}(R+S)
=\operatorname{Var}(R)+\operatorname{Var}(S)+2\operatorname{Cov}(R,S).
$$

若 $R,S$ independent，则 covariance 为 0。对一组变量，pairwise independence 已足以使所有交叉 covariance 为 0，从而方差相加；不需要 mutual independence。indicator $I\sim\operatorname{Bernoulli}(p)$ 的方差为 $p(1-p)$。

### 4.6.8 Practice with Variance — Online O33-10–O33-11

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.8_practice-with-variance|4.6.8]]。

> [!question]- O33-10–O33-11
> **O33-10** $p(1-p)=1/4-(p-1/2)^2$，最大值 $1/4$，在 $p=1/2$ 达到。
>
> **O33-11** 独立 $X,Y$ 时
> $$\operatorname{Var}(aX+bY+c)=a^2\operatorname{Var}(X)+b^2\operatorname{Var}(Y).$$

### 4.6.9 Flipping Coins — Online O33-12–O33-15

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.9_flipping-coins|4.6.9]]。$X\sim\operatorname{Bin}(100,1/2)$。

> [!question]- O33-12–O33-15
> **O33-12** $\mathbb E[X]=50$。
>
> **O33-13** Markov：$\Pr(X\ge70)\le50/70=5/7$。
>
> **O33-14** 独立 indicators 方差相加：$\operatorname{Var}(X)=100(1/4)=25$。
>
> **O33-15** $X<30$ 或 $X>70$ 蕴含 $|X-50|\ge20$；Chebyshev 给 $25/20^2=1/16$。

### 4.6.10 Practice with Bounds — Online O33-16–O33-17

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.10_practice-with-bounds|4.6.10]]。120 人均分 90。

> [!question]- O33-16–O33-17
> **O33-16** 若分数非负，Markov 给至少 180 分的比例至多 $90/180=1/2$，人数至多 60。
>
> **O33-17** 若最低 30，对 $R-30$ 用 Markov：比例至多 $(90-30)/(180-30)=2/5$，人数至多 48。两个界都可由两点分布达到，故在给定信息下最优。

### 4.6.11 Implications of Expectation — Online O33-18–O33-21

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S33_4.6.11_implications-of-expectation|4.6.11]]。$X$ 为非负整数且 $\mathbb E[X]=5$。

> [!question]- O33-18–O33-21
> **O33-18** $\Pr(X=5)$ 可以为 0，例如 $X=0,10$ 各半；也可为 1，例如常数 5，所以唯一保证项是“could be 0”。
>
> **O33-19** 必有正概率使 $X\le5$，否则 $X>5$ 几乎处处会令均值大于 5。
>
> **O33-20** $\mathbb E[X^2]$ 可以为 100：令 $X=20$ 概率 $1/4$，否则 0。且 $\mathbb E[X^2]\ge\mathbb E[X]^2=25$；官方反馈把这一式误写成自反不等式。
>
> **O33-21** Markov 给 $\Pr(X\ge1000)\le5/1000=1/200$；令 $X=1000$ 概率 $1/200$、否则 0 可达到，故不能保证更小通用界。

### CP33 — 非官方独立题解（5 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp33.pdf#page=1|cp33]]。

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

> [!question]- 三道自检
> 1. Markov 的非负条件为何不可省？
> 2. Chebyshev 的 $a$ 是方差单位还是随机变量单位？
> 3. 方差和何时只需 pairwise independence？
>
> [!success]- 自检答案
> 1. 负值可抵消均值却不减少正尾。2. 是随机变量单位；分母为 $a^2$。3. 有限和且各变量二阶矩存在时，pairwise independence 使每个 covariance 为 0。

**知识链**：[[Expected Value|均值]] → [[Variance|方差]] → [[Markov Inequality Proof|Markov 证明]] → [[Chebyshev Inequality Proof|Chebyshev 证明]] → sampling bounds。

---

## Session 34 — Sampling & Confidence

### 本节问题、前置知识与资源

**学习问题**：样本均值为何随 $n$ 集中到总体均值？confidence 是关于参数还是关于程序？生日碰撞为什么可用 pairwise independence 计算方差？

**前置知识**：期望、方差、Chebyshev、pairwise independence。核心卡片：[[Law of Large Numbers|大数定律]]、[[Weak Law of Large Numbers Proof|弱大数定律证明]]、[[Confidence Level|置信水平]]、[[Confidence Interval|置信区间]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session34.pdf#page=1|Session 34 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp34.pdf#page=1|cp34, pp. 1–4]]

### 4.7.1 Law of Large Numbers

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_LawLrgeNumbr.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-yo3YxfY47g.pdf#page=1|transcript]]

设 $X_1,X_2,\dots$ pairwise independent，且共同均值 $\mu$、共同有限方差 $\sigma^2$。样本均值

$$
A_n=\frac1n\sum_{i=1}^{n}X_i
$$

满足 $\mathbb E[A_n]=\mu$，并因 pairwise independence，

$$
\operatorname{Var}(A_n)
=\frac1{n^2}\sum_{i=1}^{n}\operatorname{Var}(X_i)
=\frac{\sigma^2}{n}.
$$

Chebyshev 给

$$
\Pr(|A_n-\mu|\ge\varepsilon)
\le\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0.
$$

这证明 $A_n$ [[Convergence in Probability|依概率收敛]]到 $\mu$，即 weak law；它不等同于“每条无限样本路径最终都收敛”的 strong law。

### 4.7.2 Not So Strong — Online O34-01

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.2_not-so-strong|4.7.2]]。

> [!question]- O34-01
> $X_i$ 以相同概率 $f(i)$ 取 $\pm i2^i$，其余取 0。正负项对消，$\mathbb E[X_i]=0$；有限线性性给 $\mathbb E[S_n]=0$。这并不推出 $S_n$ 集中在 0，因为方差可能极大。

### 4.7.3 Independent Sampling Theorem

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_IndependSmpling.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/MMn7q1M7pGI.pdf#page=1|transcript]]

对 Bernoulli samples $X_i\in\{0,1\}$，$\mathbb E[X_i]=p$、$\operatorname{Var}(X_i)=p(1-p)\le1/4$。若 pairwise independent，则样本比例 $A_n$ 满足

$$
\Pr(|A_n-p|\ge\varepsilon)
\le\frac{1}{4n\varepsilon^2}.
$$

要把 failure probability 控制在 $\delta$ 以下，一个分布无关的充分条件是

$$
n\ge\frac1{4\delta\varepsilon^2}.
$$

所需样本量主要取决于误差容忍度与置信失败率，不直接随总体人数增长；这依赖均匀、独立或合适的无放回抽样设计。

### 4.7.4 Sampling Coin Tosses — Online O34-02–O34-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.4_sampling-coin-tosses|4.7.4]]。

> [!question]- O34-02–O34-04
> **O34-02** 公平币样本比例 $\mathbb E[A_n]=1/2$。
>
> **O34-03** $\operatorname{Var}(A_n)=1/(4n)$；要求 $\Pr(|A_n-1/2|>1/10)\le1/25$，Chebyshev 上界 $25/n$，故 $n\ge625$。
>
> **O34-04** $n=100$ 时令 $[1/(4n)]/\delta^2\le1/4$，得到 $\delta\ge1/10$；题问最小保证值为 $1/10$。

### 4.7.5 Birthday Matching

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Birthday.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TWVntUfXsKs.pdf#page=1|transcript]]

若 $n$ 人生日独立均匀落在 $d$ 天，令 $I_{ij}$ 指示第 $i,j$ 人同生日，匹配 pair 数

$$
M=\sum_{i<j}I_{ij}.
$$

每个 $\mathbb E[I_{ij}]=1/d$，故

$$
\mathbb E[M]=\binom n2\frac1d.
$$

不同 pair indicators 即使共享一个人也 pairwise independent：例如给定第 $i$ 人生日后，第 $j,k$ 人各自匹配的概率仍为 $1/d$，联合为 $1/d^2$。因此

$$
\operatorname{Var}(M)=\binom n2\frac1d\left(1-\frac1d\right).
$$

但整组 indicators 不 mutually independent，例如三个人的三对匹配事件存在传递约束。

### 4.7.6 Birthdays on Naboo — Online O34-05–O34-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.6_birthdays-on-naboo|4.7.6]]。200 个孩子、199 天。

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

### 4.7.7 Sampling & Confidence

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SmplingConfid.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Q-6Cw8tYVeY.pdf#page=1|transcript]]

未知总体比例 $p$ 是固定常数，抽样比例 $\hat p$ 才是随机变量。若抽样前证明

$$
\Pr_p(|\hat p-p|\le\varepsilon)\ge1-\delta,
$$

则随机区间 $[\hat p-\varepsilon,\hat p+\varepsilon]$ 在重复执行程序时至少以 $1-\delta$ 的比例覆盖 $p$。观察数据后，不能在频率学派解释下说“固定的 $p$ 有 $1-\delta$ 概率落在这一个已实现区间”；应说该构造方法有相应 coverage/confidence。

### 4.7.8 Confidence — Online O34-08–O34-10

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.8_confidence|4.7.8]]。100 个样本给 $\pm1$、95% confidence，其他条件固定。

> [!question]- O34-08–O34-10
> **O34-08** 总体从 5000 增至 10000：独立均匀抽样的分布无关界不变，仍 100。
>
> **O34-09** tolerance 从 1 加倍到 2，样本量按 $1/\varepsilon^2$ 缩为四分之一，得 25。
>
> **O34-10** failure probability 从 0.05 降为 0.01，Chebyshev 型样本量按 $1/\delta$ 放大 5 倍，得 500。

### 4.7.9 Random Sampling — Online O34-11–O34-12

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.9_random-sampling|4.7.9]]。

> [!question]- O34-11–O34-12
> **O34-11** 正确事实：某个固定选民的偏好已确定，概率是 0 或 1；随机抽法使每位选民成为第 3 个样本的概率相同；若第 2 人与第 1 人来自同州，其支持概率可能不等于全国 $p$。错误项把固定对象与随机抽中者混淆，或忽略重复抽样与条件选择。
>
> **O34-12** 观察 $\hat p=0.53$ 且程序保证 $\Pr(|\hat p-p|\le0.04)\ge0.95$ 后，正确表述是：“$p$ 在 $0.53\pm0.04$ 内，或者发生了至多 5% 的异常抽样结果”；也可说“我们对该区间有 95% confidence”。不能说 $p$ 本身现在以 95% 概率落入区间。

### 4.7.10 Above Average Number of Fingers — Online O34-13

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S34_4.7.10_above-average-number-of-fingers|4.7.10]]。

> [!question]- O34-13
> 多数人手指数高于平均数，是因为缺失手指的总量多于额外手指的总量；“缺失远多于额外”与较弱的“至少略多”都足以解释。分布偏斜时，大多数观测高于均值并不矛盾；均值不是中位数。

### CP34 — 非官方独立题解（6 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp34.pdf#page=1|cp34]]。

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

> [!question]- 三道自检
> 1. weak law 的收敛对象是什么？
> 2. 95% confidence 是给参数赋 95% 后验概率吗？
> 3. birthday pair indicators 为何可 pairwise independent 却不 mutually independent？
>
> [!success]- 自检答案
> 1. 对每个固定 $\varepsilon>0$，偏差概率趋 0。2. 不是；它描述随机区间程序的覆盖率。3. 任意两对的联合仍分解，但三对之间有“前两对匹配则第三对必匹配”的约束。

**知识链**：[[Variance|样本方差缩放]] → [[Weak Law of Large Numbers Proof|弱大数定律]] → independent sampling → [[Confidence Level|置信水平]] / [[Confidence Interval|置信区间]] → 选择偏差。

---

## Session 35 — Random Walks & PageRank

### 本节问题、前置知识与资源

**学习问题**：图上的随机移动怎样写成矩阵？stationary distribution 存在、唯一与收敛是同一件事吗？PageRank 为什么是一个长期访问概率？

**前置知识**：有向图、矩阵乘法、条件概率、递推。核心卡片：[[Random Walk on a Graph|图上随机游走]]、[[Markov Chain|Markov 链]]、[[Stationary Distribution|平稳分布]]、[[PageRank]]。

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session35.pdf#page=1|Session 35 reading]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp35.pdf#page=1|cp35, pp. 1–3]]

### 4.8.1 Random Walks

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RandomWalks.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/-j7MoM3P_J8.pdf#page=1|transcript]]

有限状态 random walk 的 transition matrix $P$ 满足

$$
P_{uv}=\Pr(X_{t+1}=v\mid X_t=u),\qquad
P_{uv}\ge0,\qquad \sum_vP_{uv}=1.
$$

Markov property 是：给定当前状态后，下一步与更早历史无关。若行向量 $d_t$ 表示时刻 $t$ 的状态分布，则 total probability 给

$$
d_{t+1}=d_tP,\qquad d_t=d_0P^t.
$$

路径 $v_0\to\cdots\to v_t$ 的概率是 $d_0(v_0)\prod_{i=0}^{t-1}P_{v_iv_{i+1}}$。

### 4.8.2 Stationary Distributions

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_StatinaryDist.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iZX8WEGZTVw.pdf#page=1|transcript]]

分布 $\pi$ stationary，当且仅当

$$
\pi=\pi P,\qquad \pi_v\ge0,\qquad\sum_v\pi_v=1.
$$

它表达每个顶点的流入概率等于其质量。必须区分：

- **存在**：有限 Markov chain 至少有一个 stationary distribution；
- **唯一**：有限 irreducible chain 有唯一 stationary distribution；
- **从任意初值收敛**：还需排除周期振荡，常用充分条件是 irreducible 且 aperiodic。

两节点确定性交替链有 stationary $(1/2,1/2)$，但从单点出发的分布永远在两个单点间跳动，不收敛。这直接否定“有 stationary 就会趋近 stationary”。

### 4.8.3 PageRank

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pagerank.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QKO_2WQkZ0k.pdf#page=1|transcript]]

在网页 digraph 上，基础 Google walk 从页面 $u$ 均匀选择一个 outgoing link，$P_{uv}=1/\operatorname{outdeg}(u)$。[[PageRank]] 把 stationary mass 解释为长期访问频率：被许多高质量页面指向的页面得到较大流入。

实际模型加入 teleportation：

$$
P_\alpha=\alpha P+(1-\alpha)\frac1n\mathbf1\mathbf1^\mathsf T,\qquad0<\alpha<1,
$$

并为 dangling pages 指定合法转移。正的 teleportation 使有限链 irreducible 且 aperiodic，因此 stationary distribution 唯一并从任意初值收敛。排名仍取决于链接图与 damping/teleportation 选择，不是网页“客观价值”的无假设度量。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit04-random-walk-pagerank.png|900]]

读图：左侧沿边传播当前概率，右侧 stationary 向量满足“传播一次保持不变”；PageRank 再加入少量全局跳转，打破封闭类与周期。

### 4.8.4 Random Walks — Online O35-01–O35-04

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.4_random-walks|4.8.4]]。

> [!question]- O35-01–O35-04
> **O35-01** 图 1 中 $x\leftrightarrow y$ 确定性交替，stationary 的 $d(x)=1/2$。
>
> **O35-02** 归一化给 $d(y)=1/2$。
>
> **O35-03** 图 2 中 $w\to z$ 概率 1，$z\to w,z$ 概率 $0.9,0.1$。平衡式 $\pi_w=0.9\pi_z$ 与归一化给 $\pi_z=10/19$。
>
> **O35-04** 从 $x$ 出发不会接近 $(1/2,1/2)$；偶数时刻全在 $x$，奇数时刻全在 $y$，周期为 2。

### 4.8.5 Random Walks Continued — Online O35-05–O35-07

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S35_4.8.5_random-walks-cont|4.8.5]]。

> [!question]- O35-05–O35-07
> **O35-05** 图 2 从 $w$ 出发会趋近 stationary。前几步 $(1,0)\to(0,1)\to(0.9,0.1)\to(0.09,0.91)\to(0.819,0.181)$；振荡幅度按 $0.9^t$ 衰减。
>
> **O35-06** 图 3 有两个 absorbing states $a,d$；任意 $\lambda\delta_a+(1-\lambda)\delta_d$ 都 stationary，$\lambda\in[0,1]$，故有不可数无限多个。
>
> **O35-07** 令 $h_b,h_c$ 为从 $b,c$ 最终吸收到 $d$ 的概率。由图
> $$h_b=\tfrac12h_c,\qquad h_c=\tfrac12h_b+\tfrac12,$$
> 解得 $h_b=1/3$。所以从 $b$ 长期在 $d$ 的概率趋近 $1/3$。

### CP35 — 非官方独立题解（3 道）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp35.pdf#page=1|cp35]]。

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

> [!question]- 三道自检
> 1. stationary distribution 存在是否保证从单点出发收敛？
> 2. irreducible 与 aperiodic 分别解决什么问题？
> 3. PageRank 中 teleportation 的数学作用是什么？
>
> [!success]- 自检答案
> 1. 不保证，周期链可永久振荡。2. irreducible 排除多个封闭类并给唯一性；aperiodic 排除周期振荡。3. 使有限链连通且非周期，从而得到唯一、可收敛的 stationary distribution。

**知识链**：[[Random Walk on a Graph|图上随机游走]] → [[Markov Matrix|转移矩阵]] → [[Stationary Distribution|平稳分布]] → irreducibility/aperiodicity → [[PageRank]]。

---

## 全章方法地图

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

## 覆盖与资源核对

- 官方在线题：O28-01–11（11）、O29-01–06（6）、O30-01–05（5）、O31-01–21（21）、O32-01–12（12）、O33-01–21（21）、O34-01–13（13）、O35-01–07（7），合计 **96**。
- Classroom Problems：C28-1–6、C29-1–4、C30-1–4、C31-1–5、C32-1–5、C33-1–5、C34-1–6、C35-1–3，合计 **38**；本地 cp32 的主题错配已经显式标注。
- 作业：Problem Set 11 的 P11-1–3 与 Problem Set 12 的 P12-1–3，合计 **6** 个编号主问题，全部子问均解答。
- 官方 blocks：Session 28–35 依次为 8、10、7、8、13、11、10、5 个，共 **72**；每个 video/exercise 按原顺序出现。
- 教学图：7 张 PNG 均在首次展开相关概念处以宽度 900 嵌入，并附中文读图说明。

> [!summary] 一句话收束
> **概率空间**定义可能性，**条件化**更新信息，**独立性**允许乘法，**随机变量**把结果数值化，**期望与方差**描述中心和离散，**集中界与抽样**把局部随机变成总体保证，**stationary distribution**则把同一思想延伸到长期随机过程。
