---
aliases:
  - MIT 6.042J course map
  - Mathematics for Computer Science course map
  - 离散数学课程地图
tags:
  - discrete-mathematics
  - mit-ocw
  - course-note
---

# MIT OCW 6.042J course map

> [!info] 课程来源与版本
> 本套笔记对应 MIT OpenCourseWare **6.042J / 18.062J Mathematics for Computer Science, Spring 2015**，授课教师为 Albert R. Meyer 与 Adam Chlipala。
>
> - [Official course](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)
> - [Syllabus](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/syllabus/)
> - [Official readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/)
> - [Open Learning Library courseware](https://openlearninglibrary.mit.edu/courses/course-v1%3AOCW%2B6.042J%2B2T2019/course/)
> - [Lecture videos](https://www.youtube.com/playlist?list=PLUl4u3cNGP60UlabZBeeqOuoLuj_KNphQ)
> <!-- bilingual-en:start -->
> These notes correspond to MIT OpenCourseWare **6.042J / 18.062J Mathematics for Computer Science, Spring 2015**, taught by Albert R. Meyer and Adam Chlipala.
>
> - [Official course](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)
> - [Syllabus](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/syllabus/)
> - [Official readings](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/readings/)
> - [Open Learning Library courseware](https://openlearninglibrary.mit.edu/courses/course-v1%3AOCW%2B6.042J%2B2T2019/course/)
> - [Lecture videos](https://www.youtube.com/playlist?list=PLUl4u3cNGP60UlabZBeeqOuoLuj_KNphQ)
> <!-- bilingual-en:end -->

## 从哪里开始
<!-- bilingual-en:start -->
*Where to begin*
<!-- bilingual-en:end -->

- 第一次学习：按 Proofs → Structures → Counting → Probability → Final Review 的顺序阅读。
- 每个 Session：先明确本节问题，再读理论与证明，然后完成在线反馈题、课堂题和三道自检题。
- 复习某种证明：从 [[01_Proofs#证明方法总图|证明方法总图]] 或本页“题型入口”进入。
- 考前复习：进入 [[05_Review and exam roadmap|全课程复习与期末考试]]。
- 查原始资料：进入 [[MIT_OCW_6.042J_Materials/index|课程材料总索引]]。
- 查可复用知识卡：进入 [[Mathematics for Computer Science Course Atlas|Discrete Mathematics Hub]]。
<!-- bilingual-en:start -->
- First study: read Proofs → Structures → Counting → Probability → Final Review in order.
- For each session: identify the question first, then read the theory and proofs, and finally complete the online feedback exercises, in-class questions, and three self-check questions.
- To review a proof method: start from [[01_Proofs#证明方法总图|the proof-method map]] or the problem-type entry points on this page.
- Exam review: open [[05_Review and exam roadmap|Whole-course review and final exam]].
- Original materials: open [[MIT_OCW_6.042J_Materials/index|Course materials master index]].
- Reusable knowledge notes: open [[Mathematics for Computer Science Course Atlas|Discrete Mathematics Hub]].
<!-- bilingual-en:end -->

## 五篇主笔记
<!-- bilingual-en:start -->
*The five main notes*
<!-- bilingual-en:end -->

| 顺序 | 主笔记 | Sessions | 核心问题 | 阶段验收 |
|---:|---|---:|---|---|
| 1 | [[01_Proofs]] | 1–11 | 怎样把“看起来正确”变成量词、假设和推导都可检查的证明？ | Midterm 1、PS1–4 |
| 2 | [[02_Structures]] | 12–22 | 数论、图和偏序怎样描述计算、加密、调度与匹配？ | Midterm 2、PS5–8 |
| 3 | [[03_Counting]] | 23–27 | 不逐个列举，怎样可靠计算对象数量并比较增长阶？ | Midterm 3、PS9–10 |
| 4 | [[04_Probability]] | 28–35 | 怎样从样本空间出发计算条件概率、期望、偏差与随机过程？ | PS11–12 |
| 5 | [[05_Review and exam roadmap]] | 全课程 | 怎样识别综合题所属结构、选择工具并交叉验算？ | Final Exam |
<!-- bilingual-en:start -->
| Order | Main note | Sessions | Central question | Stage checkpoint |
|---:|---|---:|---|---|
| 1 | [[01_Proofs]] | 1–11 | How do we turn something that looks true into a proof whose quantifiers, assumptions, and deductions can all be checked? | Midterm 1, PS1–4 |
| 2 | [[02_Structures]] | 12–22 | How do number theory, graphs, and partial orders describe computation, encryption, scheduling, and matching? | Midterm 2, PS5–8 |
| 3 | [[03_Counting]] | 23–27 | How can we count objects reliably and compare growth rates without listing every case? | Midterm 3, PS9–10 |
| 4 | [[04_Probability]] | 28–35 | How do we begin with a sample space and calculate conditional probabilities, expectations, deviations, and stochastic processes? | PS11–12 |
| 5 | [[05_Review and exam roadmap]] | Whole course | How do we identify the structure behind a mixed problem, select a tool, and cross-check the result? | Final Exam |
<!-- bilingual-en:end -->

## 课程真正训练的四种能力
<!-- bilingual-en:start -->
*The four capabilities this course actually develops*
<!-- bilingual-en:end -->

### 1. 把自然语言翻译成精确命题
<!-- bilingual-en:start -->
*1. Translate natural language into precise propositions*
<!-- bilingual-en:end -->

必须分清：
<!-- bilingual-en:start -->
You must distinguish:
<!-- bilingual-en:end -->

- 对象的定义域；
- “对所有”与“存在”；
- 假设与结论；
- 必要条件、充分条件和双条件；
- 一个反例究竟否定了哪一句话。
<!-- bilingual-en:start -->
- the domain of the objects;
- "for all" from "there exists";
- assumptions from conclusions;
- necessary, sufficient, and biconditional conditions;
- exactly which statement a counterexample refutes.
<!-- bilingual-en:end -->

### 2. 选择证明结构
<!-- bilingual-en:start -->
*2. Select a proof structure*
<!-- bilingual-en:end -->

证明不是从公式堆砌开始，而是先看结论的逻辑形状：
<!-- bilingual-en:start -->
A proof does not begin by piling up formulas. It begins by inspecting the logical form of the conclusion:
<!-- bilingual-en:end -->

| 目标形状 | 首选入口 | 核心检查 |
|---|---|---|
| $P\Rightarrow Q$ | direct proof / contrapositive | 是否只在假设 $P$ 下推出 $Q$ |
| “不存在” | contradiction / invariant | 矛盾是否来自假设而非偷偷加入的新条件 |
| 对所有 $n\in\mathbb N$ | induction / well ordering | 基础情形与归纳步是否覆盖全部整数 |
| 程序永不进入坏状态 | state-machine invariant | 初始成立、每一步保持 |
| 递归对象都有性质 $P$ | structural induction | 每个构造规则都保持 $P$ |
| 两个有限集合等势 | bijection / double counting | 映射是否同时单射和满射 |
<!-- bilingual-en:start -->
| Shape of the goal | First method to consider | Core check |
|---|---|---|
| $P\Rightarrow Q$ | direct proof / contrapositive | Is $Q$ derived using only the assumption $P$? |
| "There is no..." | contradiction / invariant | Does the contradiction follow from the assumption rather than an unannounced extra condition? |
| For every $n\in\mathbb N$ | induction / well ordering | Do the base case and inductive step cover all integers? |
| A program never reaches a bad state | state-machine invariant | Is the invariant true initially and preserved by every transition? |
| Every recursively defined object has property $P$ | structural induction | Does each construction rule preserve $P$? |
| Two finite sets have the same cardinality | bijection / double counting | Is the map both injective and surjective? |
<!-- bilingual-en:end -->

### 3. 把计算问题改写成离散结构
<!-- bilingual-en:start -->
*3. Rewrite computational problems as discrete structures*
<!-- bilingual-en:end -->

- 密码与哈希：整数、同余、逆元和 Euler 定理；
- 调度与依赖：DAG、偏序、最长链和拓扑顺序；
- 网络与连通：路径、生成树、着色和匹配；
- 算法规模：和式、渐近记号、计数与递推直觉；
- 随机算法：样本空间、随机变量、期望、尾界和 Markov chain。
<!-- bilingual-en:start -->
- Cryptography and hashing: integers, congruences, inverses, and Euler's theorem.
- Scheduling and dependencies: DAGs, partial orders, longest chains, and topological order.
- Networks and connectivity: paths, spanning trees, coloring, and matching.
- Algorithmic scale: sums, asymptotic notation, counting, and recurrence intuition.
- Randomized algorithms: sample spaces, random variables, expectations, tail bounds, and Markov chains.
<!-- bilingual-en:end -->

### 4. 让答案可被反驳、复算和验证
<!-- bilingual-en:start -->
*4. Make answers refutable, reproducible, and verifiable*
<!-- bilingual-en:end -->

每个正式答案至少回答：
<!-- bilingual-en:start -->
Every formal answer should address at least:
<!-- bilingual-en:end -->

1. 使用了哪些定义和假设？
2. 每一步由什么定理或代数规则支持？
3. 是否存在未处理的边界情形？
4. 小规模实例、反例或另一种方法是否支持结果？
<!-- bilingual-en:start -->
1. Which definitions and assumptions are being used?
2. Which theorem or algebraic rule justifies each step?
3. Are any boundary cases untreated?
4. Does a small example, counterexample, or alternative method support the result?
<!-- bilingual-en:end -->

## 官方课程顺序
<!-- bilingual-en:start -->
*Official course order*
<!-- bilingual-en:end -->

### Unit 1: Proofs

| Session | 主题 | 作业或考试位置 |
|---:|---|---|
| 1 | Introduction to Proofs | |
| 2 | Proof Methods | |
| 3 | Well Ordering Principle | |
| 4 | Logic & Propositions | Problem Set 1 |
| 5 | Quantifiers & Predicate Logic | |
| 6 | Sets | Problem Set 2 |
| 7 | Binary Relations | |
| 8 | Induction | Problem Set 3；Midterm 1 |
| 9 | State Machines—Invariants | |
| 10 | Recursive Definition | |
| 11 | Infinite Sets | Problem Set 4 |
<!-- bilingual-en:start -->
| Session | Topic | Assignment or exam |
|---:|---|---|
| 1 | Introduction to Proofs | |
| 2 | Proof Methods | |
| 3 | Well Ordering Principle | |
| 4 | Logic & Propositions | Problem Set 1 |
| 5 | Quantifiers & Predicate Logic | |
| 6 | Sets | Problem Set 2 |
| 7 | Binary Relations | |
| 8 | Induction | Problem Set 3; Midterm 1 |
| 9 | State Machines—Invariants | |
| 10 | Recursive Definition | |
| 11 | Infinite Sets | Problem Set 4 |
<!-- bilingual-en:end -->

### Unit 2: Structures

| Session | 主题 | 作业或考试位置 |
|---:|---|---|
| 12 | GCDs | |
| 13 | Congruences | |
| 14 | Euler's Theorem | Problem Set 5 |
| 15 | RSA Encryption | |
| 16 | Digraphs: Walks & Paths | Problem Set 6；Midterm 2 |
| 17 | Directed Acyclic Graphs | |
| 18 | Partial Orders and Equivalence | |
| 19 | Degrees & Isomorphism | Problem Set 7 |
| 20 | Coloring & Connectivity | |
| 21 | Trees | |
| 22 | Stable Matching | Problem Set 8 |
<!-- bilingual-en:start -->
| Session | Topic | Assignment or exam |
|---:|---|---|
| 12 | GCDs | |
| 13 | Congruences | |
| 14 | Euler's Theorem | Problem Set 5 |
| 15 | RSA Encryption | |
| 16 | Digraphs: Walks & Paths | Problem Set 6; Midterm 2 |
| 17 | Directed Acyclic Graphs | |
| 18 | Partial Orders and Equivalence | |
| 19 | Degrees & Isomorphism | Problem Set 7 |
| 20 | Coloring & Connectivity | |
| 21 | Trees | |
| 22 | Stable Matching | Problem Set 8 |
<!-- bilingual-en:end -->

### Unit 3: Counting

| Session | 主题 | 作业或考试位置 |
|---:|---|---|
| 23 | Sums & Products | |
| 24 | Asymptotics | Problem Set 9；Midterm 3 |
| 25 | Counting with Bijections | |
| 26 | Repetitions & Binomial Theorem | |
| 27 | Pigeonhole Principle, Inclusion-Exclusion | Problem Set 10 |
<!-- bilingual-en:start -->
| Session | Topic | Assignment or exam |
|---:|---|---|
| 23 | Sums & Products | |
| 24 | Asymptotics | Problem Set 9; Midterm 3 |
| 25 | Counting with Bijections | |
| 26 | Repetitions & Binomial Theorem | |
| 27 | Pigeonhole Principle, Inclusion-Exclusion | Problem Set 10 |
<!-- bilingual-en:end -->

### Unit 4: Probability

| Session | 主题 | 作业或考试位置 |
|---:|---|---|
| 28 | Introduction to Discrete Probability | |
| 29 | Conditional Probability | Problem Set 11 |
| 30 | Independence & Causality | |
| 31 | Random Variables, Density Functions | |
| 32 | Expectation | Problem Set 12 |
| 33 | Deviation: Markov & Chebyshev Bounds | 无新作业 |
| 34 | Sampling & Confidence | 无新作业 |
| 35 | Random Walks & PageRank | 无新作业；随后 Final Exam |
<!-- bilingual-en:start -->
| Session | Topic | Assignment or exam |
|---:|---|---|
| 28 | Introduction to Discrete Probability | |
| 29 | Conditional Probability | Problem Set 11 |
| 30 | Independence & Causality | |
| 31 | Random Variables, Density Functions | |
| 32 | Expectation | Problem Set 12 |
| 33 | Deviation: Markov & Chebyshev Bounds | No new assignment |
| 34 | Sampling & Confidence | No new assignment |
| 35 | Random Walks & PageRank | No new assignment; Final Exam follows |
<!-- bilingual-en:end -->

## 材料版本说明
<!-- bilingual-en:start -->
*Notes on material versions*
<!-- bilingual-en:end -->

| 材料 | 版本 | 用法 |
|---|---|---|
| [[MIT_OCW_6.042J_Materials/99_Books/MIT6_042JS15_textbook.pdf|MIT6_042JS15_textbook.pdf]] | 2015-05-18，920 页 | 本课程的章节编号与 Session 阅读基准 |
| `MIT6_042JS15_Session1–35.pdf` | Spring 2015 分节摘录 | 每节最直接的阅读入口 |
| [[MIT-6-042j-pdf.pdf|MIT-6-042j-pdf.pdf]] | 2018-06-06，1048 页 | 修订后的补充教材，不能替代 2015 顺序 |
| `MIT6_042JS15_*` slides | Spring 2015 | 官方原始课堂材料 |
| `MIT6_042JS16_*` slides | 官方包中的 later/replacement decks | 保留年份并按官方 Resource Index 归入相应 Session |
<!-- bilingual-en:start -->
| Material | Version | Use |
|---|---|---|
| [[MIT_OCW_6.042J_Materials/99_Books/MIT6_042JS15_textbook.pdf|MIT6_042JS15_textbook.pdf]] | 2015-05-18, 920 pages | Baseline for chapter numbering and session readings in this course |
| `MIT6_042JS15_Session1–35.pdf` | Spring 2015 excerpts by session | Most direct reading entry for each session |
| [[MIT-6-042j-pdf.pdf|MIT-6-042j-pdf.pdf]] | 2018-06-06, 1,048 pages | Revised supplementary text; it does not replace the 2015 sequence |
| `MIT6_042JS15_*` slides | Spring 2015 | Original official classroom materials |
| `MIT6_042JS16_*` slides | Later or replacement decks in the official package | Preserve the year and assign each deck to the corresponding session according to the official Resource Index |
<!-- bilingual-en:end -->

> [!warning] 题解来源边界
> MIT 公开包没有提供 Problem Set、In-Class Questions 或考试的官方答案。本套笔记对这些材料给出的是**非官方独立题解**，并通过真值表、穷举、小规模图、模运算或概率复算进行核验。在线 feedback exercises 自带的正确选项和反馈才标为官方答案。
> <!-- bilingual-en:start -->
> The public MIT package does not provide official solutions to the problem sets, in-class questions, or exams. These notes therefore supply **unofficial, independently derived solutions**, checked with truth tables, exhaustive enumeration, small graphs, modular arithmetic, or probability recalculation. Only the correct choices and feedback embedded in the online feedback exercises are labeled official answers.
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success] 596 个问题入口已全部落入笔记
> - 在线 feedback prompts：**376**，保留官方正确答案与反馈并逐项解释；
> - In-Class Questions：**153**，给出非官方独立完整题解；
> - Problem Sets 1–12：**38** 个编号主问题及其全部子问；
> - Midterm 1–3 与 Final：**29** 个编号主问题及其全部子问。
>
> `MIT_OCW_6.042J_Materials/problem_coverage.csv` 为逐题覆盖台账；每一行均指向实际存在的 Unit、Problem Set 或 Exam 标题，并已通过标题解析检查。
> <!-- bilingual-en:start -->
> - Online feedback prompts: **376**, preserving the official correct answers and feedback with an explanation for each item;
> - In-Class Questions: **153**, with complete unofficial independent solutions;
> - Problem Sets 1–12: **38** numbered main problems and all subparts;
> - Midterms 1–3 and Final: **29** numbered main problems and all subparts.
>
> `MIT_OCW_6.042J_Materials/problem_coverage.csv` is the item-by-item coverage ledger. Every row points to an existing unit, problem-set, or exam heading and has passed heading-resolution checks.
> <!-- bilingual-en:end -->

## 本地材料体系
<!-- bilingual-en:start -->
*Local material structure*
<!-- bilingual-en:end -->

| 目录 | 预期数量 | 用途 |
|---|---:|---|
| `01_Session_Readings` | 35 PDF | Session 对应章节摘录 |
| `02_Lecture_Slides` | 113 PDF | 逐段 lecture-slide handouts |
| `03_Video_Transcripts` | 111 PDF | 本地可检索讲稿 |
| `04_Captions` | 111 SRT | 视频字幕与时间定位 |
| `05_In_Class_Questions` | 35 PDF | 团队课堂问题 |
| `06_Problem_Sets` | 12 PDF | PS1–PS12 |
| `07_Exams` | 4 PDF | Midterm 1–3 与 Final |
| `08_Courseware_Exercises` | 结构化清单 | 在线题、答案、反馈与 block 顺序 |
| `09_Courseware_Images` | 17 | 在线题所需图像 |
| `99_Books` | 1 PDF | 2015 完整教材；2018 版仍留在课程根目录 |
<!-- bilingual-en:start -->
| Directory | Expected count | Purpose |
|---|---:|---|
| `01_Session_Readings` | 35 PDFs | Chapter excerpts corresponding to sessions |
| `02_Lecture_Slides` | 113 PDFs | Lecture-slide handouts divided into segments |
| `03_Video_Transcripts` | 111 PDFs | Locally searchable transcripts |
| `04_Captions` | 111 SRT files | Video captions and time locations |
| `05_In_Class_Questions` | 35 PDFs | Team-based classroom questions |
| `06_Problem_Sets` | 12 PDFs | PS1–PS12 |
| `07_Exams` | 4 PDFs | Midterms 1–3 and Final |
| `08_Courseware_Exercises` | Structured inventory | Online problems, answers, feedback, and block order |
| `09_Courseware_Images` | 17 | Images required by online problems |
| `99_Books` | 1 PDF | Complete 2015 textbook; the 2018 edition remains in the course root |
<!-- bilingual-en:end -->

视频不重复存入 iCloud：每段保留 transcript、SRT、YouTube ID 与 Internet Archive 入口，正文仍按视频顺序组织。
<!-- bilingual-en:start -->
Videos are not duplicated in iCloud. Each segment retains its transcript, SRT file, YouTube ID, and Internet Archive entry, while the main notes still follow video order.
<!-- bilingual-en:end -->

## 题型入口
<!-- bilingual-en:start -->
*Entry points by problem type*
<!-- bilingual-en:end -->

| 看到的信号 | 先问什么 | 常用工具 |
|---|---|---|
| “证明对所有整数成立” | 命题是否有自然的前驱结构？ | induction、strong induction、WOP |
| “程序永远不会……” | 能否找到初始成立且转移保持的性质？ | invariant、derived variable |
| $ax+by$、余数、加密 | 是否在求 gcd、逆元或指数同余？ | Euclidean algorithm、Bézout、Euler |
| prerequisites / scheduling | 是否存在有向环？ | DAG、topological order、critical path |
| 图是否连通、能否着色 | 问的是路径、割、树还是局部冲突？ | connectivity、spanning tree、coloring |
| “有多少种” | 对象能否拆步骤、分情况或建立双射？ | sum/product、bijection、stars and bars |
| 至少两个对象落同一类 | 对象数是否超过容器容量？ | pigeonhole principle |
| 条件概率或更新信念 | 分母事件是什么？是否先划分样本空间？ | Bayes、total probability |
| 只要求平均结果 | 是否可拆成指示变量之和？ | linearity of expectation |
| 要控制偏离概率 | 已知的是均值、方差还是独立采样？ | Markov、Chebyshev、sampling bound |
<!-- bilingual-en:start -->
| Signal in the problem | First question to ask | Common tools |
|---|---|---|
| "Prove this for every integer" | Does the proposition have a natural predecessor structure? | induction, strong induction, WOP |
| "The program never..." | Can I find a property that holds initially and is preserved by transitions? | invariant, derived variable |
| $ax+by$, remainders, encryption | Am I seeking a gcd, inverse, or exponential congruence? | Euclidean algorithm, Bézout, Euler |
| prerequisites / scheduling | Is there a directed cycle? | DAG, topological order, critical path |
| Is a graph connected or colorable? | Is the question about paths, cuts, trees, or local conflicts? | connectivity, spanning tree, coloring |
| "How many?" | Can the object be decomposed into stages or cases, or paired by a bijection? | sum/product, bijection, stars and bars |
| At least two objects share a category | Does the number of objects exceed total container capacity? | pigeonhole principle |
| Conditional probability or belief updating | What is the conditioning event? Should the sample space be partitioned first? | Bayes, total probability |
| Only the average result is required | Can the quantity be written as a sum of indicators? | linearity of expectation |
| Control a deviation probability | Are the mean, variance, or independent samples known? | Markov, Chebyshev, sampling bound |
<!-- bilingual-en:end -->

## 学习完成标准
<!-- bilingual-en:start -->
*Completion standard*
<!-- bilingual-en:end -->

完成这门课不等于“看过全部公式”。你应能在不看答案时做到：
<!-- bilingual-en:start -->
Completing this course is not the same as having seen every formula. Without looking at answers, you should be able to:
<!-- bilingual-en:end -->

- 把自然语言命题写成量词形式并正确否定；
- 对错误证明指出第一处无效推理；
- 为归纳、不变量和结构归纳写出足够强的假设；
- 用 Euclidean algorithm 找 gcd 和 Bézout 系数，并解释 RSA 正确性；
- 在 digraph、partial order、simple graph 和 tree 之间切换；
- 从零建立一次计数或离散概率模型，而不是套一个不明来源的公式；
- 对最终数字做范围、单位、归一化或小规模枚举检查。
<!-- bilingual-en:start -->
- write a natural-language proposition with quantifiers and negate it correctly;
- identify the first invalid step in a flawed proof;
- state a sufficiently strong hypothesis for induction, invariants, and structural induction;
- use the Euclidean algorithm to find a gcd and Bézout coefficients, and explain why RSA works;
- move among digraphs, partial orders, simple graphs, and trees;
- build a counting or discrete-probability model from first principles rather than inserting an unexplained formula;
- check a final numerical result by range, units, normalization, or small-scale enumeration.
<!-- bilingual-en:end -->

**课程知识链：**逻辑语言 → 证明方法 → 归纳与不变量 → 数论与离散结构 → 计数 → 概率 → 随机过程与算法分析。
<!-- bilingual-en:start -->
**Course knowledge chain:** logical language → proof methods → induction and invariants → number theory and discrete structures → counting → probability → stochastic processes and algorithm analysis.
<!-- bilingual-en:end -->
