---
aliases:
  - MIT 6.042J Unit 1 Proofs
  - Mathematics for Computer Science Proofs
  - 离散数学证明单元
tags:
  - discrete-mathematics
  - mit-ocw
  - course-note
  - proofs
course: MIT 6.042J Mathematics for Computer Science
unit: 1
sessions: 1-11
---

# Unit 1: Proofs

> [!info] 课程与材料
> 本篇对应 MIT 6.042J / 18.062J *Mathematics for Computer Science*（Spring 2015）Unit 1，严格按官方 courseware 的 **Session 1–11** 顺序组织。课程地图见 [[00_MIT OCW 6.042J course map]]，原始材料见 [[MIT_OCW_6.042J_Materials/index|材料总索引]]。
>
> 本文会直接讲清理论；PDF、slides 与 transcript 是核对入口，不是理解正文的前提。在线 feedback exercises 的答案来自官方 courseware；In-Class Questions、Problem Sets 与 Midterm 1 没有公开答案，本文相关解答均标为**非官方独立题解**。
> <!-- bilingual-en:start -->
> This note covers Unit 1 of MIT 6.042J / 18.062J, *Mathematics for Computer Science* (Spring 2015), following the official Session 1–11 sequence. See [[00_MIT OCW 6.042J course map|the course map]] and [[MIT_OCW_6.042J_Materials/index|the complete materials index]].
>
> The note explains the theory directly. PDFs, slides, and transcripts are checkpoints for verification, not prerequisites for reading. Answers to online feedback exercises come from the official courseware; because no public solutions are available for the in-class questions, problem sets, or Midterm 1, the corresponding solutions here are explicitly identified as independent, unofficial solutions.
> <!-- bilingual-en:end -->

## 本单元要解决什么
<!-- bilingual-en:start -->
*What is addressed in this module*
<!-- bilingual-en:end -->

离散数学中的结论通常不是靠连续图像或数值近似得到，而是靠有限的逻辑链条。一个可检查的证明必须同时回答：
<!-- bilingual-en:start -->
The conclusion in discrete mathematics is usually not obtained by continuous image or numerical approximation, but by limited logic chain.  A verifiable proof must also answer:
<!-- bilingual-en:end -->

1. **对象是什么？** 变量的论域（domain）与符号是否明确？
2. **已知什么？** 哪些是定义、假设、先前定理或公理？
3. **要证什么？** 结论的量词与逻辑形状是什么？
4. **为什么每一步成立？** 是否误用了逆命题、除以零、遗漏情形或循环论证？
5. **边界是否覆盖？** 空集、零、最小规模、非终止过程与无限集合常常是漏洞所在。
<!-- bilingual-en:start -->
1. What is the**object?**Is the domain and symbol of the variable clear?
2. What is known about**?**What are definitions, assumptions, prior theorems, or axioms?
3. What is**to prove?**What are the quantifiers and logical shapes of conclusions?
4. **Why is each step valid?** Does it misuse the converse, divide by zero, omit a case, or argue circularly?
5. Does**the boundary cover?**Empty sets, zeros, minimum sizes, non-terminating processes, and infinite sets are often vulnerabilities.
<!-- bilingual-en:end -->

## 导航
<!-- bilingual-en:start -->
*navigation*
<!-- bilingual-en:end -->

- [[#Session 1 — Introduction to Proofs|S1 证明是什么]]
- [[#Session 2 — Proof Methods|S2 反证与分类]]
- [[#Session 3 — Well Ordering Principle|S3 良序原理]]
- [[#Session 4 — Logic and Propositions|S4 命题逻辑]]
- [[#Problem Set 1 — Sessions 1–4|PS1]]
- [[#Session 5 — Quantifiers and Predicate Logic|S5 量词与谓词逻辑]]
- [[#Session 6 — Sets|S6 集合]]
- [[#Problem Set 2 — Sessions 5–6|PS2]]
- [[#Session 7 — Binary Relations|S7 二元关系]]
- [[#Session 8 — Induction|S8 归纳法]]
- [[#Problem Set 3 — Sessions 7–8|PS3]]
- [[#Midterm 1 — Sessions 1–8|Midterm 1]]
- [[#Session 9 — State Machines and Invariants|S9 状态机与不变量]]
- [[#Session 10 — Recursive Definitions|S10 递归定义与结构归纳]]
- [[#Session 11 — Infinite Sets|S11 无限集合]]
- [[#Problem Set 4 — Sessions 9–11|PS4]]
<!-- bilingual-en:start -->
- [[#Session 1 — Introduction to Proofs|What is the S1 proof?]]
- [[#Session 2 — Proof Methods|S2 Contraposition, contradiction, and cases]]
- [[#Session 3 — Well Ordering Principle|S3 well-ordered principle]]
- [[#Session 4 — Logic and Propositions|S4 propositional logic]]
- [[#Problem Set 1 — Sessions 1–4|PS1]]
- [[#Session 5 — Quantifiers and Predicate Logic|S5 Quantifier and Predicate Logic]]
- [[#Session 6 — Sets|S6 Collection]]
- [[#Problem Set 2 — Sessions 5–6|PS2]]
- [[#Session 7 — Binary Relations|S7 binary relation]]
- [[#Session 8 — Induction|S8 induction]]
- [[#Problem Set 3 — Sessions 7–8|PS3]]
- [[#Midterm 1 — Sessions 1–8|Midterm 1]]
- [[#Session 9 — State Machines and Invariants|S9 State Machine and Invariant]]
- [[#Session 10 — Recursive Definitions|S10 Recursive Definition and Structure Induction]]
- [[#Session 11 — Infinite Sets|S11 infinite set]]
- [[#Problem Set 4 — Sessions 9–11|PS4]]
<!-- bilingual-en:end -->

## 官方 courseware block 顺序
<!-- bilingual-en:start -->
*Official courseware block Order*
<!-- bilingual-en:end -->

正文会合并重复解释，但以下新增知识严格按官方页面出现次序吸收；在线题仍在对应 Session 的折叠区逐项给出答案。
<!-- bilingual-en:start -->
The text merges and repeats the explanations, but the following new knowledge is absorbed strictly in the order in which the official pages appear; the online questions are still answered item by item in the fold corresponding to Session.
<!-- bilingual-en:end -->

| Session | 官方片段顺序 |
|---:|---|
| 1 | Welcome → Intro to Proofs I → Intro to Proofs II → Definitions → Modus Ponens |
| 2 | Contradiction video → Contradiction exercise → Cases video → When to use cases → Friends/Strangers → Bogus cases → Bogus contradiction |
| 3 | WOP I → Domain → WOP II → Counterexamples → WOP III → Geometric sum → Examples → Bogus WOP |
| 4 | Operators → operator exercise → Digital logic → Truth tables → Equivalence → Implies → Propositional logic → Soundness → Connectives |
| 5 | Predicate I → Predicate II → Satisfiability → Predicate III → Name that predicate → Quantifiers → Quantified propositions → Validity |
| 6 | Set definitions → Set operations → Difference |
| 7 | Relations → Range → Relational mappings → Total injection → Finite cardinality → $A\operatorname{inj}B$ → Total/Surjective → Inverse → In/Sur/Bijections → Mapping lemma |
| 8 | Induction → Bogus induction → Horses → Strong induction → Unstacking → WOP comparison → $n+3$ → Rules → Postage → Bogus proof diagnosis |
| 9 | State-machine invariants → invariant exercise → Derived variables → termination exercise → Integer multiplication → Chocolate bars |
| 10 | Recursive data → Matching brackets → F18 → Structural induction → template → case count → Recursive functions |
| 11 | Cardinality → Cantor/Schröder–Bernstein → Countable sets → Cantor theorem → Diagonal argument → Countability quiz → Halting → Russell → ZFC axioms |
<!-- bilingual-en:start -->
| Session | Official Fragment Order |
|—:|—|
| 1 | Welcome → Intro to Proofs I → Intro to Proofs II → Definitions → Modus Ponens |
| 2 | Contradiction video → Contradiction exercise → Cases video → When to use cases → Friends/Strangers → Bogus cases → Bogus contradiction |
| 3 | WOP I → Domain → WOP II → Counterexamples → WOP III → Geometric sum → Examples → Bogus WOP |
| 4 | Operators → operator exercise → Digital logic → Truth tables → Equivalence → Implies → Propositional logic → Soundness → Connectives |
| 5 | Predicate I → Predicate II → Satisfiability → Predicate III → Name that predicate → Quantifiers → Quantified propositions → Validity |
| 6 | Set definitions → Set operations → Difference |
| 7 | Relations → Range → Relational mappings → Total injection → Finite cardinality → $A\operatorname{inj}B$ → Total/Surjective → Inverse → In/Sur/Bijections → Mapping lemma |
| 8 | Induction → Bogus induction → Horses → Strong induction → Unstacking → WOP comparison → $n+3$ → Rules → Postage → Bogus proof diagnosis |
| 9 | State-machine invariants → invariant exercise → Derived variables → termination exercise → Integer multiplication → Chocolate bars |
| 10 | Recursive data → Matching brackets → F18 → Structural induction → template → case count → Recursive functions |
| 11 | Cardinality → Cantor/Schröder-Bernstein → Countable sets → Cantor theorem → Diagonal argument → Countability quiz → Halting → Russell → ZFC axioms |
<!-- bilingual-en:end -->

## 证明方法总图
<!-- bilingual-en:start -->
*proof method diagram*
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-proof-method-map.png|900]]

读图：先按目标命题的逻辑形状选择入口，再沿分支检查该方法必须建立的假设与最容易遗漏的条件。
<!-- bilingual-en:start -->
Read: First, the entry is selected according to the logical shape of the target proposition, and then the assumptions that must be established and the conditions that are most likely to be missed are checked along the branches.
<!-- bilingual-en:end -->

| 目标的逻辑形状 | 首选结构 | 开头必须写出 | 最常见漏洞 |
|---|---|---|---|
| $P\Rightarrow Q$ | [[数学证明方法#命题、量词与否定|直接证明]] | 假设 $P$ 成立 | 偷偷假设 $Q$；证明了逆命题 |
| $P\Rightarrow Q$，而 $\neg Q$ 信息更强 | [[数学证明方法#直接证明、逆否与反证|逆否证明]] | 假设 $\neg Q$ | 写成 $\neg P\Rightarrow\neg Q$ |
| “不存在……”或假设会造成冲突 | [[数学证明方法#直接证明、逆否与反证|反证法]] | 假设目标命题为假 | 得到的“矛盾”不是由该假设造成 |
| 论域自然分成互斥区域 | [[数学证明方法#直接证明、逆否与反证|分类证明]] | 列出覆盖全部可能的情形 | 漏掉 $0$、空集或交界点 |
| 对每个 $n\in\mathbb N$ | [[数学证明方法#归纳法|数学归纳法]] | 命题 $P(n)$ 与基例 | 归纳步没有连接到下一个整数 |
| $P(n)$ 依赖多个更小规模 | [[数学证明方法#归纳法|强归纳法]] | 所有 $P(k),k<n$ | 基例不足以启动递推 |
| 假设存在最小反例更易下降 | [[数学证明方法#良序与最小反例|良序原理]] | 反例集合 $C\subseteq\mathbb N$ | 构造的更小对象不在 $C$ 中 |
| 程序永不进入坏状态 | [[数学证明方法#不变量与算法正确性|不变量]] | 初始成立、转移保持 | 只证明“保持”，没证明初始成立 |
| 递归生成的所有对象 | [[数学证明方法#归纳法|结构归纳]] | 每个 base/constructor | 漏掉某个构造器 |
<!-- bilingual-en:start -->
| Logical shape of target | Preferred structure | Must start with | Most common vulnerabilities |
|—|—|—|—|
| $P\Rightarrow Q$ | [[数学证明方法#命题、量词与否定|direct proof]] | Assume $P$ | Silently assumes $Q$; proves the converse instead |
| $P\Rightarrow Q$, and $\neg Q$ is easier to use | [[数学证明方法#直接证明、逆否与反证|proof by contrapositive]] | Assume $\neg Q$ | Writes $\neg P\Rightarrow\neg Q$ instead |
| A nonexistence claim, or an assumption that should lead to a conflict | [[数学证明方法#直接证明、逆否与反证|proof by contradiction]] | Assume the target statement is false | The contradiction does not actually depend on that assumption |
| The domain naturally splits into exhaustive cases | [[数学证明方法#直接证明、逆否与反证|proof by cases]] | List cases covering every possibility | Omits $0$, the empty set, or a boundary case |
| For each $n\in\mathbb N$ | [[数学证明方法#归纳法|mathematical induction]] | Proposition $P(n)$ and base case | Inductive step not connected to the next integer |
| $P(n)$ relies on multiple smaller sizes | [[数学证明方法#归纳法|strong induction]] | All $P(k),k<n$ | not enough base cases to start recursion |
| Assuming a least counterexample makes descent easier | [[数学证明方法#良序与最小反例|well-ordering principle]] | Counterexample set $C\subseteq\mathbb N$ | The constructed smaller object does not belong to $C$ |
| A program never reaches a bad state | [[数学证明方法#不变量与算法正确性|invariant]] | Establish it initially and prove every transition preserves it | Proves preservation but not the initial case |
| All recursively generated objects | [[数学证明方法#归纳法|structural induction]] | Every base case and constructor | Omits a constructor |
<!-- bilingual-en:end -->

---

## Session 1 — Introduction to Proofs

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

本节不要求前置定理。核心问题是：什么叫“数学上已经证明”，以及一个正确结论为什么仍可能配有错误证明？
<!-- bilingual-en:start -->
No prior theorem is required. The central questions are what it means for a claim to be mathematically proved and how a true conclusion can still be accompanied by an invalid proof.
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session1.pdf|Session 1 reading]]
> - 课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp1.pdf|CP1]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/wIq4CssPoO0.pdf|Welcome]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/GyFVgJZ0hIs.pdf|Intro to Proofs I]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/wfr4XbR5VP8.pdf|Intro to Proofs II]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Welcome6.042.pdf|Welcome]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Introduction.pdf|Introduction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ProofExample.pdf|Proof Example]]

### 1.1 命题、谓词与证明
<!-- bilingual-en:start -->
*1.1 Proposition, Predicate and Proof*
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|命题（proposition）]]是一个具有确定真值的陈述；它要么真，要么假。例如“$2+3=5$”是命题，“$x+3=5$”在没有指定 $x$ 时不是命题。
<!-- bilingual-en:start -->
[[数学证明方法#命题、量词与否定|proposition]] is a statement with a definite truth value; it is either true or false.  For example, "$2+3=5$" is a proposition, and "$x+3=5$" is not a proposition when $x$ is not specified.
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|谓词（predicate）]] $P(x)$ 是含自由变量的真假条件。给定论域并代入 $x$，或给变量加量词后，它才成为命题。例如
<!-- bilingual-en:start -->
[[数学证明方法#命题、量词与否定|predicate]] $P(x)$ is a true and false condition with free variables.  It becomes a proposition only when a domain is given and replaced by a $x$, or a variable is quantified.  For example
<!-- bilingual-en:end -->

$$
P(x)\;:\Longleftrightarrow\;x^2=4
$$

在论域 $\mathbb Z$ 中，$P(2)$ 为真，$P(1)$ 为假；$\exists x\,P(x)$ 是真命题。
<!-- bilingual-en:start -->
In the universe $\mathbb Z$, $P(2)$ is true, $P(1)$ is false, and $\exists x\,P(x)$ is a true proposition.
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|公理（axiom）]]是在某个形式系统内接受为起点的命题；[[数学证明方法#命题、量词与否定|定理（theorem）]]是重要且已证明的命题；[[数学证明方法#命题、量词与否定|引理（lemma）]]为后续定理服务；[[数学证明方法#命题、量词与否定|推论（corollary）]]由已有定理经过很短推导得到。
<!-- bilingual-en:start -->
[[数学证明方法#命题、量词与否定|axiom]] is a proposition that is accepted as the starting point in a formal system; [[数学证明方法#命题、量词与否定|theorem]] is an important and proven proposition; [[数学证明方法#命题、量词与否定|lemma]] serves as a follow-up theorem; and [[数学证明方法#命题、量词与否定|corollary]] is derived from existing theorems in a very short time.
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|证明（proof）]]是从定义、公理、假设及已经证明的结论出发，按有效推理规则得到目标命题的有限序列。这里有两个关键限定：
<!-- bilingual-en:start -->
A [[数学证明方法#命题、量词与否定|proof]] is a finite sequence of valid inferences that derives a target proposition from definitions, axioms, assumptions, and previously proved results. Two qualifications matter:
<!-- bilingual-en:end -->

- **结论真，不代表给出的论证有效。** 一个错误论证可能恰好得到真命题。
- **例子多，不等于全称命题成立。** 检验一百万个输入仍不能证明“对所有输入”；但一个合法反例足以否定全称命题。
<!-- bilingual-en:start -->
- **A true conclusion does not make the argument valid.** An invalid argument can happen to reach a true statement.
- **Many examples do not establish a universal statement.** Checking a million inputs still does not prove a claim about all inputs, but one valid counterexample is enough to refute it.
<!-- bilingual-en:end -->

### 1.2 蕴含与 Modus Ponens
<!-- bilingual-en:start -->
*1.2 Implication and modus ponens*
<!-- bilingual-en:end -->

蕴含 $P\Rightarrow Q$ 声明：只要前件（antecedent）$P$ 成立，后件（consequent）$Q$ 必须成立。它唯一为假的情形是 $P$ 真而 $Q$ 假。
<!-- bilingual-en:start -->
The implication $P\Rightarrow Q$ states that whenever the antecedent $P$ is true, the consequent $Q$ must also be true. It is false only when $P$ is true and $Q$ is false.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-proof-implication.png|900]]

读图：重点看 $P$ 为真而 $Q$ 为假的唯一失败格；其余三格都不构成对蕴含的反例。
<!-- bilingual-en:start -->
Read: The only failure lattice where $P$ is true and $Q$ is false; none of the remaining three lattices constitute a counterexample to implication.
<!-- bilingual-en:end -->

最基本的推理规则是肯定前件（Modus Ponens）：
<!-- bilingual-en:start -->
The most basic rule of reasoning is the Modus Ponens:
<!-- bilingual-en:end -->

$$
\frac{P,\qquad P\Rightarrow Q}{Q}.
$$

这不是代数分式，而是“已知横线上方两项，即可推出横线下方结论”。相反，由 $P\Rightarrow Q$ 和 $Q$ 推出 $P$ 是**肯定后件谬误**。
<!-- bilingual-en:start -->
This is not an algebraic fraction, but rather "the above two terms of the known horizon are used to derive the conclusion below the horizon".  Instead, the introduction of $P$ by $P\Rightarrow Q$ and $Q$ is**sure of the fallacy**behind it.
<!-- bilingual-en:end -->

### 1.3 直接证明的标准结构
<!-- bilingual-en:start -->
*1.3 Standard Structure of Direct Proof*
<!-- bilingual-en:end -->

要证明 $P\Rightarrow Q$：
<!-- bilingual-en:start -->
To prove $P\Rightarrow Q$:
<!-- bilingual-en:end -->

1. 明确声明“假设 $P$”；
2. 只用该假设、定义和已知定理变形；
3. 到达 $Q$；
4. 用 $\square$ 或 QED 结束。
<!-- bilingual-en:start -->
1. Explicitly state "hypothetical $P$";
2. Only use the assumption, definition and known theorem deformation;
3. Arrival at $Q$;
4. Exit with $\square$ or QED.
<!-- bilingual-en:end -->

**示例：若整数 $n$ 为偶数，则 $n^2$ 为偶数。**
<!-- bilingual-en:start -->
**Example: If the integer $n$ is even, then $n^2$ is even.**
<!-- bilingual-en:end -->

目标是 $P(n)\Rightarrow Q(n)$，其中 $P(n)$ 为“$n$ 偶”，$Q(n)$ 为“$n^2$ 偶”。假设 $n$ 偶，按定义存在 $k\in\mathbb Z$ 使 $n=2k$。于是
<!-- bilingual-en:start -->
The target is $P(n)\Rightarrow Q(n)$, where $P(n)$ is "$n$ even" and $Q(n)$ is "$n^2$ even".  Suppose that $n$ is even, there is $k\in\mathbb Z$ by definition to make $n=2k$.  therefore
<!-- bilingual-en:end -->

$$
n^2=(2k)^2=4k^2=2(2k^2).
$$

因 $2k^2\in\mathbb Z$，所以 $n^2$ 可写成 $2$ 乘某整数，故为偶数。$\square$
<!-- bilingual-en:start -->
Because $2k^2\in\mathbb Z$, $n^2$ can be written as $2$ times an integer, and thus even.  $\square$
<!-- bilingual-en:end -->

### 1.4 先找“第一处错误”
<!-- bilingual-en:start -->
*1.4 Find the First Error*
<!-- bilingual-en:end -->

审查证明时，不要只说“答案不对”，而要找**第一处不再由前文推出的步骤**。常见来源：
<!-- bilingual-en:start -->
When auditing a proof, do not merely say that the conclusion is wrong. Locate **the first step that no longer follows from the preceding statements**. Common sources include:
<!-- bilingual-en:end -->

- 对零作除法；
- 对负数乘除却未反转不等号；
- 把 $\sqrt{ab}=\sqrt a\sqrt b$ 用到负数；
- 从最终恒真式倒推，却未验证每一步可逆；
- 用图形直觉时暗中假设不重叠、无缝、角度或长度关系。
<!-- bilingual-en:start -->
- Division by zero;
- Multiplying or dividing by a negative number without reversing the inequality sign;
- Applying $\sqrt{ab}=\sqrt a\sqrt b$ to negative numbers;
- Reasoning backward from a true final identity without checking that every preceding step is reversible;
- Silently assuming that regions do not overlap or leave gaps, or assuming unproved angle or length relations from a diagram.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（13 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S01_1.1.4_definitions-to-know-for-proofs|1.1.4 Definitions To Know For Proofs]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S01_1.1.5_modus-ponens|1.1.5 Modus Ponens]]
>
> **Definitions 1–7：**“真假确定的陈述”= proposition；“真值依赖变量”= predicate；“接受为真”= axiom；“从公理和旧结论到目标的逻辑序列”= proof；重要真命题 = theorem；服务于后续的预备命题 = lemma；由定理数步得到 = corollary。
>
> **Modus Ponens 1–6：**`IF P THEN Q` 是 implication；$P,P\Rightarrow Q\vdash Q$ 是 Modus Ponens；横线上方是 antecedent，横线下方既是 consequent 也是 conclusion；$P\Rightarrow Q$ 的逆否为 $\neg Q\Rightarrow\neg P$；证明通常以 QED 或 $\Box$ 收尾。官方反馈特别提醒：逆否命题与原命题等价，但 converse 不等价。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S01_1.1.4_definitions-to-know-for-proofs|1.1.4 Definitions To Know For Proofs]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S01_1.1.5_modus-ponens|1.1.5 Modus Ponens]]
> **Definitions 1–7:** A statement with a definite truth value is a proposition; a truth-valued expression depending on variables is a predicate; an axiom is accepted as true; a proof is a valid chain from axioms and established results to the target statement; a theorem is an important proved statement; a lemma is a supporting result used later; and a corollary follows from a theorem in a few steps.
> **Modus Ponens 1–6:** `IF P THEN Q` is an implication; $P,P\Rightarrow Q\vdash Q$ is Modus Ponens. Above the inference bar are the premises; below it is the conclusion. The contrapositive of $P\Rightarrow Q$ is $\neg Q\Rightarrow\neg P$. A proof usually ends with QED or $\Box$. The official feedback emphasizes that a statement is logically equivalent to its contrapositive, but not generally to its converse.
> <!-- bilingual-en:end -->

> [!example]- CP1 非官方独立题解（5 道）
> **1. 拼图版勾股定理。** 两种排法使用完全相同的四个直角三角形与一块小正方形。第一种把全部拼块排成外边长 $c$ 的正方形；相邻三角形的直角边相减，说明小正方形边长为 $|b-a|$。第二种把同一批拼块排成互不重叠的 $a\times a$ 与 $b\times b$ 两个正方形。面积在刚性移动与重新排列下保持，所以第一种总面积 $c^2$ 等于第二种总面积 $a^2+b^2$。当 $a=b$ 时小正方形退化为边长 0 的区域，论证仍可由极限情形或直接排法理解。隐含事实包括三角形全等、直角/补角关系、外框确为正方形、拼块无缝且内部不重叠，以及面积有限可加。
>
> **2. $1=-1$ 伪证。** $\sqrt{(-1)(-1)}=\sqrt{-1}\sqrt{-1}$ 非法：实数平方根乘法规则只对非负因子成立；在复数主值平方根下该规则也不普遍成立。命题“若 $1=-1$，则 $2=1$”在实数中为真，是因为前件与已知事实 $1\ne-1$ 矛盾；由矛盾可推出任意结论（principle of explosion）。不能假装从普通环代数直接算出 $2=1$：在 characteristic 2 的代数结构中 $1=-1$ 可成立而 $2=0\ne1$。对 $r,s>0$，令 $u=\sqrt r\sqrt s>0$，则 $u^2=rs$；正平方根的唯一性给出 $u=\sqrt{rs}$。
>
> **3. 三个伪证。** (a) $\log_{10}(1/2)<0$，乘不等式时方向应反转；(b) $(10\text{ cents})^2$ 的单位是 cents$^2$，不能等同金额，且 $10^2=100$ 也遗漏单位缩放；(c) 由 $a=b$ 得 $a-b=0$，从 $(a-b)(a+b)=(a-b)b$ 约去 $a-b$ 等于除以零。
>
> **4. AM–GM。** 原论证从待证式一路推到真式；若中间有不可逆步骤，就不能倒推。本题因 $a,b\ge0$，$a+b$ 与 $2\sqrt{ab}$ 非负，平方是等价变形，所有步骤其实可逆。修复方法是从 $(a-b)^2\ge0$ 正向展开：$a^2-2ab+b^2\ge0\Rightarrow(a+b)^2\ge4ab$，再对非负两边开方得 $(a+b)/2\ge\sqrt{ab}$。
>
> **5. Surprise quiz。** 学生把“若此前没考则周五可预测”这种条件知识反向递归使用；但一旦推导出“本周不可能考试”，周四晚并不会保持“老师承诺必考且此前未考”的共同知识。自然语言中的 surprise 还依赖认知状态，不能当作静态命题逐日消去。
> <!-- bilingual-en:start -->
> **1. A dissection proof of the Pythagorean theorem.** Both arrangements use exactly the same four congruent right triangles and one small square. In the first arrangement the pieces fill a square of side $c$; subtracting adjacent legs shows that the inner square has side $|b-a|$. In the second, the same pieces form two nonoverlapping squares of sides $a$ and $b$. Rigid motions and rearrangement preserve area, so the first total area $c^2$ equals the second total area $a^2+b^2$. When $a=b$, the inner square degenerates to side length zero, and the same conclusion follows from the limiting or direct arrangement. The argument relies on triangle congruence, the right-angle and supplementary-angle relations, the outer boundary being a square, nonoverlapping exact tiling, and finite additivity of area.
> **2. A bogus proof that $1=-1$.** The step $\sqrt{(-1)(-1)}=\sqrt{-1}\sqrt{-1}$ is invalid: the product rule for real square roots requires nonnegative factors, and it is not generally valid for principal complex square roots either. The implication “if $1=-1$, then $2=1$” is true over the reals because its premise contradicts the known fact $1\ne-1$; by the principle of explosion, any conclusion follows from a contradiction. But ordinary ring algebra does not yield $2=1$: in characteristic $2$, $1=-1$ can hold while $2=0\ne1$. For $r,s>0$, let $u=\sqrt r\sqrt s>0$. Then $u^2=rs$, so uniqueness of the positive square root gives $u=\sqrt{rs}$.
> **3. Three bogus proofs.** (a) Since $\log_{10}(1/2)<0$, multiplying an inequality by it reverses the inequality sign. (b) $(10\text{ cents})^2$ has units of cents$^2$, not money, and the calculation also mishandles the unit conversion. (c) From $a=b$ we get $a-b=0$; cancelling $a-b$ from $(a-b)(a+b)=(a-b)b$ therefore divides by zero.
> **4. AM–GM.** The original argument works backward from the desired inequality to a true statement. Such reasoning is valid only if every step is reversible. Here $a,b\ge0$, so both $a+b$ and $2\sqrt{ab}$ are nonnegative and squaring is an equivalence. A cleaner proof starts from $(a-b)^2\ge0$: $a^2-2ab+b^2\ge0\Rightarrow(a+b)^2\ge4ab$, then takes square roots of the nonnegative sides to obtain $(a+b)/2\ge\sqrt{ab}$.
> **5. Surprise quiz.** The students recursively reason backward from “if no quiz has occurred earlier, a Friday quiz would be predictable.” Once they conclude that no quiz can occur during the week, however, Thursday evening no longer preserves the common-knowledge state “the teacher will give the promised quiz and none has occurred yet.” Surprise in ordinary language also depends on agents' knowledge states, so it cannot be eliminated by treating each day as a static proposition.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. “$n^2$ 偶 $\Rightarrow n$ 偶”的直接证明为什么不宜从 $n=2k$ 开始？
> 2. 说明“前 1000 个素数都是奇数”为什么本身是假命题，并指出反例。
> 3. 从 $x=y$ 推到 $x^2=xy$ 是否可逆？在什么条件下？
>
> <!-- bilingual-en:start -->
> 1. Why is it invalid to begin a direct proof of “$n^2$ even $\Rightarrow n$ even” by writing $n=2k$?
> 2. Explain why “the first 1000 primes are odd” is false, and give a counterexample.
> 3. Is the step from $x=y$ to $x^2=xy$ reversible? Under what additional condition?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. $n=2k$ 正是待证结论，等于预设结论；应证逆否“$n$ 奇则 $n^2$ 奇”或用素因子性质。2. 素数 $2$ 为偶数。3. 正向恒可；反向由 $x(x-y)=0$ 只能得到 $x=0$ 或 $x=y$，若另知 $x\ne0$ 才可推出 $x=y$。
> <!-- bilingual-en:start -->
> 1. Writing $n=2k$ assumes exactly the conclusion to be proved. Instead prove the contrapositive, “$n$ odd implies $n^2$ odd,” or use prime factorisation. 2. The prime number $2$ is even. 3. The forward implication always holds. In reverse, $x(x-y)=0$ gives only $x=0$ or $x=y$; the additional assumption $x\ne0$ is needed to conclude $x=y$.
> <!-- bilingual-en:end -->

**知识链：**命题/谓词 → 蕴含 → 有效推理 → 直接证明 → 错误定位。
<!-- bilingual-en:start -->
**Knowledge chain:** proposition/predicate → implication → valid inference → direct proof → locating an invalid step.
<!-- bilingual-en:end -->

---

## Session 2 — Proof Methods

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

当直接从 $P$ 推到 $Q$ 很困难时，怎样在不改变目标真值的前提下重写证明任务？前置是 Session 1 的蕴含与反例意识。
<!-- bilingual-en:start -->
When a direct derivation from $P$ to $Q$ is difficult, how can the proof task be reformulated without changing its truth? The prerequisites are Session 1's treatment of implication and counterexamples.
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session2.pdf|Session 2 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp2.pdf|CP2]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CpW0ZJ7i0oc.pdf|Proof by Contradiction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/vzpFQ3uNyPo.pdf|Proof by Cases]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ProofContrad.pdf|Contradiction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ProofCases.pdf|Cases]]

### 2.1 逆否证明
<!-- bilingual-en:start -->
*2.1 Inverse proof*
<!-- bilingual-en:end -->

命题 $P\Rightarrow Q$ 与逆否命题 $\neg Q\Rightarrow\neg P$ 等价，因为
<!-- bilingual-en:start -->
The implication $P\Rightarrow Q$ is equivalent to its contrapositive $\neg Q\Rightarrow\neg P$ because
<!-- bilingual-en:end -->

$$
P\Rightarrow Q\equiv \neg P\lor Q\equiv \neg Q\Rightarrow\neg P.
$$

**示例：若 $n^2$ 为偶数，则 $n$ 为偶数。** 证逆否：假设 $n$ 为奇数，即 $n=2k+1$。则
<!-- bilingual-en:start -->
**Example: if $n^2$ is even, then $n$ is even.** Prove the contrapositive. Assume that $n$ is odd, so $n=2k+1$.
<!-- bilingual-en:end -->

$$
n^2=4k^2+4k+1=2(2k^2+2k)+1,
$$

仍为奇数。因此“$n$ 非偶 $\Rightarrow n^2$ 非偶”，原命题成立。
<!-- bilingual-en:start -->
which is still odd. Therefore, odd $n$ implies odd $n^2$, so the original statement follows by contraposition.
<!-- bilingual-en:end -->

### 2.2 反证法
<!-- bilingual-en:start -->
*2.2 Proof by contradiction*
<!-- bilingual-en:end -->

证明命题 $T$ 时，[[数学证明方法#直接证明、逆否与反证|反证法]]假设 $\neg T$，再推出某命题 $R$ 与 $\neg R$ 同时成立。逻辑骨架是
<!-- bilingual-en:start -->
To prove a statement $T$ by [[数学证明方法#直接证明、逆否与反证|contradiction]], assume $\neg T$ and derive both some statement $R$ and its negation $\neg R$. The logical skeleton is
<!-- bilingual-en:end -->

$$
\neg T\Rightarrow\bot\quad\Longrightarrow\quad T.
$$

矛盾必须依赖反设；如果不用反设也能推出矛盾，说明背景假设本身不一致或推导有误。
<!-- bilingual-en:start -->
The contradiction must depend on the assumption $\neg T$. If it can be derived without that assumption, then either the background assumptions are already inconsistent or the derivation is wrong.
<!-- bilingual-en:end -->

**$\sqrt2$ 无理性的完整证明。** 假设 $\sqrt2=a/b$，其中 $a,b\in\mathbb Z$、$b>0$，且已约为最简分数。平方得 $a^2=2b^2$，所以 $a^2$ 偶。由上一节结论 $a$ 偶，写成 $a=2k$。代回：$4k^2=2b^2$，故 $b^2=2k^2$，从而 $b$ 偶。于是 $a,b$ 都被 $2$ 整除，与“最简”矛盾。因此 $\sqrt2$ 无理。
<!-- bilingual-en:start -->
**A complete proof that $\sqrt2$ is irrational.** Assume $\sqrt2=a/b$, where $a,b\in\mathbb Z$, $b>0$, and $a/b$ is in lowest terms. Squaring gives $a^2=2b^2$, so $a^2$ is even. By the previous result, $a$ is even; write $a=2k$. Substitution gives $4k^2=2b^2$, hence $b^2=2k^2$, so $b$ is even as well. Thus both $a$ and $b$ are divisible by $2$, contradicting the assumption that the fraction is in lowest terms. Therefore $\sqrt2$ is irrational.
<!-- bilingual-en:end -->

> [!warning] 严格性说明
> 证明用到了“每个有理数都有最简整数分数表示”和“若 $n^2$ 偶则 $n$ 偶”。这两个事实不能因为熟悉就隐去；前者可由最大公因数或良序原理证明。
> <!-- bilingual-en:start -->
> The proof uses two facts: every rational number has a representation by coprime integers, and $n^2$ even implies $n$ even. Familiarity is not a reason to hide these dependencies; the first follows from greatest common divisors or the well-ordering principle.
> <!-- bilingual-en:end -->

### 2.3 分类证明
<!-- bilingual-en:start -->
*2.3 Proof by cases*
<!-- bilingual-en:end -->

[[数学证明方法#直接证明、逆否与反证|分类证明]]把论域写成覆盖全部可能的若干情形 $C_1,\dots,C_k$，分别证明 $C_i\Rightarrow T$。分类不一定互斥，但必须穷尽：
<!-- bilingual-en:start -->
[[数学证明方法#直接证明、逆否与反证|proof of classification]] wrote the domain as covering all possible cases $C_1,\dots,C_k$, proving $C_i\Rightarrow T$ respectively.  Categorization is not necessarily mutually exclusive, but it must be exhausted:
<!-- bilingual-en:end -->

$$
C_1\lor\cdots\lor C_k\quad\text{必须恒真。}
$$

**朋友或陌生人。** 任取六人中的一人 $v$。其余五人与 $v$ 的关系只有 friend/stranger 两类，鸽巢原理保证至少三人与 $v$ 同类。若这三人中有一对朋友，则与 $v$（当该类是朋友）组成三朋友；否则三人两两陌生。另一类同理，因此六人中必有三人两两朋友或两两陌生，即 Ramsey 数满足 $R(3,3)\le6$。
<!-- bilingual-en:start -->
**Friends or strangers.** Choose one of the six people and call that person $v$. Each of the other five is either a friend or a stranger to $v$, so the pigeonhole principle guarantees that at least three fall in the same category. If those three are friends of $v$ and some pair among them are friends, that pair together with $v$ forms a mutually acquainted triple; if no such pair exists, the three are mutually strangers. The stranger-to-$v$ case is analogous. Hence among any six people there are three mutual friends or three mutual strangers, so $R(3,3)\le6$.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（5 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.2_proof-by-contradiction|1.2.2 Proof By Contradiction]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.4_when-to-prove-by-cases|1.2.4 When to Prove by Cases]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.5_friends-and-strangers|1.2.5 Friends and Strangers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.6_a-bogus-proof-by-cases|1.2.6 A Bogus Proof by Cases]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.7_a-bogus-proof-by-contradiction|1.2.7 A Bogus Proof by Contradiction]]
>
> 1. $\sqrt2$ 反证依赖“奇数乘积仍为奇数”（等价地素因子分解的奇偶性质）与整数素因子分解。2. 适合分类的条件：论域能拆分、各分支更易证、分支合起来覆盖全部；“直接证明不可行”本身不是充分标准。3. 朋友/陌生人论证给出 $R(3,3)\le6$。4. 伪分类“整数只有正、负两类”漏掉 $a=0$。5. $\sqrt4$ 伪反证两次误用“$4\mid n^2\Rightarrow4\mid n$”；正确只能推出 $2\mid n$。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.2_proof-by-contradiction|1.2.2 Proof By Contradiction]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.4_when-to-prove-by-cases|1.2.4 When to Prove by Cases]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.5_friends-and-strangers|1.2.5 Friends and Strangers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.6_a-bogus-proof-by-cases|1.2.6 A Bogus Proof by Cases]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S02_1.2.7_a-bogus-proof-by-contradiction|1.2.7 A Bogus Proof by Contradiction]]
> 1. The contradiction proof for $\sqrt2$ uses unique prime factorisation, equivalently the fact that a product of odd integers is odd. 2. Proof by cases is appropriate when the domain can be split into exhaustive cases that are easier to prove separately; the mere failure of a direct proof is not sufficient. 3. The friends-and-strangers argument proves $R(3,3)\le6$. 4. The bogus split “every integer is positive or negative” omits $0$. 5. The bogus proof about $\sqrt4$ twice misuses $4\mid n^2\Rightarrow4\mid n$; the valid conclusion is only $2\mid n$.
> <!-- bilingual-en:end -->

> [!example]- CP2 非官方独立题解（4 道）
> **1. 若 $ab=n$，则 $a\le\sqrt n$ 或 $b\le\sqrt n$。** 反设二者都 $>\sqrt n$。因均非负，相乘保持不等号，得 $ab>n$，与 $ab=n$ 矛盾。
>
> **2. 推广 $\sqrt3$。** 假设 $\sqrt3=a/b$ 为最简分数。得 $a^2=3b^2$。模 $3$ 的平方余数只有 $0,1$，故 $3\mid a^2\Rightarrow3\mid a$；写 $a=3k$ 后得 $b^2=3k^2$，故 $3\mid b$，矛盾。相同模板适用于非完全平方的素数 $p$。
>
> **3. 无理数的无理次幂可为有理数。** 令 $x=(\sqrt2)^{\sqrt2}$。若 $x\in\mathbb Q$，取 $a=b=\sqrt2$ 即得例子；若 $x\notin\mathbb Q$，取 $a=x,b=\sqrt2$，则 $a^b=((\sqrt2)^{\sqrt2})^{\sqrt2}=(\sqrt2)^2=2$。两种情形穷尽。
>
> **4. 构造性例子。** 令 $a=\sqrt2$，$b=2\log_2 3$。若 $b=r/s\in\mathbb Q$（$s>0$），则 $2^{r/s}=3^2=9$，故 $2^r=9^s=3^{2s}$；唯一素因子分解使左边只有素因子 $2$、右边只有 $3$，不可能。因此 $b$ 无理，且 $a^b=2^{b/2}=2^{\log_2 3}=3$。
> <!-- bilingual-en:start -->
> **1. If $ab=n$, then $a\le\sqrt n$ or $b\le\sqrt n$.** Suppose instead that both are greater than $\sqrt n$. Since they are nonnegative, multiplication preserves the inequalities and gives $ab>n$, contradicting $ab=n$.
> **2. Generalising to $\sqrt3$.** Assume that $\sqrt3=a/b$ in lowest terms. Then $a^2=3b^2$. The only quadratic residues modulo $3$ are $0$ and $1$, so $3\mid a^2$ implies $3\mid a$. Writing $a=3k$ gives $b^2=3k^2$, hence $3\mid b$, a contradiction. The same template applies to any prime $p$ that is not a perfect square.
> **3. An irrational number raised to an irrational power can be rational.** Let $x=(\sqrt2)^{\sqrt2}$. If $x\in\mathbb Q$, choose $a=b=\sqrt2$. If $x\notin\mathbb Q$, choose $a=x$ and $b=\sqrt2$; then $a^b=((\sqrt2)^{\sqrt2})^{\sqrt2}=(\sqrt2)^2=2$. These two cases are exhaustive.
> **4. Constructive examples.**$a=\sqrt2$, $b=2\log_2 3$.  If $b=r/s\in\mathbb Q$($s>0$), then $2^{r/s}=3^2=9$, so $2^r=9^s=3^{2s}$; the only prime factorization makes the left prime factor $2$, the right only $3$, impossible.  So $b$ is irrational, and $a^b=2^{b/2}=2^{\log_2 3}=3$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 反证法与逆否证明的假设分别是什么？
> 2. “所有实数非负或非正”是否构成合法分类？
> 3. 为何由 $p\mid n^2$ 推 $p\mid n$ 要求 $p$ 为素数？
>
> <!-- bilingual-en:start -->
> 1. What assumptions do proof by contradiction and proof by contrapositive make, respectively?
> 2. Do the cases “nonnegative” and “nonpositive” form a valid case split for all real numbers?
> 3. Why does the implication $p\mid n^2\Rightarrow p\mid n$ require $p$ to be prime?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 反证假设整个目标为假；逆否证明在 $P\Rightarrow Q$ 中假设 $\neg Q$ 并证 $\neg P$。2. 合法且覆盖全部实数，$0$ 同时落入两类不妨碍证明。3. 合数反例：$4\mid 2^2$，但 $4\nmid2$。
> <!-- bilingual-en:start -->
> 1. Proof by contradiction assumes that the target statement is false; proof by contrapositive assumes $\neg Q$ and proves $\neg P$ when the target is $P\Rightarrow Q$. 2. Yes. The cases cover all real numbers, and their overlap at $0$ does not invalidate the proof. 3. A composite counterexample is $4\mid2^2$ but $4\nmid2$.
> <!-- bilingual-en:end -->

**知识链：**蕴含等价变形 → 逆否 → 反证 → 穷尽分类 → 识别伪证。
<!-- bilingual-en:start -->
**Knowledge chain:** equivalent forms of implication → contraposition → contradiction → exhaustive case analysis → diagnosing bogus proofs.
<!-- bilingual-en:end -->

---

## Session 3 — Well Ordering Principle

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

如何把“若有反例，取最小反例并把它变得更小”写成严密证明？前置是反证法和非负整数。
<!-- bilingual-en:start -->
How can “if a counterexample exists, choose the least one and construct a still smaller counterexample” be turned into a rigorous proof? The prerequisites are proof by contradiction and the nonnegative integers.
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session3.pdf|Session 3 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp3.pdf|CP3]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/fV3v6qQ3w4A.pdf|WOP I]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/I1HpgnWQI7I.pdf|WOP II]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/hNrtGiCFPGs.pdf|WOP III]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_WellOrdering1.pdf|WOP I]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_WellOrdering2.pdf|WOP II]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_WellOrdering3.pdf|WOP III]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_GeometricSum.pdf|Geometric Sum]]

### 3.1 原理与最小反例模板
<!-- bilingual-en:start -->
*3.1 The principle and the least-counterexample template*
<!-- bilingual-en:end -->

[[数学证明方法#良序与最小反例|良序原理（Well-Ordering Principle, WOP）]]：每个非空的非负整数集合 $S\subseteq\mathbb N$ 都有最小元素。
<!-- bilingual-en:start -->
[[数学证明方法#良序与最小反例|Well-Ordering Principle (WOP)]]: Each non-empty non-negative integer collection $S\subseteq\mathbb N$ has a minimum element.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-well-ordering-descent.png|900]]

读图：从假定的最小反例 $m$ 构造同一反例集合中的更小对象 $m'$；两者同时存在才真正违背最小性。
<!-- bilingual-en:start -->
Reading the diagram: starting from a hypothetical least counterexample $m$, construct a smaller object $m'$ in the same counterexample set. Only after both membership and $m'<m$ have been established is minimality contradicted.
<!-- bilingual-en:end -->

证明 $\forall n\in\mathbb N,P(n)$ 的模板：
<!-- bilingual-en:start -->
A template for proving $\forall n\in\mathbb N,P(n)$:
<!-- bilingual-en:end -->

1. 定义反例集合 $C=\{n\in\mathbb N:\neg P(n)\}$；
2. 反设 $C\ne\varnothing$；
3. WOP 给出 $m=\min C$；
4. 用 $m$ 的性质构造 $m'<m$，并严格验证 $m'\in C$；
5. 与 $m$ 最小矛盾，因此 $C=\varnothing$。
<!-- bilingual-en:start -->
1. Define the counterexample set $C=\{n\in\mathbb N:\neg P(n)\}$;
2. Suppose for contradiction that $C\ne\varnothing$;
3. WOP gives $m=\min C$;
4. Construct $m'<m$ with the properties of $m$ and strictly verify $m'\in C$;
5. This contradicts the minimality of $m$, so $C=\varnothing$.
<!-- bilingual-en:end -->

“$m'<m$”本身不够：若 $m'$ 为负数、未满足问题定义域、或根本不是反例，就没有矛盾。
<!-- bilingual-en:start -->
"$m'<m$" is not enough per se: there is no contradiction if $m'$ is negative, does not satisfy the problem domain, or is not at all a counterexample.
<!-- bilingual-en:end -->

### 3.2 每个大于 1 的整数都是素数乘积
<!-- bilingual-en:start -->
*3.2 Each integer greater than 1 is a prime product*
<!-- bilingual-en:end -->

令 $C$ 为不能写成素数乘积的、$>1$ 的整数集合。若非空，取最小 $m$。$m$ 不可能是素数，否则它本身就是一个素数乘积。所以 $m=ab$，其中 $1<a,b<m$。由 $m$ 的最小性，$a,b$ 都能写成素数乘积；相乘便给出 $m$ 的素数乘积分解，与 $m\in C$ 矛盾。
<!-- bilingual-en:start -->
Let $C$ be a collection of $>1$ integers that cannot be written as a product of primes.  If not empty, take the smallest $m$.  $m$ cannot be a prime number, otherwise it is itself a prime product.  So $m=ab$, where $1<a,b<m$.  From the minimality of $m$, $a,b$ can be written as a prime product, and the solution of $m$'s prime product is given by the product, which is contradictory to $m\in C$.
<!-- bilingual-en:end -->

这段证明的下降量是整数大小；严格边界 $1<a,b<m$ 保证可调用最小性。
<!-- bilingual-en:start -->
The amount of descent for this proof is an integer size; the strict bound $1<a,b<m$ guarantees callable minimality.
<!-- bilingual-en:end -->

### 3.3 几何级数公式
<!-- bilingual-en:start -->
*3.3 Geometric Series Formulas*
<!-- bilingual-en:end -->

对 $r\ne1$，目标为
<!-- bilingual-en:start -->
For $r\ne1$, the target is
<!-- bilingual-en:end -->

$$
1+r+\cdots+r^n=\frac{r^{n+1}-1}{r-1}.
$$

若有反例，取最小 $m$。$m\ne0$，因两边均为 $1$。于是 $m-1\ge0$ 且不是反例：
<!-- bilingual-en:start -->
If there is a counterexample, the smallest $m$ is taken.  $m\ne0$, since both sides are $1$.  So $m-1\ge0$, not the counter example:
<!-- bilingual-en:end -->

$$
1+\cdots+r^{m-1}=\frac{r^m-1}{r-1}.
$$

两边加 $r^m$：
<!-- bilingual-en:start -->
Add $r^m$ on both sides:
<!-- bilingual-en:end -->

$$
1+\cdots+r^m
=\frac{r^m-1+r^m(r-1)}{r-1}
=\frac{r^{m+1}-1}{r-1},
$$

所以 $m$ 也不是反例，矛盾。条件 $r\ne1$ 是因分母；$r=1$ 时和为 $n+1$，需单独处理。
<!-- bilingual-en:start -->
So $m$'s not the opposite. Contradiction.  The condition $r\ne1$ is denominator; $r=1$ and is $n+1$ and should be treated separately.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（5 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.2_domain-for-well-ordering-principle|1.3.2 Domain for Well Ordering Principle]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.4_well-ordering-proofs-and-counterexamples|1.3.4 Well Ordering Proofs and Counterexamples]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.6_wop-proof-for-geometric-sum|1.3.6 WOP Proof for Geometric Sum]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.7_well-ordering-principle-examples|1.3.7 Well Ordering Principle — Examples]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.8_a-bogus-well-ordering-principle-proof|1.3.8 A Bogus Well Ordering Principle Proof]]
>
> 1. WOP 的论域是 nonnegative integers。2. 素数乘积分解证明最终说明反例集合为空。3. 几何和证明的矛盾是“最小反例 $m$ 实际也满足公式”。4. well-ordered 的选项包括：整数下界截尾集、有限集合、分母有统一有限上界的正有理数集合、$\{n/(n+1)\}$、$\mathbb N\cup\{n/(n+1)\}$，以及“每个元素下方仅有限多元素”的实数子集；$\{1/n:n\ge1\}$ 和 $\mathbb Q\cap[\sqrt2,\infty)$ 没有最小元。5. “所有 Fibonacci 数均偶”的伪 WOP 证明漏掉 $m=1$。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.2_domain-for-well-ordering-principle|1.3.2 Domain for Well Ordering Principle]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.4_well-ordering-proofs-and-counterexamples|1.3.4 Well Ordering Proofs and Counterexamples]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.6_wop-proof-for-geometric-sum|1.3.6 WOP Proof for Geometric Sum]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.7_well-ordering-principle-examples|1.3.7 Well Ordering Principle — Examples]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S03_1.3.8_a-bogus-well-ordering-principle-proof|1.3.8 A Bogus Well Ordering Principle Proof]]
> 1. WOP is applied to subsets of the nonnegative integers. 2. The prime-factorisation argument ultimately proves that the counterexample set is empty. 3. In the geometric proof, the contradiction is that the least counterexample $m$ also satisfies the formula. 4. Well-ordered examples include a lower-truncated subset of the integers, any finite set, positive rationals whose denominators have a common finite upper bound, $\{n/(n+1)\}$, $\mathbb N\cup\{n/(n+1)\}$, and any subset of the reals with only finitely many elements below each member. By contrast, $\{1/n:n\ge1\}$ and $\mathbb Q\cap[\sqrt2,\infty)$ have no least element. 5. The bogus WOP proof that all Fibonacci numbers are even overlooks $m=1$.
> <!-- bilingual-en:end -->

> [!example]- CP3 非官方独立题解（5 道）
> **1. 6¢/15¢ 邮资必被 3 整除。** $C=\{n\in\mathbb N:S(n)\land3\nmid n\}$。若 $m=\min C$，则 $m>0$；任何正邮资组合至少含一张 6¢ 或 15¢ 邮票，去掉它得到 $S(m-6)$ 或 $S(m-15)$。该数更小且非负，故不是反例，于是被 3 整除；加回 6 或 15 后 $m$ 也被 3 整除，矛盾。故 $C=\varnothing$。
>
> **2. 平方和。** 令 $C$ 为不满足 $\sum_{k=0}^n k^2=n(n+1)(2n+1)/6$ 的 $n$。若 $m=\min C$，则 $m>0$。对 $m-1$ 用公式并加 $m^2$：
> $$\frac{(m-1)m(2m-1)}6+m^2=\frac{m[(m-1)(2m-1)+6m]}6=\frac{m(m+1)(2m+1)}6,$$
> 矛盾。
>
> **3. Lehman 方程无正整数解。** 若 $8a^4+4b^4+2c^4=d^4$ 有正整数解，取 $a$ 最小的一组。
>
> 先模 $2$：左边为偶数，所以 $d^4$ 偶，进而 $d$ 偶；写 $d=2d_1$。代回并除以 $2$：
> $$4a^4+2b^4+c^4=8d_1^4.$$
> 再模 $2$ 得 $c^4$ 偶，所以 $c=2c_1$。代回原式并除以 $4$：
> $$2a^4+b^4+8c_1^4=4d_1^4.$$
> 模 $2$ 得 $b^4$ 偶，所以 $b=2b_1$。再代回原式并除以 $8$：
> $$a^4+8b_1^4+4c_1^4=2d_1^4.$$
> 模 $2$ 得 $a^4$ 偶，所以 $a=2a_1$。因此 $a,b,c,d$ 全偶。把四者分别写成原来的一半，原方程两边都含因子 $2^4$；约去后得到同型正整数解 $(a_1,b_1,c_1,d_1)$，且 $a_1<a$，与最小性矛盾。
>
> **4. 二进制信封。** 反设某 $m$ 失败，取最小失败 $m$。$m=0$ 时金额 $0,1$ 可由空集或 $1$ 信封得到，故 $m>0$。对目标 $t<2^{m+1}$：若 $t<2^m$，由 $m-1$ 性质表示；若 $t\ge2^m$，先选 $2^m$ 信封，余数 $t-2^m<2^m$ 由此前信封表示。故 $m$ 不失败。
>
> **5. $n\ge30$ 可由 $6,10,15$ 表示。** 若反例非空，取最小 $m\ge30$。直接验证 $30=15+15,31=15+10+6,32=10+10+6+6,33=15+6+6+6,34=10+6\cdot4,35=15+10+10$，故 $m\ge36$。于是 $m-6\ge30$ 且比 $m$ 小，可表示；再加一张 6¢，矛盾。
> <!-- bilingual-en:start -->
> **1. Every 6¢/15¢ postage total is divisible by 3.** Let $C=\{n\in\mathbb N:S(n)\land3\nmid n\}$. If $m=\min C$, then $m>0$. Any positive postage combination contains a 6¢ or 15¢ stamp; remove one to obtain $S(m-6)$ or $S(m-15)$. This smaller nonnegative total is not a counterexample, so it is divisible by 3. Adding back 6 or 15 preserves divisibility by 3, contradicting $m\in C$. Hence $C=\varnothing$.
> **2. Sum of squares.** Let $C$ be the set of $n$ for which $\sum_{k=0}^n k^2=n(n+1)(2n+1)/6$ fails. If $m=\min C$, then $m>0$. Apply the formula to $m-1$ and add $m^2$:
> $$\frac{(m-1)m(2m-1)}6+m^2=\frac{m[(m-1)(2m-1)+6m]}6=\frac{m(m+1)(2m+1)}6,$$
> which is exactly the claimed formula for $m$, a contradiction.
> **3. No positive integer solution to the Lehman equation.** Suppose $8a^4+4b^4+2c^4=d^4$ has a positive integer solution and choose one with minimal $a$. Modulo $2$, the left side is even, so $d$ is even; write $d=2d_1$ and divide by $2$:
> $$4a^4+2b^4+c^4=8d_1^4.$$
> Reducing modulo $2$ shows that $c$ is even; write $c=2c_1$, substitute, and divide by $4$:
> $$2a^4+b^4+8c_1^4=4d_1^4.$$
> Reducing modulo $2$ again shows that $b$ is even; write $b=2b_1$, substitute, and divide by $8$:
> $$a^4+8b_1^4+4c_1^4=2d_1^4.$$
> A final reduction modulo $2$ shows that $a$ is even; write $a=2a_1$. Thus $a,b,c,d$ are all even. Cancelling the common factor $2^4$ from the original equation gives a smaller positive integer solution $(a_1,b_1,c_1,d_1)$ with $a_1<a$, contradicting minimality.
> **4. Binary envelopes.** Suppose some $m$ fails and choose the least such $m$. For $m=0$, the amounts $0$ and $1$ are represented by the empty set and the 1-unit envelope, so $m>0$. For any $t<2^{m+1}$, either $t<2^m$ and the induction hypothesis applies, or $t\ge2^m$, in which case choose the $2^m$ envelope and represent the remainder $t-2^m<2^m$. Thus $m$ cannot fail.
> **5. Every $n\ge30$ is representable using $6,10,15$.** If counterexamples exist, choose the least $m\ge30$. Directly verify $30=15+15,31=15+10+6,32=10+10+6+6,33=15+6+6+6,34=10+6\cdot4,35=15+10+10$, so $m\ge36$. Then $m-6\ge30$ is smaller than $m$ and therefore representable; adding one 6¢ stamp represents $m$, a contradiction.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 为什么实数集合 $(0,1)$ 不能直接使用 WOP？
> 2. 最小反例法中，怎样使用“$m$ 最小”？
> 3. 证明 $2^n\ge n+1$ 时，下降到 $m-1$ 还需检查什么？
>
> <!-- bilingual-en:start -->
> 1. Why can WOP not be applied directly to the real interval $(0,1)$?
> 2. In a least-counterexample proof, what exactly does the minimality of $m$ allow you to conclude?
> 3. When proving $2^n\ge n+1$, what must be checked before descending from $m$ to $m-1$?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 它非空但没有最小元素。2. 只能对同一反例集合中所有严格小于 $m$ 的合法对象断言它们不是反例。3. 先排除 $m=0$，确保 $m-1\in\mathbb N$；再验证由 $P(m-1)$ 可推出 $P(m)$。
> <!-- bilingual-en:start -->
> 1. The interval is nonempty but has no least element. 2. Every valid object in the same domain that is strictly smaller than $m$ is not a counterexample. 3. First exclude $m=0$ so that $m-1\in\mathbb N$; then verify that $P(m-1)$ implies $P(m)$.
> <!-- bilingual-en:end -->

**知识链：**反证 → 反例集合 → 最小反例 → 严格下降 → 归纳法的等价思想。
<!-- bilingual-en:start -->
**Knowledge chain:** contradiction → counterexample set → least counterexample → strict descent → induction as an equivalent principle.
<!-- bilingual-en:end -->

---

## Session 4 — Logic and Propositions

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

如何把自然语言规格翻译成可由真值表检查的公式？如何区分“某次为真”“总为真”和“可同时为真”？
<!-- bilingual-en:start -->
How do I translate natural language specifications into formulas that can be checked by truth tables?  How do I distinguish between "once true", "always true" and "can be true at the same time"?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session4.pdf|Session 4 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp4.pdf|CP4]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/0exBzsexUoI.pdf|Propositional Operators]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/eMWG-jTh-GE.pdf|Digital Logic]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/3WDzxt5p8c.pdf|Truth Tables]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_PropositOper.pdf|Operators]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_DigitalLogic.pdf|Digital Logic]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_TruthTables.pdf|Truth Tables]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Implies.pdf|Implies]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_PropositLogic.pdf|Propositional Logic]]

### 4.1 逻辑运算符
<!-- bilingual-en:start -->
*4.1 Logical Operators*
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|命题逻辑]]的基本运算：
<!-- bilingual-en:start -->
Basic operation of [[数学证明方法#命题、量词与否定|propositional logic]]:
<!-- bilingual-en:end -->

| 公式 | 何时为真 | 常用等价式 |
|---|---|---|
| $\neg P$ | $P$ 假 | — |
| $P\land Q$ | 二者都真 | — |
| $P\lor Q$ | 至少一个真 | inclusive OR |
| $P\oplus Q$ | 恰有一个真 | $(P\lor Q)\land\neg(P\land Q)$ |
| $P\Rightarrow Q$ | 除 $P=T,Q=F$ 外 | $\neg P\lor Q$ |
| $P\Leftrightarrow Q$ | 真值相同 | $(P\Rightarrow Q)\land(Q\Rightarrow P)$ |
<!-- bilingual-en:start -->
|Formula|When is true|Common Equivalents|
|—|—|—|
| $\neg P$ | $P$ False | — |
| $P\land Q$ | Both | — |
| $P\lor Q$ | At least one true | inclusive OR |
| $P\oplus Q$ | Exactly one true | $(P\lor Q)\land\neg(P\land Q)$ |
| $P\Rightarrow Q$ | Except for $P=T,Q=F$ | $\neg P\lor Q$ |
| $P\Leftrightarrow Q$ | Same Truth | $(P\Rightarrow Q)\land(Q\Rightarrow P)$ |
<!-- bilingual-en:end -->

自然语言“$P$ only if $Q$”是 $P\Rightarrow Q$；“$P$ if $Q$”是 $Q\Rightarrow P$；“$P$ iff $Q$”才是双向蕴含。
<!-- bilingual-en:start -->
The natural language "$P$ only if $Q$" is $P\Rightarrow Q$, "$P$ if $Q$" is $Q\Rightarrow P$, and "$P$ iff $Q$" is bidirectional.
<!-- bilingual-en:end -->

### 4.2 真值表、有效、可满足与等价
<!-- bilingual-en:start -->
*4.2 Truth Table, Valid, Satisfiable and Equivalent*
<!-- bilingual-en:end -->

有 $n$ 个命题变量时，[[数学证明方法#命题、量词与否定|真值表]]有 $2^n$ 行。两个公式在每个环境（truth assignment）下真值相同，称[[数学证明方法#命题、量词与否定|逻辑等价]]。
<!-- bilingual-en:start -->
With $n$ propositional variables, a [[数学证明方法#命题、量词与否定|truth table]] has $2^n$ rows. Two formulas are [[数学证明方法#命题、量词与否定|logically equivalent]] when they have the same truth value under every truth assignment.
<!-- bilingual-en:end -->

- **valid / tautology：**每个环境都真；
- **satisfiable：**至少一个环境为真；
- **unsatisfiable：**每个环境都假；
- 一组公式 **consistent：**存在一个环境使它们同时为真。
<!-- bilingual-en:start -->
- **valid / tautology:** true under every assignment;
- **satisfiable:** true under at least one assignment;
- **unsatisfiable:** false under every assignment;
- A set of formulas is **consistent** if some assignment makes all of them true simultaneously.
<!-- bilingual-en:end -->

关系是：$P$ valid 当且仅当 $\neg P$ unsatisfiable。要判断推理
<!-- bilingual-en:start -->
The relationship is: $P$ valid if and only if $\neg P$ unsatisfiable.  To judge the reasoning
<!-- bilingual-en:end -->

$$
P_1,\dots,P_k\vdash Q
$$

是否 sound，只需检查公式 $(P_1\land\cdots\land P_k)\Rightarrow Q$ 是否 valid。
<!-- bilingual-en:start -->
Whether sound, simply check that the formula $(P_1\land\cdots\land P_k)\Rightarrow Q$ is valid.
<!-- bilingual-en:end -->

### 4.3 De Morgan 与分配律
<!-- bilingual-en:start -->
*4.3 De Morgan and the Law of Distribution*
<!-- bilingual-en:end -->

$$
\neg(P\land Q)\equiv\neg P\lor\neg Q,
\qquad
\neg(P\lor Q)\equiv\neg P\land\neg Q.
$$

$$
P\lor(Q\land R)\equiv(P\lor Q)\land(P\lor R).
$$

每条都可用真值表证明。代数式“看起来像”并非证明；逻辑等价要求最后一列逐行相同。
<!-- bilingual-en:start -->
Each can be proved by a truth table.  Algebraic "looks" are not proofs; logical equivalence requires that the last column be identical row by row.
<!-- bilingual-en:end -->

### 4.4 数字电路：逻辑不是抽象装饰
<!-- bilingual-en:start -->
*4.4 Digital circuits: logic is not abstract decoration*
<!-- bilingual-en:end -->

用 $T/F$ 表示 $1/0$。一位全加器输入 $a_i,b_i,c_i$，输出
<!-- bilingual-en:start -->
$T/F$ for $1/0$.  One-bit full adder input $a_i,b_i,c_i$, output
<!-- bilingual-en:end -->

$$
s_i=a_i\oplus b_i\oplus c_i,
$$

进位为“至少两个输入为 1”：
<!-- bilingual-en:start -->
Round to At least two inputs are 1:
<!-- bilingual-en:end -->

$$
c_{i+1}=(a_i\land b_i)\lor(a_i\land c_i)\lor(b_i\land c_i).
$$

串行 ripple-carry 的关键路径为 $\Theta(n)$；后续作业的并行结构用预计算降低深度，这正是“公式等价但电路代价不同”的例子。
<!-- bilingual-en:start -->
The critical path for serial ripple-carry is $\Theta(n)$; the parallel structure of subsequent jobs is reduced in depth with precomputation, which is an example of "formula equivalent but different circuit costs".
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（10 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.2_propositional-operators|1.4.2 Propositional Operators]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.5_equivalence-and-truth-table|1.4.5 Equivalence and Truth Table]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.8_soundness-and-validity|1.4.8 Soundness and Validity]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.9_logical-connectives|1.4.9 Logical Connectives]]
>
> **Operators 1–4：**逐个环境代入。$(T,T)$ 时真：AND、OR、$P\lor\neg Q$；$(F,F)$ 时真：$\neg(P\land Q),\neg(P\lor Q),\neg P,\neg Q,P\lor\neg Q$；$(T,F)$ 时真：XOR、OR、$\neg(P\land Q),\neg Q,P\lor\neg Q$；$(F,T)$ 时真：XOR、OR、$\neg(P\land Q),\neg P,\neg P\land Q$。
>
> **Equivalence：**比较 truth tables 的 final column。**Soundness：**当前件全真时后件必真，等价于相应蕴含 valid。**Connectives 1–4：**$P\Rightarrow Q\equiv\neg P\lor Q$；IFF $\equiv(P\land Q)\lor(\neg P\land\neg Q)$；XOR $\equiv(P\land\neg Q)\lor(\neg P\land Q)$；NOR $\equiv\neg P\land\neg Q$。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.2_propositional-operators|1.4.2 Propositional Operators]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.5_equivalence-and-truth-table|1.4.5 Equivalence and Truth Table]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.8_soundness-and-validity|1.4.8 Soundness and Validity]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S04_1.4.9_logical-connectives|1.4.9 Logical Connectives]]
> **Operators 1-4:**Comes environment-by-environment.  $(T,T)$:AND, OR, $P\lor\neg Q$;$(F,F)$:$\neg(P\land Q),\neg(P\lor Q),\neg P,\neg Q,P\lor\neg Q$;$(T,F)$:XOR, OR, $\neg(P\land Q),\neg Q,P\lor\neg Q$;$(F,T)$:XOR, OR, $\neg(P\land Q),\neg P,\neg P\land Q$.
> **Equivalence:**Compare the final column of the truth tables.**Soundness:**When the current piece is true, the subsequent piece must be true, which is equivalent to the corresponding implication of valid.**Connectives 1-4:**$P\Rightarrow Q\equiv\neg P\lor Q$;IFF $\equiv(P\land Q)\lor(\neg P\land\neg Q)$;XOR $\equiv(P\land\neg Q)\lor(\neg P\land Q)$;NOR $\equiv\neg P\land\neg Q$.
> <!-- bilingual-en:end -->

> [!example]- CP4 非官方独立题解（4 道）
> **1. OR 对 AND 分配。** 八行真值表可逐行核对；更紧凑地按 $P$ 分类：若 $P=T$，两式皆 $T$；若 $P=F$，两式都化为 $Q\land R$。
>
> **2. 文件系统规格。** 公式为 $\neg L\Rightarrow Q$、$\neg L\Rightarrow B$、$\neg L\Leftrightarrow N$、$\neg Q\Rightarrow B$、$\neg B$。由 $\neg B$ 与第二式得 $L$；双条件随之要求 $N=F$。第四式在 $B=F$ 下要求 $Q=T$。唯一赋值为 $(L,Q,B,N)=(T,T,F,F)$，逐式均真。
>
> **3. 二进制加法器。** 加一个比特 $b$：$c_0=b$，$s_i=a_i\oplus c_i$，$c_{i+1}=a_i\land c_i$，最终 $c=c_{n+1}$。两数相加时 $s_i=a_i\oplus b_i\oplus c_i$，$c_{i+1}=(a_i\land b_i)\lor(c_i\land(a_i\oplus b_i))$。每位可用两个 XOR、两个 AND、一个 OR，因此 $n+1$ 位共 $2(n+1)$ XOR、$2(n+1)$ AND、$n+1$ OR（若用三项 majority 公式，门计数会依选定实现变化，必须声明实现）。
>
> **4. 数学家与母亲。** 数学陈述旨在表达必要条件：不可导可能还有别的原因，所以只有 $D\Rightarrow C$。母亲的规则在会话语用中同时表达许可与禁止，即把做作业作为看电视的充要条件，故建模为 $H\Leftrightarrow T$ 合理；自然语言的 IF–THEN 需结合意图建模。
> <!-- bilingual-en:start -->
> **1. Distributivity of OR over AND.** The eight-row truth table can be checked directly. More compactly, split on $P$: if $P=T$, both expressions are true; if $P=F$, both reduce to $Q\land R$.
> **2. File system specifications.**Formulas are $\neg L\Rightarrow Q$, $\neg L\Rightarrow B$, $\neg L\Leftrightarrow N$, $\neg Q\Rightarrow B$, $\neg B$.  $L$ from $\neg B$ and second; the dual condition then requires $N=F$.  Formula 4 requires $Q=T$ under $B=F$.  The unique assignment is $(L,Q,B,N)=(T,T,F,F)$, which is verbose.
> **3. Binary adders.**Add a bit $b$:$c_0=b$, $s_i=a_i\oplus c_i$, $c_{i+1}=a_i\land c_i$, and eventually $c=c_{n+1}$.  The sum of the two numbers is $s_i=a_i\oplus b_i\oplus c_i$,$c_{i+1}=(a_i\land b_i)\lor(c_i\land(a_i\oplus b_i))$.  Two XORs, two ANDs, one ORs are available for each, so the $n+1$ bits are $2(n+1)$ XOR, $2(n+1)$ AND, $n+1$ OR (with three majority formulas, the gate count varies by the selected implementation, which must be declared).
> **4. Mathematicians and mothers.** The mathematical statement expresses only a necessary condition: a function may fail to be differentiable for other reasons, so the correct formalisation is merely $D\Rightarrow C$. In ordinary conversation, a mother's rule can convey both permission and prohibition—homework is treated as both necessary and sufficient for watching television—so $H\Leftrightarrow T$ is a reasonable model. Natural-language “if–then” statements must be interpreted in context.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 写出 $P\Rightarrow(Q\land R)$ 的无蕴含形式。
> 2. satisfiable 是否推出 valid？给反例。
> 3. 为什么“前件假”的蕴含定义为真？
>
> <!-- bilingual-en:start -->
> 1. Rewrite $P\Rightarrow(Q\land R)$ without using implication.
> 2. Is valid available in satisfiable?  Give the counterexample.
> 3. Why is an implication with a false premise defined to be true?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. $\neg P\lor(Q\land R)$。2. 不推出；$P$ 在环境 $P=T$ 下可满足，但不是恒真。3. 蕴含只禁止“承诺条件出现而结论失败”的环境；当前件未发生时没有反例。
> <!-- bilingual-en:start -->
> 1. $\neg P\lor(Q\land R)$. 2. No. The proposition $P$ is satisfiable in a valuation with $P=T$, but it is not valid. 3. An implication rules out only valuations in which its premise is true and its conclusion false; a false premise therefore creates no counterexample.
> <!-- bilingual-en:end -->

**知识链：**逻辑运算 → 真值环境 → 等价/有效/可满足 → 规格一致性 → 数字电路。
<!-- bilingual-en:start -->
**Knowledge chain:**Logical operations → Truth environment → Equivalence/Effectiveness/Satisfiability → Specification consistency → Digital circuits.
<!-- bilingual-en:end -->

---

## Problem Set 1 — Sessions 1–4

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps1.pdf|Problem Set 1 PDF]]。以下 4 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps1.pdf|Problem Set 1 PDF]].  The following 4 questions are unofficial and independent.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：证明 $\log_4 6$ 无理
> **已知：**$4,6>1$。**目标：**$\log_4 6\notin\mathbb Q$。
>
> 反设 $\log_4 6=m/n$，其中 $m,n\in\mathbb Z$、$n>0$，且因对数为正可取 $m>0$。由定义
> $$4^{m/n}=6\quad\Longrightarrow\quad4^m=6^n.$$
> 左边的素因子分解为 $2^{2m}$，右边为 $2^n3^n$。唯一素因子分解要求右边的 $3$ 次数也为 $0$，即 $n=0$，与 $n>0$ 矛盾。因此 $\log_4 6$ 无理。$\square$
>
> **检查：**不能只说 $4^m$ 与 $6^n$“显然不同”；矛盾来自素因子 $3$。
> <!-- bilingual-en:start -->
> **Known:**$4,6>1$.**Target:**$\log_4 6\notin\mathbb Q$.
> Inversely, $\log_4 6=m/n$, where $m,n\in\mathbb Z$, $n>0$, and the positive logarithm is $m>0$.  by definition
> $$4^{m/n}=6\quad\Longrightarrow\quad4^m=6^n.$$
> The prime factorization on the left is $2^{2m}$ and on the right is $2^n3^n$.  Unique prime factorization requires that the right $3$ number is also $0$, that is, $n=0$, which is inconsistent with $n>0$.  So $\log_4 6$ is irrational.  $\square$
> **Check:**It is not enough to say that $4^m$ and $6^n$ are "obviously different"; the contradiction comes from the prime factor $3$.
> <!-- bilingual-en:end -->

> [!example]- Problem 2：用 WOP 证明 $n\le3^{n/3}$
> 目标对所有 $n\in\mathbb N$ 成立。令
> $$C=\{n\in\mathbb N:n>3^{n/3}\}.$$
> 反设 $C\ne\varnothing$，取 $m=\min C$。直接检查
> $$0\le1,\qquad1\le3^{1/3},\qquad2\le3^{2/3},\qquad3=3,\qquad4<3^{4/3},$$
> 故 $m\ge5$，于是 $m-3\in\mathbb N$ 且 $m-3<m$。最小性给出
> $$m-3\le3^{(m-3)/3}.$$
> 乘 $3$ 得 $3(m-3)\le3^{m/3}$；而 $m\ge5$ 意味着 $m\le3m-9=3(m-3)$。所以 $m\le3^{m/3}$，与 $m\in C$ 矛盾。故 $C$ 为空。
> <!-- bilingual-en:start -->
> Proof with WOP
> Target is true for all $n\in\mathbb N$.
> $$C=\{n\in\mathbb N:n>3^{n/3}\}.$$
> Optionally, set $C\ne\varnothing$ to $m=\min C$.  direct inspection
> $$0\le1,\qquad1\le3^{1/3},\qquad2\le3^{2/3},\qquad3=3,\qquad4<3^{4/3},$$
> So $m\ge5$, so $m-3\in\mathbb N$ and $m-3<m$.  give out minimally
> $$m-3\le3^{(m-3)/3}.$$
> Multiply $3$ to get $3(m-3)\le3^{m/3}$; $m\ge5$ means $m\le3m-9=3(m-3)$.  So $m\le3^{m/3}$, contradicts $m\in C$.  $C$ is null.
> <!-- bilingual-en:end -->

> [!example]- Problem 3：有效、等价、可满足与一致
> **(a)** 真值表：
>
> | $P$ | $Q$ | $P\Rightarrow Q$ | $Q\Rightarrow P$ | 二者 OR |
> |---|---|---|---|---|
> | T | T | T | T | T |
> | T | F | F | T | T |
> | F | T | T | F | T |
> | F | F | T | T | T |
>
> 最后一列恒真，故公式 valid。
>
> **(b)** 取
> $$R=(P\land Q)\lor(\neg P\land\neg Q).$$
> 它在且仅在 $P,Q$ 真值相同时为真。因此 $R$ valid 当且仅当 $P,Q$ 在每个环境中同值，即二者等价。
>
> **(c)** $P$ valid 意味着不存在使 $P$ 假的环境；这正等于不存在使 $\neg P$ 真的环境，即 $\neg P$ unsatisfiable。
>
> **(d)** $P_1,\dots,P_k$ 不一致，当且仅当 $P_1\land\cdots\land P_k$ 不可满足，当且仅当
> $$S=\neg(P_1\land\cdots\land P_k)$$
> valid。
> <!-- bilingual-en:start -->
> **(a)**Truth table:
> | $P$ | $Q$ | $P\Rightarrow Q$ | $Q\Rightarrow P$ | Both OR |
> |—|—|—|—|—|
> | T | T | T | T | T |
> | T | F | F | T | T |
> | F | T | T | F | T |
> | F | F | T | T | T |
> The last line is constant, so the formula valid.
> **(b)**take
> $$R=(P\land Q)\lor(\neg P\land\neg Q).$$
> It is true, and only if the $P,Q$ truth value is the same.  Thus, $R$ valid is equivalent if and only if $P,Q$ is the same value in each environment.
> **(c)**$P$ valid means that there is no environment that falsifies $P$; this is exactly the same as there is no environment that makes $\neg P$ true, $\neg P$ unsatisfiable.
> **(d)**$P_1,\dots,P_k$ is inconsistent if and only if $P_1\land\cdots\land P_k$ is unsatisfiable if and only if
> $$S=\neg(P_1\land\cdots\land P_k)$$
> valid.
> <!-- bilingual-en:end -->

> [!example]- Problem 4：并行 half-adder
> **(a)** 一位 add1 输入 $a_0$，输出 $a_0+1$ 的两位表示：
> $$p_0=\neg a_0,\qquad c=a_0.$$
>
> **(b)** 若控制位 $b=0$，half-adder 应原样输出 $a_i$；若 $b=1$，应选择 add1 预计算的 $p_i$：
> $$o_i=(\neg b\land a_i)\lor(b\land p_i).$$
>
> **(c)** 低半段溢出 $c^{(1)}=1$ 时才需要给高半段加一；整个双倍模块溢出又要求高半段预计算也溢出，因此
> $$c=c^{(1)}\land c^{(2)}.$$
>
> **(d)** 对 $n+1\le i\le2n+1$，令 $j=i-(n+1)$：
> $$p_i=(\neg c^{(1)}\land a_i)\lor(c^{(1)}\land r_j).$$
>
> **(e)** 设 $D(N)$ 为 $N$ 位 add1 任一输出的最大逻辑深度。倍增构造只在子模块后增加常数个 AND/OR/NOT 层，故
> $$D(2N)\le D(N)+O(1),$$
> 解得 $D(N)=O(\log N)$。ripple-carry 的最高位必须等待 $N$ 次进位传播，深度 $\Theta(N)$；所以并行版本在电路深度上指数改进。门总数仍为多项式，题目比较的是**最长依赖链**而非物理时间的无条件保证。
> <!-- bilingual-en:start -->
> **(a)**One add1 input $a_0$, output $a_0+1$ two-bit representation:
> $$p_0=\neg a_0,\qquad c=a_0.$$
> **(b)**If the control bit is $b=0$, half-adder should output $a_i$ as it is; If $b=1$, add1 should choose $p_i$:
> $$o_i=(\neg b\land a_i)\lor(b\land p_i).$$
> **(c)**The lower half overflows $c^{(1)}=1$ and the upper half needs to be added one; the entire double module overflows and requires the upper half to be pre-calculated, so
> $$c=c^{(1)}\land c^{(2)}.$$
> **(d)**For $n+1\le i\le2n+1$, let $j=i-(n+1)$:
> $$p_i=(\neg c^{(1)}\land a_i)\lor(c^{(1)}\land r_j).$$
> **(e)**Let $D(N)$ be the maximum logical depth for any output of the $N$ bit add1.  The multiplication construct only adds a constant number of AND/OR/NOT layers after the sub-module, so
> $$D(2N)\le D(N)+O(1),$$
> $D(N)=O(\log N)$.  The highest bit of the ripple-carry must wait for the $N$ carry to propagate at a depth of $\Theta(N)$; so the parallel version improves exponentially in circuit depth.  The total number of gates is still polynomial and the topic is compared**the longest dependent chain**rather than the unconditional guarantee of physical time.
> <!-- bilingual-en:end -->

---

## Session 5 — Quantifiers and Predicate Logic

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

同一句“每个程序都有一个输入”与“有一个输入适用于每个程序”为什么完全不同？本节把自由变量、论域和量词写进公式。
<!-- bilingual-en:start -->
Why is the same sentence "one input per program" different from "one input per program"?  This section writes free variables, discourse and quantifiers into formulas.
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session5.pdf|Session 5 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp5.pdf|CP5]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/UroprmQHTLc.pdf|Predicate Logic I]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/T1AtlGrCoU8.pdf|Predicate Logic II]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L5uBeAGJV1k.pdf|Predicate Logic III]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate1.pdf|Predicate I]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate2.pdf|Predicate II]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate3.pdf|Predicate III]]

### 5.1 论域、自由变量与绑定变量
<!-- bilingual-en:start -->
*5.1 Discourse Domain, Free Variable and Binding Variable*
<!-- bilingual-en:end -->

[[01_Proofs#Session 5 — Quantifiers and Predicate Logic|谓词逻辑（predicate logic）]]允许对象变量、谓词和量词。$P(x,y)$ 中未被量词控制的变量是自由变量；在 $\forall x\exists y\,P(x,y)$ 中二者都被绑定，整个式子才是命题。
<!-- bilingual-en:start -->
[[01_Proofs#Session 5 — Quantifiers and Predicate Logic|predicate logic]] allows object variables, predicates, and quantifiers.  In $P(x,y)$, the variables which are not controlled by quantifiers are free variables; in $\forall x\exists y\,P(x,y)$, both are bound, and the whole formula is proposition.
<!-- bilingual-en:end -->

论域必须明确。同一公式 $\exists x(x^2=2)$ 在 $\mathbb Q$ 中为假，在 $\mathbb R$ 中为真。若论域为空，$\forall xP(x)$ 真（没有反例），$\exists xP(x)$ 假；本课程通常默认非空论域，但证明时仍应知道这一边界。
<!-- bilingual-en:start -->
The context must be clear.  The same formula $\exists x(x^2=2)$ is false in $\mathbb Q$ and true in $\mathbb R$.  If the domain is empty, $\forall xP(x)$ is true (there is no counterexample) and $\exists xP(x)$ is false; this course usually defaults to non-empty domains, but this boundary should still be known at the time of proof.
<!-- bilingual-en:end -->

### 5.2 量词的语义与顺序
<!-- bilingual-en:start -->
*5.2 The Semantics and Order of Quantifiers*
<!-- bilingual-en:end -->

$$
\forall x\,P(x):\text{论域内每个 }x\text{ 都满足 }P,
$$

$$
\exists x\,P(x):\text{至少存在一个见证 }x\text{ 满足 }P.
$$

同类量词可交换：$\forall x\forall yP\equiv\forall y\forall xP$，$\exists x\exists yP\equiv\exists y\exists xP$。异类通常不可交换：
<!-- bilingual-en:start -->
Similar quantifiers are interchangeable: $\forall x\forall yP\equiv\forall y\forall xP$, $\exists x\exists yP\equiv\exists y\exists xP$.  Heterogeneous is usually not interchangeable:
<!-- bilingual-en:end -->

$$
\forall x\exists y\;(y>x)
$$

允许 $y$ 随 $x$ 改变；$\exists y\forall x(y>x)$ 要求一个统一的最大上界。在 $\mathbb N$ 中前者真、后者假。
<!-- bilingual-en:start -->
Allow $y$ to change with $x$; $\exists y\forall x(y>x)$ requires a uniform maximum upper bound.  In $\mathbb N$, the former is true and the latter is false.
<!-- bilingual-en:end -->

### 5.3 否定穿过量词
<!-- bilingual-en:start -->
*5.3 Negative Through Quantifier*
<!-- bilingual-en:end -->

De Morgan 规律推广为
<!-- bilingual-en:start -->
De Morgan's law is generalized as
<!-- bilingual-en:end -->

$$
\neg\forall xP(x)\equiv\exists x\neg P(x),
\qquad
\neg\exists xP(x)\equiv\forall x\neg P(x).
$$

因此，“并非每个算法都终止”是“存在一个算法不终止”，不是“每个算法都不终止”。否定多层量词时逐层交换 $\forall\leftrightarrow\exists$ 并否定最内层谓词。
<!-- bilingual-en:start -->
Therefore, "not every algorithm terminates" is "there is an algorithm that does not terminate", not "every algorithm does not terminate".  When you negate a multilevel quantifier, you exchange $\forall\leftrightarrow\exists$ layer by layer and negate the innermost predicate.
<!-- bilingual-en:end -->

### 5.4 可满足性与见证
<!-- bilingual-en:start -->
*5.4 Satisfiability and Witness*
<!-- bilingual-en:end -->

谓词 $P(x)$ satisfiable 意味着 $\exists xP(x)$。证明存在命题最直接的方法是给出见证并验证；否定存在命题则必须对任意候选排除。
<!-- bilingual-en:start -->
The predicate $P(x)$ satisfiable means $\exists xP(x)$.  The most direct way to prove the existence of a proposition is to give a witness and verify it; negation of the existence of a proposition requires the exclusion of any candidate.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（7 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.3_satisfiability|1.5.3 Satisfiability]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.5_name-that-predicate|1.5.5 Name That Predicate!]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.6_quantifiers|1.5.6 Quantifiers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.7_propositions-with-quantifiers|1.5.7 Propositions with Quantifiers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.8_predicate-logic|1.5.8 Predicate Logic]]
>
> 1. 保证 $P$ 可满足的式子：$\forall xP(x)$、$\exists xP(x)$、$\neg\forall x\neg P(x)$、$\neg\exists x\neg P(x)$（最后一项等价 $\forall xP$，默认论域非空）。
>
> 2. $\neg\exists x\exists yQ(x,y)$ 等价 $\forall x\forall y\neg Q(x,y)$。
>
> 3. 有效式：$\exists x\exists yP\Rightarrow\exists y\exists xP$、$\exists x\forall yR\Rightarrow\forall y\exists xR$、$\neg\exists xS\Leftrightarrow\forall x\neg S$。
>
> 4. 论域 $\mathbb N$ 时，题给五式中为真的是 $\forall x\exists y(2x-y=0)$ 与 $\forall x[x<10\Rightarrow\forall y(y<x\Rightarrow y<9)]$。
>
> 5. 论域 $\mathbb Z$ 时，除上两式外，$\forall x\exists y[y>x\land\exists z(y+z=100)]$ 也真；$x-2y=0$ 仍被奇数反例否定。
>
> 6. 论域 $\mathbb R$ 时，为真的是 $\forall x\exists y(2x-y=0)$、$\forall x\exists y(x-2y=0)$ 与上一条的“找更大 $y$”公式；课程反馈给 $x=9.5$ 作为整界式的反例。
>
> 7. “每个整数都有某个素因子”的量词顺序是 $\forall i\exists p$，不能要求同一素数除尽所有整数。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.3_satisfiability|1.5.3 Satisfiability]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.5_name-that-predicate|1.5.5 Name That Predicate!]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.6_quantifiers|1.5.6 Quantifiers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.7_propositions-with-quantifiers|1.5.7 Propositions with Quantifiers]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S05_1.5.8_predicate-logic|1.5.8 Predicate Logic]]
> 1. Formulae that guarantee $P$ to be satisfied: $\forall xP(x)$, $\exists xP(x)$, $\neg\forall x\neg P(x)$, $\neg\exists x\neg P(x)$ (last equivalent to $\forall xP$, default domain is non-empty).
> 2. $\neg\exists x\exists yQ(x,y)$ is equivalent to $\forall x\forall y\neg Q(x,y)$.
> 3. Valid formulas: $\exists x\exists yP\Rightarrow\exists y\exists xP$, $\exists x\forall yR\Rightarrow\forall y\exists xR$ and $\neg\exists xS\Leftrightarrow\forall x\neg S$.
> 4. When the universe is $\mathbb N$, the title given to the five is really $\forall x\exists y(2x-y=0)$ and $\forall x[x<10\Rightarrow\forall y(y<x\Rightarrow y<9)]$.
> 5. When the universe is $\mathbb Z$, $\forall x\exists y[y>x\land\exists z(y+z=100)]$ is true except for the above two; $x-2y=0$ is still negated by the odd counter-example.
> 6. When discussing $\mathbb R$, it is really $\forall x\exists y(2x-y=0)$, $\forall x\exists y(x-2y=0)$ and the "find a bigger $y$" formula of the previous article; the course feedback to $x=9.5$ as a counterexample to the whole bound.
> 7. The quantifier order for “each integer has a prime factor” is $\forall i\exists p$; it does not require one prime to divide every integer.
> <!-- bilingual-en:end -->

> [!example]- CP5 非官方独立题解（5 道）
> **1. 论域真值表。**
>
> | 公式 | $\mathbb N$ | $\mathbb Z$ | $\mathbb Q$ | $\mathbb R$ | $\mathbb C$ |
> |---|:---:|:---:|:---:|:---:|:---:|
> | $\exists x:x^2=2$ | F | F | F | T | T |
> | $\forall x\exists y:x^2=y$ | T | T | T | T | T |
> | $\forall y\exists x:x^2=y$ | F | F | F | F | T |
> | $\forall x\ne0\exists y:xy=1$ | F | F | T | T | T |
> | $\exists x,y:x+2y=2\land2x+4y=5$ | F | F | F | F | F |
>
> 最后一行因第二个左边恒为第一个左边的两倍，应等于 $4$ 而非 $5$。
>
> **2. 二进制串翻译。** 令 `PREFIX`、`SUBSTRING`、`NO-1S` 如题定义：
> (a) $\exists y(x=yyy)$；
> (b) $\exists y(x=yy\land\mathrm{NO\mbox{-}1S}(y))$；
> (c) $\neg(\mathrm{SUBSTRING}(0,x)\land\mathrm{SUBSTRING}(1,x))$；
> (d) $x=10\lor\exists y(x=1y1\land\mathrm{NO\mbox{-}1S}(y))$；
> (e) 若 $x$ 是全 0 串，显然 $x$ 是 $0x$ 前缀。反之若 $x$ 含首个 1，比较 $x$ 与 $0x$ 的相应位置会迫使这个 1 等于前一位；向左反复传播最终迫使首位既为 0 又为 1，矛盾。因此 `PREFIX(x,0x)` 恰刻画全 0 串（含空串）。
>
> **3. 至多给两位其他同学发邮件。**
> $$
> \exists x\exists y\exists z\forall w\,
> ((E(x,w)\land w\ne x)\Rightarrow(w=y\lor w=z)).
> $$
> $y,z$ 可以相同，也可等于 $x$，所以该式同时覆盖 0、1、2 位收件人。
>
> **4. 交换量词。** 第一式无效：在论域 $\{1,2\}$ 令 $P(x,y)\equiv(x\ne y)$，则每个 $x$ 都有不同的 $y$，但不存在一个 $y$ 与所有 $x$ 不同。第二式有效：若有固定 $y_0$ 使所有 $x$ 满足 $P(x,y_0)$，则对每个 $x$ 取同一个 $y_0$ 即可。
>
> **5. Cabal（补充题）。** (a) 至少三人；(b) Siggi 与 Annie 不能同时在；(c) Elizabeth 若在则所有人都在；(d) Annie 在则 Siggi 在；(e) Ben 或 Albert 在则 Tom 不在；(f) Ben 或 Siggi 在则 Adam 不在。由 (b)(c) 得 Elizabeth 不在；由 (b)(d) 得 Annie 不在。若 Ben 不在，进一步分 Siggi 在/不在都会与“至少三人”和 (e)(f) 冲突，故 Ben 在。于是 (e)(f) 给出 Tom、Adam 不在；要达到三人，只能 Albert、Siggi 都在。唯一 cabal 是 $\{\text{Ben, Albert, Siggi}\}$。
> <!-- bilingual-en:start -->
> **1. Truth table of discourse.**
> | Formula | $\mathbb N$ | $\mathbb Z$ | $\mathbb Q$ | $\mathbb R$ | $\mathbb C$ |
> |—|:—:|:—:|:—:|:—:|:—:|
> | $\exists x:x^2=2$ | F | F | F | T | T |
> | $\forall x\exists y:x^2=y$ | T | T | T | T | T |
> | $\forall y\exists x:x^2=y$ | F | F | F | F | T |
> | $\forall x\ne0\exists y:xy=1$ | F | F | T | T | T |
> | $\exists x,y:x+2y=2\land2x+4y=5$ | F | F | F | F | F |
> The last row, since the second left side is always twice the first left side, should equal $4$ instead of $5$.
> **2. Translation of binary strings.**Let `PREFIX`, `SUBSTRING`, `NO-1S` as defined:
> (a)$\exists y(x=yyy)$;
> (b)$\exists y(x=yy\land\mathrm{NO\mbox{-}1S}(y))$;
> (c)$\neg(\mathrm{SUBSTRING}(0,x)\land\mathrm{SUBSTRING}(1,x))$;
> (d)$x=10\lor\exists y(x=1y1\land\mathrm{NO\mbox{-}1S}(y))$;
> (e) If $x$ is an all-zero string, then clearly $x$ is a prefix of $0x$. Conversely, if $x$ contains a first 1, comparing the corresponding positions in $x$ and $0x$ propagates equality one place to the left until that first 1 is forced to equal 0, a contradiction. Thus, `PREFIX(x,0x)` describes exactly the all-zero strings, including the empty string.
> **3. Email no more than two other classmates.**
> The $y,z$ can be the same or equal to $x$, so it covers 0, 1, and 2 recipients at the same time.
> **4. Exchange quantifiers.**Formula one is invalid: if $\{1,2\}$ is $P(x,y)\equiv(x\ne y)$ in the domain, then each $x$ has a different $y$, but there is no $y$ that is different from all $x$.  The second is valid: if there is a fixed $y_0$ to make all $x$ satisfy $P(x,y_0)$, then take the same $y_0$ for each $x$.
> **5.  Cabal.**(a) at least three people; (b) Siggi and Annie cannot be present at the same time; (c) Elizabeth is present, all people are present; (d) Annie is present, Siggi is present; (e) Ben or Albert is not present, Tom is not present; (f) Ben or Siggi is not present, Adam is not present.  Elizabeth is absent from (b)(c) and Annie is absent from (b)(d).  If Ben is absent, further differentiation as to whether Siggi is present or absent will result in a conflict with "at least three" and (e)(f), so Ben is present.  So (e) (f) gives Tom and Adam are not there; to reach three, only Albert and Siggi are there.  The only cabal is $\{\text{Ben, Albert, Siggi}\}$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 否定 $\forall x\exists yP(x,y)$。
> 2. 在 $\mathbb Z$ 中，$\forall x\exists y(x=2y)$ 真吗？
> 3. 证明 $\exists xP(x)$ 时，一个例子何时足够？
>
> <!-- bilingual-en:start -->
> 1. Negative $\forall x\exists yP(x,y)$.
> 2. In $\mathbb Z$, is $\forall x\exists y(x=2y)$ really?
> 3. When is an example sufficient to prove $\exists xP(x)$?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. $\exists x\forall y\neg P(x,y)$。2. 假；$x=1$ 没有整数见证。3. 当例子确属声明的论域并逐项满足谓词时足够。
> <!-- bilingual-en:start -->
> 1. $\exists x\forall y\neg P(x,y)$.  2. False; $x=1$ has no integer witness.  3. It is sufficient when the example is indeed the domain of the declaration and the predicate is satisfied item by item.
> <!-- bilingual-en:end -->

**知识链：**谓词 → 论域 → 量词顺序 → 量词否定 → 见证/反模型。
<!-- bilingual-en:start -->
**knowledge chain:**predicate→domain→order of quantifier→negation of quantifier→witness/inverse model.
<!-- bilingual-en:end -->

---

## Session 6 — Sets

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

集合运算为什么与逻辑运算一一对应？怎样用逐元素双向蕴含证明集合相等？
<!-- bilingual-en:start -->
Why does a set operation correspond to a logical operation one to one?  How to use element-by-element bidirectional implication to prove the equality of sets?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session6.pdf|Session 6 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp6.pdf|CP6]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/KZ7jjLTQ9r4.pdf|Set Definitions]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Mhip1rljvRo.pdf|Set Operations]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_SetsDefinit.pdf|Definitions]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_SetsOperation.pdf|Operations]]

### 6.1 基本定义
<!-- bilingual-en:start -->
*6.1 Basic definitions*
<!-- bilingual-en:end -->

[[01_Proofs#Session 6 — Sets|集合（set）]]由其元素决定；$x\in A$ 表示成员关系。外延性（extensionality）给出
<!-- bilingual-en:start -->
[[01_Proofs#Session 6 — Sets|set]] is determined by its elements; $x\in A$ represents the membership.  extensionality
<!-- bilingual-en:end -->

$$
A=B\quad\Longleftrightarrow\quad\forall x(x\in A\Leftrightarrow x\in B).
$$

子集 $A\subseteq B$ 意味着 $\forall x(x\in A\Rightarrow x\in B)$；真子集还要求 $A\ne B$。空集 $\varnothing$ 是每个集合的子集，因为不存在使蕴含失败的 $x\in\varnothing$。
<!-- bilingual-en:start -->
Subset $A\subseteq B$ means $\forall x(x\in A\Rightarrow x\in B)$; true subsets also require $A\ne B$.  An empty set $\varnothing$ is a subset of each set because there is no $x\in\varnothing$ that fails the entailment.
<!-- bilingual-en:end -->

幂集 $\operatorname{pow}(A)=\{S:S\subseteq A\}$。若有限集 $|A|=n$，每个元素独立选择“进/不进”子集，所以 $|\operatorname{pow}(A)|=2^n$。
<!-- bilingual-en:start -->
The power set $\operatorname{pow}(A)=\{S:S\subseteq A\}$.  If the finite set $|A|=n$, each element independently selects the "in/out" subset, so $|\operatorname{pow}(A)|=2^n$.
<!-- bilingual-en:end -->

### 6.2 运算与逻辑对应
<!-- bilingual-en:start -->
*6.2 Operations and Logical Correspondence*
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-set-operations.png|900]]

读图：把每个阴影区域逐点翻译成成员谓词，交、并、差分别对应 AND、OR 与 AND-NOT。
<!-- bilingual-en:start -->
Read Figure: Each shadow area is translated into a member predicate point by point, and the intersection, union, and difference correspond to AND, OR, and AND-NOT.
<!-- bilingual-en:end -->

| 集合条件 | 元素谓词 | 逻辑运算 |
|---|---|---|
| $x\in A\cap B$ | $x\in A$ 且 $x\in B$ | AND |
| $x\in A\cup B$ | 至少属于一个 | OR |
| $x\in\overline A$ | $x\notin A$ | NOT |
| $x\in A-B$ | $x\in A$ 且 $x\notin B$ | $P\land\neg Q$ |
| $x\in A\triangle B$ | 恰属于一个 | XOR |
<!-- bilingual-en:start -->
|Collection Condition|Element Predicate|Logical Operation|
|—|—|—|
| $x\in A\cap B$ | $x\in A$ and $x\in B$ | AND |
| $x\in A\cup B$ | belongs to at least one | OR |
| $x\in\overline A$ | $x\notin A$ | NOT |
| $x\in A-B$ | $x\in A$ and $x\notin B$ | $P\land\neg Q$ |
| $x\in A\triangle B$ | of exactly one | XOR |
<!-- bilingual-en:end -->

所以逻辑等价可逐字搬到集合恒等式。例如
<!-- bilingual-en:start -->
So the logical equivalence can be transposed verbatim to set identities.  For example
<!-- bilingual-en:end -->

$$
A-(B\cup C)=(A-B)\cap(A-C)
$$

来自 $P\land\neg(Q\lor R)\equiv(P\land\neg Q)\land(P\land\neg R)$。
<!-- bilingual-en:start -->
From $P\land\neg(Q\lor R)\equiv(P\land\neg Q)\land(P\land\neg R)$.
<!-- bilingual-en:end -->

### 6.3 集合相等证明模板
<!-- bilingual-en:start -->
*6.3 Templates for proving set equality*
<!-- bilingual-en:end -->

任选任意 $x$，建立 IFF 链：
<!-- bilingual-en:start -->
Choose any $x$ to create an IFF chain:
<!-- bilingual-en:end -->

$$
x\in L
\Leftrightarrow\cdots
\Leftrightarrow x\in R.
$$

由于 $x$ 任意，外延性给出 $L=R$。另一种写法是分别证 $L\subseteq R$ 与 $R\subseteq L$；不能只画 Venn 图替代一般证明。
<!-- bilingual-en:start -->
Since $x$ is arbitrary, epitaxy gives $L=R$.  The other is to prove $L\subseteq R$ and $R\subseteq L$ separately; you cannot just draw Venn diagrams instead of general proofs.
<!-- bilingual-en:end -->

### 6.4 有序对为什么不能用 $\{a,b\}$
<!-- bilingual-en:start -->
*6.4 Ordered Pairs Why Not $\{a,b\}$*
<!-- bilingual-en:end -->

$\{a,b\}=\{b,a\}$，会丢失顺序。课程使用
<!-- bilingual-en:start -->
$\{a,b\}=\{b,a\}$, the order is lost.  Course Usage
<!-- bilingual-en:end -->

$$
\operatorname{pair}(a,b)=\{a,\{a,b\}\}.
$$

在所采用的集合论基础下，这个编码能恢复首、次项。关键不是必须记住某一种编码，而是知道“序列”可由集合构造，并且编码要保持顺序与重复项。
<!-- bilingual-en:start -->
Based on the adopted set theory, this code can recover the first and second terms.  The key is not to have to remember a particular kind of code, but to know that the "sequence" can be constructed by a set and that the code is to be kept in order and repetition.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercise（1 prompt）
> **本地逐题入口：** [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S06_1.6.3_difference|1.6.3 Difference]]
>
> $A-B$ 唯一正确的谓词形式是 $\{x:(x\in A)\land(x\notin B)\}$；它不是交换运算，且必须同时保留“在 $A$ 中”和“不在 $B$ 中”。
> <!-- bilingual-en:start -->
> **Local entry:**[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S06_1.6.3_difference|1.6.3 Difference]]
> The only correct predicate form for $A-B$ is $\{x:(x\in A)\land(x\notin B)\}$; it is not a commutation and must both be "in $A$" and "not in $B$".
> <!-- bilingual-en:end -->

> [!example]- CP6 非官方独立题解（4 道）
> **1. 分解恒等式。** 命题式应为 $(P\land\neg Q)\lor(P\land Q)\equiv P$。任取 $x$：
> $$x\in(A-B)\cup(A\cap B)\Leftrightarrow[(x\in A\land x\notin B)\lor(x\in A\land x\in B)]\Leftrightarrow x\in A.$$
>
> **2. 只用 $\in$ 的集合论公式。**
> (a) $x=\varnothing:\ \forall z(z\notin x)$；
> (b) $x=\{y,z\}:\ \forall w(w\in x\Leftrightarrow(w=y\lor w=z))$；
> (c) $x\subseteq y:\ \forall w(w\in x\Rightarrow w\in y)$；
> (d) $x=y\cup z:\ \forall w(w\in x\Leftrightarrow(w\in y\lor w\in z))$；
> (e) $x=y-z$：把 OR 换成 $w\in y\land w\notin z$；
> (f) $x=\operatorname{pow}(y):\ \forall w(w\in x\Leftrightarrow w\subseteq y)$；
> (g) $x=\bigcup y:\ \forall w(w\in x\Leftrightarrow\exists z(z\in y\land w\in z))$。
>
> **3. 有序对编码。** $\{a,b\}$ 对调不变；$\{a,\{b\}\}$ 也可能混淆，例如不同层次的集合恰好相等。对 $\{a,\{a,b\}\}$，若 $a=b$ 则编码为 $\{a\}$，仍恢复 $(a,a)$；若 $a\ne b$，编码中唯一同时作为外层元素并出现在二元内层集合的对象是 $a$，余者为 $b$。基础公理排除成员环造成的伪识别。
>
> **4. 四元素 subset take-away。** 设 $A=\{1,2,3,4\}$，按先手第一步大小分类。
>
> - 若先手取 1 元集（或 3 元集），后手取其补集。此后合法选择全部落在剩余的 3 元底集内，局面等价于题干已经验证的三元素初始局面，轮到先手且为必败态。
> - 若先手取 2 元集，不妨为 $\{1,2\}$，后手取补集 $\{3,4\}$。此后合法集合恰为四个单点和四条“交叉二元集”$\{1,3\},\{1,4\},\{2,3\},\{2,4\}$。后手按固定配对回应：$\{1\}\leftrightarrow\{2\}$、$\{3\}\leftrightarrow\{4\}$、$\{1,3\}\leftrightarrow\{2,4\}$、$\{1,4\}\leftrightarrow\{2,3\}$。每个配对的两个集合在此前选择下同时合法；取走一个后，另一个仍不包含任何旧选择。故先手每走一步后手都有回应，最终先手先无合法步。
>
> 三种第一步规模已经穷尽所有非空真子集，所以四元素情形第二位玩家必胜。
> <!-- bilingual-en:start -->
> **1. Decomposition of the identities.**The proposition should be $(P\land\neg Q)\lor(P\land Q)\equiv P$.  Anything $x$:
> $$x\in(A-B)\cup(A\cap B)\Leftrightarrow[(x\in A\land x\notin B)\lor(x\in A\land x\in B)]\Leftrightarrow x\in A.$$
> **2. We only use $\in$'s set-theoretic formulas.**
> (a)$x=\varnothing:\ \forall z(z\notin x)$;
> (b)$x=\{y,z\}:\ \forall w(w\in x\Leftrightarrow(w=y\lor w=z))$;
> (c)$x\subseteq y:\ \forall w(w\in x\Rightarrow w\in y)$;
> (d)$x=y\cup z:\ \forall w(w\in x\Leftrightarrow(w\in y\lor w\in z))$;
> (e) $x=y-z$: replace OR with $w\in y\land w\notin z$;
> (f)$x=\operatorname{pow}(y):\ \forall w(w\in x\Leftrightarrow w\subseteq y)$;
> (g)$x=\bigcup y:\ \forall w(w\in x\Leftrightarrow\exists z(z\in y\land w\in z))$.
> **3. Ordered pair coding.**The $\{a,b\}$ tune is unchanged; $\{a,\{b\}\}$ can also be confused, for example, when sets of different levels are exactly equal.  For $\{a,\{a,b\}\}$, if $a=b$ is encoded as $\{a\}$, $(a,a)$ is still restored; if $a\ne b$, the only object encoded as an outer element that appears in the set of binary inner layers at the same time is $a$, the other is $b$.  Fundamental axioms exclude the false identities caused by member loops.
> **4. The four-element subset take-away.**Set $A=\{1,2,3,4\}$ to sort by first-hand first-step size.
> - If you take a 1-tuple set (or 3-tuple set) first, then take its complement.  After that, the legal choice is all in the remaining three-element bottom set. The situation is equivalent to the initial situation of the three elements which has been verified by the question stem.
> - If you take a 2-tuple set first, you might want to take $\{1,2\}$, then take the complement set $\{3,4\}$.  Thereafter the legal set is just four single points and four "cross binary sets" $\{1,3\},\{1,4\},\{2,3\},\{2,4\}$.  The hindhand pushes a fixed pair response: $\{1\}\leftrightarrow\{2\}$, $\{3\}\leftrightarrow\{4\}$, $\{1,3\}\leftrightarrow\{2,4\}$, $\{1,4\}\leftrightarrow\{2,3\}$.  Both sets of each pair are legal under the previous selection; after one is taken, the other still does not contain any old selections.  Therefore, the first hand has a response after every step, and ultimately the first hand has no legal step.
> Three first-step sizes have exhausted all non-empty true subsets, so the four-element scenario will be won by the second player.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. $\varnothing\in A$ 与 $\varnothing\subseteq A$ 有何差别？
> 2. 证明 $A\cap(B\cup C)=(A\cap B)\cup(A\cap C)$。
> 3. $|\operatorname{pow}(\varnothing)|$ 是多少？
>
> <!-- bilingual-en:start -->
> 1. What is the difference between $\varnothing\in A$ and $\varnothing\subseteq A$?
> 2. Proof $A\cap(B\cup C)=(A\cap B)\cup(A\cap C)$.
> 3. What is $|\operatorname{pow}(\varnothing)|$?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 前者要求空集本身是 $A$ 的元素，不总成立；后者对每个 $A$ 都成立。2. 对任意 $x$，把成员条件化成 $P\land(Q\lor R)\equiv(P\land Q)\lor(P\land R)$。3. $1$，唯一子集是 $\varnothing$。
> <!-- bilingual-en:start -->
> 1. The former requires that the empty set itself be an element of the $A$, and does not always hold; the latter holds for each $A$.  2. For any $x$, the membership is reduced to a $P\land(Q\lor R)\equiv(P\land Q)\lor(P\land R)$.  3.  $1$, the only subset is $\varnothing$.
> <!-- bilingual-en:end -->

**知识链：**成员谓词 → 逻辑运算 → 集合运算 → 外延性 → 数据类型编码。
<!-- bilingual-en:start -->
**Knowledge Chain:**Member Predicates→Logical Operations→Set Operations→Extensibility→Data Type Encoding.
<!-- bilingual-en:end -->

---

## Problem Set 2 — Sessions 5–6

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps2.pdf|Problem Set 2 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps2.pdf|Problem Set 2 PDF]].  The following 3 questions are unofficial and independent.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：用集合论表达 pair
> 定义
> $$\mathrm{Members}(p,a,b):\Longleftrightarrow\forall x(x\in p\Leftrightarrow(x=a\lor x=b)).$$
> 于是
> $$\mathrm{Pair}(p,a,b):\Longleftrightarrow\exists q(\mathrm{Members}(q,a,b)\land\mathrm{Members}(p,a,q)).$$
> 这里 $q=\{a,b\}$，而 $p=\{a,q\}=\{a,\{a,b\}\}$。最后
> $$\mathrm{Second}(p,b):\Longleftrightarrow\exists a\,\mathrm{Pair}(p,a,b).$$
> 每个辅助式都只是成员谓词、量词与逻辑运算的缩写，所以最终确为集合论公式。
> <!-- bilingual-en:start -->
> by Set Theory
> defined
> $$\mathrm{Members}(p,a,b):\Longleftrightarrow\forall x(x\in p\Leftrightarrow(x=a\lor x=b)).$$
> therefore
> $$\mathrm{Pair}(p,a,b):\Longleftrightarrow\exists q(\mathrm{Members}(q,a,b)\land\mathrm{Members}(p,a,q)).$$
> Here's $q=\{a,b\}$, and $p=\{a,q\}=\{a,\{a,b\}\}$.  last
> $$\mathrm{Second}(p,b):\Longleftrightarrow\exists a\,\mathrm{Pair}(p,a,b).$$
> Each auxiliary is simply an abbreviation for member predicates, quantifiers and logical operations, so it is indeed a set theory formula.
> <!-- bilingual-en:end -->

> [!example]- Problem 2：集合版 De Morgan
> 补集相对于同一全集。任选 $x$：
> $$
> \begin{aligned}
> x\in\overline{A\cap B}
> &\Leftrightarrow x\notin A\cap B\\
> &\Leftrightarrow\neg(x\in A\land x\in B)\\
> &\Leftrightarrow(x\notin A\lor x\notin B)\\
> &\Leftrightarrow(x\in\overline A\lor x\in\overline B)\\
> &\Leftrightarrow x\in\overline A\cup\overline B.
> \end{aligned}
> $$
> 由外延性，$\overline{A\cap B}=\overline A\cup\overline B$。
> <!-- bilingual-en:start -->
> The complement is relative to the same set.  Optional $x$:
> By epitaxy, $\overline{A\cap B}=\overline A\cup\overline B$.
> <!-- bilingual-en:end -->

> [!example]- Problem 3：star-free languages
> 记 $B=\{0,1\}^*$，补集均相对 $B$。
>
> **(a)** $R\cap S=\overline{\overline R\cup\overline S}$，故 c-d 语言对交封闭。
>
> **(b)** 以 0 开头且以 1 结尾：$0B\cap B1$。
>
> **(c)** 全 0 串语言：$0^*=\overline{B1B}$，因为右边恰是不含 1 的所有串，包括空串。
>
> **(d)**
> $$
> (01)^*=\{\lambda\}\cup\left(0B\cap B1\cap\overline{B00B\cup B11B}\right).
> $$
> 非空串从 0 开始、以 1 结束且没有相邻相同位，当且仅当它交替为若干个 01。
>
> **(e)** $(00)^*\cap0^*$ 含无限多个偶长度全 0 串；其补集与 $0^*$ 的交又含无限多个奇长度全 0 串，所以它和补集都不是 0-finite，即不 boring。
>
> **(f)** 若 $R,S$ 都 0-finite，则并仍有限；若至少一个（如 $R$）是 co-0-finite，则 $\overline{R\cup S}\cap0^*\subseteq\overline R\cap0^*$ 有限。故并保持 boring。
>
> **(g)** 若 $R,S$ 都 0-finite，能拼成全 0 串的候选来自两个有限集合，故乘积 0-finite。若某一语言没有全 0 串，乘积也没有。剩余情形中，至少一方（设 $R$）包含除有限个外的全部全 0 串，另一方含某个固定 $0^k$；于是充分大的 $0^n$ 都可写成 $0^{n-k}0^k\in R\cdot S$。所以乘积的补集只漏有限个全 0 串，仍 boring。
>
> **(h)** 有限语言均 0-finite；(f)(g) 证明 union/concatenation 保持 boring，complement 只交换“自身 0-finite”与“补集 0-finite”。结构归纳得全部 c-d 语言 boring。因 (e) 的 $(00)^*$ 不 boring，它不是 c-d。
> <!-- bilingual-en:start -->
> Note that $B=\{0,1\}^*$, the complement is relative to $B$.
> **(a)**$R\cap S=\overline{\overline R\cup\overline S}$, so the c-d language is closed.
> **(b)**$0B\cap B1$ starts with 0 and ends with 1.
> **(c)**All 0-string language: $0^*=\overline{B1B}$, because on the right is exactly all strings without a 1, including empty strings.
> **(d)**
> A non-empty string starts with 0, ends with 1, and has no adjacent identical bits if and only if it alternates with several 01.
> **(e)**$(00)^*\cap0^*$ contains infinite number of even-length all-0 strings; its complement intersects $0^*$ with infinite number of odd-length all-0 strings, so neither it nor the complement is 0-finite, that is, not boring.
> **(f)**If both $R,S$ are 0-finite, then they are still finite; if at least one (e.g., $R$) is co-0-finite, then $\overline{R\cup S}\cap0^*\subseteq\overline R\cap0^*$ is finite.  So keep boring.
> **(g)**If $R,S$ are all 0-finite, the candidates that can be spelled into all 0 strings are from two finite sets, so the product 0-finite.  If a language does not have an all-zero string, the product does not.  In the remaining case, at least one of the $R$ includes all but a limited number of all zeros, and the other includes a fixed $0^k$; therefore, a sufficiently large $0^n$ can be written as $0^{n-k}0^k\in R\cdot S$.  So the product's tween only leaves out a limited number of all-0 strings, still boring.
> **(h)**Finite languages are all 0-finite;(f)(g) Prove that union/concatenation keeps boring and complement only exchanges "self 0-finite" and "complement 0-finite".  The structure is summarized into the whole c-d language boring.  Because the $(00)^*$ of (e) is not boring, it is not c-d.
> <!-- bilingual-en:end -->

---

## Session 7 — Binary Relations

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

函数只是关系的一种。怎样用“每个点有几条入边/出边”统一定义 total、function、injection、surjection 和 bijection？
<!-- bilingual-en:start -->
Functions are just a kind of relation.  How do I define total, function, injection, surjection, and bijection uniformly with "How many in/out edges per point"?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session7.pdf|Session 7 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp7.pdf|CP7]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/FkfsmwAtDdY.pdf|Relations]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/gFD1Lp6zK3w.pdf|Relational Mappings]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/fpy5Hsz5t6E.pdf|Finite Cardinality]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Relations.pdf|Relations]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RelationalMap.pdf|Mappings]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_FiniteCardi.pdf|Finite Cardinality]]

### 7.1 关系、像与逆像
<!-- bilingual-en:start -->
*7.1 Relationship, Image and Inversion*
<!-- bilingual-en:end -->

[[01_Proofs#Session 7 — Binary Relations|二元关系（binary relation）]] $R$ from $A$ to $B$ 是笛卡尔积 $A\times B$ 的子集。写 $aRb$ 表示 $(a,b)\in R$。$A$ 是声明的 domain，$B$ 是 codomain；真正有出边的元素集合与真正被命中的 range 不一定等于二者。
<!-- bilingual-en:start -->
A [[01_Proofs#Session 7 — Binary Relations|binary relation]] $R$ from $A$ to $B$ is a subset of the Cartesian product $A\times B$. Write $aRb$ for $(a,b)\in R$. Here $A$ is the declared domain and $B$ is the codomain; the subset of $A$ that actually has outgoing edges and the range actually reached in $B$ need not equal them.
<!-- bilingual-en:end -->

对 $X\subseteq A$，像为
<!-- bilingual-en:start -->
For $X\subseteq A$, like
<!-- bilingual-en:end -->

$$
R(X)=\{b\in B:\exists a\in X,\ aRb\}.
$$

逆关系 $R^{-1}\subseteq B\times A$ 由 $bR^{-1}a\Leftrightarrow aRb$ 定义；图上就是把所有箭头反向。
<!-- bilingual-en:start -->
The inverse $R^{-1}\subseteq B\times A$ is defined by $bR^{-1}a\Leftrightarrow aRb$; on the diagram, all arrows are reversed.
<!-- bilingual-en:end -->

### 7.2 用箭头计数定义映射性质
<!-- bilingual-en:start -->
*7.2 Defining Mapping Properties with Arrow Counts*
<!-- bilingual-en:end -->

课程把各性质用于一般关系，而不预设“函数必 total”：
<!-- bilingual-en:start -->
The course uses each property in a general relation, without presetting the "function must total":
<!-- bilingual-en:end -->

| 性质 | 每个 domain 点出边 | 每个 codomain 点入边 |
|---|---:|---:|
| function | $\le1$ | 无限制 |
| total | $\ge1$ | 无限制 |
| injection | 无限制 | $\le1$ |
| surjection | 无限制 | $\ge1$ |
| bijection | $=1$ | $=1$ |
<!-- bilingual-en:start -->
| Nature | Edges per domain point | Edges per codomain point |
|—|—:|—:|
| function | $\le1$ | Unlimited|
| total | $\ge1$ | Unlimited|
| injection | Unlimited | $\le1$ |
| surjection | Unlimited | $\ge1$ |
| bijection | $=1$ | $=1$ |
<!-- bilingual-en:end -->

因此常规数学中的“函数 $f:A\to B$”对应这里的 **total function**。inverse 会交换 in/out：
<!-- bilingual-en:start -->
So "function $f:A\to B$" in general mathematics corresponds to**total function**.  inverse swaps in/out:
<!-- bilingual-en:end -->

$$
R\text{ total}\Leftrightarrow R^{-1}\text{ surjective},
\qquad
R\text{ function}\Leftrightarrow R^{-1}\text{ injective}.
$$

### 7.3 有限基数与 Mapping Rule
<!-- bilingual-en:start -->
*7.3 Finite Cardinality and Mapping Rule*
<!-- bilingual-en:end -->

[[01_Proofs#Session 11 — Infinite Sets|基数（cardinality）]] $|A|$ 表示有限集元素数。若存在 $A$ 到 $B$ 的 bijection，则 $|A|=|B|$。若存在 total injection，则 $|A|\le|B|$；若存在 total surjective function，则 $|A|\ge|B|$。
<!-- bilingual-en:start -->
[[01_Proofs#Session 11 — Infinite Sets|cardinality]] $|A|$ indicates the number of finite set elements.  If there is a bijection from $A$ to $B$, $|A|=|B|$.  $|A|\le|B|$ if total injection is present, $|A|\ge|B|$ if total surjective function is present.
<!-- bilingual-en:end -->

证明的箭头计数本质：total injection 从每个 $a$ 发出一条、不同 $a$ 不能落到同一 $b$，所以 $B$ 至少有 $|A|$ 个落点。
<!-- bilingual-en:start -->
The arrowhead count nature of the proof: total injection emits one $a$ from each $a$ and different $b$ cannot fall to the same $B$, so there are at least $|A|$ drop points.
<!-- bilingual-en:end -->

### 7.4 组合的性质不能凭感觉传递
<!-- bilingual-en:start -->
*7.4 The nature of the combination cannot be transmitted by sensation*
<!-- bilingual-en:end -->

若 $h=f\circ g$：
<!-- bilingual-en:start -->
If $h=f\circ g$:
<!-- bilingual-en:end -->

- $h$ surjective 强迫 $f$ surjective，因为 $h(A)\subseteq f(B)$；
- $h$ injective 强迫 $g$ injective，因为 $g$ 若碰撞，$h$ 必碰撞；
- 反方向通常只对 $g(A)$ 范围内有效，不能约束未被 $g$ 命中的 $B$ 元素。
<!-- bilingual-en:start -->
- $h$ surjective forces $f$ surjective because $h(A)\subseteq f(B)$;
- $h$ injective forces $g$ injective, because if $g$ collides, $h$ must collide;
- The reverse direction is usually only valid in the $g(A)$ range and cannot constrain $B$ elements that are not hit by a $g$.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（22 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.2_range-of-a-relation|1.7.2 Range of a Relation]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.4_total-injection|1.7.4 Total Injection]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.6_a-inj-b|1.7.6 A inj B]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.7_total-relations|1.7.7 Total Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.8_surjective-relations|1.7.8 Surjective Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.9_inverse-relations|1.7.9 Inverse Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.10_in-sur-and-bijections|1.7.10 In-, Sur-, and Bijections]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.11_mapping-lemma-sizes-of-domains-and-codomains|1.7.11 Mapping Lemma: Sizes of Domains and Codomains]]
>
> 1. range 不大于 codomain。2–3. total injection 是 $\ge1$ out、$\le1$ in。4. total 等价 $R^{-1}(B)=A$ 与 $\ge1$ out。5. surjective 等价 $R(A)=B$。6–10. $|R(A)|\le|B|$；surjective 时 $|A|\ge|B|$、$|R(A)|=|B|$；injective 时 $|R(A)|=|A|$；bijective 时 $|A|=|B|$。
>
> 11–14. inverse 对应：function ↔ injection，surjection ↔ total，injection ↔ function，bijection ↔ bijection。15–21（论域/陪域均 $\mathbb R$）：$x+2$ B，$2x$ B，$x^2$ N，$x^3$ B，$\sin x$ N，$x\sin x$ S-not-B，$e^x$ I-not-B。22. 对有限集，$A\operatorname{inj}B$ 等价：存在 total injective relation $A\to B$、surjective function $B\to A$、$A$ 与 $B$ 某子集间 bijection，以及 $|A|\le|B|$。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.2_range-of-a-relation|1.7.2 Range of a Relation]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.4_total-injection|1.7.4 Total Injection]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.6_a-inj-b|1.7.6 A inj B]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.7_total-relations|1.7.7 Total Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.8_surjective-relations|1.7.8 Surjective Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.9_inverse-relations|1.7.9 Inverse Relations]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.10_in-sur-and-bijections|1.7.10 In-, Sur-, and Bijections]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S07_1.7.11_mapping-lemma-sizes-of-domains-and-codomains|1.7.11 Mapping Lemma: Sizes of Domains and Codomains]]
> 1. range is no greater than codomain.  2-3. total injection is $\ge1$ out, $\le1$ in.  4. total is equivalent to $R^{-1}(B)=A$ and $\ge1$ out.  5. surjective is equivalent to $R(A)=B$.  6-10.  $|R(A)|\le|B|$; $|A|\ge|B|$, $|R(A)|=|B|$ for surjective; $|R(A)|=|A|$ for injective; $|A|=|B|$ for bijective.
> 11-14. inverse mappings: function ↔ injection, surjection ↔ total, injection ↔ function, bijection ↔ bijection.  15-21 (domain/codomain $\mathbb R$): $x+2$ B, $2x$ B, $x^2$ N, $x^3$ B, $\sin x$ N, $x\sin x$ S-not-B, $e^x$ I-not-B.  22. For finite sets, $A\operatorname{inj}B$ is equivalent: there is bijection between a subset of total injective relation $A\to B$, surjective function $B\to A$, $A$ and $B$, and $|A|\le|B|$.
> <!-- bilingual-en:end -->

> [!example]- CP7 非官方独立题解（5 道）
> **1. 逆关系表。** total ↔ inverse surjection；function ↔ inverse injection；surjection ↔ inverse total；injection ↔ inverse function；bijection ↔ inverse bijection。每项都只是“出边”反向后变成“入边”。
>
> **2. $|A\times B|=mn$。** 设 $A=\{a_0,\dots,a_{n-1}\}$、$B=\{b_0,\dots,b_{m-1}\}$。定义
> $$f(a_i,b_j)=im+j.$$
> 值域是 $0,\dots,mn-1$。若 $im+j=i'm+j'$，因 $0\le j,j'<m$，Euclidean division 的商余数唯一，故 $i=i',j=j'$；反之任一 $k<mn$ 唯一写为 $k=im+j$。所以 $f$ bijective。
>
> **3. 最强比较符号。** (a) $|f(A)|\le|B|$；(b) surjective 时 $|A|\ge|B|$；(c) $|f(A)|=|B|$；(d) injective 时 $|f(A)|=|A|$；(e) bijective 时 $|A|=|B|$。
>
> **4. 广义 Mapping Rule。** 设从 $X$ 发出的箭头数为 $E$。因 $R$ 是 function，每个 $x\in X$ 至多一条出边，故 $E\le|X|$；每个 $R(X)$ 中元素至少接到一条来自 $X$ 的边，故 $|R(X)|\le E$。所以 $|R(X)|\le|X|$。
>
> **5. inj/surj 的组合。** (a) 合成两个 surjective relations：对每个 $c$，先取 $bRc$，再取 $aSb$，得到 $a$ 到 $c$，故仍 surjective。(b) $A\operatorname{surj}B$ 当且仅当把箭头反向得到 $B\operatorname{inj}A$。(c) 对 (a) 取 inverse，得到 injection 的传递性。(d) 从 total injective relation 每个 $a$ 选一条出边；删边不会破坏 $\le1$ in，得到 total injective function。有限情形可逐点选择；任意集合的统一陈述涉及 choice。
> <!-- bilingual-en:start -->
> **1. Table of inverse relations.**total ↔ inverse surjection;function ↔ inverse injection;surjection ↔ inverse total;injection ↔ inverse function;bijection ↔ inverse bijection.  Each becomes Inset only after the Out edge is flipped.
> **2.  $|A\times B|=mn$.**$A=\{a_0,\dots,a_{n-1}\}$, $B=\{b_0,\dots,b_{m-1}\}$.  defined
> $$f(a_i,b_j)=im+j.$$
> The value domain is $0,\dots,mn-1$.  If $im+j=i'm+j'$, because $0\le j,j'<m$, the quotient remainder of Euclidean division is unique, $i=i',j=j'$; conversely, any $k<mn$ is unique as $k=im+j$.  So, $f$ bijective.
> **3. Strongest comparison sign.**(a)$|f(A)|\le|B|$;(b)$|A|\ge|B|$ at surjective;(c)$|f(A)|=|B|$;(d)$|f(A)|=|A|$ at injective;(e)$|A|=|B|$ at bijective.
> **4. Generic Mapping Rule.**Sets the number of arrows from $X$ to $E$.  Since $R$ is function and each $x\in X$ has at most one outside edge, $E\le|X|$; elements in each $R(X)$ are attached to at least one edge from $X$, $|R(X)|\le E$.  So, $|R(X)|\le|X|$.
> **5. Combining injection and surjection properties.** (a) The composition of two surjective relations is surjective: for each $c$, choose $b$ with $bRc$, then choose $a$ with $aSb$, producing a relation from $a$ to $c$. (b) $A\operatorname{surj}B$ if and only if reversing all arrows gives $B\operatorname{inj}A$. (c) Taking inverses in part (a) gives transitivity of injection. (d) From a total injective relation, choose one outgoing edge for each $a$. Removing other edges preserves the at-most-one-incoming-edge property and produces a total injective function. For finite sets the choices can be made one at a time; the uniform statement for arbitrary sets invokes a choice principle.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 一个 relation 可以同时是 function 但不 total 吗？
> 2. $f:A\to B$ 为 injection 时，为何不必 $f(A)=B$？
> 3. 若 $A$ 空、$B$ 非空，空关系有哪些性质？
>
> <!-- bilingual-en:start -->
> 1. Can a relation be function but not total?
> 2. Why is $f(A)=B$ not required when $f:A\to B$ is injection?
> 3. If $A$ null and $B$ non-null, what is the nature of the null relationship?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 可以，例如 $A=\{1,2\},R=\{(1,a)\}$。2. injection 只限制碰撞，不要求覆盖。3. 它是 function 和 injection（条件真空成立），不是 total，也不是 surjection。
> <!-- bilingual-en:start -->
> 1. Yes, for example, $A=\{1,2\},R=\{(1,a)\}$.  2. injection only restricts the collision and does not require coverage.  3. It is function and injection (conditional vacuum holds), not total, not surjection.
> <!-- bilingual-en:end -->

**知识链：**笛卡尔积 → relation → 箭头约束 → inverse/composition → finite cardinality。
<!-- bilingual-en:start -->
**Knowledge Chain:**Cartesian Product→relation→Arrow Constraint→inverse/composition→finite cardinality.
<!-- bilingual-en:end -->

---

## Session 8 — Induction

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

怎样证明无限多个整数命题而只写有限证明？归纳假设为什么必须足够强、基例为什么必须足够多？
<!-- bilingual-en:start -->
How to prove infinitely many integer propositions and write only finite proofs?  Why do generalization assumptions have to be strong enough, and why do basis examples have to be strong enough?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session8.pdf|Session 8 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp8.pdf|CP8]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/XnV8GAuAqJM.pdf|Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/D3E5CKebKuQ.pdf|Bogus Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TUueMeRooBk.pdf|Strong Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/K8ZfzNN1miQ.pdf|WOP vs Induction]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Induction.pdf|Induction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BogusInductn.pdf|Bogus]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StrongInduct.pdf|Strong]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_WOPvsInductn.pdf|WOP comparison]]

### 8.1 普通归纳
<!-- bilingual-en:start -->
*8.1 General induction*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|普通归纳原理]]：若
<!-- bilingual-en:start -->
[[数学证明方法#归纳法|general induction principle]]: If
<!-- bilingual-en:end -->

$$
P(0),\qquad\forall n\in\mathbb N\,[P(n)\Rightarrow P(n+1)],
$$

则 $\forall n\in\mathbb N,P(n)$。
<!-- bilingual-en:start -->
$\forall n\in\mathbb N,P(n)$.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-induction-dominoes.png|900]]

读图：基例推倒第一块骨牌，归纳步只保证任意一块倒下会带动下一块；两项缺一都不能覆盖整条序列。
<!-- bilingual-en:start -->
Reading: The base case pushes down the first domino, and the induction step only ensures that any one piece falls down and drives the next piece; neither of the two items can cover the entire sequence.
<!-- bilingual-en:end -->

证明流程必须分开：
<!-- bilingual-en:start -->
The proof must keep these stages separate:
<!-- bilingual-en:end -->

1. 定义 $P(n)$；
2. base case：直接证 $P(0)$（或题目的起点）；
3. induction step：固定任意 $n$，**假设** $P(n)$，证明 $P(n+1)$；
4. 调用归纳原理。
<!-- bilingual-en:start -->
1. Define $P(n)$.
2. Base case: prove $P(0)$, or the appropriate initial case.
3. Inductive step: fix an arbitrary $n$, assume $P(n)$, and prove $P(n+1)$.
4. Invoke the induction principle.
<!-- bilingual-en:end -->

归纳假设不是把目标当已知：它只在“证明相邻传递规则”的局部步骤中暂时成立。
<!-- bilingual-en:start -->
The induction hypothesis does not assume the whole theorem; it is a temporary assumption used only to prove the one-step implication.
<!-- bilingual-en:end -->

### 8.2 强归纳与普通归纳等价
<!-- bilingual-en:start -->
*8.2 Strong Induction and General Induction Equivalence*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|强归纳]]的归纳步允许假设所有更小情形：
<!-- bilingual-en:start -->
[[数学证明方法#归纳法|strong induction]]'s induction step allows for all the smaller scenarios:
<!-- bilingual-en:end -->

$$
\left(\forall k<n\,P(k)\right)\Rightarrow P(n).
$$

它并不更“有证明力”。令 $Q(n)=\forall k\le nP(k)$，对 $Q$ 做普通归纳即可模拟强归纳。反过来，普通归纳只是强归纳中只使用 $P(n)$ 的特例。
<!-- bilingual-en:start -->
It is not more "proving".  Let $Q(n)=\forall k\le nP(k)$ do the general induction to $Q$ and then simulate the strong induction.  Conversely, general induction is a special case of strong induction where only $P(n)$ is used.
<!-- bilingual-en:end -->

### 8.3 基例的数量由递推跨度决定
<!-- bilingual-en:start -->
*8.3 The number of base cases is determined by the recurrence span*
<!-- bilingual-en:end -->

若归纳步是 $P(n)\Rightarrow P(n+3)$，只证明 $P(0)$ 只能覆盖 $0,3,6,\dots$。要覆盖所有足够大的整数，必须为模 $3$ 的三个余数类各提供一个起点，例如 $P(5),P(6),P(7)$。
<!-- bilingual-en:start -->
If the induction step is $P(n)\Rightarrow P(n+3)$, it is proved that $P(0)$ can only cover $0,3,6,\dots$.  To override all sufficiently large integers, you must provide a starting point for each of the three remainder classes modulo $3$, for example, $P(5),P(6),P(7)$.
<!-- bilingual-en:end -->

### 8.4 “所有马同色”漏洞
<!-- bilingual-en:start -->
*8.4 "All Horses Same Color" Vulnerability*
<!-- bilingual-en:end -->

伪证把 $n+1$ 匹马分成前 $n$ 与后 $n$，声称两组各同色且因重叠而颜色一致。由 $n=1$ 推 $n=2$ 时两组没有共同马，所以“重叠传递颜色”第一次失效。真正的审查目标是第一处无依据的推断，不是只说结论荒谬。
<!-- bilingual-en:start -->
The bogus proof divides the $n+1$ horses into the first $n$ and the last $n$, claims that each group has one colour, and then uses their overlap to conclude that the colours agree. In the step from $n=1$ to $n=2$, however, the two groups have no horse in common, so the overlap argument fails for the first time. The point is to locate this first unsupported inference, not merely to observe that the conclusion is absurd.
<!-- bilingual-en:end -->

### 8.5 与 WOP 的关系
<!-- bilingual-en:start -->
*8.5 Relationship with WOP*
<!-- bilingual-en:end -->

若归纳结论失败，WOP 可取最小反例；它之前所有整数都成立，归纳步会推出最小反例也成立。反向也可用归纳证明非空自然数集合有最小元。因此三种方法在课程范围内等价，但表达便利不同。
<!-- bilingual-en:start -->
If the conclusion of induction fails, the WOP can obtain the smallest counterexample; all the integers before it hold, and the step of induction can also obtain the smallest counterexample.  Inversion can also be used to prove that the set of non-empty natural numbers has a minimum element.  Therefore, the three methods are equivalent within the curriculum, but the expression is convenient.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（13 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.3_same-colored-horses|1.8.3 Same Colored Horses]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.5_unstacking-game-score|1.8.5 Unstacking Game Score]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.7_strong-vs-ordinary-induction-vs-wop|1.8.7 Strong vs Ordinary Induction vs WOP (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.8_induction-by-n-3|1.8.8 Induction by n+3]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.9_induction-rules|1.8.9 Induction Rules]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.10_postage-by-induction|1.8.10 Postage by Induction]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.11_a-bogus-induction|1.8.11 A Bogus Induction]]
>
> 1. Horses 伪证在 $P(1)\Rightarrow P(2)$ 处失败。2. unstacking 总分与策略无关且为 $n(n-1)/2$。3–4. ordinary/strong/WOP 等价；强归纳转普通归纳用 $Q(n)=\forall k\le nP(k)$。5–6. $P(n)\Rightarrow P(n+3)$ 从 $P(5)$ 得 $5,8,11,\dots$；覆盖所有 $n\ge5$ 需三个不同模 3 余数的基例，如 $5,6,7$。7–11. 五条规则依次是 Strong、Ordinary、WOP、None（漏掉 $0\to1$）、Strong（$m=0$ 时前件真空成立提供基例）。12. 邮资题三种方法均可，但 strong/WOP 更自然。13. Fibonacci 伪强归纳漏证 $P(1)$，最后调用归纳原理无效。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.3_same-colored-horses|1.8.3 Same Colored Horses]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.5_unstacking-game-score|1.8.5 Unstacking Game Score]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.7_strong-vs-ordinary-induction-vs-wop|1.8.7 Strong vs Ordinary Induction vs WOP (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.8_induction-by-n-3|1.8.8 Induction by n+3]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.9_induction-rules|1.8.9 Induction Rules]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.10_postage-by-induction|1.8.10 Postage by Induction]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S08_1.8.11_a-bogus-induction|1.8.11 A Bogus Induction]]
> 1. The all-horses-the-same-colour proof fails at $P(1)\Rightarrow P(2)$. 2. The total unstacking score is independent of strategy and equals $n(n-1)/2$. 3–4. Ordinary induction, strong induction, and WOP are equivalent; convert strong induction to ordinary induction by defining $Q(n)=\forall k\le nP(k)$. 5–6. From $P(5)$ and $P(n)\Rightarrow P(n+3)$ we obtain only $5,8,11,\dots$; covering all $n\ge5$ requires base cases in all three residue classes modulo $3$, such as $5,6,7$. 7–11. The five rules are Strong, Ordinary, WOP, None (the step $0\to1$ is missing), and Strong (when $m=0$, the vacuous hypothesis supplies the base case). 12. The postage problem can be proved by all three methods, though strong induction or WOP is more natural. 13. The bogus strong-induction proof for Fibonacci numbers omits $P(1)$, so its final invocation of induction is invalid.
> <!-- bilingual-en:end -->

> [!example]- CP8 非官方独立题解（4 道）
> **1. 证明 $\sum_{k=1}^n1/k^2<2-1/n$（$n>1$）。** 基例 $n=2$：$5/4<3/2$。假设对 $n$ 成立，则
> $$\sum_{k=1}^{n+1}\frac1{k^2}<2-\frac1n+\frac1{(n+1)^2}.$$
> 因 $1/(n+1)^2<1/[n(n+1)]=1/n-1/(n+1)$，右边 $<2-1/(n+1)$，完成归纳。
>
> **2. L-tromino 铺砖。** (a) 对 $n$ 归纳。$2^0\times2^0$ 只有角落 statue，无需砖。把 $2^{n+1}$ 大院分成四个 $2^n$ 象限；含 statue 的象限由归纳假设铺好，在中心用一块 L 砖占据其他三个象限各自的中心角，随后这三象限也分别成为“角落缺一格”的 $2^n$ 问题，均可铺好。(b) statue 在中心时，把它看成某一象限的中心角，在另三中心格放 L 砖，四个象限都调用 (a)。
>
> **3. 3¢/7¢ 邮资。** 用强归纳。$12=4\cdot3$、$13=2\cdot3+7$、$14=2\cdot7$。对 $N\ge15$，$N-3\ge12$，由强归纳假设可表示，再加一张 3¢。因此所有 $N\ge12$ 可表示；换成题目记号就是所有 $S(n)$ 成立。
>
> **4. 素数整除乘积的伪证。** 首次无依据处发生在由 $n=1$ 推 $n+1=2$：看到 $p\mid x_1x_2$ 后，论证又对“两项乘积”调用强归纳，但 $P(2)$ 正是当前要证，不能由只含 $P(1)$ 的假设得到。修复：把 Euclid lemma（两因子情形）作为独立基例/已知定理，再从 $n\ge2$ 做归纳。
> <!-- bilingual-en:start -->
> **1. Proof $\sum_{k=1}^n1/k^2<2-1/n$ ($n>1$).**Base case $n=2$:$5/4<3/2$.  Assuming it is true for $n$, then
> $$\sum_{k=1}^{n+1}\frac1{k^2}<2-\frac1n+\frac1{(n+1)^2}.$$
> $1/(n+1)^2<1/[n(n+1)]=1/n-1/(n+1)$, right $<2-1/(n+1)$, summed up.
> **2.  L-tromino tile.**(a) An induction into $n$.  $2^0\times2^0$ statue corners, no tiles.  The $2^{n+1}$ compound is divided into four $2^n$ quadrants; the quadrants containing statue are laid by inductive hypothesis, and a piece of L-bricks occupies the central corners of the other three quadrants in the center, and then these three quadrants become "one lattice missing in the corner" $2^n$ problems respectively, and all the quadrants can be laid.  (b) When statue is in the center, consider it as the central angle of a quadrant, and place the L-bricks in the other three centers, and all four quadrants call (a).
> **3. 3 Zim/7 Zinc.**Summarize strong.  $12=4\cdot3$,$13=2\cdot3+7$,$14=2\cdot7$.  For $N\ge15$, $N-3\ge12$, it can be represented by a strong inductive hypothesis plus a 3.  Therefore, all $N\ge12$ are expressible; replacing the title mark with the title mark means that all $S(n)$ are true.
> **4. Falsification of products of prime integers and divisions.**First unsupported occurrence in the $n+1=2$ by $n=1$: After seeing $p\mid x_1x_2$, the argument calls a strong induction for the "two products", but $P(2)$ is the current proof and cannot be derived from the assumption that only $P(1)$ exists.  Repair: Take Euclid lemma (two-factor case) as an independent basis/known theorem, and then generalize from $n\ge2$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. $P(0)$ 与 $P(n)\Rightarrow P(n+2)$ 能推出哪些 $P$？
> 2. 强归纳中能否假设 $P(n)$ 来证明 $P(n)$？
> 3. 递推使用 $P(n-2)$ 时，至少要检查哪些基例？
>
> <!-- bilingual-en:start -->
> 1. What $P$ will be available for $P(0)$ and $P(n)\Rightarrow P(n+2)$?
> 2. In strong induction, can we suppose $P(n)$ to prove $P(n)$?
> 3. At least which base cases should be checked when using $P(n-2)$ recursively?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 只能推出偶数指标。2. 不能；假设只包括严格更小指标。3. 通常需连续两个起始指标，使每条递推链都能启动。
> <!-- bilingual-en:start -->
> 1. Only even indicators can be introduced.  2. No; the assumption is that only strictly smaller indicators are included.  3. Typically, two consecutive starting indicators are required so that each recursive chain can be activated.
> <!-- bilingual-en:end -->

**知识链：**WOP 最小反例 ↔ 普通归纳 ↔ 强归纳 → 足够强的假设 → 递归对象。
<!-- bilingual-en:start -->
**Knowledge chain:**WOP minimal counterexample↔general induction↔strong induction→strong enough hypothesis→recursive object.
<!-- bilingual-en:end -->

---

## Problem Set 3 — Sessions 7–8

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps3.pdf|Problem Set 3 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps3.pdf|Problem Set 3 PDF]].  The following 3 questions are unofficial and independent.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：Binet 公式
> 设
> $$p=\frac{1+\sqrt5}{2},\qquad q=\frac{1-\sqrt5}{2}.$$
> 二者满足 $x^2=x+1$。用强归纳证明 $F_n=(p^n-q^n)/\sqrt5$。
>
> $n=0$ 时右边 $0$；$n=1$ 时 $(p-q)/\sqrt5=1$。对 $n\ge2$，由递推和归纳假设：
> $$
> \begin{aligned}
> F_n&=F_{n-1}+F_{n-2}\\
> &=\frac{p^{n-1}+p^{n-2}-q^{n-1}-q^{n-2}}{\sqrt5}\\
> &=\frac{p^{n-2}(p+1)-q^{n-2}(q+1)}{\sqrt5}\\
> &=\frac{p^n-q^n}{\sqrt5}.
> \end{aligned}
> $$
> 最后一步用了 $p+1=p^2,q+1=q^2$。
> <!-- bilingual-en:start -->
> if
> $$p=\frac{1+\sqrt5}{2},\qquad q=\frac{1-\sqrt5}{2}.$$
> Both meet $x^2=x+1$.  $F_n=(p^n-q^n)/\sqrt5$ is proved by strong induction.
> $0$ for $n=0$, $(p-q)/\sqrt5=1$ for $n=1$.  For $n\ge2$, it is assumed by recursion and induction:
> $p+1=p^2,q+1=q^2$.
> <!-- bilingual-en:end -->

> [!example]- Problem 2：Block Stacking
> 单栈高度 $k$ 的势能为 $\phi(k)=k(k-1)/2$。一次把 $a+b$ 拆成 $a,b$：
> $$
> \phi(a+b)-\phi(a)-\phi(b)
> =\frac{(a+b)(a+b-1)-a(a-1)-b(b-1)}2=ab,
> $$
> 恰等于该步得分。因此任意动作序列的总分是势能下降的望远镜和：
> $$\mathrm{score}=\Phi(A)-\Phi(B).$$
> 游戏结束时全为单块，$\Phi(B)=0$；初始为一栈 $n$，故总分恒为 $n(n-1)/2$。不存在更优策略：所有合法完整策略都并列最优。
> <!-- bilingual-en:start -->
> The potential energy of $k$ is $\phi(k)=k(k-1)/2$.  Break the $a+b$ into a $a,b$ at a time:
> That's exactly the point of the step.  Therefore, the total score of the arbitrary action sequence is the telescope with the descending potential energy and:
> $$\mathrm{score}=\Phi(A)-\Phi(B).$$
> At the end of the game, the game is all single, $\Phi(B)=0$; initially a stack of $n$, so the total score is $n(n-1)/2$.  There is no better policy: all legitimate full policies are optimized side by side.
> <!-- bilingual-en:end -->

> [!example]- Problem 3：函数复合
> 设 $h=f\circ g$。
>
> **(a) 真。** 任取 $c\in C$，$h$ surjective 给出 $a$ 使 $f(g(a))=c$；令 $b=g(a)$，故 $f$ 命中 $c$。
>
> **(b) 假。** $A=\{1\},B=\{1,2\},C=\{c\}$，令 $g(1)=1$、$f(1)=f(2)=c$。$h$ 满射但 $g$ 未命中 2。
>
> **(c) 假。** 同一反例中，单元素域上的 $h$ 单射，而 $f(1)=f(2)$ 非单射。
>
> **(d) 真。** 若 $g(a)=g(a')$，因 $f$ total，二者均可代入并得 $h(a)=f(g(a))=f(g(a'))=h(a')$；$h$ injective 给出 $a=a'$，所以 $g$ injective。
> <!-- bilingual-en:start -->
> Set $h=f\circ g$.
> **(a) True.**If $c\in C$, $h$ surjective gives $a$ to make $f(g(a))=c$; $b=g(a)$ to make $f$ hit $c$.
> **(b) False.**$A=\{1\},B=\{1,2\},C=\{c\}$, $g(1)=1$, $f(1)=f(2)=c$.  $h$ shot full, but $g$ missed 2.
> **(c) False.**In the same counterexample, the $h$ is monochromatic and the $f(1)=f(2)$ is non-monochromatic.
> **(d) True.**If $g(a)=g(a')$, because $f$ total, both can be substituted and merged into $h(a)=f(g(a))=f(g(a'))=h(a')$; $h$ injective gives $a=a'$, so $g$ injective.
> <!-- bilingual-en:end -->

---

## Midterm 1 — Sessions 1–8

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm1.pdf|Midterm 1 PDF]]。以下 5 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm1.pdf|Midterm 1 PDF]].  The following five questions are unofficial and independent.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：$\sqrt[7]{35}$ 无理
> 反设 $\sqrt[7]{35}=a/b$，其中 $a,b$ 为互素正整数。则
> $$a^7=35b^7=5\cdot7\cdot b^7.$$
> 比较素因子 $5$ 的指数：左边是 $7v_5(a)$，为 $7$ 的倍数；右边是 $1+7v_5(b)$，模 $7$ 余 $1$。唯一素因子分解不允许二者相等，矛盾。因此该数无理。
> <!-- bilingual-en:start -->
> Suppose for contradiction that $\sqrt[7]{35}=a/b$, where $a$ and $b$ are coprime positive integers.
> $$a^7=35b^7=5\cdot7\cdot b^7.$$
> Comparing the exponent of the prime $5$, the left side has exponent $7v_5(a)$, a multiple of $7$, whereas the right side has exponent $1+7v_5(b)$, which is congruent to $1$ modulo $7$. Unique factorisation makes equality impossible. Therefore $\sqrt[7]{35}$ is irrational.
> <!-- bilingual-en:end -->

> [!example]- Problem 2：WOP 排除 $3a^4+9b^4=c^4$
> 假设有正整数解，选取 $a$ 最小的一组。模 $3$ 得 $3\mid c$，写 $c=3c_1$。原式除以 $3$：
> $$a^4+3b^4=27c_1^4,$$
> 模 $3$ 得 $3\mid a$，写 $a=3a_1$；代回并除以 $3$ 得
> $$27a_1^4+b^4=9c_1^4,$$
> 模 $3$ 又得 $3\mid b$，写 $b=3b_1$。所以 $a,b,c$ 都被 3 整除。代回原式并约去 $3^4$，$(a_1,b_1,c_1)$ 仍是同方程的正整数解，但 $a_1<a$，与最小性矛盾。
> <!-- bilingual-en:start -->
> Suppose a positive integer solution exists and choose one with minimal $a$. Reducing modulo $3$ gives $3\mid c$; write $c=3c_1$ and divide the original equation by $3$:
> $$a^4+3b^4=27c_1^4,$$
> Reducing modulo $3$ gives $3\mid a$; write $a=3a_1$, substitute, and divide by $3$:
> $$27a_1^4+b^4=9c_1^4,$$
> Reducing modulo $3$ again gives $3\mid b$; write $b=3b_1$. Thus $a,b,c$ are all divisible by $3$. After substituting and cancelling $3^4$, $(a_1,b_1,c_1)$ is another positive integer solution of the same equation with $a_1<a$, contradicting minimality.
> <!-- bilingual-en:end -->

> [!example]- Problem 3：不用数字常量写谓词
> 论域为 $\mathbb N$。先定义
> $$\mathrm{One}(x):\Longleftrightarrow\forall y(xy=y).$$
> **(a)** $x=1$ 即 $\mathrm{One}(x)$。
>
> **(b)** $m\mid n$：$\exists k(mk=n)$。
>
> **(c)**
> $$\mathrm{Prime}(n):\Longleftrightarrow\neg\mathrm{One}(n)\land\forall m(m\mid n\Rightarrow(\mathrm{One}(m)\lor m=n)).$$
>
> **(d)** “$n$ 是某素数的正整数次幂”可不用指数写成
> $$\exists p\,[\mathrm{Prime}(p)\land p\mid n\land\forall q((\mathrm{Prime}(q)\land q\mid n)\Rightarrow q=p)].$$
> 唯一素因子分解保证只有一个素因子的正整数恰为该素数的幂；$1$ 因没有素因子、$0$ 因被所有素数整除，均被排除。
> <!-- bilingual-en:start -->
> The domain is $\mathbb N$.  Define first
> $$\mathrm{One}(x):\Longleftrightarrow\forall y(xy=y).$$
> **(a)**$x=1$ is $\mathrm{One}(x)$.
> **(b)**$m\mid n$:$\exists k(mk=n)$.
> **(c)**
> $$\mathrm{Prime}(n):\Longleftrightarrow\neg\mathrm{One}(n)\land\forall m(m\mid n\Rightarrow(\mathrm{One}(m)\lor m=n)).$$
> **(d)**"$n$ is a positive integer power of a prime number" can be written without exponentiation
> $$\exists p\,[\mathrm{Prime}(p)\land p\mid n\land\forall q((\mathrm{Prime}(q)\land q\mid n)\Rightarrow q=p)].$$
> Unique factorisation says that a positive integer with exactly one prime factor is a power of that prime. The definition excludes $1$, which has no prime factors, and $0$, which is divisible by every prime.
> <!-- bilingual-en:end -->

> [!example]- Problem 4：$\mathbb Z\to\mathbb Z$ 的映射类型
> (a) $x^2$：N（$x,-x$ 碰撞且负整数未命中）；(b) $x+2$：B，逆为 $y-2$；(c) $2x$：I，奇数未命中；(d) $-x$：B，自身为逆；(e) $\lfloor x/2\rfloor$：S，每个 $k$ 被 $2k$ 命中，但 $2k,2k+1$ 碰撞。
> <!-- bilingual-en:start -->
> (a) $x^2$: neither injective nor surjective, because $x$ and $-x$ collide and negative integers are not attained; (b) $x+2$: bijective, with inverse $y-2$; (c) $2x$: injective but not surjective, because odd integers are not attained; (d) $-x$: bijective and its own inverse; (e) $\lfloor x/2\rfloor$: surjective but not injective, because every $k$ is attained by $2k$ while $2k$ and $2k+1$ collide.
> <!-- bilingual-en:end -->

> [!example]- Problem 5：4 人与 7 人团队
> 用强归纳证明 $S(n)$：“$n+18$ 人可分为 4/7 人团队”。连续基例：
> $$18=7+7+4,\qquad19=7+4+4+4,\qquad20=5\cdot4,\qquad21=3\cdot7.$$
> 对 $n\ge4$，$(n+18)-4=(n-4)+18$，由强归纳假设可组队；再增加一个 4 人队即可。因此所有 $n\ge0$ 的 $S(n)$ 成立。
> <!-- bilingual-en:start -->
> $S(n)$: "$n+18$ people can be divided into 4/7 teams".  Continuous base case:
> $$18=7+7+4,\qquad19=7+4+4+4,\qquad20=5\cdot4,\qquad21=3\cdot7.$$
> For $n\ge4$, $(n+18)-4=(n-4)+18$, the team can be formed by the strong induction hypothesis, and an additional team of four can be formed.  Therefore, all $n\ge0$'s $S(n)$ are valid.
> <!-- bilingual-en:end -->

---

## Session 9 — State Machines and Invariants

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

怎样证明一个可能运行任意久的过程永远不会到达坏状态？怎样把“值一直下降”变成终止性证明？
<!-- bilingual-en:start -->
How do you prove that a process that may run for any length of time never reaches a bad state?  How do you turn "value keeps dropping" into a termination certificate?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session9.pdf|Session 9 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp9.pdf|CP9]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VuG2JNcRXYg.pdf|State Machines and Invariants]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/a7JUH-EtHtI.pdf|Derived Variables]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StateMachine.pdf|State Machines]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_DerivedVaria.pdf|Derived Variables]]

### 9.1 状态机模型
<!-- bilingual-en:start -->
*9.1 state machine model*
<!-- bilingual-en:end -->

[[数学证明方法#不变量与算法正确性|状态机（state machine）]]由状态集合 $Q$、开始状态 $q_0$、转移关系 $\delta\subseteq Q\times Q$ 构成。状态 $r$ **reachable**，若存在有限路径
<!-- bilingual-en:start -->
[[数学证明方法#不变量与算法正确性|state machine]] is composed of state collection $Q$, start state $q_0$ and transition relation $\delta\subseteq Q\times Q$.  Status $r$**reachable**if finite path exists
<!-- bilingual-en:end -->

$$
q_0\to q_1\to\cdots\to r.
$$

建模时要分清：哪些信息足以决定下一步、什么是合法转移、何时无转移。若状态遗漏必要历史，所谓“不变量”可能根本无法表达。
<!-- bilingual-en:start -->
Modeling is about identifying what information is sufficient to determine the next step, what is a legal transfer, and when there is no transfer.  If that state omit the necessary history, the so-called" invariant" may not be express at all.
<!-- bilingual-en:end -->

### 9.2 preserved invariant 与 invariant
<!-- bilingual-en:start -->
*9.2 preserved invariant and invariant*
<!-- bilingual-en:end -->

谓词 $P$ 是 **preserved invariant**，若每条转移 $q\to q'$ 都满足
<!-- bilingual-en:start -->
The predicate $P$ is**preserved invariant**if each transition $q\to q'$ satisfies
<!-- bilingual-en:end -->

$$
P(q)\Rightarrow P(q').
$$

若再有 $P(q_0)$，Floyd invariant principle 给出所有 reachable $r$ 均满足 $P(r)$。证明是对路径长度归纳：长度 0 为开始状态；长度 $n+1$ 的路径由长度 $n$ 的可达状态再走一步，应用保持性。
<!-- bilingual-en:start -->
If there is another $P(q_0)$, Floyd invariant principle gives that all reachable $r$ satisfy $P(r)$.  The proof is to conclude the length of the path: the length of 0 is the beginning state; the length of $n+1$ path from the length of $n$ can reach the state of another step, the application of retention.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-state-machine-invariant.png|900]]

读图：初始状态位于不变量区域内，而每条合法转移都留在该区域；因此区域外的坏状态不可达。
<!-- bilingual-en:start -->
Read: The initial state is within the invariant region, and each legal transition remains within the region; therefore, a bad state outside the region is not reachable.
<!-- bilingual-en:end -->

> [!warning] 两个词不能混用
> preserved 只表示“一旦成立就保持”，未保证开始时成立；invariant 是“所有可达状态都成立”。
> <!-- bilingual-en:start -->
> preserved simply states "hold as soon as it is established" and there is no guarantee that it will be established at the beginning; invariant is "all reachable states are established".
> <!-- bilingual-en:end -->

### 9.3 找不变量：从转移变化量入手
<!-- bilingual-en:start -->
*9.3 Looking for the invariant: Starting with the quantity of change*
<!-- bilingual-en:end -->

若状态为向量 $(x_1,\dots,x_k)$，先列每类转移的 $\Delta x_i$，再寻找线性组合 $a_1x_1+\cdots+a_kx_k$ 使每类变化为 0，或模 $m$ 为 0。整数网格机器人适合模不变量，资源守恒适合线性等式，排列谜题常用 parity。
<!-- bilingual-en:start -->
If the state is vector $(x_1,\dots,x_k)$, first list the $\Delta x_i$ of each class, and then look for the linear combination $a_1x_1+\cdots+a_kx_k$ to make each class change to 0, or the module $m$ is 0.  Integer grid robots are suitable for modular invariants, resource conservation is suitable for linear equations, and puzzles are often arranged by parity.
<!-- bilingual-en:end -->

### 9.4 partial correctness 与 termination 分开
<!-- bilingual-en:start -->
*9.4 partial correctness separate from termination*
<!-- bilingual-en:end -->

**部分正确性（partial correctness）**只说“若算法停，则答案正确”；**全正确性**还要证一定停。
<!-- bilingual-en:start -->
**partial correctness**just say "if the algorithm stops, then the answer is correct";**complete correctness**and verify that it stops.
<!-- bilingual-en:end -->

[[数学证明方法#不变量与算法正确性|秩函数/derived variable]] $V:Q\to W$ 若每步严格下降，而 $W$ well-ordered，则不存在无限转移链。仅 weakly decreasing 不足，例如保持常数的过程可以永远运行。
<!-- bilingual-en:start -->
If [[数学证明方法#不变量与算法正确性|rank function/derived variable]] $V:Q\to W$ drops strictly at each step, and $W$ well-ordered, there is no infinite transfer chain.  Only weakly decreasing is insufficient, for example, a constant-keeping procedure can run forever.
<!-- bilingual-en:end -->

### 9.5 俄式整数乘法
<!-- bilingual-en:start -->
*9.5 Russian integer multiplication*
<!-- bilingual-en:end -->

状态 $(r,s,a)$ 从 $(x,y,0)$ 开始；$s$ 偶时转到 $(2r,s/2,a)$，$s$ 奇时转到 $(2r,(s-1)/2,a+r)$。不变量为
<!-- bilingual-en:start -->
Status $(r,s,a)$ starts with $(x,y,0)$; $s$ goes to $(2r,s/2,a)$ even, $s$ goes to $(2r,(s-1)/2,a+r)$ odd.  The invariant is
<!-- bilingual-en:end -->

$$
rs+a=xy.
$$

偶转移后 $(2r)(s/2)+a=rs+a$；奇转移后
<!-- bilingual-en:start -->
$(2r)(s/2)+a=rs+a$ after even transfer; after odd transfer
<!-- bilingual-en:end -->

$$
(2r)\frac{s-1}{2}+(a+r)=r(s-1)+a+r=rs+a.
$$

终态只能有 $s=0$，于是 $a=xy$。同时 $s$ 每步变为 $\lfloor s/2\rfloor$，故转移次数 $O(\log(y+1))$。
<!-- bilingual-en:start -->
There's only $s=0$, so $a=xy$.  At the same time, $s$ becomes $\lfloor s/2\rfloor$ in each step, so the number of transfer is $O(\log(y+1))$.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（14 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.2_state-machine-invariants|1.9.2 State Machine Invariants]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.4_derived-variables-and-termination|1.9.4 Derived Variables and Termination]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.5_integer-multiplication|1.9.5 Integer Multiplication]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.6_chocolate-bars|1.9.6 Chocolate Bars]]
>
> 1–4. Die Hard 状态是两壶水量 $(b,l)$；例中每壶水量始终被 3 整除；Diagonal Robot 合法示例为 $(-11,3),(3,7)$；Floyd 原理是对路径长度的归纳，开始成立且 preserved 即对所有 reachable 状态成立。
>
> 5–8. derived variable 是给状态赋实值的函数；机器人例为 $(x+y)\bmod2$；constant 等价同时 weakly increasing/decreasing；取值于 well-ordered set 且每步 strictly decreasing 保证终止。
>
> 9–10. 乘法算法唯一同时被两类命令保持的是 $xy+p=ab$；每步严格变小的是 $x$，而 $xy$ 仅 weakly decreasing。11–14. 巧克力拆分中 $s=p-1$ 与 $s\ne p$ preserved；$mn-p$ 每步减小；终态 $p=mn$；因此拆分次数 $s=mn-1$。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.2_state-machine-invariants|1.9.2 State Machine Invariants]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.4_derived-variables-and-termination|1.9.4 Derived Variables and Termination]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.5_integer-multiplication|1.9.5 Integer Multiplication]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S09_1.9.6_chocolate-bars|1.9.6 Chocolate Bars]]
> 1-4.  The Die Hard state is that that amount of water in the two jugs is $(b,l)$; in the example, the amount of water in each jug is divide by 3 all the time; the legal example of Diagonal Robot is $(-11,3),(3,7)$; the Floyd principle is the induction of the length of the path, and the preserved is established for all the reachable states at first.
> 5-8. derived variable is a function that assigns a real value to the state;the robot case is $(x+y)\bmod2$;constant is equivalent to weakly increasing/decreasing;the value is well-ordered set and strictly decreasing is guaranteed to terminate in each step.
> 9-10. The only multiplication algorithm that is maintained by both types of commands is $xy+p=ab$; each step is strictly smaller than $x$, while $xy$ is only weakly decreasing.  11-14. $s=p-1$ and $s\ne p$ preserved in chocolate splitting; $mn-p$ is reduced in each step; the final state is $p=mn$; therefore, the splitting times are $s=mn-1$.
> <!-- bilingual-en:end -->

> [!example]- CP9 非官方独立题解（4 道）
> **1. shift-and-add。** (a) 上文已逐类验证 $rs+a=xy$ preserved。(b) 无转移意味着 $s=0$，代入不变量得 $a=xy$。(c) 每步把正整数 $s$ 变为 $\lfloor s/2\rfloor$，至多 $\lfloor\log_2y\rfloor+1$ 步归零。
>
> **2. Fifteen Puzzle。** (a) 初态列表为 $(1,2,\dots,15)$、空格 $(4,4)$；目标列表为 $(15,14,\dots,1)$、空格 $(4,4)$。(b) 初态 inversion 数 $0$，加空格行号为偶数；目标 reversal 的 inversion 数为 $\binom{15}{2}=105$，$105+4=109$ 为奇数。(c) 水平移动不改变去掉空格后的列表，也不改变行号；竖直移动使某牌在列表中跨过 3 张牌，inversion parity 翻转，同时空格行号变化 1，也翻转。二者之和模 2 不变，故目标不可达。
>
> **3. Zakim bridge。** 入桥转移（条件 $A-B<T_0$）为 $(A,B,C)\to(A+3,B,C+1)$；出桥（$C>0$）为 $(A,B,C)\to(A,B+2,C-1)$。
>
> derived variables 分类：$A$ WI，$B$ WI，$A+B$ SI，$A-B$ N，$3C-A$ WD，$2A-3B$ N，$B+3C$ N，$2A-3B-6C$ constant，$2A-2B-3C$ N。关键变化量：
>
> | 量 | 入桥 $\Delta$ | 出桥 $\Delta$ |
> |---|---:|---:|
> | $A-B-3C$ | $0$ | $+1$ |
> | $2A-3B-6C$ | $0$ | $0$ |
>
> 令 $J=A-B-3C$，则 $J\ge J_0=A_0-B_0-3C_0$ preserved。若 $C=1000$，便有
> $$A-B\ge A_0-B_0+3(1000-C_0)=T_0,$$
> 所以入桥条件失败；出桥只会降低 $C$。故 $P:(C\le1000\land J\ge J_0)$ 是开始成立且排除 collapsed state 的不变量。
>
> deadlock 确可达：先让初始 $C_0$ 辆车全部驶出，此时 $C=0$、$A-B=A_0-B_0-2C_0<T_0$。反复“放一车进入、立即让它驶出”，每轮回到 $C=0$ 且 $A-B$ 净增 $1$。当增到 $T_0$ 时，既无车可出，又因严格不等式失败而无车可入。
>
> **4. Beaver flu（补充题）。** 把已感染格子的 perimeter 定义为感染区域与未感染/教室外部之间的单位边数。新感染格若有 $k\ge2$ 个已感染邻居，perimeter 变化 $4-2k\le0$。若初始感染 $m<n$ 人，perimeter 至多 $4m<4n$；若最终全班感染，$n\times n$ 方形 perimeter 恰为 $4n$。非增量不可能从 $<4n$ 到 $4n$，故至少一人永不感染。
> <!-- bilingual-en:start -->
> **1. shift-and-add.**(a) The $rs+a=xy$ preserved has been validated class by class above.  (b) No transfer means $s=0$, substituting the invariant as $a=xy$.  (c) The positive integer $s$ is changed to $\lfloor s/2\rfloor$ in each step, and at most $\lfloor\log_2y\rfloor+1$ is returned to zero.
> **2.  Fifteen Puzzle.**(a) Initial list is $(1,2,\dots,15)$, space $(4,4)$; target list is $(15,14,\dots,1)$, space $(4,4)$.  (b) The initial state inversion number $0$, plus the space line number is even; the target reversal inversion number is $\binom{15}{2}=105$, and $105+4=109$ is odd.  (c) Horizontal movement does not change the list after the blanks are removed, nor does it change the line numbers; vertical movement causes a card to be flipped across three cards in the list, with inversion parity flipped, while the blanks line number changes 1, also flipped.  The sum of the two modes 2 is constant, so the target is unreachable.
> **3.  Zakim bridge.**In-bridge transfer (condition $A-B<T_0$) is $(A,B,C)\to(A+3,B,C+1)$; Out-bridge ($C>0$) is $(A,B,C)\to(A,B+2,C-1)$.
> derived variables classification: $A$ WI, $B$ WI, $A+B$ SI, $A-B$ N, $3C-A$ WD, $2A-3B$ N, $B+3C$ N, $2A-3B-6C$ constant, $2A-2B-3C$ N.  Key Variations:
> | Quantity | In Bridge $\Delta$ | Out Bridge $\Delta$ |
> |—|—:|—:|
> | $A-B-3C$ | $0$ | $+1$ |
> | $2A-3B-6C$ | $0$ | $0$ |
> Let $J=A-B-3C$, then $J\ge J_0=A_0-B_0-3C_0$ preserved.  If $C=1000$, yes
> $$A-B\ge A_0-B_0+3(1000-C_0)=T_0,$$
> So the bridge entry condition fails; the bridge exit will only decrease $C$.  So $P:(C\le1000\land J\ge J_0)$ is the invariant that starts and excludes collapsed state.
> The deadlock does: First let the original $C_0$ cars out, at this point $C=0$, $A-B=A_0-B_0-2C_0<T_0$.  Repeatedly "put a car in, let it out immediately", each round back to $C=0$ and $A-B$ net $1$.  When it increases to $T_0$, there is no car to go out, and no car to go in because of the failure of strict inequality.
> **4.  Beaver flu.**Defines the number of unit edges between the infected area and the uninfected/outside of the classroom as the perimeter of the infected grid.  If there are $k\ge2$ infected neighbors in the newly infected cell, the change of perimeter is $4-2k\le0$.  perimeter is $4m<4n$ $n\times n$ at most if $m<n$ is initially infected, and $4n$ is perimeter squared if the entire class is eventually infected.  Non-incremental cannot go from $<4n$ to $4n$, so at least one person is never infected.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. preserved invariant 还缺哪个条件才能推出安全性？
> 2. $V$ 每步减少且始终为非负实数是否必终止？
> 3. invariant 排除坏状态，是否说明所有好状态都 reachable？
>
> <!-- bilingual-en:start -->
> 1. What else does preserved invariant lack to introduce security?
> 2. Does $V$ have to terminate per-step reduction and always be a non-negative real number?
> 3. Does invariant rule out bad states, which means all good states are reachable?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 它在 start state 成立。2. 不一定，如 $1,1/2,1/4,\dots$；需要 well-founded/良序下降。3. 不说明；不变量通常只是可达集的 over-approximation。
> <!-- bilingual-en:start -->
> 1. It was established in start state.  2. Not necessarily, e.g. $1,1/2,1/4,\dots$; well-founded/well-ordered decline required.  3. No description; invariants are usually only over-approximation of reachable sets.
> <!-- bilingual-en:end -->

**知识链：**归纳 → 路径长度 → preserved invariant → safety；良序下降 → termination。
<!-- bilingual-en:start -->
**Knowledge chain:**Summarization→Path length→preserved invariant→safety; Well-ordered descent→termination.
<!-- bilingual-en:end -->

---

## Session 10 — Recursive Definitions

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

对象若不是按整数大小而是按语法树生成，怎样定义“全部对象”，又怎样证明每种对象都有性质 $P$？
<!-- bilingual-en:start -->
How to define "all objects" and how to prove that each object has a property $P$?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session10.pdf|Session 10 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp10.pdf|CP10]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TXNXT3oBROw.pdf|Recursive Data]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VWIDwHCGJDQ.pdf|Structural Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/tOsdeaYDCMk.pdf|Recursive Functions]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RecursiveData.pdf|Recursive Data]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StructuralInd.pdf|Structural Induction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RecursiveFunc.pdf|Recursive Functions]]

### 10.1 递归数据类型
<!-- bilingual-en:start -->
*10.1 Recursive Data Types*
<!-- bilingual-en:end -->

[[01_Proofs#Session 10 — Recursive Definitions|递归定义（recursive definition）]]包含 base cases、constructor cases 与 closure clause。例如匹配括号串 $M$：
<!-- bilingual-en:start -->
[[01_Proofs#Session 10 — Recursive Definitions|recursive definition]] includes base cases, constructor cases, and closure clause.  For example, match the parenthetical string $M$:
<!-- bilingual-en:end -->

- $\lambda\in M$；
- 若 $s,t\in M$，则 $[s]t\in M$；
- 只有有限次使用以上规则生成的串才在 $M$ 中。
<!-- bilingual-en:start -->
- $\lambda\in M$;
- If $s,t\in M$, $[s]t\in M$;
- Strings generated using the above rule only a limited number of times are in $M$.
<!-- bilingual-en:end -->

最后一条排除无限构造链产生的非有限串，也防止凭直觉额外加入对象。
<!-- bilingual-en:start -->
The last one eliminates the non-finite string generated by the infinite construction chain and also prevents the addition of objects by intuition.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-recursive-structure.png|900]]

读图：从叶端的 base objects 向上经过 constructors 形成有限构造树，结构归纳正是沿同一依赖方向传播性质。
<!-- bilingual-en:start -->
Reading map:From the base objects of the leaf end through the constructors to form the finite structure tree, the structure induction is propagating along the same direction of dependence.
<!-- bilingual-en:end -->

### 10.2 结构归纳原理
<!-- bilingual-en:start -->
*10.2 Structural Inductive Principles*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|结构归纳]]要求：对每个 base object 证明 $P$；对每个 constructor，假设直接子对象均满足 $P$，证明构造结果满足 $P$。证明依据是有限构造树，而不是表面的数值大小。若有 3 个 bases、4 个 constructors，就有 7 个逻辑分支。
<!-- bilingual-en:start -->
[[数学证明方法#归纳法|structure induction]] requires proving $P$ for each base object, and for each constructor, assuming that the direct child objects satisfy $P$ and the result of proof construction satisfies $P$.  The proof is based on the finite construction tree, not the numerical size of the surface.  If there are 3 bases, 4 constructors, there are 7 logical branches.
<!-- bilingual-en:end -->

### 10.3 递归函数与良定义
<!-- bilingual-en:start -->
*10.3 Recursive Functions and Good Definitions*
<!-- bilingual-en:end -->

对二叉树，`size`、`flatten` 等函数也按 base/constructor 定义：叶子直接给值，内部节点从左右子树的结果组合。若语法可能给同一对象多种解析，还需证明不同解析结果相同，或改用 unambiguous grammar。
<!-- bilingual-en:start -->
For binary trees, `size`, `flatten` and other functions are also defined in terms of base/constructor: the leaves give values directly, and the internal nodes are combined from the results of the left and right sub-trees.  If the syntax may give more than one parse to the same object, you may want to prove that different parses produce the same result, or use unambiguous grammar instead.
<!-- bilingual-en:end -->

普通归纳适合唯一数值前驱，强归纳适合任意更小规模，结构归纳适合直接组成部件。结构归纳可转成对构造树高度的归纳，但直接写更贴近信息流。
<!-- bilingual-en:start -->
The general induction is suitable for the unique value precursor, the strong induction is suitable for any smaller scale, and the structural induction is suitable for the direct component.  Structural induction can be converted into the induction of the height of the tectonic tree, but direct writing is closer to the information flow.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（9 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.2_matching-parentheses|1.10.2 Matching Parentheses]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.3_functions-of-f18|1.10.3 Functions of F18]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.5_structural-induction-definition|1.10.5 Structural Induction: Definition]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.6_counting-cases|1.10.6 Counting Cases]]
>
> 1. matched-parentheses 中：没有串以右括号开头、集合无限、每串长度偶数；“每串以左括号开头”被空串反例否定。2–3. 用 $\cos x=\sin(x+\pi/2)$ 需要 identity、constant、sine 三个 base，并使用 addition 与 composition。4–8. 结构归纳模板五个 placeholder 依次为 $x,b,R,c(x),x$。9. 3 个 base、4 个 constructor 共需 7 个证明分支。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.2_matching-parentheses|1.10.2 Matching Parentheses]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.3_functions-of-f18|1.10.3 Functions of F18]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.5_structural-induction-definition|1.10.5 Structural Induction: Definition]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S10_1.10.6_counting-cases|1.10.6 Counting Cases]]
> 1. For matched-parenthesis strings, no string begins with a closing parenthesis, the set is infinite, and every string has even length. The claim that every string begins with an opening parenthesis is refuted by the empty string. 2–3. Expressing $\cos x=\sin(x+\pi/2)$ requires the identity, constant, and sine base cases, together with addition and composition. 4–8. The five placeholders in the structural-induction template are $x,b,R,c(x),x$, in that order. 9. Three base cases and four constructors require seven proof branches in total.
> <!-- bilingual-en:end -->

> [!example]- CP10 非官方独立题解（4 道）
> **1. Elementary 18.01 Functions。** (a) $e(x)=2^x$ 是 F18，inverse $\ell=e^{-1}=\log_2x$ 也是 F18；故在 $x>0$ 上 $2^{-\ell(x)}=1/x$ 是 F18。在 $x<0$ 分支写成 $-2^{-\log_2(-x)}$。课程把它们视为带自然定义域的 elementary branches；$\mathrm{id}^{-1}=\mathrm{id}$，不能误当 reciprocal。
>
> (b) 对构造做结构归纳。bases：$\mathrm{id}'=1$、常数导数 $0$；$\sin'x=\cos x=\sin(x+\pi/2)$。若 $f',g'$ 已是 F18：
> $$(f+g)'=f'+g',\quad(fg)'=f'g+fg',\quad(2^g)'=(\ln2)2^gg',\quad(f\circ g)'=(f'\circ g)g'.$$
> 若 $h=f^{-1}$ 且局部可逆、$f'(h(x))\ne0$，则 $h'(x)=1/f'(h(x))$，由 (a) 和 composition 仍是 F18。可逆区间与非零导数是必要条件。
>
> **2. Erasable = RecMatch。** 对 RecMatch 结构归纳：$\lambda$ 可擦除；若 $s,t$ 可擦除，则 $[s]t$ 先把 $s$ 擦空，擦掉 `[]`，再擦 $t$，故 RecMatch $\subseteq$ Erasable。
>
> 反向对串长强归纳。长度 0 只有 $\lambda$。若一次擦除把 $x$ 变成 $y\in$ RecMatch：$y=\lambda$ 时 $x=[]$；否则 $y=[s]t$。补回的 `[]` 只可能在最前面、$s$ 内或 $t$ 内。第一种 $x=[][s]t=[\lambda]([s]t)$；后两种产生更短的 erasable 子串，由强归纳进入 RecMatch，再用 constructor。三类按语法位置穷尽。
>
> **3. 递归定义。** (a) $S$：base $1$；constructors $x\mapsto2x,3x,5x$。(b) $T$：base $1$；constructors $x\mapsto18x,15x,5x$，分别对应增加 $k,m,n$。(c) $L'$：base $(0,0)$；constructors $(a,b)\mapsto(a\pm1,b\pm1)$（同号）及 $(a,b)\mapsto(a\pm3,b)$。(d) 每个 constructor 保持 $a-b\equiv0\pmod3$，故 $L'\subseteq L$。(e) 若 $a-b=3k$，先同步移动到 $(b,b)$，再把第一坐标移动 $3k$，故 $L\subseteq L'$。(f) 要无歧义，可用唯一正规形 $(k,b)\mapsto(b+3k,b)$，并把整数 $k,b$ 各自编码为 sign 加 natural magnitude。
>
> **4. binary-2PTG（补充题）。** 图中
> $$G=\langle bintree,\langle bintree,\langle leaf,win\rangle,\langle bintree,\langle leaf,lose\rangle,\langle leaf,win\rangle\rangle\rangle,\langle leaf,win\rangle\rangle.$$
> `flatten(leaf,l)=(l)`；`flatten(bintree,G1,G2)=flatten(G1)++flatten(G2)`。结构归纳：叶子时 $2\cdot1=1+1$；内部节点由 IH：
> $$2(\ell_1+\ell_2)=(|G_1|+1)+(|G_2|+1)=|G_1|+|G_2|+2=|G|+1.$$
> <!-- bilingual-en:start -->
> **1.  Elementary 18.01 Functions.**(a) $e(x)=2^x$ is F18 and inverse $\ell=e^{-1}=\log_2x$ is F18; therefore, $2^{-\ell(x)}=1/x$ is F18 on $x>0$.  In the $x<0$ branch, write $-2^{-\log_2(-x)}$.  The course treats them as elementary branches;$\mathrm{id}^{-1}=\mathrm{id}$, not reciprocal.
> (b) Structural induction of the structure.  bases:$\mathrm{id}'=1$, constant derivative $0$; $\sin'x=\cos x=\sin(x+\pi/2)$.  If $f',g'$ is already F18:
> $$(f+g)'=f'+g',\quad(fg)'=f'g+fg',\quad(2^g)'=(\ln2)2^gg',\quad(f\circ g)'=(f'\circ g)g'.$$
> If $h=f^{-1}$ and locally reversible, $f'(h(x))\ne0$, then $h'(x)=1/f'(h(x))$, from (a) and composition is still F18.  Reversible interval and non-zero derivative are necessary conditions.
> **2.  Erasable = RecMatch.**Summarize the structure of RecMatch: $\lambda$ can erase; if $s,t$ can erase, $[s]t$ erases $s$, erases `[]`, erases $t$, so RecMatch $\subseteq$ Erasable.
> Inverse induction of string length.  Length 0 is only $\lambda$.  $x=[]$ if one erasure changes $x$ to $y\in$ RecMatch:$y=\lambda$; otherwise $y=[s]t$.  A `[]` that is patched may only be in the first, $s$, or $t$.  The first $x=[][s]t=[\lambda]([s]t)$, the latter two produce shorter erasable substrings, which are induced into RecMatch and then constructor.  The three categories are exhausted by grammatical position.
> **3. Recursive definitions.**(a)$S$:base $1$;constructors $x\mapsto2x,3x,5x$.  (b) $T$:base $1$; constructors $x\mapsto18x,15x,5x$, corresponding to an increase of $k,m,n$, respectively.  (c) $L'$:base $(0,0)$; constructors $(a,b)\mapsto(a\pm1,b\pm1)$ (same number) and $(a,b)\mapsto(a\pm3,b)$.  (d) $a-b\equiv0\pmod3$ per constructor, so $L'\subseteq L$.  (e) If $a-b=3k$, first synchronously move to $(b,b)$ and then move the first coordinate to $3k$, so $L\subseteq L'$.  (f) To be unambiguous, the unique regular form $(k,b)\mapsto(b+3k,b)$ may be used, and the integers $k,b$ are each encoded as sign plus natural magnitude.
> **4. binary-2PTG.**in the picture
> $$G=\langle bintree,\langle bintree,\langle leaf,win\rangle,\langle bintree,\langle leaf,lose\rangle,\langle leaf,win\rangle\rangle\rangle,\langle leaf,win\rangle\rangle.$$
> `flatten(leaf,l)=(l)`;`flatten(bintree,G1,G2)=flatten(G1)++flatten(G2)`.  Structure induction:Leaf time $2\cdot1=1+1$;Internal nodes by IH:
> $$2(\ell_1+\ell_2)=(|G_1|+1)+(|G_2|+1)=|G_1|+|G_2|+2=|G|+1.$$
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 为什么只证明所有 base cases 不足？
> 2. structural induction 的假设针对谁？
> 3. 若允许无限次 constructor，会发生什么？
>
> <!-- bilingual-en:start -->
> 1. Why only prove that all base cases are insufficient?
> 2. Who is the hypothesis for structural induction?
> 3. What happens if you allow an infinite number of constructor?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. constructors 会产生新对象，必须证性质在每个 constructor 下保持。2. 当前 constructor 的直接子对象。3. 可能加入无限树/无限串，有限结构归纳不再自动适用。
> <!-- bilingual-en:start -->
> 1. constructors generates a new object and must be certified under each constructor.  2. The immediate child object of the current constructor.  3. Infinite trees/infinite strings may be added, and finite structure induction is no longer automatically applicable.
> <!-- bilingual-en:end -->

**知识链：**induction → construction tree → recursive datatype → structural induction → recursive function。
<!-- bilingual-en:start -->
**Knowledge Chain:**induction → construction tree → recursive datatype → structural induction → recursive function.
<!-- bilingual-en:end -->

---

## Session 11 — Infinite Sets

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

两个无限集合怎样比较大小？为什么整数、整数对和有理数一样多，而实数更多？为什么“所有集合的集合”和“万能程序分析器”会触发同一种对角矛盾？
<!-- bilingual-en:start -->
How do two infinite sets compare in size?  Why are there as many integers, pairs of integers, and rational numbers as there are real numbers?  Why do the "set of all sets" and the "universal analyser" trigger the same kind of diagonal contradiction?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session11.pdf|Session 11 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp11.pdf|CP11]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QzSCf62kzjE.pdf|Cardinality]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/AipSRi3CyLg.pdf|Countable Sets]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/4dj1ogUwTEM.pdf|Cantor's Theorem]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/WQHOImO0pX0.pdf|Halting Problem]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/5hETv64GIuE.pdf|Russell's Paradox]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/zcvsyL7GtH4.pdf|Set Theory Axioms]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Cardinality.pdf|Cardinality]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_CountableSets.pdf|Countable Sets]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_CantorsTheo.pdf|Cantor]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_haltingproblm.pdf|Halting]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RussellsParad.pdf|Russell]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_theroyaxioms.pdf|Axioms]]

### 11.1 无限基数与 bijection
<!-- bilingual-en:start -->
*11.1 Infinite Cardinality and bijection*
<!-- bilingual-en:end -->

两个集合等势，定义为存在 bijection。有限集不可能与增加一个新元素后的集合等势；无限集的特征之一恰是可以做到。例如
<!-- bilingual-en:start -->
Two sets have the same cardinality precisely when there is a bijection between them. A finite set cannot be equinumerous with the result of adjoining one new element, whereas an infinite set may be. For example,
<!-- bilingual-en:end -->

$$
f:\mathbb N\cup\{d\}\to\mathbb N,
\qquad f(d)=0,\quad f(n)=n+1.
$$

[[01_Proofs#11.3 Cantor 对角论证|Cantor–Schröder–Bernstein 定理]]：若 $A$ 可单射进 $B$ 且 $B$ 可单射进 $A$，则存在 $A\leftrightarrow B$ 的 bijection。课程也用等价的 surjection 版本；对非空集合，$A\twoheadrightarrow B$ 的逆关系给出 $B\hookrightarrow A$。
<!-- bilingual-en:start -->
The [[01_Proofs#11.3 Cantor 对角论证|Cantor–Schröder–Bernstein theorem]] says that injections $A\to B$ and $B\to A$ imply a bijection $A\leftrightarrow B$. The course also uses an equivalent surjection formulation; for nonempty sets, choosing one preimage for each element under a surjection $A\twoheadrightarrow B$ yields an injection $B\hookrightarrow A$.
<!-- bilingual-en:end -->

### 11.2 可数集合
<!-- bilingual-en:start -->
*11.2 Countable sets*
<!-- bilingual-en:end -->

[[01_Proofs#11.2 可数集合|可数（countable）]]集合是有限集或可与 $\mathbb N$ 的某个子集建立 bijection 的集合；countably infinite 则与 $\mathbb N$ 本身等势。
<!-- bilingual-en:start -->
A [[01_Proofs#11.2 可数集合|countable]] set is finite or is in bijection with a subset of $\mathbb N$; a countably infinite set is in bijection with $\mathbb N$ itself.
<!-- bilingual-en:end -->

整数可按 $0,1,-1,2,-2,\dots$ 枚举。整数对按对角线 $a+b=0,1,2,\dots$ 枚举，因此 $\mathbb N^2$ 可数。正有理数是正整数对映射 $(a,b)\mapsto a/b$ 的像；虽有重复，但取每个有理数首次出现位置即可得到 injection，故 $\mathbb Q$ 可数。
<!-- bilingual-en:start -->
Integers can be enumerated by $0,1,-1,2,-2,\dots$.  Integer pairs are enumerated diagonally by $a+b=0,1,2,\dots$, so $\mathbb N^2$ is countable.  A positive rational number is an image of a positive integer pair mapped to $(a,b)\mapsto a/b$. Although there is a repetition, injection can be obtained by taking the first occurrence position of each rational number, so $\mathbb Q$ is countable.
<!-- bilingual-en:end -->

> [!warning] “稠密”与“基数大”不同
> $\mathbb Q$ 在任意实数区间中都有点，却仍可数；拓扑上的稠密性和集合基数是不同性质。
> <!-- bilingual-en:start -->
> The results show that $\mathbb Q$ is a bit in any real number interval, but it is still countable, and the density and cardinality are different in topology.
> <!-- bilingual-en:end -->

### 11.3 Cantor 对角论证
<!-- bilingual-en:start -->
*11.3 Cantor Diagonal Argumentation*
<!-- bilingual-en:end -->

[[01_Proofs#11.3 Cantor 对角论证|Cantor 对角论证]]证明无限二进制序列集合 $\{0,1\}^{\omega}$ 不可数。反设存在列表
<!-- bilingual-en:start -->
[[01_Proofs#11.3 Cantor 对角论证|Cantor diagonal argument]] proved that the infinite set of binary sequences, $\{0,1\}^{\omega}$, is uncountable.  Invert Presence List
<!-- bilingual-en:end -->

$$
s_0,s_1,s_2,\dots,
$$

定义新序列 $d$ 的第 $n$ 位为 $1-s_n(n)$。于是 $d$ 与 $s_n$ 在第 $n$ 位不同，所以不等于任何 $s_n$，与列表完备矛盾。
<!-- bilingual-en:start -->
The $n$ bit that defines the new sequence $d$ is $1-s_n(n)$.  So $d$ and $s_n$ are different in $n$ bit, so it is not equal to any $s_n$, contradicting with complete list.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-cantor-diagonal.png|900]]

读图：沿列表主对角线逐位取反得到 $d$；第 $n$ 个反转位保证 $d$ 与列表第 $n$ 行至少有一位不同。
<!-- bilingual-en:start -->
Read: $d$ is inverted bit by bit along the main diagonal of the list; the $n$-th inverted bit guarantees that $d$ is at least one bit different from the $n$-th row of the list.
<!-- bilingual-en:end -->

更一般的 [[01_Proofs#11.3 Cantor 对角论证|Cantor 定理]]：对任何集合 $A$，不存在 surjection $f:A\to\operatorname{pow}(A)$。若有，定义
<!-- bilingual-en:start -->
More general [[01_Proofs#11.3 Cantor 对角论证|Cantor theorem]]: No surjection $f:A\to\operatorname{pow}(A)$ exists for any collection $A$.  if so, definition
<!-- bilingual-en:end -->

$$
D=\{a\in A:a\notin f(a)\}.
$$

满射给出 $d$ 使 $f(d)=D$，于是
<!-- bilingual-en:start -->
Surjectivity gives some $d$ such that $f(d)=D$. Therefore,
<!-- bilingual-en:end -->

$$
d\in D\Longleftrightarrow d\notin f(d)=D,
$$

矛盾。因此 $|A|<|\operatorname{pow}(A)|$。这不是“找不到合适函数”，而是排除了所有候选函数。
<!-- bilingual-en:start -->
This is a contradiction. Therefore $|A|<|\operatorname{pow}(A)|$. The argument does not merely fail to find a suitable function; it rules out every possible candidate.
<!-- bilingual-en:end -->

### 11.4 Russell 悖论与受限 comprehension
<!-- bilingual-en:start -->
*11.4 Russell Paradox and Restricted comprehension*
<!-- bilingual-en:end -->

若假设任意谓词都定义一个集合，令 $W=\{x:x\notin x\}$，便有
<!-- bilingual-en:start -->
Suppose that any predicate defines a set, let $W=\{x:x\notin x\}$, there is
<!-- bilingual-en:end -->

$$
W\in W\Longleftrightarrow W\notin W.
$$

正确结论不是逻辑失效，而是“$W$ 是集合”的 unrestricted comprehension 假设不允许。ZFC 的 separation 只允许从已有集合 $S$ 中切出 $\{x\in S:P(x)\}$，不承诺“所有对象”构成集合。
<!-- bilingual-en:start -->
The correct conclusion is not that logic is ineffective, but that the unrestricted comprehension assumption that "$W$ is a collection" is not allowed.  ZFC's separation only allows $\{x\in S:P(x)\}$ to be cut out of an existing collection, $S$, and does not promise "all objects" to form a collection.
<!-- bilingual-en:end -->

### 11.5 Halting Problem

[[01_Proofs#11.5 Halting Problem|停机问题]]问：给程序编码 $P$ 与输入 $x$，程序是否会停止？反设存在总判定器 $H(P,x)$。构造程序 $D(P)$：若 $H(P,P)$ 说会停，则进入无限循环；若说不停，则立即停止。把 $D$ 输入自身：
<!-- bilingual-en:start -->
The [[01_Proofs#11.5 Halting Problem|Halting Problem]] asks whether a program encoded by $P$ will halt on input $x$. Suppose, for contradiction, that there is a total decision procedure $H(P,x)$. Construct a program $D(P)$ that enters an infinite loop if $H(P,P)$ predicts that $P$ halts on itself, and halts immediately if $H(P,P)$ predicts that it does not. Now run $D$ on its own encoding:
<!-- bilingual-en:end -->

- 若 $H(D,D)$ 说停，定义使 $D(D)$ 不停；
- 若说不停，定义使 $D(D)$ 停。
<!-- bilingual-en:start -->
- If $H(D,D)$ says “halts,” the definition makes $D(D)$ run forever;
- if $H(D,D)$ says “does not halt,” the definition makes $D(D)$ halt immediately.
<!-- bilingual-en:end -->

两种都矛盾，所以不存在这样的总且永远正确的判定器。与 Cantor/Russell 相同，核心是让对象在“描述自身行为”的对角位置反转。
<!-- bilingual-en:start -->
Both cases are contradictory, so no total decision procedure can always answer correctly. As in Cantor's and Russell's arguments, the core is diagonal self-reference: make the object reverse the predicted description of its own behavior.
<!-- bilingual-en:end -->

### 11.6 ZFC 只需知道的边界
<!-- bilingual-en:start -->
*11.6 ZFC only needs to know the boundaries*
<!-- bilingual-en:end -->

课程不是集合论公理课，但以下职责要分清：extensionality 由成员决定集合；pairing/union/power set/infinity 允许受控构造；separation 在已有集合中筛选；foundation 排除无限向下成员环；choice 支持对任意非空集合族同时选择代表。使用“从无限集合持续选出不同元素”时，严格背景可能涉及 choice。
<!-- bilingual-en:start -->
The course is not an axiomatic course in set theory, but the following are important to clarify: extensionality determines sets by members; pairing/union/power set/infinity allows controlled constructs; separation filters in existing sets; foundation excludes infinite downward member loops; choice supports simultaneous selection of representatives for any family of non-empty sets.  When using Persistent Selection of Different Elements from an Infinite Set, strict backgrounds may involve choice.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercises（23 prompts）
> **本地逐题入口：**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.2_cantor-schroeder-bernstein|1.11.2 Cantor, Schroeder–Bernstein]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.5_cantor-s-diagonal-argument|1.11.5 Cantor's Diagonal Argument]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.6_countable-and-uncountable-sets|1.11.6 Countable and Uncountable Sets]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.8_halting-problem-basics|1.11.8 Halting Problem Basics (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.10_russell-s-paradox-and-zfc-optional|1.11.10 Russell's Paradox and ZFC (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.12_set-theory-axioms|1.11.12 Set Theory Axioms (optional)]]
>
> 1–3. 等势由 bijection 定义；Schröder–Bernstein 的 surjection 版本是双向 surjection 推出 bijection；无限集可与加入一个新元素后的集合 bijective。4–8. diagonal argument 构造一个不在列表中的无限序列；因此 $\mathbb N$ 不能 surject 到全部无限比特串；Cantor 定理给出 $A\operatorname{strict}\operatorname{pow}(A)$。
>
> 9. 可数选项：$\mathbb N,\mathbb Z,\mathbb N^2,\mathbb Q,\mathbb Z^+$ 以及任意固定有限长度的 bit strings；$\mathbb R,\mathbb C,\{0,1\}^{\omega},\mathbb Q^{\omega}$ 不可数。10. Halting Problem 是判断程序是否停机的问题，并且 undecidable；对角/反证是证明方法。
>
> 11–12. Russell 矛盾来自错误假设 $W$ 是集合；ZFC 避免 unrestricted comprehension。ZFC 用谓词逻辑表达，含 extensionality、power set、foundation 等公理，但不允许无约束造集合。13–23. 四条公理识别依次包含 Power Set、Comprehension/Separation、Foundation、Extensionality；对题给成员环，$\varnothing,T,\{T\},\{V,T\},\{U,V,T\},\{T,U,V\},\{\varnothing,T,U,V\}$ 是否含 $\in$-minimal 元素的答案依次为 No、Maybe、Yes、Yes、No、No、Yes。
> <!-- bilingual-en:start -->
> **Local entry:**
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.2_cantor-schroeder-bernstein|1.11.2 Cantor, Schroeder–Bernstein]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.5_cantor-s-diagonal-argument|1.11.5 Cantor's Diagonal Argument]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.6_countable-and-uncountable-sets|1.11.6 Countable and Uncountable Sets]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.8_halting-problem-basics|1.11.8 Halting Problem Basics (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.10_russell-s-paradox-and-zfc-optional|1.11.10 Russell's Paradox and ZFC (optional)]]
> - [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S11_1.11.12_set-theory-axioms|1.11.12 Set Theory Axioms (optional)]]
> 1–3. Equal cardinality is defined by a bijection. The surjection version of Schröder–Bernstein says that surjections in both directions imply a bijection. An infinite set may be in bijection with the result of adjoining one new element. 4–8. A diagonal argument constructs an infinite sequence missing from any proposed list; therefore $\mathbb N$ cannot surject onto all infinite bit strings. Cantor's theorem states that $|A|<|\mathcal P(A)|$.
> 9. The countable choices are $\mathbb N,\mathbb Z,\mathbb N^2,\mathbb Q,\mathbb Z^+$, and the set of bit strings of any fixed finite length. The sets $\mathbb R,\mathbb C,\{0,1\}^{\omega},\mathbb Q^{\omega}$ are uncountable. 10. The Halting Problem asks whether a program eventually halts; it is undecidable, and the proof uses diagonalization and contradiction.
> 11–12. Russell's paradox arises from the false assumption that $W$ is a set. ZFC avoids unrestricted comprehension: it is formulated in predicate logic and includes axioms such as extensionality, power set, and foundation, but it does not permit unrestricted set formation. 13–23. The four axioms identified in order are Power Set, Comprehension/Separation, Foundation, and Extensionality. For the given membership cycles, the answers to whether $\varnothing,T,\{T\},\{V,T\},\{U,V,T\},\{T,U,V\},\{\varnothing,T,U,V\}$ contain an $\in$-minimal element are respectively No, Maybe, Yes, Yes, No, No, and Yes.
> <!-- bilingual-en:end -->

> [!example]- CP11 非官方独立题解（4 道）
> **1. 无限集与序列。** (a) “继续挑一个新元素”在标准 ZFC 中可形式化；但在不含 choice 的弱集合论里，“每个无限集都含可数无限子集”并非自动成立，因此担忧有意义。(b) 对选出的互异序列 $a_0,a_1,\dots$，定义 $f(a_n)=n$，其余 $a\in A$ 映到 0；这是 $A\twoheadrightarrow\mathbb N$。
>
> **2. surjection image 可数。** 若 $f:\mathbb N\twoheadrightarrow S$，对每个 $s\in S$ 定义
> $$g(s)=\min\{n\in\mathbb N:f(n)=s\}.$$
> 该集合非空，WOP 保证最小元。若 $g(s)=g(t)$，则 $s=f(g(s))=f(g(t))=t$，所以 $g:S\hookrightarrow\mathbb N$，$S$ 可数。这就是“枚举时过滤重复”的数学形式。
>
> **3. 正有理数可数。** (a) 对 $a,b\in\mathbb Z^+$，Cantor pairing
> $$\pi(a,b)=\frac{(a+b-2)(a+b-1)}2+a$$
> 是到 $\mathbb Z^+$ 的 bijection：每个对角线 $a+b$ 占连续区段，区段内由 $a$ 唯一定位。(b) 映射 $(a,b)\mapsto a/b$ 从 $\mathbb Z^+\times\mathbb Z^+$ surject 到 $\mathbb Q^+$；由 Problem 2，像可数。
>
> **4. 不可识别语言。** (a) recognizer 顺次检查长度为偶数且每对字符相同并属于 `a`–`z`，否则 False。(b) 每个字符串 $s$ 编译为过程 $P_s$，$f(s)$ 是它识别的语言，所以 $\operatorname{range}(f)$ 正是所有 recognizable languages。(c) 令 $N=\{s:s\notin f(s)\}$。若 $N$ 可识别，则 $N=f(t)$ 对某个程序串 $t$；于是
> $$t\in N\Longleftrightarrow t\notin f(t)=N,$$
> 矛盾。(d) 因此不存在能对任意程序精确判断所有非平凡行为集合的万能 analyzer；具体受限程序类仍可能可判定，结论不是“任何分析都不可能”。
> <!-- bilingual-en:start -->
> **1. Infinite sets and sequences.**(a) "Continue to pick a new element" is formalized in standard ZFC; but in weak set theory without choice, "every infinite set has countable infinite subsets" is not automatic, so the concern is relevant.  (b) For the selected distinct sequence $a_0,a_1,\dots$, define $f(a_n)=n$, the remaining $a\in A$ map to 0; this is $A\twoheadrightarrow\mathbb N$.
> **2. surjection image is countable.**If $f:\mathbb N\twoheadrightarrow S$, define a value for each $s\in S$
> $$g(s)=\min\{n\in\mathbb N:f(n)=s\}.$$
> The set is non-empty, and WOP guarantees minimal elements.  If $g(s)=g(t)$, then $s=f(g(s))=f(g(t))=t$, so $g:S\hookrightarrow\mathbb N$,$S$ is countable.  This is the mathematical form of "filter duplicates when enumerating".
> **3. There are positive rational numbers.**(a) For $a,b\in\mathbb Z^+$, Cantor pairing
> $$\pi(a,b)=\frac{(a+b-2)(a+b-1)}2+a$$
> is the bijection to $\mathbb Z^+$: each diagonal $a+b$ occupies a continuous segment within which $a$ uniquely locates.  (b) Map $(a,b)\mapsto a/b$ from $\mathbb Z^+\times\mathbb Z^+$ surject to $\mathbb Q^+$; by Problem 2, like countable.
> **4. An unrecognizable language.** (a) The recognizer checks that the string has even length and that every adjacent pair consists of the same letter from `a`–`z`; otherwise it returns False. (b) Compile each string $s$ as a program $P_s$, and let $f(s)$ be the language recognized by that program. Then $\operatorname{range}(f)$ is exactly the class of recognizable languages. (c) Define $N=\{s:s\notin f(s)\}$. If $N$ were recognizable, then $N=f(t)$ for some program string $t$; hence
> $$t\in N\Longleftrightarrow t\notin f(t)=N,$$
> Contradiction.  (d) There is therefore no universal analyzer that can accurately determine the set of all non-trivial behaviors for any program; the specific restricted program class may still be decidable and the conclusion is not "impossible for any analysis".
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 一个 surjection $\mathbb N\to S$ 为什么可能重复但仍足以说明 $S$ 可数？
> 2. Cantor 集合 $D$ 的定义为什么必须依赖 $f(a)$？
> 3. 停机不可判定是否等于任何实例都不能判断？
>
> <!-- bilingual-en:start -->
> 1. Why is a surjection $\mathbb N\to S$ likely to be duplicated but still sufficient to indicate that a $S$ is countable?
> 2. Why must the definition of the Cantor collection $D$ rely on $f(a)$?
> 3. Does undecidability of the Halting Problem mean that no individual instance can be decided?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->
> [!success]- 答案
> 1. 为每个 $s$ 取首次出现位置得到 injection。2. 它要在每个候选 $f(a)$ 的“自身坐标”上反转，才能保证与每个候选不同。3. 不是；许多具体程序显然停或不停，不存在的是覆盖全部程序输入的总正确算法。
> <!-- bilingual-en:start -->
> 1. Take the first occurrence position for each $s$ to get injection.  2. It needs to be reversed on the "own coordinates" of each candidate $f(a)$ to be guaranteed to be different from each candidate.  3. No; many specific programs apparently stop or do not stop, and what does not exist is a total correct algorithm that covers all program inputs.
> <!-- bilingual-en:end -->

**知识链：**bijection → countability → diagonalization → power set → Russell/Halting self-reference。
<!-- bilingual-en:start -->
**Knowledge Chain:**bijection → countability → diagonalization → power set → Russell/Halting self-reference.
<!-- bilingual-en:end -->

---

## Problem Set 4 — Sessions 9–11

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps4.pdf|Problem Set 4 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps4.pdf|Problem Set 4 PDF]].  The following 3 questions are unofficial and independent.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：网格机器人不可达 $(1,1)$
> 四种动作是 $\pm(2,-1)$ 与 $\pm(1,3)$。考察
> $$I(x,y)=3x-y\pmod7.$$
> 对 $(2,-1)$，$\Delta I=3\cdot2-(-1)=7\equiv0$；对 $(1,3)$，$\Delta I=3-3=0$，反向动作同样为 0。因此 $I$ 是 preserved invariant。起点 $I(0,0)=0$，而目标 $I(1,1)=2\not\equiv0\pmod7$，所以不可达。
> <!-- bilingual-en:start -->
> The four actions are $\pm(2,-1)$ and $\pm(1,3)$.  inspection
> $$I(x,y)=3x-y\pmod7.$$
> For $(2,-1)$, $\Delta I=3\cdot2-(-1)=7\equiv0$; for $(1,3)$, $\Delta I=3-3=0$, the reverse action is also 0.  So $I$ is preserved invariant.  The starting point is $I(0,0)=0$ and the target is $I(1,1)=2\not\equiv0\pmod7$, so it is not reachable.
> <!-- bilingual-en:end -->

> [!example]- Problem 2：unique-label LBT
> 对 LBT 结构归纳。base $\langle l,leaf\rangle$ 有 $f_B=1,n_B=0$，所以 $f_B=n_B+1$。
>
> constructor $T=\langle l,B,C\rangle$。唯一标签条件保证 $B,C$ 的 leaf-label 集合互不相交，故
> $$f_T=f_B+f_C.$$
> 它也保证两子树 internal-label 集合互不相交且新标签 $l$ 未出现，故
> $$n_T=n_B+n_C+1.$$
> 用 IH：
> $$f_T=(n_B+1)+(n_C+1)=n_B+n_C+2=n_T+1.$$
> “唯一标签”正是在把 union 的基数写成和时使用；若标签重合，该等式会重复计数。
> <!-- bilingual-en:start -->
> Summarize the LBT structure.  base $\langle l,leaf\rangle$ has $f_B=1,n_B=0$, so $f_B=n_B+1$.
> constructor $T=\langle l,B,C\rangle$.  The unique label condition guarantees that $B,C$'s leaf-label sets do not intersect each other, so
> $$f_T=f_B+f_C.$$
> it also ensure that that set of two subtrees internal-label are not intersect and the new label $l$ does not appear, so
> $$n_T=n_B+n_C+1.$$
> With IH:
> $$f_T=(n_B+1)+(n_C+1)=n_B+n_C+2=n_T+1.$$
> Unique Tags is used precisely when writing the cardinality of a union as and; if the tags are coincident, the equation is repeated.
> <!-- bilingual-en:end -->

> [!example]- Problem 3：$(0,1]$ 与非负平面等势
> **(a)**
> $$h:(0,1]\to[0,\infty),\qquad h(x)=\frac1x-1,$$
> 逆为 $h^{-1}(y)=1/(y+1)$，故 bijective。
>
> **(b)** 把 long digit sequence $(d_1,d_2,\dots)$ 映到 $0.d_1d_2\dots$。不以全 0 结尾的约定为每个 $(0,1]$ 实数选取唯一表示：终止小数改用尾随 9 的表示，$1=0.999\dots$；$0$ 没有 long 表示。
>
> **(c)** 在那些奇数位、偶数位子序列都 long 的输入上定义
> $$F(d_1d_2d_3d_4\dots)=((d_1,d_3,\dots),(d_2,d_4,\dots)).$$
> 它可为 partial function；任给两条 long 序列，交错后得到其原像，所以 $F:L\twoheadrightarrow L^2$。
>
> **(d)** 若 $f:A\to B$ bijective，则 $(a_1,a_2)\mapsto(f(a_1),f(a_2))$ bijective，逆逐坐标用 $f^{-1}$。由 (b) 得 $L^2\leftrightarrow(0,1]^2$。
>
> **(e)** 组合 (b)(c)(d) 得 $(0,1]\twoheadrightarrow(0,1]^2$；反向投影 $(x,y)\mapsto x$ 也是 surjection。Schröder–Bernstein 的等价 surjection 版本给出 bijection $(0,1]\leftrightarrow(0,1]^2$。
>
> **(f)** 对上一步 bijection 的两个坐标分别复合 (a) 的 $h$，得到
> $$(0,1]\leftrightarrow(0,1]^2\leftrightarrow[0,\infty)^2.$$
> <!-- bilingual-en:start -->
> **(a)**
> $$h:(0,1]\to[0,\infty),\qquad h(x)=\frac1x-1,$$
> The inverse is $h^{-1}(y)=1/(y+1)$, so bijective.
> **(b)** Map a long digit sequence $(d_1,d_2,\dots)$ to $0.d_1d_2\dots$. The convention that a representation does not end in all zeros selects a unique decimal expansion for every real number in $(0,1]$: use the trailing-$9$ representation for terminating decimals, including $1=0.999\dots$. The number $0$ has no long representation.
> **(c)** On inputs whose odd-position and even-position subsequences are both long, define
> $$F(d_1d_2d_3d_4\dots)=((d_1,d_3,\dots),(d_2,d_4,\dots)).$$
> This is a partial function on all digit strings, but for any pair of long sequences, interleaving their digits produces a preimage. Thus $F:L\twoheadrightarrow L^2$.
> **(d)** If $f:A\to B$ is a bijection, then $(a_1,a_2)\mapsto(f(a_1),f(a_2))$ is also a bijection, with $f^{-1}$ applied coordinatewise as its inverse. Part (b) therefore gives $L^2\leftrightarrow(0,1]^2$.
> **(e)** Combining (b), (c), and (d) gives a surjection $(0,1]\twoheadrightarrow(0,1]^2$. Projection $(x,y)\mapsto x$ is a surjection in the other direction. The surjection form of Schröder–Bernstein therefore gives a bijection $(0,1]\leftrightarrow(0,1]^2$.
> **(f)** Compose the bijection from part (a) with each coordinate of the preceding bijection to obtain
> $$(0,1]\leftrightarrow(0,1]^2\leftrightarrow[0,\infty)^2.$$
> <!-- bilingual-en:end -->

---

## Unit 1 总结：怎样独立完成一份证明
<!-- bilingual-en:start -->
*Unit 1 Summary: How to Complete a Proof Independently*
<!-- bilingual-en:end -->

拿到新题时按以下顺序，不要先堆公式：
<!-- bilingual-en:start -->
When approaching a new problem, use the following order instead of immediately piling up formulas:
<!-- bilingual-en:end -->

1. 写论域、自由变量与量词；
2. 把目标化成 $P\Rightarrow Q$、集合相等、全称命题、不可达性或等势问题；
3. 按目标形状选择 direct / contrapositive / contradiction / cases / induction / invariant / bijection；
4. 把方法所需对象写出来：反例集合、归纳谓词、状态与转移、不变量、映射及其逆；
5. 单独检查零、空集、最小规模、定义域、除法分母、基例数量和终止性；
6. 用一个小实例或反例逆向审查结论强度。
<!-- bilingual-en:start -->
1. State the universe of discourse, free variables, and quantifiers.
2. Recast the goal as an implication $P\Rightarrow Q$, a set equality, a universal statement, an unreachability claim, or an equinumerosity problem.
3. Choose a method that matches the goal: direct proof, contrapositive, contradiction, cases, induction, invariant, or bijection.
4. Write down the objects that method requires: a counterexample set, induction predicate, states and transitions, invariant, or a map and its inverse.
5. Check zero, the empty set, the smallest case, the domain, possible zero denominators, the required base cases, and termination separately.
6. Test the strength of the conclusion against a small example or counterexample.
<!-- bilingual-en:end -->

### 覆盖核对
<!-- bilingual-en:start -->
*coverage check*
<!-- bilingual-en:end -->

| 来源 | 已覆盖数量 | 位置 |
|---|---:|---|
| Official online feedback prompts | 122 | 各 Session “官方在线 feedback exercises” |
| In-Class main problems | 48 | CP1–CP11 折叠题解 |
| Problem Set main problems | 13 | PS1–PS4 |
| Midterm 1 main problems | 5 | Midterm 1 |
| Session self-checks | 33 | 每节 3 道 |
<!-- bilingual-en:start -->
|Source|Overridden Quantity|Location|
|—|—:|—|
| Official online feedback prompts | 122 | each Session "Official Online feedback exercises" |
| In-Class main problems | 48 | CP1-CP89% discount stack |
| Problem Set main problems | 13 | PS1-PS4 |
| Midterm 1 main problems | 5 | Midterm 1 |
| Session self-checks | 33 | 3 tracks per section |
<!-- bilingual-en:end -->

下一单元从证明语言转入可计算结构：[[02_Structures|Unit 2: Structures]]。
<!-- bilingual-en:start -->
The next unit moves from the proof language to the computable structure: [[02_Structures|Unit 2: Structures]].
<!-- bilingual-en:end -->

## 课程笔记反链
<!-- bilingual-en:start -->
*course note inverse chain*
<!-- bilingual-en:end -->

```dataview
LIST
FROM "01_Math/07-Mathematics for Computer Science"
WHERE contains(file.outlinks, this.file.link)
SORT file.name ASC
```
