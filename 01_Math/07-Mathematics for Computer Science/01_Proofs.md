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
*What this unit is designed to answer*
<!-- bilingual-en:end -->

离散数学中的结论通常不是靠连续图像或数值近似得到，而是靠有限的逻辑链条。一个可检查的证明必须同时回答：
<!-- bilingual-en:start -->
Conclusions in discrete mathematics usually come from finite chains of logical reasoning rather than continuous graphs or numerical approximation. A checkable proof must answer all of the following questions:
<!-- bilingual-en:end -->

1. **对象是什么？** 变量的论域（domain）与符号是否明确？
2. **已知什么？** 哪些是定义、假设、先前定理或公理？
3. **要证什么？** 结论的量词与逻辑形状是什么？
4. **为什么每一步成立？** 是否误用了逆命题、除以零、遗漏情形或循环论证？
5. **边界是否覆盖？** 空集、零、最小规模、非终止过程与无限集合常常是漏洞所在。
<!-- bilingual-en:start -->

&nbsp;
**1.** **What are the objects?** Are the variables, their domains, and the notation clearly specified?<br>
**2.** **What is already known?** Which statements are definitions, assumptions, earlier theorems, or axioms?<br>
**3.** **What must be proved?** What are the quantifiers and logical form of the conclusion?<br>
**4.** **Why is each step valid?** Does it misuse the converse, divide by zero, omit a case, or argue circularly?<br>
**5.** **Have all boundary cases been covered?** Empty sets, zero, the smallest possible size, nonterminating processes, and infinite sets are common sources of gaps.<br>
<!-- bilingual-en:end -->

## 导航

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

## 官方 courseware block 顺序

正文会合并重复解释，但以下新增知识严格按官方页面出现次序吸收；在线题仍在对应 Session 的折叠区逐项给出答案。
<!-- bilingual-en:start -->
The main text consolidates repeated explanations, but introduces new material in the order used on the official courseware pages. Online exercises remain answered item by item in the collapsible section for the corresponding session.
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

## 证明方法总图
<!-- bilingual-en:start -->
*Proof-method map*
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-proof-method-map.png|900]]

读图：先按目标命题的逻辑形状选择入口，再沿分支检查该方法必须建立的假设与最容易遗漏的条件。
<!-- bilingual-en:start -->
How to read the map: begin with the logical form of the statement you want to prove, then follow the relevant branch to check the assumptions that method requires and the conditions most often overlooked.
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
| Logical form of the target | Preferred structure | What the proof must establish first | Most common failure mode |
|---|---|---|---|
| $P\Rightarrow Q$ | [[数学证明方法#命题、量词与否定|direct proof]] | Assume $P$ | Silently assumes $Q$; proves the converse instead |
| $P\Rightarrow Q$, and $\neg Q$ is easier to use | [[数学证明方法#直接证明、逆否与反证|proof by contrapositive]] | Assume $\neg Q$ | Writes $\neg P\Rightarrow\neg Q$ instead |
| A nonexistence claim, or an assumption that should lead to a conflict | [[数学证明方法#直接证明、逆否与反证|proof by contradiction]] | Assume the target statement is false | The contradiction does not actually depend on that assumption |
| The domain naturally splits into exhaustive cases | [[数学证明方法#直接证明、逆否与反证|proof by cases]] | List cases covering every possibility | Omits $0$, the empty set, or a boundary case |
| For each $n\in\mathbb N$ | [[数学证明方法#归纳法|mathematical induction]] | Proposition $P(n)$ and base case | Inductive step not connected to the next integer |
| $P(n)$ relies on multiple smaller cases | [[数学证明方法#归纳法|strong induction]] | All $P(k),k<n$ | Too few base cases to start the recurrence |
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
*1.1 Propositions, predicates, and proofs*
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|命题（proposition）]]是一个具有确定真值的陈述；它要么真，要么假。例如“$2+3=5$”是命题，“$x+3=5$”在没有指定 $x$ 时不是命题。
<!-- bilingual-en:start -->
A [[数学证明方法#命题、量词与否定|proposition]] is a statement with a definite truth value: it is either true or false. For example, "$2+3=5$" is a proposition, whereas "$x+3=5$" is not a proposition unless $x$ is specified.
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|谓词（predicate）]] $P(x)$ 是含自由变量的真假条件。给定论域并代入 $x$，或给变量加量词后，它才成为命题。例如
<!-- bilingual-en:start -->
A [[数学证明方法#命题、量词与否定|predicate]] $P(x)$ is a truth-valued condition containing one or more free variables. Once a domain has been specified, substituting a value for $x$ or binding $x$ with a quantifier turns it into a proposition. For example,
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
An [[数学证明方法#命题、量词与否定|axiom]] is a proposition accepted as a starting point within a formal system; a [[数学证明方法#命题、量词与否定|theorem]] is an important proposition that has been proved; a [[数学证明方法#命题、量词与否定|lemma]] is a supporting result used to prove a later theorem; and a [[数学证明方法#命题、量词与否定|corollary]] follows from an established theorem in only a few steps.
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
How to read the table: focus on the single cell in which $P$ is true and $Q$ is false. None of the other three truth-value combinations is a counterexample to the implication.
<!-- bilingual-en:end -->

最基本的推理规则是肯定前件（Modus Ponens）：
<!-- bilingual-en:start -->
The most basic rule of inference is modus ponens:
<!-- bilingual-en:end -->

$$
\frac{P,\qquad P\Rightarrow Q}{Q}.
$$

这不是代数分式，而是“已知横线上方两项，即可推出横线下方结论”。相反，由 $P\Rightarrow Q$ 和 $Q$ 推出 $P$ 是**肯定后件谬误**。
<!-- bilingual-en:start -->
This is not an algebraic fraction. It says that the two premises above the inference bar justify the conclusion below it. By contrast, inferring $P$ from $P\Rightarrow Q$ and $Q$ is the **fallacy of affirming the consequent**.
<!-- bilingual-en:end -->

### 1.3 直接证明的标准结构
<!-- bilingual-en:start -->
*1.3 The standard structure of a direct proof*
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

&nbsp;
**1.** Explicitly state, “Assume $P$.”<br>
**2.** Reason using only that assumption, definitions, and established results.<br>
**3.** Derive $Q$.<br>
**4.** End with $\square$ or QED.<br>
<!-- bilingual-en:end -->

**示例：若整数 $n$ 为偶数，则 $n^2$ 为偶数。**
<!-- bilingual-en:start -->
**Example: If the integer $n$ is even, then $n^2$ is even.**
<!-- bilingual-en:end -->

目标是 $P(n)\Rightarrow Q(n)$，其中 $P(n)$ 为“$n$ 偶”，$Q(n)$ 为“$n^2$ 偶”。假设 $n$ 偶，按定义存在 $k\in\mathbb Z$ 使 $n=2k$。于是
<!-- bilingual-en:start -->
The target has the form $P(n)\Rightarrow Q(n)$, where $P(n)$ means “$n$ is even” and $Q(n)$ means “$n^2$ is even.” Assume that $n$ is even. By definition, there is some $k\in\mathbb Z$ such that $n=2k$. Therefore,
<!-- bilingual-en:end -->

$$
n^2=(2k)^2=4k^2=2(2k^2).
$$

因 $2k^2\in\mathbb Z$，所以 $n^2$ 可写成 $2$ 乘某整数，故为偶数。$\square$
<!-- bilingual-en:start -->
Because $2k^2\in\mathbb Z$, $n^2$ is twice an integer and is therefore even. $\square$
<!-- bilingual-en:end -->

### 1.4 先找“第一处错误”
<!-- bilingual-en:start -->
*1.4 Find the first error*
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
>
> &nbsp;
> **1.** Why is it invalid to begin a direct proof of “$n^2$ even $\Rightarrow n$ even” by writing $n=2k$?<br>
> **2.** Explain why “the first 1000 primes are odd” is false, and give a counterexample.<br>
> **3.** Is the step from $x=y$ to $x^2=xy$ reversible? Under what additional condition?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. $n=2k$ 正是待证结论，等于预设结论；应证逆否“$n$ 奇则 $n^2$ 奇”或用素因子性质。2. 素数 $2$ 为偶数。3. 正向恒可；反向由 $x(x-y)=0$ 只能得到 $x=0$ 或 $x=y$，若另知 $x\ne0$ 才可推出 $x=y$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Writing $n=2k$ assumes exactly the conclusion to be proved. Instead prove the contrapositive, “$n$ odd implies $n^2$ odd,” or use prime factorisation.<br>
> **2.** The prime number $2$ is even.<br>
> **3.** The forward implication always holds. In reverse, $x(x-y)=0$ gives only $x=0$ or $x=y$; the additional assumption $x\ne0$ is needed to conclude $x=y$.<br>
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
*2.1 Proof by contraposition*
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
A [[数学证明方法#直接证明、逆否与反证|proof by cases]] divides the domain into cases $C_1,\dots,C_k$ that cover every possibility and proves $C_i\Rightarrow T$ for each case. The cases need not be mutually exclusive, but they must be exhaustive:
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
>
> &nbsp;
> **1.** The contradiction proof for $\sqrt2$ uses unique prime factorisation, equivalently the fact that a product of odd integers is odd.<br>
> **2.** Proof by cases is appropriate when the domain can be split into exhaustive cases that are easier to prove separately; the mere failure of a direct proof is not sufficient.<br>
> **3.** The friends-and-strangers argument proves $R(3,3)\le6$.<br>
> **4.** The bogus split “every integer is positive or negative” omits $0$.<br>
> **5.** The bogus proof about $\sqrt4$ twice misuses $4\mid n^2\Rightarrow4\mid n$; the valid conclusion is only $2\mid n$.<br>
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
> **4. A constructive example.** Let $a=\sqrt2$ and $b=2\log_2 3$. If $b=r/s\in\mathbb Q$ with $s>0$, then $2^{r/s}=3^2=9$, so $2^r=9^s=3^{2s}$. This is impossible by unique prime factorisation: the left-hand side has only the prime factor $2$, whereas the right-hand side has only the prime factor $3$. Thus $b$ is irrational, while $a^b=2^{b/2}=2^{\log_2 3}=3$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 反证法与逆否证明的假设分别是什么？
> 2. “所有实数非负或非正”是否构成合法分类？
> 3. 为何由 $p\mid n^2$ 推 $p\mid n$ 要求 $p$ 为素数？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What assumptions do proof by contradiction and proof by contrapositive make, respectively?<br>
> **2.** Do the cases “nonnegative” and “nonpositive” form a valid case split for all real numbers?<br>
> **3.** Why does the implication $p\mid n^2\Rightarrow p\mid n$ require $p$ to be prime?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 反证假设整个目标为假；逆否证明在 $P\Rightarrow Q$ 中假设 $\neg Q$ 并证 $\neg P$。2. 合法且覆盖全部实数，$0$ 同时落入两类不妨碍证明。3. 合数反例：$4\mid 2^2$，但 $4\nmid2$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Proof by contradiction assumes that the target statement is false; proof by contrapositive assumes $\neg Q$ and proves $\neg P$ when the target is $P\Rightarrow Q$.<br>
> **2.** Yes. The cases cover all real numbers, and their overlap at $0$ does not invalidate the proof.<br>
> **3.** A composite counterexample is $4\mid2^2$ but $4\nmid2$.<br>
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
The [[数学证明方法#良序与最小反例|Well-Ordering Principle (WOP)]] states that every nonempty set of nonnegative integers $S\subseteq\mathbb N$ has a least element.
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

&nbsp;
**1.** Define the counterexample set $C=\{n\in\mathbb N:\neg P(n)\}$;<br>
**2.** Suppose for contradiction that $C\ne\varnothing$;<br>
**3.** WOP gives $m=\min C$;<br>
**4.** Use the properties of $m$ to construct $m'<m$, and verify carefully that $m'\in C$;<br>
**5.** This contradicts the minimality of $m$, so $C=\varnothing$.<br>
<!-- bilingual-en:end -->

“$m'<m$”本身不够：若 $m'$ 为负数、未满足问题定义域、或根本不是反例，就没有矛盾。
<!-- bilingual-en:start -->
The inequality $m'<m$ is not enough by itself. There is no contradiction if $m'$ is negative, lies outside the problem's domain, or is not actually a counterexample.
<!-- bilingual-en:end -->

### 3.2 每个大于 1 的整数都是素数乘积
<!-- bilingual-en:start -->
*3.2 Every integer greater than 1 is a product of primes*
<!-- bilingual-en:end -->

令 $C$ 为不能写成素数乘积的、$>1$ 的整数集合。若非空，取最小 $m$。$m$ 不可能是素数，否则它本身就是一个素数乘积。所以 $m=ab$，其中 $1<a,b<m$。由 $m$ 的最小性，$a,b$ 都能写成素数乘积；相乘便给出 $m$ 的素数乘积分解，与 $m\in C$ 矛盾。
<!-- bilingual-en:start -->
Let $C$ be the set of integers greater than $1$ that cannot be written as a product of primes. Suppose $C$ is nonempty and let $m$ be its least element. The number $m$ cannot itself be prime, because then it would already be a product of primes. Hence $m=ab$ for some $1<a,b<m$. By the minimality of $m$, both $a$ and $b$ can be written as products of primes. Multiplying those factorisations gives a prime factorisation of $m$, contradicting $m\in C$.
<!-- bilingual-en:end -->

这段证明的下降量是整数大小；严格边界 $1<a,b<m$ 保证可调用最小性。
<!-- bilingual-en:start -->
The descent measure in this proof is the integer's value. The strict bounds $1<a,b<m$ are what allow the minimality of $m$ to be applied to both factors.
<!-- bilingual-en:end -->

### 3.3 几何级数公式
<!-- bilingual-en:start -->
*3.3 The geometric-series formula*
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
If a counterexample exists, let $m$ be the least one. We have $m\ne0$, because for $m=0$ both sides equal $1$. Hence $m-1\ge0$, and by the minimality of $m$, $m-1$ is not a counterexample:
<!-- bilingual-en:end -->

$$
1+\cdots+r^{m-1}=\frac{r^m-1}{r-1}.
$$

两边加 $r^m$：
<!-- bilingual-en:start -->
Add $r^m$ to both sides:
<!-- bilingual-en:end -->

$$
1+\cdots+r^m
=\frac{r^m-1+r^m(r-1)}{r-1}
=\frac{r^{m+1}-1}{r-1},
$$

所以 $m$ 也不是反例，矛盾。条件 $r\ne1$ 是因分母；$r=1$ 时和为 $n+1$，需单独处理。
<!-- bilingual-en:start -->
Thus $m$ is not a counterexample after all, a contradiction. The condition $r\ne1$ is needed because the formula divides by $r-1$. When $r=1$, the sum is $n+1$ and must be handled separately.
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
>
> &nbsp;
> **1.** WOP is applied to subsets of the nonnegative integers.<br>
> **2.** The prime-factorisation argument ultimately proves that the counterexample set is empty.<br>
> **3.** In the geometric proof, the contradiction is that the least counterexample $m$ also satisfies the formula.<br>
> **4.** Well-ordered examples include any subset of the integers that is bounded below, any finite set, positive rationals whose denominators have a common finite upper bound, $\{n/(n+1)\}$, $\mathbb N\cup\{n/(n+1)\}$, and any subset of the reals with only finitely many elements below each member. By contrast, $\{1/n:n\ge1\}$ and $\mathbb Q\cap[\sqrt2,\infty)$ have no least element.<br>
> **5.** The bogus WOP proof that all Fibonacci numbers are even overlooks $m=1$.<br>
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
>
> &nbsp;
> **1.** Why can WOP not be applied directly to the real interval $(0,1)$?<br>
> **2.** In a least-counterexample proof, what exactly does the minimality of $m$ allow you to conclude?<br>
> **3.** When proving $2^n\ge n+1$, what must be checked before descending from $m$ to $m-1$?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 它非空但没有最小元素。2. 只能对同一反例集合中所有严格小于 $m$ 的合法对象断言它们不是反例。3. 先排除 $m=0$，确保 $m-1\in\mathbb N$；再验证由 $P(m-1)$ 可推出 $P(m)$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The interval is nonempty but has no least element.<br>
> **2.** Every valid object in the same domain that is strictly smaller than $m$ is not a counterexample.<br>
> **3.** First exclude $m=0$ so that $m-1\in\mathbb N$; then verify that $P(m-1)$ implies $P(m)$.<br>
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
How can natural-language specifications be translated into formulas that truth tables can check? How do we distinguish a formula that is true under one assignment, one that is true under every assignment, and a collection of formulas that can all be true simultaneously?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session4.pdf|Session 4 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp4.pdf|CP4]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/0exBzsexUoI.pdf|Propositional Operators]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/eMWG-jTh-GE.pdf|Digital Logic]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/3WDzxt5p8c.pdf|Truth Tables]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_PropositOper.pdf|Operators]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_DigitalLogic.pdf|Digital Logic]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_TruthTables.pdf|Truth Tables]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Implies.pdf|Implies]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_PropositLogic.pdf|Propositional Logic]]

### 4.1 逻辑运算符
<!-- bilingual-en:start -->
*4.1 Logical operators*
<!-- bilingual-en:end -->

[[数学证明方法#命题、量词与否定|命题逻辑]]的基本运算：
<!-- bilingual-en:start -->
The basic operators of [[数学证明方法#命题、量词与否定|propositional logic]] are:
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
| Formula | When it is true | Common equivalent form |
|---|---|---|
| $\neg P$ | $P$ is false | — |
| $P\land Q$ | Both are true | — |
| $P\lor Q$ | At least one true | inclusive OR |
| $P\oplus Q$ | Exactly one true | $(P\lor Q)\land\neg(P\land Q)$ |
| $P\Rightarrow Q$ | Except for $P=T,Q=F$ | $\neg P\lor Q$ |
| $P\Leftrightarrow Q$ | They have the same truth value | $(P\Rightarrow Q)\land(Q\Rightarrow P)$ |
<!-- bilingual-en:end -->

自然语言“$P$ only if $Q$”是 $P\Rightarrow Q$；“$P$ if $Q$”是 $Q\Rightarrow P$；“$P$ iff $Q$”才是双向蕴含。
<!-- bilingual-en:start -->
In natural language, “$P$ only if $Q$” means $P\Rightarrow Q$, “$P$ if $Q$” means $Q\Rightarrow P$, and “$P$ iff $Q$” expresses a biconditional.
<!-- bilingual-en:end -->

### 4.2 真值表、有效、可满足与等价
<!-- bilingual-en:start -->
*4.2 Truth tables, validity, satisfiability, and equivalence*
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
A formula $P$ is valid if and only if $\neg P$ is unsatisfiable. To test whether the inference
<!-- bilingual-en:end -->

$$
P_1,\dots,P_k\vdash Q
$$

是否 sound，只需检查公式 $(P_1\land\cdots\land P_k)\Rightarrow Q$ 是否 valid。
<!-- bilingual-en:start -->
is sound, check whether $(P_1\land\cdots\land P_k)\Rightarrow Q$ is valid.
<!-- bilingual-en:end -->

### 4.3 De Morgan 与分配律
<!-- bilingual-en:start -->
*4.3 De Morgan's laws and distributivity*
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
Each identity can be proved with a truth table. Merely “looking algebraically similar” is not a proof: logical equivalence requires the final columns to match in every row.
<!-- bilingual-en:end -->

### 4.4 数字电路：逻辑不是抽象装饰
<!-- bilingual-en:start -->
*4.4 Digital circuits: logic is not abstract decoration*
<!-- bilingual-en:end -->

用 $T/F$ 表示 $1/0$。一位全加器输入 $a_i,b_i,c_i$，输出
<!-- bilingual-en:start -->
Represent $1$ and $0$ by $T$ and $F$. A one-bit full adder takes $a_i,b_i,c_i$ as input and outputs
<!-- bilingual-en:end -->

$$
s_i=a_i\oplus b_i\oplus c_i,
$$

进位为“至少两个输入为 1”：
<!-- bilingual-en:start -->
The carry bit is $1$ when at least two inputs are $1$:
<!-- bilingual-en:end -->

$$
c_{i+1}=(a_i\land b_i)\lor(a_i\land c_i)\lor(b_i\land c_i).
$$

串行 ripple-carry 的关键路径为 $\Theta(n)$；后续作业的并行结构用预计算降低深度，这正是“公式等价但电路代价不同”的例子。
<!-- bilingual-en:start -->
A serial ripple-carry adder has critical-path depth $\Theta(n)$. A later problem set uses precomputation to reduce the depth of a parallel design, illustrating that logically equivalent formulas can have different circuit costs.
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
> **Operators 1–4:** Evaluate each expression under the given assignment. For $(T,T)$, the true expressions are AND, OR, and $P\lor\neg Q$; for $(F,F)$, they are $\neg(P\land Q),\neg(P\lor Q),\neg P,\neg Q,P\lor\neg Q$; for $(T,F)$, they are XOR, OR, $\neg(P\land Q),\neg Q,P\lor\neg Q$; and for $(F,T)$, they are XOR, OR, $\neg(P\land Q),\neg P,\neg P\land Q$.
> **Equivalence:** Compare the final columns of the truth tables. **Soundness:** Whenever all premises are true, the conclusion must also be true; equivalently, the corresponding implication is valid. **Connectives 1–4:** $P\Rightarrow Q\equiv\neg P\lor Q$; IFF $\equiv(P\land Q)\lor(\neg P\land\neg Q)$; XOR $\equiv(P\land\neg Q)\lor(\neg P\land Q)$; NOR $\equiv\neg P\land\neg Q$.
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
> **2. File-system specifications.** The formulas are $\neg L\Rightarrow Q$, $\neg L\Rightarrow B$, $\neg L\Leftrightarrow N$, $\neg Q\Rightarrow B$, and $\neg B$. From $\neg B$ and the second formula we obtain $L$; the biconditional then requires $N=F$. Under $B=F$, the fourth formula requires $Q=T$. The unique assignment is $(L,Q,B,N)=(T,T,F,F)$, and it satisfies every formula.
> **3. Binary adders.** To add one bit $b$, set $c_0=b$, $s_i=a_i\oplus c_i$, $c_{i+1}=a_i\land c_i$, and finally $c=c_{n+1}$. To add two numbers, use $s_i=a_i\oplus b_i\oplus c_i$ and $c_{i+1}=(a_i\land b_i)\lor(c_i\land(a_i\oplus b_i))$. Each bit can be implemented with two XOR gates, two AND gates, and one OR gate, giving $2(n+1)$ XOR, $2(n+1)$ AND, and $n+1$ OR gates for $n+1$ bits. If the three-term majority formula is used instead, the count depends on the chosen implementation and must be stated.
> **4. Mathematicians and mothers.** The mathematical statement expresses only a necessary condition: a function may fail to be differentiable for other reasons, so the correct formalisation is merely $D\Rightarrow C$. In ordinary conversation, a mother's rule can convey both permission and prohibition—homework is treated as both necessary and sufficient for watching television—so $H\Leftrightarrow T$ is a reasonable model. Natural-language “if–then” statements must be interpreted in context.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 写出 $P\Rightarrow(Q\land R)$ 的无蕴含形式。
> 2. satisfiable 是否推出 valid？给反例。
> 3. 为什么“前件假”的蕴含定义为真？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Rewrite $P\Rightarrow(Q\land R)$ without using implication.<br>
> **2.** Does satisfiability imply validity? Give a counterexample.<br>
> **3.** Why is an implication with a false premise defined to be true?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. $\neg P\lor(Q\land R)$。2. 不推出；$P$ 在环境 $P=T$ 下可满足，但不是恒真。3. 蕴含只禁止“承诺条件出现而结论失败”的环境；当前件未发生时没有反例。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\neg P\lor(Q\land R)$.<br>
> **2.** No. The proposition $P$ is satisfiable in a valuation with $P=T$, but it is not valid.<br>
> **3.** An implication rules out only valuations in which its premise is true and its conclusion false; a false premise therefore creates no counterexample.<br>
> <!-- bilingual-en:end -->

**知识链：**逻辑运算 → 真值环境 → 等价/有效/可满足 → 规格一致性 → 数字电路。
<!-- bilingual-en:start -->
**Knowledge chain:** logical operators → truth assignments → equivalence, validity, and satisfiability → consistency of specifications → digital circuits.
<!-- bilingual-en:end -->

---

## Problem Set 1 — Sessions 1–4

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps1.pdf|Problem Set 1 PDF]]。以下 4 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps1.pdf|Problem Set 1 PDF]]. The four solutions below are independent and unofficial.
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
> **Given:** $4,6>1$. **Goal:** $\log_4 6\notin\mathbb Q$.
> Suppose for contradiction that $\log_4 6=m/n$, where $m,n\in\mathbb Z$ and $n>0$. Because the logarithm is positive, we may also take $m>0$. By definition,
> $$4^{m/n}=6\quad\Longrightarrow\quad4^m=6^n.$$
> The prime factorisation on the left is $2^{2m}$, whereas the right-hand side is $2^n3^n$. By uniqueness of prime factorisation, the exponent of $3$ would have to be $0$, forcing $n=0$ and contradicting $n>0$. Therefore $\log_4 6$ is irrational. $\square$
> **Check:** It is not enough to say that $4^m$ and $6^n$ are “obviously different”; the contradiction comes from the prime factor $3$.
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
> **Proof using WOP.** The goal is to prove the inequality for every $n\in\mathbb N$. Let
> $$C=\{n\in\mathbb N:n>3^{n/3}\}.$$
> Suppose for contradiction that $C\ne\varnothing$, and let $m=\min C$. Direct calculation gives
> $$0\le1,\qquad1\le3^{1/3},\qquad2\le3^{2/3},\qquad3=3,\qquad4<3^{4/3},$$
> Hence $m\ge5$, so $m-3\in\mathbb N$ and $m-3<m$. By the minimality of $m$,
> $$m-3\le3^{(m-3)/3}.$$
> Multiplying by $3$ gives $3(m-3)\le3^{m/3}$. Since $m\ge5$, we also have $m\le3m-9=3(m-3)$. Therefore $m\le3^{m/3}$, contradicting $m\in C$. Thus $C=\varnothing$.
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
> **(a)** Truth table:
> | $P$ | $Q$ | $P\Rightarrow Q$ | $Q\Rightarrow P$ | Their disjunction |
> |---|---|---|---|---|
> | T | T | T | T | T |
> | T | F | F | T | T |
> | F | T | T | F | T |
> | F | F | T | T | T |
> The final column is true in every row, so the formula is valid.
> **(b)** Let
> $$R=(P\land Q)\lor(\neg P\land\neg Q).$$
> The formula is true exactly when $P$ and $Q$ have the same truth value. Hence $R$ is valid if and only if $P$ and $Q$ agree under every assignment, which is precisely logical equivalence.
> **(c)** $P$ is valid exactly when no assignment makes $P$ false. Equivalently, no assignment makes $\neg P$ true, so $\neg P$ is unsatisfiable.
> **(d)** $P_1,\dots,P_k$ are inconsistent if and only if $P_1\land\cdots\land P_k$ is unsatisfiable, which holds if and only if
> $$S=\neg(P_1\land\cdots\land P_k)$$
> is valid.
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
> **(a)** A one-bit `add1` circuit takes $a_0$ as input and outputs the two-bit representation of $a_0+1$:
> $$p_0=\neg a_0,\qquad c=a_0.$$
> **(b)** If the control bit is $b=0$, the half-adder should output $a_i$ unchanged; if $b=1$, it should select the precomputed `add1` output $p_i$:
> $$o_i=(\neg b\land a_i)\lor(b\land p_i).$$
> **(c)** The high half needs to be incremented only when the low half overflows, so $c^{(1)}=1$. The full doubled module overflows only when the precomputed high half also overflows, hence
> $$c=c^{(1)}\land c^{(2)}.$$
> **(d)** For $n+1\le i\le2n+1$, let $j=i-(n+1)$:
> $$p_i=(\neg c^{(1)}\land a_i)\lor(c^{(1)}\land r_j).$$
> **(e)** Let $D(N)$ be the maximum logical depth of any output of the $N$-bit `add1` circuit. The doubling construction adds only a constant number of AND/OR/NOT layers after each submodule, so
> $$D(2N)\le D(N)+O(1),$$
> which solves to $D(N)=O(\log N)$. The highest bit of a ripple-carry circuit must wait for $N$ carry propagations, giving depth $\Theta(N)$; the parallel construction therefore gives an exponential improvement in circuit depth. The total gate count remains polynomial. The comparison concerns the **longest dependency chain**, not an unconditional guarantee about physical running time.
> <!-- bilingual-en:end -->

---

## Session 5 — Quantifiers and Predicate Logic

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

同一句“每个程序都有一个输入”与“有一个输入适用于每个程序”为什么完全不同？本节把自由变量、论域和量词写进公式。
<!-- bilingual-en:start -->
Why are “every program has some input” and “there is one input that works for every program” completely different statements? This section makes free variables, domains, and quantifiers explicit in formulas.
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session5.pdf|Session 5 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp5.pdf|CP5]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/UroprmQHTLc.pdf|Predicate Logic I]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/T1AtlGrCoU8.pdf|Predicate Logic II]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/L5uBeAGJV1k.pdf|Predicate Logic III]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate1.pdf|Predicate I]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate2.pdf|Predicate II]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Predicate3.pdf|Predicate III]]

### 5.1 论域、自由变量与绑定变量
<!-- bilingual-en:start -->
*5.1 Domain of discourse, free variables, and bound variables*
<!-- bilingual-en:end -->

[[01_Proofs#Session 5 — Quantifiers and Predicate Logic|谓词逻辑（predicate logic）]]允许对象变量、谓词和量词。$P(x,y)$ 中未被量词控制的变量是自由变量；在 $\forall x\exists y\,P(x,y)$ 中二者都被绑定，整个式子才是命题。
<!-- bilingual-en:start -->
[[01_Proofs#Session 5 — Quantifiers and Predicate Logic|Predicate logic]] allows object variables, predicates, and quantifiers. In $P(x,y)$, variables not governed by a quantifier are free. In $\forall x\exists y\,P(x,y)$, both variables are bound, so the entire formula is a proposition.
<!-- bilingual-en:end -->

论域必须明确。同一公式 $\exists x(x^2=2)$ 在 $\mathbb Q$ 中为假，在 $\mathbb R$ 中为真。若论域为空，$\forall xP(x)$ 真（没有反例），$\exists xP(x)$ 假；本课程通常默认非空论域，但证明时仍应知道这一边界。
<!-- bilingual-en:start -->
The domain must be explicit. The same formula $\exists x(x^2=2)$ is false over $\mathbb Q$ but true over $\mathbb R$. If the domain is empty, $\forall xP(x)$ is true because it has no counterexample, while $\exists xP(x)$ is false. This course usually assumes a nonempty domain, but a rigorous proof should still recognise this boundary case.
<!-- bilingual-en:end -->

### 5.2 量词的语义与顺序
<!-- bilingual-en:start -->
*5.2 The semantics and order of quantifiers*
<!-- bilingual-en:end -->

$$
\forall x\,P(x):\text{论域内每个 }x\text{ 都满足 }P,
$$

$$
\exists x\,P(x):\text{至少存在一个见证 }x\text{ 满足 }P.
$$

同类量词可交换：$\forall x\forall yP\equiv\forall y\forall xP$，$\exists x\exists yP\equiv\exists y\exists xP$。异类通常不可交换：
<!-- bilingual-en:start -->
Quantifiers of the same kind commute: $\forall x\forall yP\equiv\forall y\forall xP$ and $\exists x\exists yP\equiv\exists y\exists xP$. Quantifiers of different kinds generally do not:
<!-- bilingual-en:end -->

$$
\forall x\exists y\;(y>x)
$$

允许 $y$ 随 $x$ 改变；$\exists y\forall x(y>x)$ 要求一个统一的最大上界。在 $\mathbb N$ 中前者真、后者假。
<!-- bilingual-en:start -->
The first formula allows $y$ to depend on $x$; the second requires a single $y$ greater than every $x$. Over $\mathbb N$, the former is true and the latter is false.
<!-- bilingual-en:end -->

### 5.3 否定穿过量词
<!-- bilingual-en:start -->
*5.3 Negating quantifiers*
<!-- bilingual-en:end -->

De Morgan 规律推广为
<!-- bilingual-en:start -->
De Morgan's laws extend to quantifiers as follows:
<!-- bilingual-en:end -->

$$
\neg\forall xP(x)\equiv\exists x\neg P(x),
\qquad
\neg\exists xP(x)\equiv\forall x\neg P(x).
$$

因此，“并非每个算法都终止”是“存在一个算法不终止”，不是“每个算法都不终止”。否定多层量词时逐层交换 $\forall\leftrightarrow\exists$ 并否定最内层谓词。
<!-- bilingual-en:start -->
Therefore, “not every algorithm terminates” means “there exists an algorithm that does not terminate,” not “every algorithm fails to terminate.” To negate several nested quantifiers, swap $\forall\leftrightarrow\exists$ one layer at a time and negate the innermost predicate.
<!-- bilingual-en:end -->

### 5.4 可满足性与见证
<!-- bilingual-en:start -->
*5.4 Satisfiability and witnesses*
<!-- bilingual-en:end -->

谓词 $P(x)$ satisfiable 意味着 $\exists xP(x)$。证明存在命题最直接的方法是给出见证并验证；否定存在命题则必须对任意候选排除。
<!-- bilingual-en:start -->
A predicate $P(x)$ is satisfiable when $\exists xP(x)$ is true. The most direct way to prove an existential statement is to provide a witness and verify that it satisfies the predicate. To refute an existential statement, every possible candidate must be ruled out.
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
>
> &nbsp;
> **1.** Formulas that guarantee $P$ is satisfiable are $\forall xP(x)$, $\exists xP(x)$, $\neg\forall x\neg P(x)$, and $\neg\exists x\neg P(x)$. The last is equivalent to $\forall xP(x)$, and the domain is assumed nonempty.<br>
> **2.** $\neg\exists x\exists yQ(x,y)$ is equivalent to $\forall x\forall y\neg Q(x,y)$.<br>
> **3.** Valid formulas: $\exists x\exists yP\Rightarrow\exists y\exists xP$, $\exists x\forall yR\Rightarrow\forall y\exists xR$ and $\neg\exists xS\Leftrightarrow\forall x\neg S$.<br>
> **4.** Over $\mathbb N$, the true formulas among the five given are $\forall x\exists y(2x-y=0)$ and $\forall x[x<10\Rightarrow\forall y(y<x\Rightarrow y<9)]$.<br>
> **5.** Over $\mathbb Z$, those two remain true, and $\forall x\exists y[y>x\land\exists z(y+z=100)]$ is also true. The formula involving $x-2y=0$ still fails for odd $x$.<br>
> **6.** Over $\mathbb R$, the true formulas are $\forall x\exists y(2x-y=0)$, $\forall x\exists y(x-2y=0)$, and the preceding “find a larger $y$” formula. The course feedback uses $x=9.5$ as a counterexample to the formula with integer bounds.<br>
> **7.** The quantifier order for “each integer has a prime factor” is $\forall i\exists p$; it does not require one prime to divide every integer.<br>
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
> **1. Truth values across domains.**
> | Formula | $\mathbb N$ | $\mathbb Z$ | $\mathbb Q$ | $\mathbb R$ | $\mathbb C$ |
> |---|:---:|:---:|:---:|:---:|:---:|
> | $\exists x:x^2=2$ | F | F | F | T | T |
> | $\forall x\exists y:x^2=y$ | T | T | T | T | T |
> | $\forall y\exists x:x^2=y$ | F | F | F | F | T |
> | $\forall x\ne0\exists y:xy=1$ | F | F | T | T | T |
> | $\exists x,y:x+2y=2\land2x+4y=5$ | F | F | F | F | F |
> In the last row, the left-hand side of the second equation is always twice that of the first, so it would have to equal $4$, not $5$.
> **2. Translating binary-string statements.** Let `PREFIX`, `SUBSTRING`, and `NO-1S` be defined as in the problem:
> (a)$\exists y(x=yyy)$;
> (b)$\exists y(x=yy\land\mathrm{NO\mbox{-}1S}(y))$;
> (c)$\neg(\mathrm{SUBSTRING}(0,x)\land\mathrm{SUBSTRING}(1,x))$;
> (d)$x=10\lor\exists y(x=1y1\land\mathrm{NO\mbox{-}1S}(y))$;
> (e) If $x$ is an all-zero string, then clearly $x$ is a prefix of $0x$. Conversely, if $x$ contains a first 1, comparing the corresponding positions in $x$ and $0x$ propagates equality one place to the left until that first 1 is forced to equal 0, a contradiction. Thus, `PREFIX(x,0x)` describes exactly the all-zero strings, including the empty string.
> **3. Email at most two other classmates.**
> The variables $y$ and $z$ may be equal to each other or to $x$, so the formula also covers the cases of zero or one recipient.
> **4. Swapping quantifiers.** The first formula is invalid. On the domain $\{1,2\}$, let $P(x,y)\equiv(x\ne y)$. Every $x$ has some different $y$, but there is no single $y$ that differs from every $x$. The second formula is valid: if a fixed $y_0$ makes $P(x,y_0)$ true for every $x$, then each $x$ can use that same witness $y_0$.
> **5. Cabal.** (a) At least three people belong; (b) Siggi and Annie cannot both belong; (c) if Elizabeth belongs, everyone belongs; (d) if Annie belongs, Siggi belongs; (e) if Ben or Albert belongs, Tom does not; (f) if Ben or Siggi belongs, Adam does not. Conditions (b) and (c) imply that Elizabeth is absent, while (b) and (d) imply that Annie is absent. If Ben were absent, splitting on whether Siggi is present would in either case conflict with the requirement of at least three members together with (e) and (f). Thus Ben is present. Conditions (e) and (f) then exclude Tom and Adam; to reach three members, Albert and Siggi must both be present. The unique cabal is $\{\text{Ben, Albert, Siggi}\}$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 否定 $\forall x\exists yP(x,y)$。
> 2. 在 $\mathbb Z$ 中，$\forall x\exists y(x=2y)$ 真吗？
> 3. 证明 $\exists xP(x)$ 时，一个例子何时足够？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Negate $\forall x\exists yP(x,y)$.<br>
> **2.** Is $\forall x\exists y(x=2y)$ true over $\mathbb Z$?<br>
> **3.** When is an example sufficient to prove $\exists xP(x)$?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. $\exists x\forall y\neg P(x,y)$。2. 假；$x=1$ 没有整数见证。3. 当例子确属声明的论域并逐项满足谓词时足够。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** $\exists x\forall y\neg P(x,y)$.<br>
> **2.** No; when $x=1$, there is no integer witness $y$.<br>
> **3.** One example is sufficient when it lies in the stated domain and satisfies every condition in the predicate.<br>
> <!-- bilingual-en:end -->

**知识链：**谓词 → 论域 → 量词顺序 → 量词否定 → 见证/反模型。
<!-- bilingual-en:start -->
**Knowledge chain:** predicate → domain → quantifier order → negating quantifiers → witness or countermodel.
<!-- bilingual-en:end -->

---

## Session 6 — Sets

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

集合运算为什么与逻辑运算一一对应？怎样用逐元素双向蕴含证明集合相等？
<!-- bilingual-en:start -->
Why do set operations correspond directly to logical operators? How can elementwise implications in both directions prove that two sets are equal?
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
A [[01_Proofs#Session 6 — Sets|set]] is determined by its elements, and $x\in A$ denotes membership. The axiom of extensionality states that
<!-- bilingual-en:end -->

$$
A=B\quad\Longleftrightarrow\quad\forall x(x\in A\Leftrightarrow x\in B).
$$

子集 $A\subseteq B$ 意味着 $\forall x(x\in A\Rightarrow x\in B)$；真子集还要求 $A\ne B$。空集 $\varnothing$ 是每个集合的子集，因为不存在使蕴含失败的 $x\in\varnothing$。
<!-- bilingual-en:start -->
The relation $A\subseteq B$ means $\forall x(x\in A\Rightarrow x\in B)$; a proper subset additionally requires $A\ne B$. The empty set $\varnothing$ is a subset of every set because no $x\in\varnothing$ can make the implication fail.
<!-- bilingual-en:end -->

幂集 $\operatorname{pow}(A)=\{S:S\subseteq A\}$。若有限集 $|A|=n$，每个元素独立选择“进/不进”子集，所以 $|\operatorname{pow}(A)|=2^n$。
<!-- bilingual-en:start -->
The power set is $\operatorname{pow}(A)=\{S:S\subseteq A\}$. If $A$ is finite and $|A|=n$, each element can independently be included in or excluded from a subset, so $|\operatorname{pow}(A)|=2^n$.
<!-- bilingual-en:end -->

### 6.2 运算与逻辑对应
<!-- bilingual-en:start -->
*6.2 Set operations and their logical counterparts*
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-set-operations.png|900]]

读图：把每个阴影区域逐点翻译成成员谓词，交、并、差分别对应 AND、OR 与 AND-NOT。
<!-- bilingual-en:start -->
How to read the figure: translate each shaded region point by point into a membership predicate. Intersection, union, and difference correspond respectively to AND, OR, and AND-NOT.
<!-- bilingual-en:end -->

| 集合条件 | 元素谓词 | 逻辑运算 |
|---|---|---|
| $x\in A\cap B$ | $x\in A$ 且 $x\in B$ | AND |
| $x\in A\cup B$ | 至少属于一个 | OR |
| $x\in\overline A$ | $x\notin A$ | NOT |
| $x\in A-B$ | $x\in A$ 且 $x\notin B$ | $P\land\neg Q$ |
| $x\in A\triangle B$ | 恰属于一个 | XOR |
<!-- bilingual-en:start -->
| Set-membership condition | Element predicate | Logical operator |
|---|---|---|
| $x\in A\cap B$ | $x\in A$ and $x\in B$ | AND |
| $x\in A\cup B$ | belongs to at least one | OR |
| $x\in\overline A$ | $x\notin A$ | NOT |
| $x\in A-B$ | $x\in A$ and $x\notin B$ | $P\land\neg Q$ |
| $x\in A\triangle B$ | belongs to exactly one | XOR |
<!-- bilingual-en:end -->

所以逻辑等价可逐字搬到集合恒等式。例如
<!-- bilingual-en:start -->
Logical equivalences can therefore be translated directly into set identities. For example,
<!-- bilingual-en:end -->

$$
A-(B\cup C)=(A-B)\cap(A-C)
$$

来自 $P\land\neg(Q\lor R)\equiv(P\land\neg Q)\land(P\land\neg R)$。
<!-- bilingual-en:start -->
This identity follows from $P\land\neg(Q\lor R)\equiv(P\land\neg Q)\land(P\land\neg R)$.
<!-- bilingual-en:end -->

### 6.3 集合相等证明模板
<!-- bilingual-en:start -->
*6.3 Templates for proving set equality*
<!-- bilingual-en:end -->

任选任意 $x$，建立 IFF 链：
<!-- bilingual-en:start -->
Choose an arbitrary $x$ and build a chain of biconditionals:
<!-- bilingual-en:end -->

$$
x\in L
\Leftrightarrow\cdots
\Leftrightarrow x\in R.
$$

由于 $x$ 任意，外延性给出 $L=R$。另一种写法是分别证 $L\subseteq R$ 与 $R\subseteq L$；不能只画 Venn 图替代一般证明。
<!-- bilingual-en:start -->
Because $x$ was arbitrary, extensionality gives $L=R$. Alternatively, prove $L\subseteq R$ and $R\subseteq L$ separately. A Venn diagram alone cannot replace a general proof.
<!-- bilingual-en:end -->

### 6.4 有序对为什么不能用 $\{a,b\}$
<!-- bilingual-en:start -->
*6.4 Why an ordered pair cannot be represented by $\{a,b\}$*
<!-- bilingual-en:end -->

$\{a,b\}=\{b,a\}$，会丢失顺序。课程使用
<!-- bilingual-en:start -->
Because $\{a,b\}=\{b,a\}$, an ordinary two-element set loses the order. The course uses the encoding
<!-- bilingual-en:end -->

$$
\operatorname{pair}(a,b)=\{a,\{a,b\}\}.
$$

在所采用的集合论基础下，这个编码能恢复首、次项。关键不是必须记住某一种编码，而是知道“序列”可由集合构造，并且编码要保持顺序与重复项。
<!-- bilingual-en:start -->
Within the underlying set theory, this encoding lets us recover the first and second coordinates. The important point is not to memorise one particular encoding, but to understand that sequences can be constructed from sets and that the encoding must preserve both order and repeated entries.
<!-- bilingual-en:end -->

> [!question]- 官方在线 feedback exercise（1 prompt）
> **本地逐题入口：** [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S06_1.6.3_difference|1.6.3 Difference]]
>
> $A-B$ 唯一正确的谓词形式是 $\{x:(x\in A)\land(x\notin B)\}$；它不是交换运算，且必须同时保留“在 $A$ 中”和“不在 $B$ 中”。
> <!-- bilingual-en:start -->
> The correct predicate form for $A-B$ is $\{x:(x\in A)\land(x\notin B)\}$. Set difference is not commutative, and the definition must retain both conditions: membership in $A$ and nonmembership in $B$.
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
> **1. Decomposition identity.** The corresponding propositional identity is $(P\land\neg Q)\lor(P\land Q)\equiv P$. For an arbitrary $x$,
> $$x\in(A-B)\cup(A\cap B)\Leftrightarrow[(x\in A\land x\notin B)\lor(x\in A\land x\in B)]\Leftrightarrow x\in A.$$
> **2. Set-theoretic formulas using only $\in$.**
> (a)$x=\varnothing:\ \forall z(z\notin x)$;
> (b)$x=\{y,z\}:\ \forall w(w\in x\Leftrightarrow(w=y\lor w=z))$;
> (c)$x\subseteq y:\ \forall w(w\in x\Rightarrow w\in y)$;
> (d)$x=y\cup z:\ \forall w(w\in x\Leftrightarrow(w\in y\lor w\in z))$;
> (e) $x=y-z$: replace OR with $w\in y\land w\notin z$;
> (f)$x=\operatorname{pow}(y):\ \forall w(w\in x\Leftrightarrow w\subseteq y)$;
> (g)$x=\bigcup y:\ \forall w(w\in x\Leftrightarrow\exists z(z\in y\land w\in z))$.
> **3. Encoding ordered pairs.** The set $\{a,b\}$ is unchanged when its entries are swapped. The encoding $\{a,\{b\}\}$ can also be ambiguous when sets at different nesting levels happen to be equal. For $\{a,\{a,b\}\}$, if $a=b$, the encoding collapses to $\{a\}$ but still identifies the pair $(a,a)$. If $a\ne b$, then $a$ is the unique object that appears both as an outer element and inside the two-element inner set; the remaining object is $b$. The foundational axioms rule out spurious identifications caused by membership cycles.
> **4. Four-element subset take-away.** Let $A=\{1,2,3,4\}$ and split according to the size of the first player's opening move.
> - If the first player takes a one-element set or a three-element set, the second player takes its complement. Every subsequent legal choice then lies inside the remaining three-element ground set, reducing the game to the three-element losing position established in the problem, with the first player to move.
> - If the first player takes a two-element set, assume without loss of generality that it is $\{1,2\}$; the second player takes the complement $\{3,4\}$. The remaining legal sets are the four singletons and the four “cross pairs” $\{1,3\},\{1,4\},\{2,3\},\{2,4\}$. The second player responds according to the fixed pairing $\{1\}\leftrightarrow\{2\}$, $\{3\}\leftrightarrow\{4\}$, $\{1,3\}\leftrightarrow\{2,4\}$, and $\{1,4\}\leftrightarrow\{2,3\}$. The two sets in each pair are legal together under all earlier choices; after one is taken, its partner still contains none of the previously selected sets. Thus every move by the first player has a paired response, and the first player is eventually the first to have no legal move.
> These three possible opening sizes exhaust all nonempty proper subsets, so the second player wins the four-element game.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. $\varnothing\in A$ 与 $\varnothing\subseteq A$ 有何差别？
> 2. 证明 $A\cap(B\cup C)=(A\cap B)\cup(A\cap C)$。
> 3. $|\operatorname{pow}(\varnothing)|$ 是多少？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What is the difference between $\varnothing\in A$ and $\varnothing\subseteq A$?<br>
> **2.** Prove $A\cap(B\cup C)=(A\cap B)\cup(A\cap C)$.<br>
> **3.** What is $|\operatorname{pow}(\varnothing)|$?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 前者要求空集本身是 $A$ 的元素，不总成立；后者对每个 $A$ 都成立。2. 对任意 $x$，把成员条件化成 $P\land(Q\lor R)\equiv(P\land Q)\lor(P\land R)$。3. $1$，唯一子集是 $\varnothing$。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** The former says that the empty set itself is an element of $A$, which need not hold; the latter holds for every $A$.<br>
> **2.** For an arbitrary $x$, reduce the membership condition to $P\land(Q\lor R)\equiv(P\land Q)\lor(P\land R)$.<br>
> **3.** It is $1$; the only subset is $\varnothing$.<br>
> <!-- bilingual-en:end -->

**知识链：**成员谓词 → 逻辑运算 → 集合运算 → 外延性 → 数据类型编码。
<!-- bilingual-en:start -->
**Knowledge chain:** membership predicates → logical operators → set operations → extensionality → encoding data types.
<!-- bilingual-en:end -->

---

## Problem Set 2 — Sessions 5–6

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps2.pdf|Problem Set 2 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps2.pdf|Problem Set 2 PDF]]. The three solutions below are independent and unofficial.
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
> Define
> $$\mathrm{Members}(p,a,b):\Longleftrightarrow\forall x(x\in p\Leftrightarrow(x=a\lor x=b)).$$
> Then
> $$\mathrm{Pair}(p,a,b):\Longleftrightarrow\exists q(\mathrm{Members}(q,a,b)\land\mathrm{Members}(p,a,q)).$$
> Here $q=\{a,b\}$ and $p=\{a,q\}=\{a,\{a,b\}\}$. Finally,
> $$\mathrm{Second}(p,b):\Longleftrightarrow\exists a\,\mathrm{Pair}(p,a,b).$$
> Each auxiliary expression abbreviates a formula built only from membership predicates, quantifiers, and logical operators, so the final result is genuinely a formula of set theory.
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
> Complements are taken relative to the same universal set. Choose an arbitrary $x$:
> By extensionality, $\overline{A\cap B}=\overline A\cup\overline B$.
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
> Let $B=\{0,1\}^*$, with all complements taken relative to $B$.
> **(a)** $R\cap S=\overline{\overline R\cup\overline S}$, so the class of c-d languages is closed under intersection.
> **(b)** The language of strings that begin with $0$ and end with $1$ is $0B\cap B1$.
> **(c)** The language of all-zero strings is $0^*=\overline{B1B}$, because the right-hand side contains exactly the strings with no $1$, including the empty string.
> **(d)**
> A nonempty string begins with $0$, ends with $1$, and has no equal adjacent bits exactly when it is a repetition of $01$.
> **(e)** $(00)^*\cap0^*$ contains infinitely many even-length all-zero strings, while its complement intersects $0^*$ in infinitely many odd-length all-zero strings. Thus neither the language nor its complement is 0-finite, so it is not boring.
> **(f)** If both $R$ and $S$ are 0-finite, their union is still 0-finite. If at least one, say $R$, is co-0-finite, then $\overline{R\cup S}\cap0^*\subseteq\overline R\cap0^*$ is finite. Hence union preserves boring languages.
> **(g)** If both $R$ and $S$ are 0-finite, only finitely many pairs of their all-zero strings can be concatenated, so the product is 0-finite. If either language contains no all-zero string, neither does the product. In the remaining case, one language, say $R$, contains all but finitely many all-zero strings, while the other contains some fixed $0^k$. Every sufficiently long $0^n$ can then be written as $0^{n-k}0^k\in R\cdot S$. The complement of the product therefore omits only finitely many all-zero strings, so the product is still boring.
> **(h)** Every finite language is 0-finite. Parts (f) and (g) show that union and concatenation preserve the boring property, while complement merely swaps “the language is 0-finite” with “its complement is 0-finite.” Structural induction therefore proves that every c-d language is boring. Since $(00)^*$ from part (e) is not boring, it is not c-d.
> <!-- bilingual-en:end -->

---

## Session 7 — Binary Relations

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

函数只是关系的一种。怎样用“每个点有几条入边/出边”统一定义 total、function、injection、surjection 和 bijection？
<!-- bilingual-en:start -->
A function is only one kind of relation. How can totality, functionality, injectivity, surjectivity, and bijectivity all be defined by counting the incoming and outgoing edges at each point?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session7.pdf|Session 7 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp7.pdf|CP7]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/FkfsmwAtDdY.pdf|Relations]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/gFD1Lp6zK3w.pdf|Relational Mappings]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/fpy5Hsz5t6E.pdf|Finite Cardinality]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Relations.pdf|Relations]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RelationalMap.pdf|Mappings]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_FiniteCardi.pdf|Finite Cardinality]]

### 7.1 关系、像与逆像
<!-- bilingual-en:start -->
*7.1 Relations, images, and inverses*
<!-- bilingual-en:end -->

[[01_Proofs#Session 7 — Binary Relations|二元关系（binary relation）]] $R$ from $A$ to $B$ 是笛卡尔积 $A\times B$ 的子集。写 $aRb$ 表示 $(a,b)\in R$。$A$ 是声明的 domain，$B$ 是 codomain；真正有出边的元素集合与真正被命中的 range 不一定等于二者。
<!-- bilingual-en:start -->
A [[01_Proofs#Session 7 — Binary Relations|binary relation]] $R$ from $A$ to $B$ is a subset of the Cartesian product $A\times B$. Write $aRb$ for $(a,b)\in R$. Here $A$ is the declared domain and $B$ is the codomain; the subset of $A$ that actually has outgoing edges and the range actually reached in $B$ need not equal them.
<!-- bilingual-en:end -->

对 $X\subseteq A$，像为
<!-- bilingual-en:start -->
For $X\subseteq A$, its image is
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
*7.2 Defining mapping properties by counting arrows*
<!-- bilingual-en:end -->

课程把各性质用于一般关系，而不预设“函数必 total”：
<!-- bilingual-en:start -->
The course defines these properties for arbitrary relations, without assuming in advance that every function is total:
<!-- bilingual-en:end -->

| 性质 | 每个 domain 点出边 | 每个 codomain 点入边 |
|---|---:|---:|
| function | $\le1$ | 无限制 |
| total | $\ge1$ | 无限制 |
| injection | 无限制 | $\le1$ |
| surjection | 无限制 | $\ge1$ |
| bijection | $=1$ | $=1$ |
<!-- bilingual-en:start -->
| Property | Outgoing edges per domain point | Incoming edges per codomain point |
|---|---:|---:|
| function | $\le1$ | Unlimited|
| total | $\ge1$ | Unlimited|
| injection | Unlimited | $\le1$ |
| surjection | Unlimited | $\ge1$ |
| bijection | $=1$ | $=1$ |
<!-- bilingual-en:end -->

因此常规数学中的“函数 $f:A\to B$”对应这里的 **total function**。inverse 会交换 in/out：
<!-- bilingual-en:start -->
Thus what standard mathematical usage calls a “function $f:A\to B$” is a **total function** in this terminology. Taking the inverse swaps incoming and outgoing edges:
<!-- bilingual-en:end -->

$$
R\text{ total}\Leftrightarrow R^{-1}\text{ surjective},
\qquad
R\text{ function}\Leftrightarrow R^{-1}\text{ injective}.
$$

### 7.3 有限基数与 Mapping Rule
<!-- bilingual-en:start -->
*7.3 Finite cardinality and the Mapping Rule*
<!-- bilingual-en:end -->

[[01_Proofs#Session 11 — Infinite Sets|基数（cardinality）]] $|A|$ 表示有限集元素数。若存在 $A$ 到 $B$ 的 bijection，则 $|A|=|B|$。若存在 total injection，则 $|A|\le|B|$；若存在 total surjective function，则 $|A|\ge|B|$。
<!-- bilingual-en:start -->
For a finite set, its [[01_Proofs#Session 11 — Infinite Sets|cardinality]] $|A|$ is its number of elements. A bijection from $A$ to $B$ gives $|A|=|B|$; a total injection gives $|A|\le|B|$; and a total surjective function gives $|A|\ge|B|$.
<!-- bilingual-en:end -->

证明的箭头计数本质：total injection 从每个 $a$ 发出一条、不同 $a$ 不能落到同一 $b$，所以 $B$ 至少有 $|A|$ 个落点。
<!-- bilingual-en:start -->
The proof is fundamentally an arrow-counting argument. A total injection sends at least one arrow from every $a\in A$, and no two distinct elements of $A$ can land on the same $b\in B$. Therefore $B$ must contain at least $|A|$ distinct targets.
<!-- bilingual-en:end -->

### 7.4 组合的性质不能凭感觉传递
<!-- bilingual-en:start -->
*7.4 Composition does not preserve properties automatically*
<!-- bilingual-en:end -->

若 $h=f\circ g$：
<!-- bilingual-en:start -->
If $h=f\circ g$:
<!-- bilingual-en:end -->

- $h$ surjective 强迫 $f$ surjective，因为 $h(A)\subseteq f(B)$；
- $h$ injective 强迫 $g$ injective，因为 $g$ 若碰撞，$h$ 必碰撞；
- 反方向通常只对 $g(A)$ 范围内有效，不能约束未被 $g$ 命中的 $B$ 元素。
<!-- bilingual-en:start -->
- If $h$ is surjective, then $f$ must be surjective because $h(A)\subseteq f(B)$.
- If $h$ is injective, then $g$ must be injective, because any collision under $g$ would also be a collision under $h$.
- Converse claims usually hold only on the image $g(A)$ and cannot constrain elements of $B$ that $g$ never reaches.
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
>
> &nbsp;
> **1.** The range cannot be larger than the codomain.<br>
> **2–3.** A total injection has at least one outgoing edge per domain element and at most one incoming edge per codomain element.<br>
> **4.** Totality is equivalent to $R^{-1}(B)=A$, or to every domain element having at least one outgoing edge.<br>
> **5.** Surjectivity is equivalent to $R(A)=B$.<br>
> **6–10.** In general, $|R(A)|\le|B|$; for a surjection, $|A|\ge|B|$ and $|R(A)|=|B|$; for an injection, $|R(A)|=|A|$; and for a bijection, $|A|=|B|$.<br>
> **11–14.** Under inversion: function ↔ injection, surjection ↔ total, injection ↔ function, and bijection ↔ bijection. 15–21, with domain and codomain both $\mathbb R$: $x+2$ is B, $2x$ is B, $x^2$ is N, $x^3$ is B, $\sin x$ is N, $x\sin x$ is S-not-B, and $e^x$ is I-not-B.<br>
> **22.** For finite sets, $A\operatorname{inj}B$ is equivalent to each of the following: a total injective relation $A\to B$ exists; a surjective function $B\to A$ exists; $A$ is in bijection with a subset of $B$; and $|A|\le|B|$.<br>
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
> **1. Table of inverse relations.** Total ↔ inverse surjection; function ↔ inverse injection; surjection ↔ inverse total; injection ↔ inverse function; bijection ↔ inverse bijection. Each statement simply observes that reversing every arrow turns outgoing-edge conditions into incoming-edge conditions.
> **2. $|A\times B|=mn$.** Let $A=\{a_0,\dots,a_{n-1}\}$ and $B=\{b_0,\dots,b_{m-1}\}$. Define
> $$f(a_i,b_j)=im+j.$$
> Its range is $0,\dots,mn-1$. If $im+j=i'm+j'$ with $0\le j,j'<m$, uniqueness of quotient and remainder in Euclidean division gives $i=i'$ and $j=j'$. Conversely, every $k<mn$ has a unique representation $k=im+j$. Thus $f$ is bijective.
> **3. Strongest comparison sign.** (a) $|f(A)|\le|B|$; (b) if $f$ is surjective, $|A|\ge|B|$; (c) $|f(A)|=|B|$; (d) if $f$ is injective, $|f(A)|=|A|$; (e) if $f$ is bijective, $|A|=|B|$.
> **4. Generalised Mapping Rule.** Let $E$ be the number of arrows leaving $X$. Because $R$ is a function, each $x\in X$ has at most one outgoing edge, so $E\le|X|$. Every element of $R(X)$ receives at least one edge from $X$, so $|R(X)|\le E$. Hence $|R(X)|\le|X|$.
> **5. Combining injection and surjection properties.** (a) The composition of two surjective relations is surjective: for each $c$, choose $b$ with $bRc$, then choose $a$ with $aSb$, producing an edge from $a$ to $c$ in the composite relation. (b) $A\operatorname{surj}B$ if and only if reversing all arrows gives $B\operatorname{inj}A$. (c) Taking inverses in part (a) gives transitivity of injection. (d) From a total injective relation, choose one outgoing edge for each $a$. Removing other edges preserves the at-most-one-incoming-edge property and produces a total injective function. For finite sets the choices can be made one at a time; the uniform statement for arbitrary sets invokes a choice principle.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 一个 relation 可以同时是 function 但不 total 吗？
> 2. $f:A\to B$ 为 injection 时，为何不必 $f(A)=B$？
> 3. 若 $A$ 空、$B$ 非空，空关系有哪些性质？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Can a relation be a function without being total?<br>
> **2.** Why can $f(A)$ be a proper subset of $B$ when $f:A\to B$ is injective?<br>
> **3.** If $A$ is empty and $B$ is nonempty, which properties does the empty relation have?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 可以，例如 $A=\{1,2\},R=\{(1,a)\}$。2. injection 只限制碰撞，不要求覆盖。3. 它是 function 和 injection（条件真空成立），不是 total，也不是 surjection。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Yes; for example, take $A=\{1,2\}$ and $R=\{(1,a)\}$.<br>
> **2.** Injectivity only forbids collisions; it does not require the codomain to be covered.<br>
> **3.** It is a function and an injection, because those conditions hold vacuously, but it is neither total nor surjective.<br>
> <!-- bilingual-en:end -->

**知识链：**笛卡尔积 → relation → 箭头约束 → inverse/composition → finite cardinality。
<!-- bilingual-en:start -->
**Knowledge chain:** Cartesian product → relation → arrow-count constraints → inverse and composition → finite cardinality.
<!-- bilingual-en:end -->

---

## Session 8 — Induction

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

怎样证明无限多个整数命题而只写有限证明？归纳假设为什么必须足够强、基例为什么必须足够多？
<!-- bilingual-en:start -->
How can a finite proof establish infinitely many statements about integers? Why must the induction hypothesis be strong enough, and why must there be enough base cases?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session8.pdf|Session 8 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp8.pdf|CP8]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/XnV8GAuAqJM.pdf|Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/D3E5CKebKuQ.pdf|Bogus Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TUueMeRooBk.pdf|Strong Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/K8ZfzNN1miQ.pdf|WOP vs Induction]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Induction.pdf|Induction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BogusInductn.pdf|Bogus]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StrongInduct.pdf|Strong]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_WOPvsInductn.pdf|WOP comparison]]

### 8.1 普通归纳
<!-- bilingual-en:start -->
*8.1 Ordinary induction*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|普通归纳原理]]：若
<!-- bilingual-en:start -->
The [[数学证明方法#归纳法|ordinary induction principle]] states that if
<!-- bilingual-en:end -->

$$
P(0),\qquad\forall n\in\mathbb N\,[P(n)\Rightarrow P(n+1)],
$$

则 $\forall n\in\mathbb N,P(n)$。
<!-- bilingual-en:start -->
then $\forall n\in\mathbb N,P(n)$.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-induction-dominoes.png|900]]

读图：基例推倒第一块骨牌，归纳步只保证任意一块倒下会带动下一块；两项缺一都不能覆盖整条序列。
<!-- bilingual-en:start -->
How to read the diagram: the base case knocks down the first domino, while the inductive step says that any falling domino knocks down the next one. Neither condition alone covers the entire sequence.
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

&nbsp;
**1.** Define $P(n)$.<br>
**2.** Base case: prove $P(0)$, or the appropriate initial case.<br>
**3.** Inductive step: fix an arbitrary $n$, assume $P(n)$, and prove $P(n+1)$.<br>
**4.** Invoke the induction principle.<br>
<!-- bilingual-en:end -->

归纳假设不是把目标当已知：它只在“证明相邻传递规则”的局部步骤中暂时成立。
<!-- bilingual-en:start -->
The induction hypothesis does not assume the whole theorem; it is a temporary assumption used only to prove the one-step implication.
<!-- bilingual-en:end -->

### 8.2 强归纳与普通归纳等价
<!-- bilingual-en:start -->
*8.2 Equivalence of strong and ordinary induction*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|强归纳]]的归纳步允许假设所有更小情形：
<!-- bilingual-en:start -->
The inductive step in [[数学证明方法#归纳法|strong induction]] may assume every strictly smaller case:
<!-- bilingual-en:end -->

$$
\left(\forall k<n\,P(k)\right)\Rightarrow P(n).
$$

它并不更“有证明力”。令 $Q(n)=\forall k\le nP(k)$，对 $Q$ 做普通归纳即可模拟强归纳。反过来，普通归纳只是强归纳中只使用 $P(n)$ 的特例。
<!-- bilingual-en:start -->
This does not make strong induction more powerful. Defining $Q(n)=\forall k\le nP(k)$ and applying ordinary induction to $Q$ simulates strong induction. Conversely, ordinary induction is the special case of strong induction that uses only $P(n)$.
<!-- bilingual-en:end -->

### 8.3 基例的数量由递推跨度决定
<!-- bilingual-en:start -->
*8.3 The number of base cases is determined by the recurrence span*
<!-- bilingual-en:end -->

若归纳步是 $P(n)\Rightarrow P(n+3)$，只证明 $P(0)$ 只能覆盖 $0,3,6,\dots$。要覆盖所有足够大的整数，必须为模 $3$ 的三个余数类各提供一个起点，例如 $P(5),P(6),P(7)$。
<!-- bilingual-en:start -->
If the inductive step is $P(n)\Rightarrow P(n+3)$, proving only $P(0)$ covers just $0,3,6,\dots$. To cover all sufficiently large integers, provide a starting point in each of the three residue classes modulo $3$, for example $P(5),P(6),P(7)$.
<!-- bilingual-en:end -->

### 8.4 “所有马同色”漏洞
<!-- bilingual-en:start -->
*8.4 The flaw in the “all horses have the same colour” proof*
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
If an inductive claim fails, WOP supplies a least counterexample. Every smaller integer satisfies the claim, so the inductive step forces the least counterexample to satisfy it as well, a contradiction. Conversely, induction can prove that every nonempty set of natural numbers has a least element. Thus ordinary induction, strong induction, and WOP are equivalent within this course, though different problems are easier to express with different forms.
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
>
> &nbsp;
> **1.** The all-horses-the-same-colour proof fails at $P(1)\Rightarrow P(2)$.<br>
> **2.** The total unstacking score is independent of strategy and equals $n(n-1)/2$.<br>
> **3–4.** Ordinary induction, strong induction, and WOP are equivalent; convert strong induction to ordinary induction by defining $Q(n)=\forall k\le nP(k)$.<br>
> **5–6.** From $P(5)$ and $P(n)\Rightarrow P(n+3)$ we obtain only $5,8,11,\dots$; covering all $n\ge5$ requires base cases in all three residue classes modulo $3$, such as $5,6,7$.<br>
> **7–11.** The five rules are Strong, Ordinary, WOP, None (the step $0\to1$ is missing), and Strong (when $m=0$, the vacuous hypothesis supplies the base case).<br>
> **12.** The postage problem can be proved by all three methods, though strong induction or WOP is more natural.<br>
> **13.** The bogus strong-induction proof for Fibonacci numbers omits $P(1)$, so its final invocation of induction is invalid.<br>
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
> **1. Prove $\sum_{k=1}^n1/k^2<2-1/n$ for $n>1$.** The base case $n=2$ is $5/4<3/2$. Assuming the claim for $n$ gives
> $$\sum_{k=1}^{n+1}\frac1{k^2}<2-\frac1n+\frac1{(n+1)^2}.$$
> Since $1/(n+1)^2<1/[n(n+1)]=1/n-1/(n+1)$, the right-hand side is less than $2-1/(n+1)$, completing the induction.
> **2. Tiling with L-trominoes.** (a) Induct on $n$. A $2^0\times2^0$ courtyard consists only of the corner statue and needs no tile. Divide the $2^{n+1}\times2^{n+1}$ courtyard into four $2^n\times2^n$ quadrants. Tile the quadrant containing the statue by the induction hypothesis. Place one L-tromino over the three central squares belonging to the other three quadrants; each of those quadrants is now a $2^n\times2^n$ courtyard missing one corner and can also be tiled by the induction hypothesis. (b) If the statue is at the centre, regard it as the central corner of one quadrant, cover the other three central squares with an L-tromino, and apply part (a) to all four quadrants.
> **3. Postage using 3¢ and 7¢ stamps.** Use strong induction. The base cases are $12=4\cdot3$, $13=2\cdot3+7$, and $14=2\cdot7$. For $N\ge15$, we have $N-3\ge12$, so the strong induction hypothesis represents $N-3$; adding one 3¢ stamp represents $N$. Hence every $N\ge12$ is representable, which is exactly the assertion that all the stated $S(n)$ hold.
> **4. A bogus proof about a prime dividing a product.** The first unsupported step occurs when the proof tries to pass from $n=1$ to $n+1=2$. After obtaining $p\mid x_1x_2$, it invokes strong induction on a product of two factors, but $P(2)$ is the very case being proved and cannot follow from a hypothesis containing only $P(1)$. To repair the proof, establish Euclid's lemma for two factors as an independent base case or prior theorem, then induct from $n\ge2$.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. $P(0)$ 与 $P(n)\Rightarrow P(n+2)$ 能推出哪些 $P$？
> 2. 强归纳中能否假设 $P(n)$ 来证明 $P(n)$？
> 3. 递推使用 $P(n-2)$ 时，至少要检查哪些基例？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Which instances of $P$ follow from $P(0)$ and $P(n)\Rightarrow P(n+2)$?<br>
> **2.** In strong induction, can we suppose $P(n)$ to prove $P(n)$?<br>
> **3.** At least which base cases should be checked when using $P(n-2)$ recursively?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 只能推出偶数指标。2. 不能；假设只包括严格更小指标。3. 通常需连续两个起始指标，使每条递推链都能启动。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Only instances with even indices follow.<br>
> **2.** No; the hypothesis contains only cases with strictly smaller indices.<br>
> **3.** Usually two consecutive base cases are needed so that both recurrence chains can start.<br>
> <!-- bilingual-en:end -->

**知识链：**WOP 最小反例 ↔ 普通归纳 ↔ 强归纳 → 足够强的假设 → 递归对象。
<!-- bilingual-en:start -->
**Knowledge chain:** WOP and least counterexamples ↔ ordinary induction ↔ strong induction → a sufficiently strong hypothesis → recursive objects.
<!-- bilingual-en:end -->

---

## Problem Set 3 — Sessions 7–8

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps3.pdf|Problem Set 3 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps3.pdf|Problem Set 3 PDF]]. The three solutions below are independent and unofficial.
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
> Let
> $$p=\frac{1+\sqrt5}{2},\qquad q=\frac{1-\sqrt5}{2}.$$
> Both satisfy $x^2=x+1$. We prove $F_n=(p^n-q^n)/\sqrt5$ by strong induction.
> For $n=0$, the right-hand side is $0$; for $n=1$, it is $(p-q)/\sqrt5=1$. For $n\ge2$, substitute the induction hypotheses into the recurrence, as shown above. The final equality uses $p+1=p^2$ and $q+1=q^2$.
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
> Define the potential of a stack of height $k$ by $\phi(k)=k(k-1)/2$. When one stack of height $a+b$ is split into stacks of heights $a$ and $b$, the calculation above shows that the potential decreases by exactly $ab$, the score earned on that move. Therefore the total score along any sequence of moves is the telescoping decrease in total potential:
> $$\mathrm{score}=\Phi(A)-\Phi(B).$$
> At the end of the game every stack is a singleton, so $\Phi(B)=0$. The initial state is one stack of height $n$, so every complete play has total score $n(n-1)/2$. No strategy can do better: all legal strategies that finish the game tie for the optimum.
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
> **(a) True.** For any $c\in C$, surjectivity of $h$ gives an $a$ with $f(g(a))=c$. Setting $b=g(a)$ shows that $f$ reaches $c$.
> **(b) False.** Let $A=\{1\}$, $B=\{1,2\}$, and $C=\{c\}$, with $g(1)=1$ and $f(1)=f(2)=c$. Then $h$ is surjective, but $g$ does not reach $2$.
> **(c) False.** In the same counterexample, $h$ is injective because its domain has one element, whereas $f$ is not injective because $f(1)=f(2)$.
> **(d) True.** If $g(a)=g(a')$, totality of $f$ allows both values to be evaluated, giving $h(a)=f(g(a))=f(g(a'))=h(a')$. Injectivity of $h$ then gives $a=a'$, so $g$ is injective.
> <!-- bilingual-en:end -->

---

## Midterm 1 — Sessions 1–8

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm1.pdf|Midterm 1 PDF]]。以下 5 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm1.pdf|Midterm 1 PDF]]. The five solutions below are independent and unofficial.
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
> The domain is $\mathbb N$. First define
> $$\mathrm{One}(x):\Longleftrightarrow\forall y(xy=y).$$
> **(a)** The formula for $x=1$ is $\mathrm{One}(x)$.
> **(b)** The formula for $m\mid n$ is $\exists k(mk=n)$.
> **(c)**
> $$\mathrm{Prime}(n):\Longleftrightarrow\neg\mathrm{One}(n)\land\forall m(m\mid n\Rightarrow(\mathrm{One}(m)\lor m=n)).$$
> **(d)** The statement “$n$ is a positive integer power of some prime” can be written without exponentiation as
> $$\exists p\,[\mathrm{Prime}(p)\land p\mid n\land\forall q((\mathrm{Prime}(q)\land q\mid n)\Rightarrow q=p)].$$
> Unique factorisation says that a positive integer with exactly one distinct prime factor is a power of that prime. The definition excludes $1$, which has no prime factors, and $0$, which is divisible by every prime.
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
> Use strong induction on $S(n)$: “$n+18$ people can be divided into teams of size $4$ or $7$.” The consecutive base cases are
> $$18=7+7+4,\qquad19=7+4+4+4,\qquad20=5\cdot4,\qquad21=3\cdot7.$$
> For $n\ge4$, $(n+18)-4=(n-4)+18$ can be partitioned into teams by the strong induction hypothesis; adding one more four-person team handles $n+18$. Therefore $S(n)$ holds for every $n\ge0$.
> <!-- bilingual-en:end -->

---

## Session 9 — State Machines and Invariants

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

怎样证明一个可能运行任意久的过程永远不会到达坏状态？怎样把“值一直下降”变成终止性证明？
<!-- bilingual-en:start -->
How do you prove that a process that may run for any length of time never reaches a bad state? How do you turn “a value keeps decreasing” into a proof of termination?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session9.pdf|Session 9 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp9.pdf|CP9]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VuG2JNcRXYg.pdf|State Machines and Invariants]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/a7JUH-EtHtI.pdf|Derived Variables]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StateMachine.pdf|State Machines]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_DerivedVaria.pdf|Derived Variables]]

### 9.1 状态机模型
<!-- bilingual-en:start -->
*9.1 The state-machine model*
<!-- bilingual-en:end -->

[[数学证明方法#不变量与算法正确性|状态机（state machine）]]由状态集合 $Q$、开始状态 $q_0$、转移关系 $\delta\subseteq Q\times Q$ 构成。状态 $r$ **reachable**，若存在有限路径
<!-- bilingual-en:start -->
A [[数学证明方法#不变量与算法正确性|state machine]] consists of a state set $Q$, a start state $q_0$, and a transition relation $\delta\subseteq Q\times Q$. A state $r$ is **reachable** if there is a finite path
<!-- bilingual-en:end -->

$$
q_0\to q_1\to\cdots\to r.
$$

建模时要分清：哪些信息足以决定下一步、什么是合法转移、何时无转移。若状态遗漏必要历史，所谓“不变量”可能根本无法表达。
<!-- bilingual-en:start -->
Modelling requires deciding which information is sufficient to determine the next step, which transitions are legal, and when no transition is available. If the state omits necessary history, the intended invariant may not even be expressible.
<!-- bilingual-en:end -->

### 9.2 preserved invariant 与 invariant
<!-- bilingual-en:start -->
*9.2 Preserved invariants and invariants*
<!-- bilingual-en:end -->

谓词 $P$ 是 **preserved invariant**，若每条转移 $q\to q'$ 都满足
<!-- bilingual-en:start -->
A predicate $P$ is a **preserved invariant** if every transition $q\to q'$ satisfies
<!-- bilingual-en:end -->

$$
P(q)\Rightarrow P(q').
$$

若再有 $P(q_0)$，Floyd invariant principle 给出所有 reachable $r$ 均满足 $P(r)$。证明是对路径长度归纳：长度 0 为开始状态；长度 $n+1$ 的路径由长度 $n$ 的可达状态再走一步，应用保持性。
<!-- bilingual-en:start -->
If $P(q_0)$ also holds, Floyd's invariant principle implies that every reachable state $r$ satisfies $P(r)$. The proof is by induction on path length: a path of length $0$ ends at the start state; a path of length $n+1$ consists of a length-$n$ path followed by one transition, to which preservation applies.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-state-machine-invariant.png|900]]

读图：初始状态位于不变量区域内，而每条合法转移都留在该区域；因此区域外的坏状态不可达。
<!-- bilingual-en:start -->
How to read the diagram: the initial state lies inside the invariant region, and every legal transition remains inside it. Therefore bad states outside the region are unreachable.
<!-- bilingual-en:end -->

> [!warning] 两个词不能混用
> preserved 只表示“一旦成立就保持”，未保证开始时成立；invariant 是“所有可达状态都成立”。
> <!-- bilingual-en:start -->
> “Preserved” means only that once the predicate holds, every transition keeps it true; it does not say that the predicate holds initially. An invariant holds in every reachable state.
> <!-- bilingual-en:end -->

### 9.3 找不变量：从转移变化量入手
<!-- bilingual-en:start -->
*9.3 Finding invariants from transition deltas*
<!-- bilingual-en:end -->

若状态为向量 $(x_1,\dots,x_k)$，先列每类转移的 $\Delta x_i$，再寻找线性组合 $a_1x_1+\cdots+a_kx_k$ 使每类变化为 0，或模 $m$ 为 0。整数网格机器人适合模不变量，资源守恒适合线性等式，排列谜题常用 parity。
<!-- bilingual-en:start -->
If a state is a vector $(x_1,\dots,x_k)$, first list the changes $\Delta x_i$ for every type of transition. Then seek a linear combination $a_1x_1+\cdots+a_kx_k$ whose change is $0$ under every transition, either exactly or modulo $m$. Modular invariants suit robots on integer grids; linear equalities suit conservation of resources; and permutation puzzles often use parity.
<!-- bilingual-en:end -->

### 9.4 partial correctness 与 termination 分开
<!-- bilingual-en:start -->
*9.4 Separating partial correctness from termination*
<!-- bilingual-en:end -->

**部分正确性（partial correctness）**只说“若算法停，则答案正确”；**全正确性**还要证一定停。
<!-- bilingual-en:start -->
**Partial correctness** says only, “if the algorithm terminates, its answer is correct.” **Total correctness** additionally proves that the algorithm must terminate.
<!-- bilingual-en:end -->

[[数学证明方法#不变量与算法正确性|秩函数/derived variable]] $V:Q\to W$ 若每步严格下降，而 $W$ well-ordered，则不存在无限转移链。仅 weakly decreasing 不足，例如保持常数的过程可以永远运行。
<!-- bilingual-en:start -->
If a [[数学证明方法#不变量与算法正确性|rank function or derived variable]] $V:Q\to W$ decreases strictly at every step and $W$ is well ordered, then no infinite transition chain exists. Weak decrease alone is insufficient: a process can remain constant forever.
<!-- bilingual-en:end -->

### 9.5 俄式整数乘法
<!-- bilingual-en:start -->
*9.5 Russian integer multiplication*
<!-- bilingual-en:end -->

状态 $(r,s,a)$ 从 $(x,y,0)$ 开始；$s$ 偶时转到 $(2r,s/2,a)$，$s$ 奇时转到 $(2r,(s-1)/2,a+r)$。不变量为
<!-- bilingual-en:start -->
The state $(r,s,a)$ starts at $(x,y,0)$. When $s$ is even, the transition is to $(2r,s/2,a)$; when $s$ is odd, it is to $(2r,(s-1)/2,a+r)$. The invariant is
<!-- bilingual-en:end -->

$$
rs+a=xy.
$$

偶转移后 $(2r)(s/2)+a=rs+a$；奇转移后
<!-- bilingual-en:start -->
After an even transition, $(2r)(s/2)+a=rs+a$; after an odd transition,
<!-- bilingual-en:end -->

$$
(2r)\frac{s-1}{2}+(a+r)=r(s-1)+a+r=rs+a.
$$

终态只能有 $s=0$，于是 $a=xy$。同时 $s$ 每步变为 $\lfloor s/2\rfloor$，故转移次数 $O(\log(y+1))$。
<!-- bilingual-en:start -->
At a terminal state, necessarily $s=0$, so the invariant gives $a=xy$. Because each step replaces $s$ by $\lfloor s/2\rfloor$, the number of transitions is $O(\log(y+1))$.
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
>
> &nbsp;
> **1–4.** A Die Hard state records the amounts of water in the two jugs as $(b,l)$; in the example, both amounts remain divisible by $3$. Legal Diagonal Robot examples include $(-11,3)$ and $(3,7)$. Floyd's principle is proved by induction on path length: if the predicate holds initially and is preserved, then it holds in every reachable state.<br>
> **5–8.** A derived variable is a real-valued function of the state. In the robot example it is $(x+y)\bmod2$. Being constant is equivalent to being both weakly increasing and weakly decreasing. A variable taking values in a well-ordered set and decreasing strictly at every step guarantees termination.<br>
> **9–10.** The only expression preserved by both commands in the multiplication algorithm is $xy+p=ab$. The variable $x$ decreases strictly at every step, whereas $xy$ is only weakly decreasing.<br>
> **11–14.** In the chocolate-splitting process, $s=p-1$ and $s\ne p$ are preserved; $mn-p$ decreases at every step; and the terminal state has $p=mn$. Therefore the number of splits is $s=mn-1$.<br>
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
> **1. Shift-and-add.** (a) The calculation above verifies, transition by transition, that $rs+a=xy$ is preserved. (b) If no transition is available, then $s=0$, and substituting this into the invariant gives $a=xy$. (c) Each step replaces the positive integer $s$ by $\lfloor s/2\rfloor$, so it reaches zero within at most $\lfloor\log_2y\rfloor+1$ steps.
> **2. Fifteen Puzzle.** (a) The initial tile list is $(1,2,\dots,15)$ with the blank at $(4,4)$; the target list is $(15,14,\dots,1)$ with the blank again at $(4,4)$. (b) Initially the inversion count is $0$, and adding the blank's row number gives an even value. The reversed target list has $\binom{15}{2}=105$ inversions, and $105+4=109$ is odd. (c) A horizontal move changes neither the tile list with the blank removed nor the blank's row. A vertical move passes one tile across three others in that list, flipping inversion parity, while the blank moves by one row, flipping the row parity as well. Their sum modulo $2$ is therefore invariant, so the target is unreachable.
> **3. Zakim Bridge.** An entering transition, allowed when $A-B<T_0$, is $(A,B,C)\to(A+3,B,C+1)$; an exiting transition, allowed when $C>0$, is $(A,B,C)\to(A,B+2,C-1)$.
> The derived-variable classifications are: $A$ WI, $B$ WI, $A+B$ SI, $A-B$ N, $3C-A$ WD, $2A-3B$ N, $B+3C$ N, $2A-3B-6C$ constant, and $2A-2B-3C$ N. The key changes are:
> | Quantity | In Bridge $\Delta$ | Out Bridge $\Delta$ |
> |---|---:|---:|
> | $A-B-3C$ | $0$ | $+1$ |
> | $2A-3B-6C$ | $0$ | $0$ |
> Let $J=A-B-3C$. Then $J\ge J_0=A_0-B_0-3C_0$ is preserved. If $C=1000$, then
> $$A-B\ge A_0-B_0+3(1000-C_0)=T_0,$$
> so the condition for entering the bridge fails; an exit can only decrease $C$. Thus $P:(C\le1000\land J\ge J_0)$ holds initially, is preserved, and rules out the collapsed state.
> A deadlock is nevertheless reachable. First let all $C_0$ cars initially on the bridge exit. Then $C=0$ and $A-B=A_0-B_0-2C_0<T_0$. Repeatedly let one car enter and immediately exit; each round returns to $C=0$ while increasing $A-B$ by $1$. Once $A-B=T_0$, no car can exit because $C=0$, and no car can enter because the strict inequality is false.
> **4. Beaver flu.** Define the perimeter of the infected region as the number of unit edges separating an infected square from either an uninfected square or the outside of the classroom. If a newly infected square has $k\ge2$ infected neighbours, the perimeter changes by $4-2k\le0$. With $m<n$ initially infected students, the perimeter is at most $4m<4n$. If the entire $n\times n$ classroom became infected, its perimeter would be exactly $4n$. A nonincreasing quantity cannot rise from below $4n$ to $4n$, so at least one student is never infected.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. preserved invariant 还缺哪个条件才能推出安全性？
> 2. $V$ 每步减少且始终为非负实数是否必终止？
> 3. invariant 排除坏状态，是否说明所有好状态都 reachable？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** What additional condition turns a preserved predicate into a safety invariant?<br>
> **2.** If $V$ decreases at every step and remains a nonnegative real number, must the process terminate?<br>
> **3.** If an invariant excludes every bad state, does that imply that every good state is reachable?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 它在 start state 成立。2. 不一定，如 $1,1/2,1/4,\dots$；需要 well-founded/良序下降。3. 不说明；不变量通常只是可达集的 over-approximation。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** It must hold in the start state.<br>
> **2.** Not necessarily; consider $1,1/2,1/4,\dots$. The decrease must be well founded, for example in a well-ordered set.<br>
> **3.** No. An invariant is usually only an over-approximation of the reachable set.<br>
> <!-- bilingual-en:end -->

**知识链：**归纳 → 路径长度 → preserved invariant → safety；良序下降 → termination。
<!-- bilingual-en:start -->
**Knowledge chain:** induction → path length → preserved invariant → safety; well-ordered descent → termination.
<!-- bilingual-en:end -->

---

## Session 10 — Recursive Definitions

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

对象若不是按整数大小而是按语法树生成，怎样定义“全部对象”，又怎样证明每种对象都有性质 $P$？
<!-- bilingual-en:start -->
When objects are generated by a syntax tree rather than indexed by integer size, how can “all objects” be defined, and how can we prove that every generated object has property $P$?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session10.pdf|Session 10 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp10.pdf|CP10]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TXNXT3oBROw.pdf|Recursive Data]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/VWIDwHCGJDQ.pdf|Structural Induction]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/tOsdeaYDCMk.pdf|Recursive Functions]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RecursiveData.pdf|Recursive Data]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StructuralInd.pdf|Structural Induction]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RecursiveFunc.pdf|Recursive Functions]]

### 10.1 递归数据类型
<!-- bilingual-en:start -->
*10.1 Recursive data types*
<!-- bilingual-en:end -->

[[01_Proofs#Session 10 — Recursive Definitions|递归定义（recursive definition）]]包含 base cases、constructor cases 与 closure clause。例如匹配括号串 $M$：
<!-- bilingual-en:start -->
A [[01_Proofs#Session 10 — Recursive Definitions|recursive definition]] includes base cases, constructor cases, and a closure clause. For example, the set $M$ of matched-bracket strings is defined by:
<!-- bilingual-en:end -->

- $\lambda\in M$；
- 若 $s,t\in M$，则 $[s]t\in M$；
- 只有有限次使用以上规则生成的串才在 $M$ 中。
<!-- bilingual-en:start -->
- $\lambda\in M$;
- If $s,t\in M$, $[s]t\in M$;
- Only strings generated by finitely many applications of the preceding rules belong to $M$.
<!-- bilingual-en:end -->

最后一条排除无限构造链产生的非有限串，也防止凭直觉额外加入对象。
<!-- bilingual-en:start -->
The last clause excludes infinite strings arising from an endless construction chain and prevents us from adding extra objects merely because they seem intuitively related.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-recursive-structure.png|900]]

读图：从叶端的 base objects 向上经过 constructors 形成有限构造树，结构归纳正是沿同一依赖方向传播性质。
<!-- bilingual-en:start -->
How to read the diagram: base objects form the leaves, and constructors combine them upward into a finite construction tree. Structural induction propagates a property along this same direction of dependence.
<!-- bilingual-en:end -->

### 10.2 结构归纳原理
<!-- bilingual-en:start -->
*10.2 The structural-induction principle*
<!-- bilingual-en:end -->

[[数学证明方法#归纳法|结构归纳]]要求：对每个 base object 证明 $P$；对每个 constructor，假设直接子对象均满足 $P$，证明构造结果满足 $P$。证明依据是有限构造树，而不是表面的数值大小。若有 3 个 bases、4 个 constructors，就有 7 个逻辑分支。
<!-- bilingual-en:start -->
[[数学证明方法#归纳法|Structural induction]] requires proving $P$ for every base object and, for each constructor, assuming $P$ for its immediate subobjects and proving $P$ for the constructed result. The proof follows the finite construction tree rather than an object's superficial numerical size. Three base cases and four constructors therefore produce seven proof branches.
<!-- bilingual-en:end -->

### 10.3 递归函数与良定义
<!-- bilingual-en:start -->
*10.3 Recursive functions and well-definedness*
<!-- bilingual-en:end -->

对二叉树，`size`、`flatten` 等函数也按 base/constructor 定义：叶子直接给值，内部节点从左右子树的结果组合。若语法可能给同一对象多种解析，还需证明不同解析结果相同，或改用 unambiguous grammar。
<!-- bilingual-en:start -->
Functions such as `size` and `flatten` on binary trees are likewise defined by base and constructor cases: leaves receive values directly, while an internal node combines the results from its left and right subtrees. If the grammar permits multiple parses of the same object, one must prove that every parse yields the same result or replace it with an unambiguous grammar.
<!-- bilingual-en:end -->

普通归纳适合唯一数值前驱，强归纳适合任意更小规模，结构归纳适合直接组成部件。结构归纳可转成对构造树高度的归纳，但直接写更贴近信息流。
<!-- bilingual-en:start -->
Ordinary induction suits a unique numerical predecessor, strong induction suits dependence on arbitrary smaller cases, and structural induction suits dependence on immediate components. Structural induction can be recast as induction on construction-tree height, but the direct form follows the information flow more naturally.
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
>
> &nbsp;
> **1.** For matched-parenthesis strings, no string begins with a closing parenthesis, the set is infinite, and every string has even length. The claim that every string begins with an opening parenthesis is refuted by the empty string.<br>
> **2–3.** Expressing $\cos x=\sin(x+\pi/2)$ requires the identity, constant, and sine base cases, together with addition and composition.<br>
> **4–8.** The five placeholders in the structural-induction template are $x,b,R,c(x),x$, in that order.<br>
> **9.** Three base cases and four constructors require seven proof branches in total.<br>
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
> **1. Elementary 18.01 Functions.** (a) The function $e(x)=2^x$ is F18, and so is its inverse $\ell=e^{-1}=\log_2x$. Hence $2^{-\ell(x)}=1/x$ is F18 on $x>0$. On the branch $x<0$, write the function as $-2^{-\log_2(-x)}$. The course treats these as elementary branches with their natural domains. Note that $\mathrm{id}^{-1}=\mathrm{id}$; it is not the reciprocal function.
> (b) Use structural induction on the construction of the function. For the base cases, $\mathrm{id}'=1$, the derivative of a constant is $0$, and $\sin'x=\cos x=\sin(x+\pi/2)$. If $f'$ and $g'$ are already F18, then
> $$(f+g)'=f'+g',\quad(fg)'=f'g+fg',\quad(2^g)'=(\ln2)2^gg',\quad(f\circ g)'=(f'\circ g)g'.$$
> If $h=f^{-1}$ is locally invertible and $f'(h(x))\ne0$, then $h'(x)=1/f'(h(x))$, which is still F18 by part (a) and composition. An interval of invertibility and a nonzero derivative are necessary conditions.
> **2. Erasable = RecMatch.** First use structural induction on RecMatch. The empty string $\lambda$ is erasable. If $s$ and $t$ are erasable, then $[s]t$ can be erased by erasing $s$, deleting the resulting `[]`, and then erasing $t$. Hence RecMatch $\subseteq$ Erasable.
> For the reverse inclusion, use strong induction on string length. At length $0$, the only string is $\lambda$. Suppose one deletion turns $x$ into some $y\in$ RecMatch. If $y=\lambda$, then $x=[]$. Otherwise write $y=[s]t$. The restored `[]` can occur only at the front, inside $s$, or inside $t$. In the first case, $x=[][s]t=[\lambda]([s]t)$. In the other two cases, strong induction puts the shorter erasable substring in RecMatch, after which the constructor applies. These three cases exhaust the possible grammatical positions.
> **3. Recursive definitions.** (a) $S$: base $1$; constructors $x\mapsto2x,3x,5x$. (b) $T$: base $1$; constructors $x\mapsto18x,15x,5x$, corresponding respectively to increasing $k,m,n$. (c) $L'$: base $(0,0)$; constructors $(a,b)\mapsto(a\pm1,b\pm1)$ with matching signs, and $(a,b)\mapsto(a\pm3,b)$. (d) Every constructor preserves $a-b\equiv0\pmod3$, so $L'\subseteq L$. (e) If $a-b=3k$, first move both coordinates together to reach $(b,b)$, then move the first coordinate by $3k$; hence $L\subseteq L'$. (f) For an unambiguous definition, use the unique normal form $(k,b)\mapsto(b+3k,b)$ and encode each integer $k,b$ as a sign plus a natural-number magnitude.
> **4. Binary-2PTG.** In the diagram,
> $$G=\langle bintree,\langle bintree,\langle leaf,win\rangle,\langle bintree,\langle leaf,lose\rangle,\langle leaf,win\rangle\rangle\rangle,\langle leaf,win\rangle\rangle.$$
> `flatten(leaf,l)=(l)` and `flatten(bintree,G1,G2)=flatten(G1)++flatten(G2)`. Use structural induction. For a leaf, $2\cdot1=1+1$. For an internal node, the induction hypotheses give:
> $$2(\ell_1+\ell_2)=(|G_1|+1)+(|G_2|+1)=|G_1|+|G_2|+2=|G|+1.$$
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 为什么只证明所有 base cases 不足？
> 2. structural induction 的假设针对谁？
> 3. 若允许无限次 constructor，会发生什么？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why is proving all the base cases alone insufficient?<br>
> **2.** To which objects does the structural-induction hypothesis apply?<br>
> **3.** What happens if infinitely many constructor applications are allowed?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. constructors 会产生新对象，必须证性质在每个 constructor 下保持。2. 当前 constructor 的直接子对象。3. 可能加入无限树/无限串，有限结构归纳不再自动适用。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Constructors generate new objects, so the property must be proved to survive every constructor.<br>
> **2.** It applies to the immediate subobjects of the current constructor.<br>
> **3.** Infinite trees or strings may enter the class, and induction over finite construction trees no longer applies automatically.<br>
> <!-- bilingual-en:end -->

**知识链：**induction → construction tree → recursive datatype → structural induction → recursive function。
<!-- bilingual-en:start -->
**Knowledge chain:** induction → construction tree → recursive data type → structural induction → recursive function.
<!-- bilingual-en:end -->

---

## Session 11 — Infinite Sets

### 本节问题与前置
<!-- bilingual-en:start -->
*Questions and prerequisites for this section*
<!-- bilingual-en:end -->

两个无限集合怎样比较大小？为什么整数、整数对和有理数一样多，而实数更多？为什么“所有集合的集合”和“万能程序分析器”会触发同一种对角矛盾？
<!-- bilingual-en:start -->
How can two infinite sets be compared in size? Why are the integers, pairs of integers, and rational numbers all equally numerous, while the real numbers are strictly more numerous? Why do a “set of all sets” and a “universal program analyser” lead to the same kind of diagonal contradiction?
<!-- bilingual-en:end -->

> [!note] 本地材料
> - 阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session11.pdf|Session 11 reading]]；课堂题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp11.pdf|CP11]]
> - transcripts：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QzSCf62kzjE.pdf|Cardinality]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/AipSRi3CyLg.pdf|Countable Sets]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/4dj1ogUwTEM.pdf|Cantor's Theorem]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/WQHOImO0pX0.pdf|Halting Problem]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/5hETv64GIuE.pdf|Russell's Paradox]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/zcvsyL7GtH4.pdf|Set Theory Axioms]]
> - slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Cardinality.pdf|Cardinality]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_CountableSets.pdf|Countable Sets]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_CantorsTheo.pdf|Cantor]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_haltingproblm.pdf|Halting]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_RussellsParad.pdf|Russell]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_theroyaxioms.pdf|Axioms]]

### 11.1 无限基数与 bijection
<!-- bilingual-en:start -->
*11.1 Infinite cardinality and bijections*
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
The integers can be enumerated as $0,1,-1,2,-2,\dots$. Pairs of natural numbers can be enumerated along the diagonals $a+b=0,1,2,\dots$, so $\mathbb N^2$ is countable. Every positive rational occurs in the image of the map $(a,b)\mapsto a/b$ from pairs of positive integers. Although values repeat, assigning each rational its first position in the enumeration gives an injection into $\mathbb N$. Hence $\mathbb Q$ is countable.
<!-- bilingual-en:end -->

> [!warning] “稠密”与“基数大”不同
> $\mathbb Q$ 在任意实数区间中都有点，却仍可数；拓扑上的稠密性和集合基数是不同性质。
> <!-- bilingual-en:start -->
> Every real interval contains rational numbers, yet $\mathbb Q$ is still countable. Topological density and set cardinality are different properties.
> <!-- bilingual-en:end -->

### 11.3 Cantor 对角论证
<!-- bilingual-en:start -->
*11.3 Cantor's diagonal argument*
<!-- bilingual-en:end -->

[[01_Proofs#11.3 Cantor 对角论证|Cantor 对角论证]]证明无限二进制序列集合 $\{0,1\}^{\omega}$ 不可数。反设存在列表
<!-- bilingual-en:start -->
[[01_Proofs#11.3 Cantor 对角论证|Cantor's diagonal argument]] proves that the set of infinite binary sequences $\{0,1\}^{\omega}$ is uncountable. Suppose for contradiction that there is a complete list
<!-- bilingual-en:end -->

$$
s_0,s_1,s_2,\dots,
$$

定义新序列 $d$ 的第 $n$ 位为 $1-s_n(n)$。于是 $d$ 与 $s_n$ 在第 $n$ 位不同，所以不等于任何 $s_n$，与列表完备矛盾。
<!-- bilingual-en:start -->
Define the $n$th bit of a new sequence $d$ to be $1-s_n(n)$. Then $d$ differs from $s_n$ in position $n$, so $d$ is unequal to every sequence on the list, contradicting its assumed completeness.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit01-cantor-diagonal.png|900]]

读图：沿列表主对角线逐位取反得到 $d$；第 $n$ 个反转位保证 $d$ 与列表第 $n$ 行至少有一位不同。
<!-- bilingual-en:start -->
How to read the diagram: construct $d$ by flipping the entries along the list's main diagonal. The flipped $n$th bit guarantees that $d$ differs from the $n$th row in at least one position.
<!-- bilingual-en:end -->

更一般的 [[01_Proofs#11.3 Cantor 对角论证|Cantor 定理]]：对任何集合 $A$，不存在 surjection $f:A\to\operatorname{pow}(A)$。若有，定义
<!-- bilingual-en:start -->
More generally, [[01_Proofs#11.3 Cantor 对角论证|Cantor's theorem]] says that no surjection $f:A\to\operatorname{pow}(A)$ exists for any set $A$. Suppose one did, and define
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
*11.4 Russell's paradox and restricted comprehension*
<!-- bilingual-en:end -->

若假设任意谓词都定义一个集合，令 $W=\{x:x\notin x\}$，便有
<!-- bilingual-en:start -->
If we assume that every predicate defines a set and let $W=\{x:x\notin x\}$, then
<!-- bilingual-en:end -->

$$
W\in W\Longleftrightarrow W\notin W.
$$

正确结论不是逻辑失效，而是“$W$ 是集合”的 unrestricted comprehension 假设不允许。ZFC 的 separation 只允许从已有集合 $S$ 中切出 $\{x\in S:P(x)\}$，不承诺“所有对象”构成集合。
<!-- bilingual-en:start -->
The correct conclusion is not that logic has failed, but that unrestricted comprehension cannot be used to assume that $W$ is a set. ZFC's separation scheme permits $\{x\in S:P(x)\}$ to be selected only from an existing set $S$; it does not assert that “all objects” form a set.
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
*11.6 The ZFC boundaries needed here*
<!-- bilingual-en:end -->

课程不是集合论公理课，但以下职责要分清：extensionality 由成员决定集合；pairing/union/power set/infinity 允许受控构造；separation 在已有集合中筛选；foundation 排除无限向下成员环；choice 支持对任意非空集合族同时选择代表。使用“从无限集合持续选出不同元素”时，严格背景可能涉及 choice。
<!-- bilingual-en:start -->
This is not a course in axiomatic set theory, but the roles of the relevant axioms should be kept distinct. Extensionality determines sets by their members; pairing, union, power set, and infinity permit controlled constructions; separation filters an existing set; foundation excludes infinite descending membership chains; and choice supports selecting a representative simultaneously from every set in an arbitrary family of nonempty sets. A rigorous justification for repeatedly choosing distinct elements from an infinite set may therefore involve choice.
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
>
> &nbsp;
> **1–3.** Equal cardinality is defined by a bijection. The surjection version of Schröder–Bernstein says that surjections in both directions imply a bijection. An infinite set may be in bijection with the result of adjoining one new element.<br>
> **4–8.** A diagonal argument constructs an infinite sequence missing from any proposed list; therefore $\mathbb N$ cannot surject onto all infinite bit strings. Cantor's theorem states that $|A|<|\mathcal P(A)|$.<br>
> **9.** The countable choices are $\mathbb N,\mathbb Z,\mathbb N^2,\mathbb Q,\mathbb Z^+$, and the set of bit strings of any fixed finite length. The sets $\mathbb R,\mathbb C,\{0,1\}^{\omega},\mathbb Q^{\omega}$ are uncountable.<br>
> **10.** The Halting Problem asks whether a program eventually halts; it is undecidable, and the proof uses diagonalization and contradiction.<br>
> **11–12.** Russell's paradox arises from the false assumption that $W$ is a set. ZFC avoids unrestricted comprehension: it is formulated in predicate logic and includes axioms such as extensionality, power set, and foundation, but it does not permit unrestricted set formation.<br>
> **13–23.** The four axioms identified in order are Power Set, Comprehension/Separation, Foundation, and Extensionality. For the given membership cycles, the answers to whether $\varnothing,T,\{T\},\{V,T\},\{U,V,T\},\{T,U,V\},\{\varnothing,T,U,V\}$ contain an $\in$-minimal element are respectively No, Maybe, Yes, Yes, No, No, and Yes.<br>
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
> **1. Infinite sets and sequences.** (a) “Keep choosing a new element” can be formalised in standard ZFC. In weaker set theories without choice, however, the statement that every infinite set contains a countably infinite subset is not automatic, so the concern is legitimate. (b) For a chosen sequence of distinct elements $a_0,a_1,\dots$, define $f(a_n)=n$ and send every other $a\in A$ to $0$. This gives a surjection $A\twoheadrightarrow\mathbb N$.
> **2. The image of a surjection from $\mathbb N$ is countable.** If $f:\mathbb N\twoheadrightarrow S$, define for each $s\in S$
> $$g(s)=\min\{n\in\mathbb N:f(n)=s\}.$$
> The set inside the minimum is nonempty, and WOP guarantees a least element. If $g(s)=g(t)$, then $s=f(g(s))=f(g(t))=t$, so $g:S\hookrightarrow\mathbb N$ and $S$ is countable. This formalises “filtering out duplicates during an enumeration.”
> **3. The positive rationals are countable.** (a) For $a,b\in\mathbb Z^+$, the Cantor pairing function
> $$\pi(a,b)=\frac{(a+b-2)(a+b-1)}2+a$$
> is a bijection to $\mathbb Z^+$: each diagonal $a+b$ occupies a consecutive interval, and $a$ uniquely locates a pair within that interval. (b) The map $(a,b)\mapsto a/b$ is a surjection from $\mathbb Z^+\times\mathbb Z^+$ onto $\mathbb Q^+$. By Problem 2, its image is countable.
> **4. An unrecognizable language.** (a) The recognizer checks that the string has even length and that every adjacent pair consists of the same letter from `a`–`z`; otherwise it returns False. (b) Compile each string $s$ as a program $P_s$, and let $f(s)$ be the language recognized by that program. Then $\operatorname{range}(f)$ is exactly the class of recognizable languages. (c) Define $N=\{s:s\notin f(s)\}$. If $N$ were recognizable, then $N=f(t)$ for some program string $t$; hence
> $$t\in N\Longleftrightarrow t\notin f(t)=N,$$
> a contradiction. (d) Therefore no universal analyser can decide every nontrivial behavioural property of arbitrary programs exactly. Particular restricted classes of programs may still be decidable; the conclusion is not that all program analysis is impossible.
> <!-- bilingual-en:end -->

> [!question]- 三道自检
> 1. 一个 surjection $\mathbb N\to S$ 为什么可能重复但仍足以说明 $S$ 可数？
> 2. Cantor 集合 $D$ 的定义为什么必须依赖 $f(a)$？
> 3. 停机不可判定是否等于任何实例都不能判断？
>
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Why can a surjection $\mathbb N\to S$ contain repetitions and still suffice to prove that $S$ is countable?<br>
> **2.** Why must the definition of Cantor's set $D$ depend on $f(a)$?<br>
> **3.** Does undecidability of the Halting Problem mean that no individual instance can be decided?<br>
> <!-- bilingual-en:end -->
> [!success]- 答案
> 1. 为每个 $s$ 取首次出现位置得到 injection。2. 它要在每个候选 $f(a)$ 的“自身坐标”上反转，才能保证与每个候选不同。3. 不是；许多具体程序显然停或不停，不存在的是覆盖全部程序输入的总正确算法。
> <!-- bilingual-en:start -->
>
> &nbsp;
> **1.** Assign each $s$ the position of its first occurrence to obtain an injection.<br>
> **2.** The construction must flip each candidate $f(a)$ at its own coordinate $a$ so that $D$ is guaranteed to differ from every candidate.<br>
> **3.** No. Many particular programs can plainly be shown to halt or to run forever; what does not exist is a total, always-correct algorithm covering every program-input pair.<br>
> <!-- bilingual-en:end -->

**知识链：**bijection → countability → diagonalization → power set → Russell/Halting self-reference。
<!-- bilingual-en:start -->
**Knowledge chain:** bijection → countability → diagonalisation → power set → self-reference in Russell's paradox and the Halting Problem.
<!-- bilingual-en:end -->

---

## Problem Set 4 — Sessions 9–11

> [!note] 原题与答案性质
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps4.pdf|Problem Set 4 PDF]]。以下 3 道均为非官方独立题解。
> <!-- bilingual-en:start -->
> [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps4.pdf|Problem Set 4 PDF]]. The three solutions below are independent and unofficial.
> <!-- bilingual-en:end -->

> [!example]- Problem 1：网格机器人不可达 $(1,1)$
> 四种动作是 $\pm(2,-1)$ 与 $\pm(1,3)$。考察
> $$I(x,y)=3x-y\pmod7.$$
> 对 $(2,-1)$，$\Delta I=3\cdot2-(-1)=7\equiv0$；对 $(1,3)$，$\Delta I=3-3=0$，反向动作同样为 0。因此 $I$ 是 preserved invariant。起点 $I(0,0)=0$，而目标 $I(1,1)=2\not\equiv0\pmod7$，所以不可达。
> <!-- bilingual-en:start -->
> The four moves are $\pm(2,-1)$ and $\pm(1,3)$. Consider
> $$I(x,y)=3x-y\pmod7.$$
> For $(2,-1)$, $\Delta I=3\cdot2-(-1)=7\equiv0$; for $(1,3)$, $\Delta I=3-3=0$, and the reverse moves also change $I$ by $0$ modulo $7$. Thus $I$ is a preserved invariant. The start has $I(0,0)=0$, while the target has $I(1,1)=2\not\equiv0\pmod7$, so the target is unreachable.
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
> Use structural induction on the LBT. The base $\langle l,leaf\rangle$ has $f_B=1$ and $n_B=0$, so $f_B=n_B+1$.
> For the constructor $T=\langle l,B,C\rangle$, the unique-label condition guarantees that the leaf-label sets of $B$ and $C$ are disjoint, so
> $$f_T=f_B+f_C.$$
> It also guarantees that the internal-label sets of the two subtrees are disjoint and that the new label $l$ has not appeared before, so
> $$n_T=n_B+n_C+1.$$
> By the induction hypotheses,
> $$f_T=(n_B+1)+(n_C+1)=n_B+n_C+2=n_T+1.$$
> The unique-label condition is used precisely when the cardinality of a union is written as a sum. If labels overlap, this equation double-counts them.
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
> Its inverse is $h^{-1}(y)=1/(y+1)$, so $h$ is bijective.
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
*Unit 1 summary: how to complete a proof independently*
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

&nbsp;
**1.** State the universe of discourse, free variables, and quantifiers.<br>
**2.** Recast the goal as an implication $P\Rightarrow Q$, a set equality, a universal statement, an unreachability claim, or an equinumerosity problem.<br>
**3.** Choose a method that matches the goal: direct proof, contrapositive, contradiction, cases, induction, invariant, or bijection.<br>
**4.** Write down the objects that method requires: a counterexample set, induction predicate, states and transitions, invariant, or a map and its inverse.<br>
**5.** Check zero, the empty set, the smallest case, the domain, possible zero denominators, the required base cases, and termination separately.<br>
**6.** Test the strength of the conclusion against a small example or counterexample.<br>
<!-- bilingual-en:end -->

### 覆盖核对
<!-- bilingual-en:start -->
*Coverage check*
<!-- bilingual-en:end -->

| 来源 | 已覆盖数量 | 位置 |
|---|---:|---|
| Official online feedback prompts | 122 | 各 Session “官方在线 feedback exercises” |
| In-Class main problems | 48 | CP1–CP11 折叠题解 |
| Problem Set main problems | 13 | PS1–PS4 |
| Midterm 1 main problems | 5 | Midterm 1 |
| Session self-checks | 33 | 每节 3 道 |
<!-- bilingual-en:start -->
| Source | Number covered | Location |
|---|---:|---|
| Official online feedback prompts | 122 | “Official online feedback exercises” in each session |
| In-class main problems | 48 | CP1–CP11 collapsible solution sections |
| Problem Set main problems | 13 | PS1-PS4 |
| Midterm 1 main problems | 5 | Midterm 1 |
| Session self-checks | 33 | Three questions per session |
<!-- bilingual-en:end -->

下一单元从证明语言转入可计算结构：[[02_Structures|Unit 2: Structures]]。
<!-- bilingual-en:start -->
The next unit moves from the language of proof to computable structures: [[02_Structures|Unit 2: Structures]].
<!-- bilingual-en:end -->

## 课程笔记反链
<!-- bilingual-en:start -->
*Course-note backlinks*
<!-- bilingual-en:end -->

```dataview
LIST
FROM "01_Math/07-Mathematics for Computer Science"
WHERE contains(file.outlinks, this.file.link)
SORT file.name ASC
```
