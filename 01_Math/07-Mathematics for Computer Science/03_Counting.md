---
aliases:
  - MIT 6.042J Unit 3 Counting
  - MIT 6.042J 计数
  - Counting
tags:
  - course/MIT-6.042J
  - math/discrete-mathematics
  - topic/counting
course: MIT 6.042J Mathematics for Computer Science
unit: 3
sessions: 23-27
source: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/
status: complete
---

# Unit 3 — Counting

> [!abstract] 本章要解决什么
> 计数并不只是“套排列组合公式”。本章建立一条完整的方法链：先把复杂对象转成和、积、积分或已知集合；再用渐近记号只保留规模；最后用双射、除法、二项式、多项式、鸽巢与容斥精确计数。读完应能回答三类问题：**一个和有多大？一个集合有多少元素？不精确知道对象时，仍能保证什么必然发生？**
> <!-- bilingual-en:start -->
> Counting is not just a collection of nested binomial-coefficient formulas. This chapter develops a complete chain of methods: translate complex objects into sums, products, integrals, or familiar sets; use asymptotic notation when only scale matters; then count exactly with bijections, division, binomial and multinomial coefficients, the pigeonhole principle, and inclusion–exclusion. The chapter answers three kinds of questions: **What is the total and how large is it? How many objects does a set contain? What must happen even when the exact arrangement is unknown?**
> <!-- bilingual-en:end -->

本笔记严格依照 MIT OCW Spring 2015 Unit 3 的 block/video 顺序整理：Session 23 → Session 24 → Problem Set 9 → Midterm 3 → Session 25 → Session 26 → Session 27 → Problem Set 10。课程入口见 [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/)；本地总索引见 [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]]。
<!-- bilingual-en:start -->
This notebook is organized in the block/video order of MIT OCW Spring 2015 Unit 3: Session 23 → Session 24 → Problem Set 9 → Midterm 3 → Session 25 → Session 26 → Session 27 → Problem Set 10.  For the course entry see [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/); for the local total index see [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]].
<!-- bilingual-en:end -->

> [!info] 题解来源与严谨性
> - 标为“官方反馈”的在线题答案来自本地静态课程包，逐题链接原 Markdown。
> - Classroom Problems、Problem Set 9/10 与 Midterm 3 的课程包只提供题目；下文均为**非官方独立题解**，并给出可检查的推导。
> - Stirling 公式的主渐近式在正文给出证明路线；Robbins 型精细余项界被明确标成加强定理，不假装是视频中已完整证明的内容。
> <!-- bilingual-en:start -->
> - Answers to online questions marked "Official Feedback" come from a local static course package, linking to the original Markdown on a topic-by-topic basis.
> - The course package provides only the questions for the Classroom Problems, Problem Sets 9/10, and Midterm 3. The solutions below are therefore **unofficial independent solutions**, with derivations that can be checked line by line.
> - The body proves the main asymptotic form of Stirling's formula. The sharper Robbins remainder bound is explicitly marked as an additional theorem rather than being presented as something fully proved in the course video.
> <!-- bilingual-en:end -->

## 课程导航与覆盖

| 顺序 | 内容 | 官方 block | 在线 prompt | 课堂题 | 评量 |
|---:|---|---:|---:|---:|---|
| 1 | Session 23：Sums & Products | 13 | 16 | 5 | — |
| 2 | Session 24：Asymptotics | 9 | 21 | 5 | — |
| 3 | Problem Set 9 | — | — | — | 3 题 |
| 4 | Midterm 3 | — | — | — | 6 题及全部子问 |
| 5 | Session 25：Counting with Bijections | 5 | 4 | 4 | — |
| 6 | Session 26：Repetitions & Binomial Theorem | 5 | 1 | 5 | — |
| 7 | Session 27：Pigeonhole & Inclusion–Exclusion | 7 | 11 | 5 | — |
| 8 | Problem Set 10 | — | — | — | 3 题及全部子问 |
| **合计** |  | **39** | **53** | **24** | **15 大题** |

> [!tip] 推荐使用方式
> 第一次阅读按正文顺序走；每个 Session 末先做三道自检题，再展开答案。复习时直接看“知识链小结”和带编号的题解。在线题编号 `O23-01` 表示 Session 23 online prompt 1；课堂题编号 `C23-1` 表示 Session 23 classroom problem 1。
> <!-- bilingual-en:start -->
> On a first pass, read the main text in order and attempt the three self-check questions at the end of each Session before revealing the answers. For review, go directly to the “knowledge-chain summary” and the numbered solutions. `O23-01` means online prompt 1 in Session 23; `C23-1` means classroom problem 1 in Session 23.
> <!-- bilingual-en:end -->

---

## Session 23 — Sums & Products

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：怎样把长和式化成闭式？闭式不存在时怎样给出可靠上下界？乘积为什么可以通过对数变成和？阶乘究竟增长多快？
<!-- bilingual-en:start -->
**Learning questions**: How can a long sum be converted into a closed form? How can reliable upper and lower bounds be obtained when no closed form is available? Why does taking logarithms turn products into sums? How quickly does the factorial grow?
<!-- bilingual-en:end -->

**前置知识**：有限求和符号、极限、导数与定积分、归纳法、质心。首次正式使用 [[无穷级数与幂级数#数项级数与必要条件|级数]]、[[定积分与微积分基本定理#从黎曼和到定积分|积分]] 与 [[货币时间价值与贴现#年金、永续年金与增长永续年金|年金]]。
<!-- bilingual-en:start -->
**Prerequisites**: Finite Sum Symbol, Limits, Derivatives and Definite Integrals, Inductive Method, Centroid.  First live use of [[无穷级数与幂级数#数项级数与必要条件|series]], [[定积分与微积分基本定理#从黎曼和到定积分|Integral]], and [[货币时间价值与贴现#年金、永续年金与增长永续年金|Annuity]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session23.pdf#page=1|Session 23 reading, pp. 1–24]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp23.pdf#page=1|cp23, pp. 1–3]]

### 3.1.1 Arithmetic Sums — 配对扰动
<!-- bilingual-en:start -->
*3.1.1 Arithmetic Sums — Pair Disturbance*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Arithmetic.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/v6axtBS6IF8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=v6axtBS6IF8)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Arithmetic.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/v6axtBS6IF8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=v6axtBS6IF8)
<!-- bilingual-en:end -->

设等差数列首项为 $a$、公差为 $d$、共有 $n$ 项：
<!-- bilingual-en:start -->
Let an arithmetic progression have first term $a$, common difference $d$, and $n$ terms:
<!-- bilingual-en:end -->

$$
S=a+(a+d)+\cdots+[a+(n-1)d].
$$

**目标**：求 $S$。把同一和式逆序写出：
<!-- bilingual-en:start -->
**Goal:** find $S$. Write the same sum in reverse order:
<!-- bilingual-en:end -->

$$
S=[a+(n-1)d]+[a+(n-2)d]+\cdots+a.
$$

逐列相加，每一列都等于 $2a+(n-1)d$，共 $n$ 列，所以
<!-- bilingual-en:start -->
Adding the two displayed sums term by term, each of the $n$ paired terms equals $2a+(n-1)d$, so
<!-- bilingual-en:end -->

$$
2S=n[2a+(n-1)d],
\qquad
\boxed{S=\frac n2[2a+(n-1)d]}.
$$

这不是“背公式”，而是一次**扰动（perturbation）**：构造另一个容易与原式相消或配对的式子。算法分析中常用同一思想处理递推和式。
<!-- bilingual-en:start -->
This is not a formula to memorize; it is a **perturbation** argument: construct a second expression that pairs or cancels conveniently with the original. The same idea often handles recurrence sums in algorithm analysis.
<!-- bilingual-en:end -->

### 3.1.2 Perturbation by Young Gauss — 官方在线题 O23-01
<!-- bilingual-en:start -->
*3.1.2 Perturbation by Young Gauss — official online O23-01*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.2_perturbation-by-young-gauss|3.1.2]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.2_perturbation-by-young-gauss|3.1.2]]
<!-- bilingual-en:end -->

> [!question] O23-01
> 年幼的 Gauss 用什么扰动快速求从 89 开始、公差 13 的 30 个整数之和？
> <!-- bilingual-en:start -->
> What rearrangement does young Gauss use to sum quickly the 30-term arithmetic progression with first term 89 and common difference 13?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 把和式与其逆序形式相加，使对应项的和相同。官方材料写成 $30(89+479)$ 后再除以 $2$；其要点是“正序 + 逆序”，而不是先调用现成闭式。注意：若严格按“首项 89、公差 13、共 30 项”，末项应为 $89+29\cdot13=466$；官方题面中的 479 与这三个数据存在一项偏差。本笔记保留官方答案，同时把这个边界不一致明确指出。
> <!-- bilingual-en:start -->
> Add the sum in forward order to the same sum in reverse order, so that every paired term has the same total. The official material writes the result as $30(89+479)/2$; the point is the “forward + reverse” pairing, not merely invoking a memorised formula. Strictly following “first term 89, common difference 13, 30 terms” gives the last term $89+29\cdot13=466$, so 479 is inconsistent with the other three data. This notebook retains the official answer while making that discrepancy explicit.
> <!-- bilingual-en:end -->

### 3.1.3 Geometric Sums — 移位相减
<!-- bilingual-en:start -->
*3.1.3 Geometric Sums—Shift Subtraction*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_GeometricSum.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ZDQk45NQbEo.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=ZDQk45NQbEo)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_GeometricSum.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ZDQk45NQbEo.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=ZDQk45NQbEo)
<!-- bilingual-en:end -->

#### 有限几何和
<!-- bilingual-en:start -->
*finite geometric sum*
<!-- bilingual-en:end -->

[[组合计数原理#求和与渐近|几何和（geometric sum）]]的公比固定。令
<!-- bilingual-en:start -->
A [[组合计数原理#求和与渐近|geometric sum]] has a fixed common ratio. Let
<!-- bilingual-en:end -->

$$
S_n=1+x+x^2+\cdots+x^n.
$$

**目标**：让大多数项消掉。乘以 $x$ 后移位：
<!-- bilingual-en:start -->
**Goal**: arrange for most terms to cancel. Multiplying by $x$ shifts every term one place:
<!-- bilingual-en:end -->

$$
xS_n=x+x^2+\cdots+x^n+x^{n+1}.
$$

相减得到 $(1-x)S_n=1-x^{n+1}$。因此，当 $x\ne1$ 时，
<!-- bilingual-en:start -->
Subtract to get $(1-x)S_n=1-x^{n+1}$.  So, when $x\ne1$,
<!-- bilingual-en:end -->

$$
\boxed{S_n=\frac{1-x^{n+1}}{1-x}}.
$$

当 $x=1$ 时不能除以 $1-x$，必须回到原定义：$S_n=n+1$。这正是公式的适用边界。
<!-- bilingual-en:start -->
Cannot divide by $1-x$ when $x=1$, must return to the original definition: $S_n=n+1$.  This is the boundary of the formula.
<!-- bilingual-en:end -->

#### 无限几何级数及其必要条件
<!-- bilingual-en:start -->
*Infinite Geometric Series and Its Necessary Conditions*
<!-- bilingual-en:end -->

所谓无限和，是部分和的极限：
<!-- bilingual-en:start -->
The so-called infinite sum is the limit of the partial sum:
<!-- bilingual-en:end -->

$$
\sum_{i=0}^{\infty}x^i:=\lim_{n\to\infty}S_n.
$$

若 $|x|<1$，则 $x^{n+1}\to0$，所以
<!-- bilingual-en:start -->
If $|x|<1$, then $x^{n+1}\to0$, so
<!-- bilingual-en:end -->

$$
\boxed{\sum_{i=0}^{\infty}x^i=\frac1{1-x}}.
$$

若 $|x|\ge1$，通项 $x^i$ 不趋于 $0$（$x=-1$ 时来回振荡），而任何收敛级数的通项都必须趋于 $0$：若部分和 $s_n\to s$，则 $x^n=s_n-s_{n-1}\to s-s=0$。故此时不收敛。
<!-- bilingual-en:start -->
If $|x|\ge1$, the term $x^i$ does not tend to $0$ (for $x=-1$ it oscillates). Every convergent series must have terms tending to zero: if its partial sums satisfy $s_n\to s$, then $x^n=s_n-s_{n-1}\to0$. Hence the geometric series diverges in this case.
<!-- bilingual-en:end -->

#### 对几何和求导：带权和
<!-- bilingual-en:start -->
*Deriving for Geometric Sums: Weighted Sums*
<!-- bilingual-en:end -->

有限多项式可逐项求导：
<!-- bilingual-en:start -->
A finite polynomial may be differentiated term by term:
<!-- bilingual-en:end -->

$$
\frac{d}{dx}\sum_{i=0}^{n}x^i
=\sum_{i=1}^{n}i x^{i-1}.
$$

对闭式使用商法则：
<!-- bilingual-en:start -->
Applying the quotient rule to the closed form gives:
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{n}i x^{i-1}
=\frac{1-(n+1)x^n+n x^{n+1}}{(1-x)^2}.
$$

再乘 $x$：
<!-- bilingual-en:start -->
Multiply $x$:
<!-- bilingual-en:end -->

$$
\boxed{\sum_{i=1}^{n}i x^i
=\frac{x-(n+1)x^{n+1}+n x^{n+2}}{(1-x)^2}}.
$$

当 $|x|<1$，$n|x|^n\to0$，从而
<!-- bilingual-en:start -->
When $|x|<1$ $n|x|^n\to0$,
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{\infty}i x^i=\frac{x}{(1-x)^2}.
$$

### 3.1.4 Annuities — 官方在线题 O23-02
<!-- bilingual-en:start -->
*3.1.4 Annuities — official online O23-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.4_annuities|3.1.4]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.4_annuities|3.1.4]]
<!-- bilingual-en:end -->

一笔在第 $i$ 年末收到的 $m$ 元，若年利率为 $p$，今天的现值是 $m/(1+p)^i$。从一年后开始、永久每年支付 $m$ 的永续年金现值为
<!-- bilingual-en:start -->
A sum of $m$ received at the end of $i$, with an annual interest rate of $p$, is now $m/(1+p)^i$.  The present value of perpetuity for permanent annual payments of $m$, commencing after one year
<!-- bilingual-en:end -->

$$
V=\sum_{i=1}^{\infty}\frac{m}{(1+p)^i}
=\frac{m}{1+p}\cdot\frac1{1-1/(1+p)}
=\boxed{\frac mp}.
$$

若第一次支付就在今天，则还要加 $m$，现值为 $m(1+p)/p$。时间点差一年，答案便差一个 $m$，这是金融题最常见的错误源。
<!-- bilingual-en:start -->
If the first payment is today, then add $m$, the present value is $m(1+p)/p$.  By a year, the answer is a $m$, the most common source of error for financial questions.
<!-- bilingual-en:end -->

> [!question] O23-02
> 年收益率恒为 $4\%$，从一年后起每年永久支付 $10{,}000$ 美元，今天应投入多少？
> <!-- bilingual-en:start -->
> With an annual rate of return of $4\%$ and a permanent payment of $10{,}000$ dollars a year from a year later, how much should I invest today?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> $V=10{,}000/0.04=\boxed{250{,}000}$。等价直觉：$250{,}000$ 每年的 $4\%$ 利息恰为 $10{,}000$，不动本金即可永久支付。
> <!-- bilingual-en:start -->
> $V=10{,}000/0.04=\boxed{250{,}000}$.  Equivalent Intuition: $250{,}000$'s annual interest rate of $4\%$ is exactly $10{,}000$, and the principal can be paid permanently without any change.
> <!-- bilingual-en:end -->

### 3.1.5 Book Stacking — 调和数从质心出现
<!-- bilingual-en:start -->
*3.1.5 Book Stacking — harmonic numbers appear from the center of mass*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BookStacking.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CdhuVhWTSMI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CdhuVhWTSMI)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BookStacking.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CdhuVhWTSMI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CdhuVhWTSMI)
<!-- bilingual-en:end -->

每本书长度归一化为 $1$、质量相同。要使最上方 $n$ 本书在下一本书边缘上恰好稳定，它们的共同质心必须落在该边缘正上方。
<!-- bilingual-en:start -->
Each book is normalized to be $1$ in length and of the same quality.  For the topmost $n$ book to be just right on the edge of the next one, their common center of mass must fall just above that edge.
<!-- bilingual-en:end -->

设已有 $n$ 本书相对其支撑边缘的最大伸出量为 $B_n$。在下面加第 $n+1$ 本书并把上面 $n$ 本整体向右移动 $\Delta$。以新书中心为力矩原点：上面 $n$ 本的总质量为 $n$，其质心向右 $\Delta$；新书质量为 $1$，其中心距右边缘 $1/2$，向左的力臂是 $1/2-\Delta$。临界平衡要求
<!-- bilingual-en:start -->
Suppose there is already a $n$ book with a maximum protrusion of $B_n$ from its supporting edge.  Add the $n+1$ book underneath and move the $n$ book entirely to the right, $\Delta$.  Taking the center of the new book as the moment origin: the total mass of the $n$ book on the top is $n$, and its centroid is $\Delta$ to the right; the mass of the new book is $1$, the center is $1/2$ from the right edge, and the left arm of the force is $1/2-\Delta$.  critical equilibrium requirement
<!-- bilingual-en:end -->

$$
n\Delta=\frac12-\Delta,
\qquad
\Delta=\frac1{2(n+1)}.
$$

于是
<!-- bilingual-en:start -->
therefore
<!-- bilingual-en:end -->

$$
B_{n+1}=B_n+\frac1{2(n+1)},\qquad B_1=\frac12,
$$

迭代得
<!-- bilingual-en:start -->
iterative
<!-- bilingual-en:end -->

$$
\boxed{B_n=\frac12\left(1+\frac12+\cdots+\frac1n\right)=\frac12H_n},
$$

其中 [[组合计数原理#求和与渐近|调和数]] $H_n=\sum_{k=1}^{n}1/k$。
<!-- bilingual-en:start -->
where [[组合计数原理#求和与渐近|harmonic number]] $H_n=\sum_{k=1}^{n}1/k$.
<!-- bilingual-en:end -->

由于调和级数发散，理论上可以把书伸出任意远；但 $H_n$ 只像 $\ln n$ 增长，代价呈指数级。
<!-- bilingual-en:start -->
Because of the divergence of the harmonic series, the book can theoretically be extended any distance; but $H_n$ only grows like $\ln n$ at an exponential cost.
<!-- bilingual-en:end -->

### 3.1.6 Harmonic Numbers — 官方在线题 O23-03
<!-- bilingual-en:start -->
*3.1.6 Harmonic Numbers — official online O23-03*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.6_harmonic-numbers|3.1.6]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.6_harmonic-numbers|3.1.6]]
<!-- bilingual-en:end -->

> [!question] O23-03
> 在四个陈述中选真命题：调和数有简单和式定义；$H_n=\sum_{i=0}^n1/i$；$n$ 本书伸出量为 $H_n$；新增一本时 $\Delta=\frac{1/2}{n+1}$ 来自 $n\Delta=1(1/2-\Delta)$。
> <!-- bilingual-en:start -->
> Select the true proposition in four statements: harmonic numbers are simply and formally defined; $H_n=\sum_{i=0}^n1/i$; $n$ book protrusions are $H_n$; and when a new one is added, $\Delta=\frac{1/2}{n+1}$ is from $n\Delta=1(1/2-\Delta)$.
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 第 1、4 项正确。定义从 $i=1$ 开始，$H_n=\sum_{i=1}^n1/i$；最大伸出量是 $H_n/2$，不是 $H_n$。
> <!-- bilingual-en:start -->
> Item 1 and 4 are correct.  Definitions begin with $i=1$, $H_n=\sum_{i=1}^n1/i$; maximum protrusion is $H_n/2$, not $H_n$.
> <!-- bilingual-en:end -->

### 3.1.7 Integral Method — 用面积夹住离散和
<!-- bilingual-en:start -->
*3.1.7 Integral Method —Squeeze discrete and*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_IntegralMeth.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/EegG5TPL29c.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=EegG5TPL29c)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_IntegralMeth.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/EegG5TPL29c.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=EegG5TPL29c)
<!-- bilingual-en:end -->

令 $f:\mathbb R^+\to\mathbb R^+$，
<!-- bilingual-en:start -->
Let $f:\mathbb R^+\to\mathbb R^+$,
<!-- bilingual-en:end -->

$$
S=\sum_{i=1}^{n}f(i),\qquad I=\int_1^n f(x)\,dx.
$$

#### 单调递增情形的完整证明
<!-- bilingual-en:start -->
*Full proof of monotonically increasing case*
<!-- bilingual-en:end -->

若 $f$ 弱递增，则对每个整数 $i=1,\ldots,n-1$ 以及 $x\in[i,i+1]$，
<!-- bilingual-en:start -->
if $f$ is weak increment, $i=1,\ldots,n-1$ and $x\in[i,i+1]$ for each integer,
<!-- bilingual-en:end -->

$$
f(i)\le f(x)\le f(i+1).
$$

区间长度为 $1$，积分后得
<!-- bilingual-en:start -->
The interval length is $1$, the integral is
<!-- bilingual-en:end -->

$$
f(i)\le\int_i^{i+1}f(x)\,dx\le f(i+1).
$$

对 $i=1$ 到 $n-1$ 求和：
<!-- bilingual-en:start -->
Sum $i=1$ to $n-1$:
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{n-1}f(i)\le I\le\sum_{i=2}^{n}f(i).
$$

左式加 $f(n)$、右式加 $f(1)$ 并改写为 $S$：
<!-- bilingual-en:start -->
Left with $f(n)$, Right with $f(1)$ and rewritten as $S$:
<!-- bilingual-en:end -->

$$
\boxed{I+f(1)\le S\le I+f(n)}.
$$

#### 单调递减情形的完整证明
<!-- bilingual-en:start -->
*Complete proof of monotonic decreasing case*
<!-- bilingual-en:end -->

若 $f$ 弱递减，不等号方向反转：对 $x\in[i,i+1]$，
<!-- bilingual-en:start -->
If $f$ is weakly decreasing, the direction of the unequal sign is reversed: for $x\in[i,i+1]$,
<!-- bilingual-en:end -->

$$
f(i+1)\le f(x)\le f(i).
$$

同样积分、求和得到
<!-- bilingual-en:start -->
Equal Integral, Sum
<!-- bilingual-en:end -->

$$
\sum_{i=2}^{n}f(i)\le I\le\sum_{i=1}^{n-1}f(i),
$$

即
<!-- bilingual-en:start -->
that is
<!-- bilingual-en:end -->

$$
\boxed{I+f(n)\le S\le I+f(1)}.
$$

严格单调只会让相应不等号变严格，不改变界的表达式。条件“正值”保证可用上下界判断收敛；“单调”保证每个单位区间上的矩形方向一致。
<!-- bilingual-en:start -->
Strict monotonicity only makes the corresponding inequality strict, without changing the boundary expression.  The condition "positive value" guarantees convergence by upper and lower bounds, and "monotone" guarantees the rectangle direction in each unit interval is consistent.
<!-- bilingual-en:end -->

#### 例：$\sum_{i=1}^{n}\sqrt i$
<!-- bilingual-en:start -->
*Example: $\sum_{i=1}^{n}\sqrt i$*
<!-- bilingual-en:end -->

$f(x)=\sqrt x$ 递增，且
<!-- bilingual-en:start -->
$f(x)=\sqrt x$ increments, and
<!-- bilingual-en:end -->

$$
I=\int_1^n\sqrt x\,dx=\frac23(n^{3/2}-1).
$$

所以
<!-- bilingual-en:start -->
therefore
<!-- bilingual-en:end -->

$$
\boxed{\frac23n^{3/2}+\frac13
\le\sum_{i=1}^{n}\sqrt i
\le\frac23n^{3/2}+\sqrt n-\frac23}.
$$

主导项两边相同，故 $\sum_{i=1}^n\sqrt i\sim\frac23n^{3/2}$。
<!-- bilingual-en:start -->
$\sum_{i=1}^n\sqrt i\sim\frac23n^{3/2}$.
<!-- bilingual-en:end -->

#### 调和数的界与发散
<!-- bilingual-en:start -->
*Boundary and Divergence of Harmonic Numbers*
<!-- bilingual-en:end -->

对递减函数 $f(x)=1/x$，直接应用定理得
<!-- bilingual-en:start -->
For the decreasing function $f(x)=1/x$, the theorem is directly applied to get
<!-- bilingual-en:end -->

$$
\ln n+\frac1n\le H_n\le1+\ln n.
$$

把每个 $1/i$ 看成区间 $[i,i+1]$ 上 $1/x$ 的左端矩形，还可得到更常用的下界
<!-- bilingual-en:start -->
Each $1/i$ is treated as a rectangle at the left end of the $1/x$ on the interval $[i,i+1]$, and more commonly used lower bounds can also be obtained
<!-- bilingual-en:end -->

$$
\boxed{\ln(n+1)\le H_n\le1+\ln n}.
$$

两边除以 $\ln n$ 均趋于 $1$，由夹逼定理
<!-- bilingual-en:start -->
The division of $\ln n$ between two sides tends to $1$, which is determined by the pinch theorem.
<!-- bilingual-en:end -->

$$
H_n\sim\ln n.
$$

且下界 $\ln(n+1)\to\infty$，所以 [[无穷级数与幂级数#数项级数与必要条件|调和级数发散]]。
<!-- bilingual-en:start -->
and the lower bound is $\ln(n+1)\to\infty$, so [[无穷级数与幂级数#数项级数与必要条件|harmonic series divergence]].
<!-- bilingual-en:end -->

### 3.1.8 Integral Method Demystified — 官方在线题 O23-04 至 O23-11
<!-- bilingual-en:start -->
*3.1.8 Integral Method Demystified — official online questions O23-04 through O23-11*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.8_integral-method-demystified|3.1.8]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.8_integral-method-demystified|3.1.8]]
<!-- bilingual-en:end -->

| 编号 | 问题 | 官方答案 | 为什么 |
|---|---|---|---|
| O23-04 | $f$ 弱递增时 $S$ 的上界 | $I+f(n)$ | 右端矩形在曲线上方 |
| O23-05 | $f$ 弱递减时 $S$ 的上界 | $I+f(1)$ | 左端矩形在曲线上方 |
| O23-06 | $f$ 弱递增时 $S$ 的下界 | $I+f(1)$ | 去掉末项后矩形在曲线下方 |
| O23-07 | $f$ 弱递减时 $S$ 的下界 | $I+f(n)$ | 去掉首项后矩形在曲线下方 |
| O23-08 | 严格单调是否改变界的表达式 | 否 | 只把 $\le$ 改为 $<$ |
| O23-09 | $H_n$ 的上界 | $1+\ln n$ | 递减情形 |
| O23-10 | $H_n$ 的下界 | $\ln(n+1)$ | 用 $[i,i+1]$ 面积求和 |
| O23-11 | $H_n$ 的渐近等价 | $\ln n$ | 上下界之比都趋于 $1$ |
<!-- bilingual-en:start -->
|Number|Question|Official Answer|Why|
|—|—|—|—|
| O23-04 | Upper bound of $S$ when $f$ is weakly increasing | $I+f(n)$ | Right rectangle over curve |
| O23-05 | Upper bound of $S$ when $f$ is weakly decreasing | $I+f(1)$ | Left rectangle over curve |
| O23-06 | Lower Bound of $S$ for Weak Increment of $f$ | $I+f(1)$ | Rectangle Below Curve After Removing End Term |
| O23-07 | Lower Bound of $S$ when $f$ is weakly decreasing | $I+f(n)$ | Rectangle under curve after first term is removed |
| O23-08 | Strictly monotonous Whether to change the boundary expression | No | Change $\le$ to $<$ only |
| O23-09 | Upper bound for $H_n$ | $1+\ln n$ | Decrementing |
| O23-10 | Lower bound of $H_n$ | $\ln(n+1)$ | Sum by $[i,i+1]$ area |
| O23-11 | Asymptotic equivalence of $H_n$ | $\ln n$ | ratio of upper to lower bounds tends to $1$ |
<!-- bilingual-en:end -->

### 3.1.9 Stirling's Formula — 阶乘的规模
<!-- bilingual-en:start -->
*3.1.9 Stirling's Formula—Scale of the factorial*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StirlingForm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/lU_QT5GSuxI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=lU_QT5GSuxI)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StirlingForm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/lU_QT5GSuxI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=lU_QT5GSuxI)
<!-- bilingual-en:end -->

乘积先取对数：
<!-- bilingual-en:start -->
Product first logarithm:
<!-- bilingual-en:end -->

$$
\ln\prod_{i=1}^{n}f(i)=\sum_{i=1}^{n}\ln f(i).
$$

特别地，$\ln(n!)=\sum_{i=1}^n\ln i$。积分法先给出粗界
<!-- bilingual-en:start -->
In particular, $\ln(n!)=\sum_{i=1}^n\ln i$.  The coarse bound is given by the integral method
<!-- bilingual-en:end -->

$$
n\ln n-n+1\le\ln(n!)\le(n+1)\ln n-n+1,
$$

指数化后：
<!-- bilingual-en:start -->
Indexed:
<!-- bilingual-en:end -->

$$
\frac{n^n}{e^{n-1}}\le n!\le\frac{n^{n+1}}{e^{n-1}}.
$$

它说明主尺度约为 $(n/e)^n$，但缺少关键因子 $\sqrt{2\pi n}$。
<!-- bilingual-en:start -->
It shows that the main scale is about $(n/e)^n$, but the key factor $\sqrt{2\pi n}$ is missing.
<!-- bilingual-en:end -->

#### Stirling 主公式
<!-- bilingual-en:start -->
*The main Stirling formula*
<!-- bilingual-en:end -->

[[组合计数原理#求和与渐近|Stirling 公式]]给出 factorial 的相对误差渐近：
<!-- bilingual-en:start -->
[[组合计数原理#求和与渐近|Stirling's formula]] gives an asymptotic approximation with relative error tending to zero:
<!-- bilingual-en:end -->

$$
\boxed{n!\sim\sqrt{2\pi n}\left(\frac{n}{e}\right)^n}.
$$

更精细的 Robbins 形式为
<!-- bilingual-en:start -->
The finer Robbins form is
<!-- bilingual-en:end -->

$$
n!=\sqrt{2\pi n}\left(\frac{n}{e}\right)^n e^{\varepsilon_n},
\qquad
\frac1{12n+1}<\varepsilon_n<\frac1{12n}.
$$

#### 主公式的证明路线（Wallis + 收敛余项）
<!-- bilingual-en:start -->
*Proof of the main formula (Wallis product and a convergent remainder)*
<!-- bilingual-en:end -->

**目标**：证明比值
<!-- bilingual-en:start -->
**Goal**: prove that the ratio
<!-- bilingual-en:end -->

$$
a_n:=\frac{n!}{\sqrt n(n/e)^n}
$$

趋于 $\sqrt{2\pi}$。
<!-- bilingual-en:start -->
converges to $\sqrt{2\pi}$.
<!-- bilingual-en:end -->

**步骤 1：证明 $a_n$ 有有限正极限。** 计算相邻项对数差：
<!-- bilingual-en:start -->
**Step 1: Prove that $a_n$ has a finite positive limit.**Calculate the logarithmic difference between adjacent terms:
<!-- bilingual-en:end -->

$$
\ln\frac{a_{n+1}}{a_n}
=1-\left(n+\frac12\right)\ln\left(1+\frac1n\right).
$$

利用交错级数展开
<!-- bilingual-en:start -->
Using the alternating-series expansion
<!-- bilingual-en:end -->

$$
\ln(1+t)=t-\frac{t^2}{2}+\frac{t^3}{3}-\cdots\quad(0<t\le1),
$$

代入 $t=1/n$ 可得
<!-- bilingual-en:start -->
Substituting $t=1/n$ gives
<!-- bilingual-en:end -->

$$
\ln\frac{a_{n+1}}{a_n}=O\left(\frac1{n^2}\right).
$$

由于 $\sum 1/n^2$ 收敛，级数 $\sum\ln(a_{n+1}/a_n)$ 绝对收敛，故 $\ln a_n$ 收敛到某个实数，$a_n\to C>0$。
<!-- bilingual-en:start -->
Because $\sum 1/n^2$ converges, the series $\sum\ln(a_{n+1}/a_n)$ absolutely converges, so $\ln a_n$ converges to a real number, $a_n\to C>0$.
<!-- bilingual-en:end -->

**步骤 2：用 Wallis 乘积确定 $C$。** 令
<!-- bilingual-en:start -->
**Step 2: Determine $C$ using the Wallis product.** Let
<!-- bilingual-en:end -->

$$
I_m=\int_0^{\pi/2}\sin^m x\,dx.
$$

分部积分给出 $I_m=\frac{m-1}{m}I_{m-2}$；又因 $0<\sin x<1$，有 $I_{2n+1}<I_{2n}<I_{2n-1}$。把递推式展开并夹逼可得 Wallis 乘积
<!-- bilingual-en:start -->
Integration by parts gives $I_m=\frac{m-1}{m}I_{m-2}$. Since $0<\sin x<1$ on the interior of the interval, $I_{2n+1}<I_{2n}<I_{2n-1}$. Expanding the recurrence and applying the squeeze theorem yields the Wallis product
<!-- bilingual-en:end -->

$$
\frac\pi2=\prod_{k=1}^{\infty}\frac{(2k)^2}{(2k-1)(2k+1)}.
$$

将有限乘积改写为阶乘：
<!-- bilingual-en:start -->
Rewrite the finite product in terms of factorials:
<!-- bilingual-en:end -->

$$
W_n:=\prod_{k=1}^{n}\frac{(2k)^2}{(2k-1)(2k+1)}
=\frac{2^{4n}(n!)^4}{(2n)!^2(2n+1)}\longrightarrow\frac\pi2.
$$

由 $n!\sim C\sqrt n(n/e)^n$ 与 $(2n)!\sim C\sqrt{2n}(2n/e)^{2n}$，代入上式：
<!-- bilingual-en:start -->
Substitute $n!\sim C\sqrt n(n/e)^n$ and $(2n)!\sim C\sqrt{2n}(2n/e)^{2n}$ into this expression:
<!-- bilingual-en:end -->

$$
W_n\sim
\frac{2^{4n}C^4n^2(n/e)^{4n}}
{C^2(2n)(2n/e)^{4n}(2n+1)}
\longrightarrow\frac{C^2}{4}.
$$

与 $W_n\to\pi/2$ 比较，得到 $C^2/4=\pi/2$，故 $C=\sqrt{2\pi}$。主公式得证。
<!-- bilingual-en:start -->
Comparing this with $W_n\to\pi/2$ gives $C^2/4=\pi/2$, hence $C=\sqrt{2\pi}$. This proves the main asymptotic formula.
<!-- bilingual-en:end -->

> [!warning] 严格性边界
> 上述证明完整确定主渐近常数。精细的 $1/(12n+1)<\varepsilon_n<1/(12n)$ 还需对 $\ln(1+1/n)$ 余项作更细的单调估计；课程使用该加强结论时可直接引用，不能把它说成只由一次积分夹逼自动得到。
> <!-- bilingual-en:start -->
> The argument above determines the constant in the main asymptotic formula. The sharper bound $1/(12n+1)<\varepsilon_n<1/(12n)$ requires a more detailed monotonic estimate of the remainder in $\ln(1+1/n)$. It may be cited as a stronger result, but it does not follow automatically from a single integral squeeze.
> <!-- bilingual-en:end -->

### 3.1.10 Applying Stirling — 官方在线题 O23-12
<!-- bilingual-en:start -->
*3.1.10 Applying Stirling — official online O23-12*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.10_applying-stirling-s-formula|3.1.10]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.10_applying-stirling-s-formula|3.1.10]]
<!-- bilingual-en:end -->

> [!question] O23-12
> 化简 $\dfrac{(2n)!}{2^{2n}(n!)^2}$ 的渐近等价式。
> <!-- bilingual-en:start -->
> The asymptotic equivalence of $\dfrac{(2n)!}{2^{2n}(n!)^2}$ is simplified.
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success]- 官方答案与逐步化简
> $$
> \begin{aligned}
> \frac{(2n)!}{2^{2n}(n!)^2}
> &\sim\frac{\sqrt{4\pi n}(2n/e)^{2n}}
> {2^{2n}[\sqrt{2\pi n}(n/e)^n]^2}\\
> &=\frac{2\sqrt{\pi n}\,2^{2n}(n/e)^{2n}}
> {2^{2n}(2\pi n)(n/e)^{2n}}\\
> &=\boxed{\frac1{\sqrt{\pi n}}}.
> \end{aligned}
> $$
> 每一项的指数因子全部消掉，只剩平方根尺度。
> <!-- bilingual-en:start -->
> The exponential factor of each item is eliminated completely, and only the square root scale is left.
> <!-- bilingual-en:end -->

### 3.1.11 Convergence of Geometric Series — 官方在线题 O23-13
<!-- bilingual-en:start -->
*3.1.11 Convergence of Geometric Series — official online O23-13*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.11_convergence-of-geometric-series|3.1.11]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.11_convergence-of-geometric-series|3.1.11]]
<!-- bilingual-en:end -->

> [!question] O23-13
> $r\in\{0,-0.5,0.5,1\}$ 时，哪个公比使几何级数不收敛？
> <!-- bilingual-en:start -->
> For $r\in\{0,-0.5,0.5,1\}$, which common ratio makes the geometric series diverge?
> <!-- bilingual-en:end -->
<!-- bilingual-en:start -->

<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> $\boxed{r=1}$；几何级数收敛当且仅当 $|r|<1$。
> <!-- bilingual-en:start -->
> $\boxed{r=1}$;The geometric series converges if and only if $|r|<1$.
> <!-- bilingual-en:end -->

### 3.1.12 Summation — 官方在线题 O23-14、O23-15
<!-- bilingual-en:start -->
*3.1.12 Summation—official online questions O23-14, O23-15*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.12_summation|3.1.12]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.12_summation|3.1.12]]
<!-- bilingual-en:end -->

考察 $p$-级数 $\sum_{i=1}^{\infty}i^p$。对 $p\ne-1$，
<!-- bilingual-en:start -->
Examine the $p$-series $\sum_{i=1}^{\infty}i^p$.  For $p\ne-1$,
<!-- bilingual-en:end -->

$$
\int_1^N x^p\,dx=\frac{N^{p+1}-1}{p+1}.
$$

- 若 $p<-1$，$N^{p+1}\to0$，积分有限，递减正项级数由积分法收敛。
- 若 $p=-1$，就是调和级数，发散。
- 若 $p>-1$，$i^p\ge i^{-1}$（$i\ge1$），与调和级数逐项比较，发散。
<!-- bilingual-en:start -->
- If $p<-1$, then $N^{p+1}\to0$, so the improper integral is finite; the integral test gives convergence of the positive decreasing series.
- If $p=-1$, the series is harmonic and diverges.
- If $p>-1$, then $i^p\ge i^{-1}$ for $i\ge1$, so comparison with the harmonic series gives divergence.
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O23-14**：临界值 $\boxed{a=-1}$，即恰在 $p<-1$ 时收敛。
> **O23-15**：好方法是计算 $\int_1^\infty x^pdx$，以及与调和级数逐项比较；对 $p$ 或有限部分和做归纳不能解决无穷尾部。
> <!-- bilingual-en:start -->
> **O23-14:** The threshold is $\boxed{a=-1}$; the series converges exactly when $p<-1$.
> **O23-15:** A good approach is to compute $\int_1^\infty x^p\,dx$ and use termwise comparison with the harmonic series. Induction on the real parameter $p$ is not meaningful, and induction on finite partial sums cannot by itself settle the infinite tail.
> <!-- bilingual-en:end -->

### 3.1.13 Sum's Upper/Lower Bounds — 官方在线题 O23-16
<!-- bilingual-en:start -->
*3.1.13 Sum's Upper/Lower Bounds — official online O23-16*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.13_sum-s-upper-lower-bounds|3.1.13]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.13_sum-s-upper-lower-bounds|3.1.13]]
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->

$$
S=\sum_{n=1}^{57}\frac1{\sqrt[3]{n+7}},
\qquad f(x)=(x+7)^{-1/3}.
$$

$f$ 递减，所以界之差为 $f(1)-f(57)=1/2-1/4=1/4$。
<!-- bilingual-en:start -->
$f$ is decreasing, so the difference between the bounds is $f(1)-f(57)=1/2-1/4=1/4$.
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O23-16**：$\boxed{0.25}$。共同积分项为
> $$
> I=\frac32[(64)^{2/3}-(8)^{2/3}]=18,
> $$
> 因而下界 $18.25$、上界 $18.5$；本题无需先算 $I$。
> <!-- bilingual-en:start -->
> **O23-16:** $\boxed{0.25}$. The common integral term is $I=18$, so the lower and upper bounds are $18.25$ and $18.5$, respectively. Their difference can be found without first evaluating $I$.
> <!-- bilingual-en:end -->

### Session 23 易错点与反例
<!-- bilingual-en:start -->
*Session 23 Error-prone Points and Counterexamples*
<!-- bilingual-en:end -->

1. **无限和不是形式代数**：先定义部分和，再取极限；$|r|\ge1$ 时不能套 $1/(1-r)$。
2. **端点错一位**：$\sum_{i=0}^{n}$ 有 $n+1$ 项；对它求导后最高项是 $nx^{n-1}$。
3. **年金时间点**：今天首付与一年后首付相差一个 $m$。
4. **积分法必须看单调方向**：递增函数的右端矩形给上界，递减函数恰好相反。
5. **$\sim$ 不能只比较对数**：$\ln(n!)\sim n\ln n$ 并不直接推出 $n!\sim n^n$；指数化会放大较小的加性误差。
<!-- bilingual-en:start -->
1.**Infinite sums are not formal algebras**: define partial sums before taking limits; $1/(1-r)$ cannot be wrapped when $|r|\ge1$.
2.**Endpoint is out of place**: $\sum_{i=0}^{n}$ has a $n+1$ term; the highest term after derivation is $nx^{n-1}$.
3.**Annuity time point**: The difference between the down payment today and the down payment one year later is one $m$.
4.**Integral method must look at the monotonic direction**: the right rectangle of the increasing function gives the upper bound, the decreasing function is just the opposite.
5.**$\sim$ cannot compare only logarithms**: $\ln(n!)\sim n\ln n$ does not directly deduce $n!\sim n^n$; exponentiation magnifies small additive errors.
<!-- bilingual-en:end -->

### Session 23 自检
<!-- bilingual-en:start -->
*Session 23 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 23-1
> 求 $\sum_{i=0}^{n}3\cdot2^i$，并说明何时能取 $n\to\infty$。
> <!-- bilingual-en:start -->
> Request $\sum_{i=0}^{n}3\cdot2^i$ and specify when $n\to\infty$ is available.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 有限和为 $3(2^{n+1}-1)$。公比 $2$ 的绝对值不小于 $1$，故无限级数发散。
> <!-- bilingual-en:start -->
> Limited sum is $3(2^{n+1}-1)$.  The absolute value of the common ratio $2$ is not less than $1$, so the infinite series is divergent.
> <!-- bilingual-en:end -->

> [!question] 自检 23-2
> 用积分法给 $\sum_{i=1}^{n}i^2$ 一个足以证明 $\Theta(n^3)$ 的上下界。
> <!-- bilingual-en:start -->
> The $\sum_{i=1}^{n}i^2$ is given an upper bound and a lower bound sufficient to prove the $\Theta(n^3)$ by the integral method.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $f(x)=x^2$ 递增，$I=(n^3-1)/3$，所以
> $$
> \frac{n^3-1}{3}+1\le\sum_{i=1}^{n}i^2\le\frac{n^3-1}{3}+n^2.
> $$
> 两边均为 $\Theta(n^3)$。
> <!-- bilingual-en:start -->
> $f(x)=x^2$ increments, $I=(n^3-1)/3$, so
> Both sides are $\Theta(n^3)$.
> <!-- bilingual-en:end -->

> [!question] 自检 23-3
> 为什么 $H_n$ 发散却增长得非常慢？
> <!-- bilingual-en:start -->
> Why is $H_n$ diverging and growing very slowly?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 因为 $\ln(n+1)\le H_n\le1+\ln n$。下界无界保证发散，但要让 $H_n$ 增加固定量 $c$，$n$ 大约要乘以 $e^c$。
> <!-- bilingual-en:start -->
> Because of $\ln(n+1)\le H_n\le1+\ln n$.  The lower bound guarantees divergence, but to increase $H_n$ by a fixed amount of $c$, $n$ is approximately multiplied by $e^c$.
> <!-- bilingual-en:end -->

### Classroom Problems 23 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 23 — 5 Complete Independent Questions*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp23.pdf#page=1|cp23 pp. 1–3]]
<!-- bilingual-en:start -->
Original Question: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp23.pdf#page=1|cp23 pp. 1–3]]
<!-- bilingual-en:end -->

> [!example]- C23-1 酒与水反复交换
> **已知**：两个杯子各一品脱，第一杯水、第二杯酒。每轮从第一杯倒 $1/3$ 品脱到第二杯，搅匀，再倒 $1/3$ 品脱回第一杯。
> **目标**：求 $n$ 轮后第一杯的酒量及两杯极限。
>
> 令 $x_n$ 为第 $n$ 轮后第一杯酒量，$x_0=0$。一轮开始时第一杯总体积为 $1$，其中酒 $x_n$。第一次倒出后第一杯剩酒 $2x_n/3$；第二杯总体积 $4/3$，酒量 $1+x_n/3$。倒回 $1/3$ 品脱等于倒回第二杯内容的 $1/4$，带回酒
> $$
> \frac14\left(1+\frac{x_n}{3}\right).
> $$
> 因此
> $$
> x_{n+1}=\frac{2x_n}{3}+\frac14+\frac{x_n}{12}
> =\frac34x_n+\frac14.
> $$
> 固定点为 $1$。令 $y_n=1-x_n$，则 $y_{n+1}=\frac34y_n$、$y_0=1$，故
> $$
> \boxed{x_n=1-\left(\frac34\right)^n}.
> $$
> 总酒量恒为 $1$，第二杯酒量为 $(3/4)^n$。所以 $n\to\infty$ 时第一杯酒趋于 $1$ 品脱、第二杯趋于 $0$。这看似反直觉，但流程有方向性：每轮都先从第一杯取样、再从更大体积的第二杯只取回四分之一。
> <!-- bilingual-en:start -->
> **Known**: Two cups each pint, the first glass of water, the second glass of wine.  In each round, pint $1/3$ from that first cup to the second cup, stir well, then pint $1/3$ back to the first cup.
> **Target**: $n$ first round and limit to two glasses.
> Make $x_n$ the first drink after round $n$, $x_0=0$.  At the start of the round, the first glass had a total volume of $1$, of which the wine was $x_n$.  After the first pour, the first glass was $2x_n/3$, and the second glass was $4/3$ in volume and $1+x_n/3$ in volume.  Pour back $1/3$ pints is equivalent to pour back the second cup of $1/4$, bring back the wine
> therefore
> The fixed point is $1$.  $y_n=1-x_n$, $y_{n+1}=\frac34y_n$, $y_0=1$, and
> The total volume of wine was $1$, and the second volume was $(3/4)^n$.  So at $n\to\infty$, the first glass tends to be $1$ pints and the second to $0$.  It may seem counter-intuitive, but the process is directional: each round takes a sample from the first cup and then only a quarter from the second, larger one.
> <!-- bilingual-en:end -->

> [!example]- C23-2 沙漠缓存与调和数
> **(a)** 只有 $1$ 加仑且必须往返，最远走 $1/2$ 天，因为去、回各耗同样水量。
> **(b)** 两加仑时可先在距离 $1/4$ 处建立缓存，再用一加仑往返额外 $1/2$；总距离 $1/4+1/2=3/4$。
> **(c) 递推证明**：设用总计 $n$ 加仑可往返距离 $D_n$。为了把“基地”向前推进 $\delta$，前 $n-1$ 次运送都要往返，最后一次只向前；递归远行返回新基地后还需留下 $\delta$ 加仑回旧基地。新基地累计可用水为
> $$
> (n-1)(1-2\delta)+(1-\delta)=n-(2n-1)\delta.
> $$
> 它应等于递归所需的 $n-1$ 加仑再加返程的 $\delta$，故
> $$
> n-(2n-1)\delta=(n-1)+\delta
> \quad\Longrightarrow\quad
> \delta=\frac1{2n}.
> $$
> 因而
> $$
> D_n=D_{n-1}+\frac1{2n},\quad D_1=\frac12,
> $$
> 所以
> $$
> \boxed{D_n=\frac12H_n}.
> $$
> 因 $H_n\to\infty$，任何有限距离最终都能到达。
> **(d)** 要到 $d=10$，需 $H_n/2\gtrsim10$，即 $\ln n\gtrsim20$，所以
> $$
> n\gtrsim e^{20}\approx4.85\times10^8.
> $$
> 每天最多从绿洲取一加仑，单是取水次数就超过 $4.85\times10^8$ 天，约 $1.33\times10^6$ 年，确实超过一百万年。
> <!-- bilingual-en:start -->
> **(a)** With only one gallon and a required return trip, the traveller can go at most half a day away, because the outward and return legs consume equal amounts.
> **(b)** With two gallons, first establish a cache one quarter of a day away, then use the remaining gallon for a further half-day round trip. The total distance is $1/4+1/2=3/4$.
> **(c) Recursive proof.** Suppose $n$ gallons permit a round-trip distance $D_n$. To move the base forward by $\delta$, the first $n-1$ journeys traverse the new segment in both directions, while the last traverses it only outward. Accounting for the water returned to the old base gives $n\delta=1(1/2-\delta)$, hence $\delta=1/(2n)$ and $D_n=D_{n-1}+1/(2n)$. Therefore $D_n=H_n/2$. Since $H_n\to\infty$, every finite distance is eventually reachable.
> **(d)** Reaching distance $10$ requires $H_n/2\gtrsim10$, so $\ln n\gtrsim20$ and $n\gtrsim e^{20}\approx4.85\times10^8$. Even the required water-collection trips exceed a million years.
> A maximum of one gallon per day is taken from the oasis, and the number of water withdrawals alone exceeds $4.85\times10^8$ days, approximately $1.33\times10^6$ years, and indeed more than a million years.
> <!-- bilingual-en:end -->

> [!example]- C23-3 递减函数积分界证明
> 对 $x\in[i,i+1]$，递减性给出 $f(i+1)\le f(x)\le f(i)$。积分并从 $i=1$ 求和到 $n-1$：
> $$
> \sum_{i=2}^{n}f(i)\le\int_1^n f(x)dx\le\sum_{i=1}^{n-1}f(i).
> $$
> 令 $S=\sum_{i=1}^nf(i)$、$I=\int_1^nf(x)dx$，两端补回缺失项即
> $$
> \boxed{I+f(n)\le S\le I+f(1)}.
> $$
> <!-- bilingual-en:start -->
> For $x\in[i,i+1]$, $f(i+1)\le f(x)\le f(i)$ is given in decreasing order.  Integral and sum from $i=1$ to $n-1$:
> Let $S=\sum_{i=1}^nf(i)$, $I=\int_1^nf(x)dx$, both sides patch the missing items
> <!-- bilingual-en:end -->

> [!example]- C23-4 Sammy the Shark 复利债务
> 令 $D_d$ 为第 $d$ 天末债务，$D_0=m$。每天先加服务费 $f$，再乘利息因子 $p$：
> $$
> D_d=p(D_{d-1}+f).
> $$
> **(a)** $D_1=p(m+f)$。
> **(b)** $D_2=p[p(m+f)+f]=p^2m+fp(p+1)$。
> **(c)** 展开递推：
> $$
> D_d=p^dm+fp(1+p+\cdots+p^{d-1}).
> $$
> 若 $p\ne1$，
> $$
> \boxed{D_d=p^dm+fp\frac{p^d-1}{p-1}};
> $$
> 若 $p=1$，则 $D_d=m+df$。
> **(d)** $m=10,f=0.1,p=1.01,d=365$：
> $$
> D_{365}=1.01^{365}\cdot10+0.1(1.01)\frac{1.01^{365}-1}{0.01}
> \approx\boxed{749.35\text{ 美元}}.
> $$
> <!-- bilingual-en:start -->
> $D_0=m$ $D_d$ $d$ day end debt.  $f$ and then the interest factor $p$:
> **(a)**$D_1=p(m+f)$.
> **(b)**$D_2=p[p(m+f)+f]=p^2m+fp(p+1)$.
> **(c)**Expansion Recursion:
> If $p\ne1$,
> If $p=1$, $D_d=m+df$.
> **(d)**$m=10,f=0.1,p=1.01,d=365$:
> <!-- bilingual-en:end -->

> [!example]- C23-5 带权几何和
> 令 $T=z+2z^2+\cdots+nz^n$。乘 $z$：
> $$
> zT=z^2+2z^3+\cdots+(n-1)z^n+nz^{n+1}.
> $$
> 相减：
> $$
> (1-z)T=z+z^2+\cdots+z^n-nz^{n+1}.
> $$
> 其中 $z+\cdots+z^n=z(1-z^n)/(1-z)$，再除以 $1-z$：
> $$
> \boxed{T=\frac{z-(n+1)z^{n+1}+nz^{n+2}}{(1-z)^2}},\qquad z\ne1.
> $$
> $z=1$ 时回到 $T=1+2+\cdots+n=n(n+1)/2$。
> <!-- bilingual-en:start -->
> Let $T=z+2z^2+\cdots+nz^n$ and multiply it by $z$. Subtracting the shifted expression from $T$ leaves $z+z^2+\cdots+z^n-nz^{n+1}$. Since $z+\cdots+z^n=z(1-z^n)/(1-z)$, divide once more by $1-z$ to obtain the displayed formula. When $z=1$, handle the exceptional case directly: $T=1+2+\cdots+n=n(n+1)/2$.
> <!-- bilingual-en:end -->

### Session 23 知识链小结
<!-- bilingual-en:start -->
*Session 23 Knowledge Chain Summary*
<!-- bilingual-en:end -->

$$
\text{扰动配对}
\longrightarrow\text{几何和闭式}
\longrightarrow\text{年金与递推}
\longrightarrow\text{积分夹逼}
\longrightarrow H_n\sim\ln n
\longrightarrow\text{对数化乘积}
\longrightarrow\text{Stirling 公式}.
$$

---

## Session 24 — Asymptotics

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：什么时候两个函数“增长一样快”？$O$、$o$、$\Theta$ 与 $\sim$ 各自表达什么逻辑关系？怎样写出带有统一常数与阈值的证明，而不是凭最高次项猜答案？
<!-- bilingual-en:start -->
**Learning problem**: When do two functions "grow as fast"?  What is the logical relationship between $O$, $o$, $\Theta$ and $\sim$?  How to write a proof with a uniform constant and a threshold instead of guessing the answer by the highest order term?
<!-- bilingual-en:end -->

**前置知识**：函数极限、绝对值、关系的自反/对称/传递/反对称性质。首次正式使用 [[渐近记号与算法复杂度#O、Omega 与 Theta|渐近记号与 Θ 同阶关系]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: functional limits, absolute values, reflexive/symmetric/transitive/antisymmetric properties of relations.  Getting Started with [[渐近记号与算法复杂度#O、Omega 与 Theta|Asymptotic Sign and Θ Same Order Relation]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|Session 24 reading, pp. 1–8]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|cp24, pp. 1–3]]

### 3.2.1 Asymptotic Notation — 五种关系各说什么
<!-- bilingual-en:start -->
*3.2.1 Asymptotic Notation — What each of the five relationships says*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymNotation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CWkh5kb4TGc.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CWkh5kb4TGc)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymNotation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CWkh5kb4TGc.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CWkh5kb4TGc)
<!-- bilingual-en:end -->

以下默认 $f,g$ 在充分大的 $x$ 上有定义，且 $g(x)>0$；若 $f$ 可取负值，Big O 定义使用 $|f|$。
<!-- bilingual-en:start -->
The following default $f,g$ is defined on a sufficiently large $x$, and $g(x)>0$; if $f$ can take a negative value, the Big O definition uses $|f|$.
<!-- bilingual-en:end -->

#### 渐近等价 $f\sim g$
<!-- bilingual-en:start -->
*asymptotically equivalent $f\sim g$*
<!-- bilingual-en:end -->

$$
\boxed{f\sim g\iff\lim_{x\to\infty}\frac{f(x)}{g(x)}=1}.
$$

它说相对误差趋于零：$f=g(1+o(1))$。例如 $n^2+n\sim n^2$，但 $2n^2\not\sim n^2$。
<!-- bilingual-en:start -->
It says that the relative error tends to zero: $f=g(1+o(1))$.  For example, $n^2+n\sim n^2$, but $2n^2\not\sim n^2$.
<!-- bilingual-en:end -->

#### 严格低阶 $f=o(g)$
<!-- bilingual-en:start -->
*strict low-order $f=o(g)$*
<!-- bilingual-en:end -->

$$
\boxed{f=o(g)\iff\lim_{x\to\infty}\frac{|f(x)|}{g(x)}=0}.
$$

它说无论给定多小的 $\varepsilon>0$，总存在 $x_0$，使 $x\ge x_0$ 时 $|f(x)|\le\varepsilon g(x)$。
<!-- bilingual-en:start -->
It says that no matter how small a $\varepsilon>0$ is given, there is always a $x_0$ such that $x\ge x_0$ is $|f(x)|\le\varepsilon g(x)$.
<!-- bilingual-en:end -->

#### 上界关系 $f=O(g)$
<!-- bilingual-en:start -->
*upper bound relation $f=O(g)$*
<!-- bilingual-en:end -->

$$
\boxed{f=O(g)\iff
\exists c>0\ \exists x_0\ \forall x\ge x_0:
|f(x)|\le c g(x)}.
$$

若商不一定有普通极限，也可写成
<!-- bilingual-en:start -->
If quotient doesn't have to have a normal limit, it can also be written as
<!-- bilingual-en:end -->

$$
\limsup_{x\to\infty}\frac{|f(x)|}{g(x)}<\infty.
$$

Big O 只给**渐近上界**；它允许常数倍差异，也允许有界振荡。
<!-- bilingual-en:start -->
Big O only gives**an asymptotic upper bound of**; it allows constant-times difference and bounded oscillations.
<!-- bilingual-en:end -->

#### 同阶 $f=\Theta(g)$ 与下界 $\Omega$
<!-- bilingual-en:start -->
*Same Order $f=\Theta(g)$ and Lower Bound $\Omega$*
<!-- bilingual-en:end -->

$$
\boxed{f=\Theta(g)\iff f=O(g)\text{ 且 }g=O(f)}.
$$

等价地，存在 $c_1,c_2>0$ 与 $x_0$，使充分大时
<!-- bilingual-en:start -->
Equivalently, there are $c_1,c_2>0$ and $x_0$, which make the time
<!-- bilingual-en:end -->

$$
c_1g(x)\le |f(x)|\le c_2g(x).
$$

$f=\Omega(g)$ 定义为 $g=O(f)$；$f=\omega(g)$ 定义为 $g=o(f)$。
<!-- bilingual-en:start -->
$f=\Omega(g)$ is defined as $g=O(f)$; $f=\omega(g)$ is defined as $g=o(f)$.
<!-- bilingual-en:end -->

> [!example] 一眼看出逻辑强弱
> $$
> f\sim g\Longrightarrow f=\Theta(g)
> \Longrightarrow f=O(g),
> $$
> 但逆命题均不成立：$2g=\Theta(g)$ 却不与 $g$ 渐近等价；$1=O(n)$ 却不是 $\Theta(n)$。
> <!-- bilingual-en:start -->
> But neither of the converse propositions holds: $2g=\Theta(g)$ is not asymptotically equivalent to $g$; $1=O(n)$ is not $\Theta(n)$.
> <!-- bilingual-en:end -->

### 3.2.2 Asymptotics as Relations — 官方在线题 O24-01、O24-02
<!-- bilingual-en:start -->
*3.2.2 Asymptotics as Relations—official online questions O24-01, O24-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.2_asymptotics-as-relations|3.2.2]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.2_asymptotics-as-relations|3.2.2]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O24-01（哪些关系对称）**：$\sim$ 与 $\Theta$。
> **O24-02（哪些关系表示同一增长阶）**：$\sim$ 与 $\Theta$。
> $f=o(g)$ 与 $f=O(g)$ 一般都不对称；$\sim$ 比 $\Theta$ 更强，因为它要求比值趋于 $1$。
> <!-- bilingual-en:start -->
> **O24-01 (which relationships are symmetrical)**: $\sim$ and $\Theta$.
> **O24-02 (which relationships represent the same growth order)**: $\sim$ and $\Theta$.
> $f=o(g)$ and $f=O(g)$ are generally asymmetric; $\sim$ is stronger than $\Theta$ because it requires the ratio to be $1$.
> <!-- bilingual-en:end -->

### 3.2.3 Asymptotic Properties — 关系结构与增长层级
<!-- bilingual-en:start -->
*3.2.3 Asymptotic Properties — Relationship Structure and Growth Hierarchy*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymProperti.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HeyEK0TWiBw.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HeyEK0TWiBw)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymProperti.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HeyEK0TWiBw.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HeyEK0TWiBw)
<!-- bilingual-en:end -->

#### 关系性质
<!-- bilingual-en:start -->
*relation property*
<!-- bilingual-en:end -->

- $\sim$ 是等价关系：自反来自 $f/f=1$；对称来自倒数；传递来自 $(f/g)(g/h)=f/h$。
- $\Theta$ 是等价关系：自反显然；定义本身对称；Big O 的传递性给传递。
- $o$ 是严格偏序式关系：非自反且传递。若 $f=o(g)$，不可能同时 $g=O(f)$，否则充分大时 $g\le C|f|$，而 $|f|/g\to0$ 与 $1\le C|f|/g$ 冲突。
- $O$ 是预序：自反、传递，但不反对称。不同函数 $f=n$ 与 $g=2n$ 互为 Big O。
- 关系“$f=O(g)$ 且 $g\ne O(f)$”是严格偏序，表达“严格不快于”。它比 $o$ 弱，因为商可能在 $0$ 与正常数量级之间振荡。
<!-- bilingual-en:start -->
- $\sim$ is the equivalence relation: reflexive from $f/f=1$; symmetric from reciprocal; pass from $(f/g)(g/h)=f/h$.
- $\Theta$ is the equivalence relation: reflexive apparent; definition itself symmetric; transitivity of Big O to pass.
- $o$ is a strictly partial order relation: non-reflexive and transitive.  If $f=o(g)$, it is not possible to $g=O(f)$ at the same time; otherwise, it is sufficient to $g\le C|f|$ and $|f|/g\to0$ conflicts with $1\le C|f|/g$.
- $O$ is a preorder: reflexive, transitive, but not asymmetric.  The different functions $f=n$ and $g=2n$ are Big O each other.
- The relation "$f=O(g)$ and $g\ne O(f)$" is strictly partial order and expresses "strictly not faster than".  It is weaker than $o$ because quotient can oscillate between $0$ and normal orders of magnitude.
<!-- bilingual-en:end -->

#### 对数慢于任意正幂
<!-- bilingual-en:start -->
*Logarithm slower than any positive power*
<!-- bilingual-en:end -->

对任意 $\varepsilon>0$，令 $t=\ln x$，则
<!-- bilingual-en:start -->
For any $\varepsilon>0$, let $t=\ln x$, then
<!-- bilingual-en:end -->

$$
\frac{\ln x}{x^\varepsilon}=\frac{t}{e^{\varepsilon t}}.
$$

由 $e^{\varepsilon t}\ge(\varepsilon t)^2/2$，
<!-- bilingual-en:start -->
By $e^{\varepsilon t}\ge(\varepsilon t)^2/2$,
<!-- bilingual-en:end -->

$$
0\le\frac{t}{e^{\varepsilon t}}
\le\frac{2}{\varepsilon^2t}\to0.
$$

故 $\ln x=o(x^\varepsilon)$，不需要把“对数很慢”当作直觉口号。
<!-- bilingual-en:start -->
So, $\ln x=o(x^\varepsilon)$, you don't have to think of "logarithmically slow" as an intuitive slogan.
<!-- bilingual-en:end -->

#### 指数快于任意固定多项式
<!-- bilingual-en:start -->
*Exponentially faster than any fixed polynomial*
<!-- bilingual-en:end -->

设 $a>1$、$c\ge0$，考察 $u_n=n^c/a^n$：
<!-- bilingual-en:start -->
Set up $a>1$, $c\ge0$, inspect $u_n=n^c/a^n$:
<!-- bilingual-en:end -->

$$
\frac{u_{n+1}}{u_n}=\frac{(1+1/n)^c}{a}\longrightarrow\frac1a<1.
$$

取 $q$ 满足 $1/a<q<1$，则充分大时 $u_{n+1}\le q u_n$，故尾部被收敛几何数列控制，$u_n\to0$。所以
<!-- bilingual-en:start -->
If $q$ satisfies $1/a<q<1$, then $u_{n+1}\le q u_n$ when it is sufficiently large, so the tail is controlled by the convergent geometric sequence, $u_n\to0$.  therefore
<!-- bilingual-en:end -->

$$
n^c=o(a^n).
$$

典型增长层级为
<!-- bilingual-en:start -->
Typical growth levels are
<!-- bilingual-en:end -->

$$
1\prec\log n\prec n^\varepsilon\prec n^c\prec a^n\prec n!\prec n^n,
$$

其中 $\prec$ 表示左边是右边的 little o；幂指数的具体顺序需满足 $0<\varepsilon<c$。
<!-- bilingual-en:start -->
where $\prec$ denotes little o on the left and $0<\varepsilon<c$ on the right; the order of the power exponents must be specified.
<!-- bilingual-en:end -->

### 3.2.4 Little oh / Big Oh — 官方在线题 O24-03 至 O24-05
<!-- bilingual-en:start -->
*3.2.4 Little oh / Big Oh —official online questions O24-03 through O24-05*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.4_little-oh-big-oh|3.2.4]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.4_little-oh-big-oh|3.2.4]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O24-03**：以下四项为真：$f=o(g)\Rightarrow f=O(g)$；$f=o(g)\Rightarrow g\ne O(f)$；$f\sim g$ 时 $f=O(g)$；即使 $f\not\sim g$，只要 $f=o(g)$ 仍有 $f=O(g)$。`O ⇒ o` 为假。
> **O24-04**：错误陈述是“对所有整数 $a,b$ 都有 $x^a=o(x^b)$”；正确条件是 $a<b$。其余核心事实是 $\log x=o(x^\varepsilon)$ 与任意固定幂 $x^c=o(a^x)$（$a>1$）。
> **O24-05**：Big O 用 $\limsup$ 是为了容纳商没有普通极限但始终被常数控制的振荡情形；$\limsup$ 不是“更严格的普通极限”。
> <!-- bilingual-en:start -->
> **O24-03**: The following four items are true: $f=o(g)\Rightarrow f=O(g)$; $f=o(g)\Rightarrow g\ne O(f)$; $f=O(g)$ at $f\sim g$; and $f=O(g)$ even if $f\not\sim g$, as long as the $f=o(g)$ still has it.  `O ⇒ o` is fake.
> **O24-04**: Error statement is "$x^a=o(x^b)$ for all integer $a,b$"; correct condition is $a<b$.  The remaining core facts are $\log x=o(x^\varepsilon)$ and any fixed power $x^c=o(a^x)$ ($a>1$).
> **O24-05**:Big O uses $\limsup$ to accommodate oscillations in which the quotient has no ordinary limit but is always controlled by a constant; $\limsup$ is not a "tighter ordinary limit".
> <!-- bilingual-en:end -->

### 3.2.5 Theta — 官方在线题 O24-06、O24-07
<!-- bilingual-en:start -->
*3.2.5 Theta—official online questions O24-06, O24-07*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.5_theta|3.2.5]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.5_theta|3.2.5]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O24-06（若 $f=\Theta(g)$，哪些必须真）**：$g=\Theta(f)$、$f=O(g)$、$g=O(f)$。
> **O24-07（哪些可能真）**：除上述三项外，$f\sim g$ 也可能真；但 $f=o(g)$ 或 $g=o(f)$ 都不可能与互相 Big O 并存。
> <!-- bilingual-en:start -->
> **O24-06 (which must be true if $f=\Theta(g)$)**:$g=\Theta(f)$, $f=O(g)$, $g=O(f)$.
> **O24-07**: Except for the three above, $f\sim g$ may be true; however, neither $f=o(g)$ nor $g=o(f)$ may coexist with Big O.
> <!-- bilingual-en:end -->

### 3.2.6 Asymptotic Blunders — 语法错往往对应逻辑错
<!-- bilingual-en:start -->
*3.2.6 Asymptotic Blunders—Syntax Errors Match Logical Errors*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymBlunders.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Y9Blo_G-Mvg.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=Y9Blo_G-Mvg)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymBlunders.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Y9Blo_G-Mvg.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=Y9Blo_G-Mvg)
<!-- bilingual-en:end -->

1. **把关系写成数量**：$O(n^2)$ 不是一个可比较大小的数，而是一类函数或二元关系的右侧。
2. **说“至少 $O(n^2)$”**：Big O 是上界。“$f$ 至少像 $n^2$”应写 $n^2=O(f)$ 或 $f=\Omega(n^2)$。
3. **把逐项常数当统一常数**：对固定 $i$，$i=O(1)$，但这里隐藏常数依赖 $i$。求和到 $n$ 时不能用同一个常数控制所有 $i$，实际 $\sum_{i=1}^ni=\Theta(n^2)$。
4. **随意对渐近关系做非线性运算**：$f\sim g$ 不保证 $3^f=\Theta(3^g)$。例：$f=n+\sqrt n$、$g=n$，虽 $f/g\to1$，但 $3^f/3^g=3^{\sqrt n}\to\infty$。
5. **忽略常数与交叉点**：渐近阶只描述充分大输入；实际系统中 $1000n$ 可能在很长区间内慢于 $n^2$，也可能快于带巨大常数的低阶算法。
<!-- bilingual-en:start -->
1.**Write the relation as a number**: $O(n^2)$ is not a comparable-sized number, but is the right-hand side of a class of functions or binary relations.
2.**Say "at least $O(n^2)$"**: Big O is the upper bound.  '$f$ at least like $n^2$' should be written as $n^2=O(f)$ or $f=\Omega(n^2)$.
3.**Consider the term-by-term constant as a uniform constant**: For fixed $i$, $i=O(1)$, but here the hidden constant depends on $i$.  The sum to $n$ cannot control all $i$ with the same constant, the actual $\sum_{i=1}^ni=\Theta(n^2)$.
4.**Do non-linear operation on asymptotic relation at will**:$f\sim g$ does not guarantee $3^f=\Theta(3^g)$.  Examples: $f=n+\sqrt n$, $g=n$, although $f/g\to1$, but $3^f/3^g=3^{\sqrt n}\to\infty$.
5.**Ignore constants and intersections**: The asymptotic order only describes sufficiently large inputs; in real systems, $1000n$ may be slower than $n^2$ in a very long range, or faster than lower-order algorithms with large constants.
<!-- bilingual-en:end -->

### 3.2.7 Asymptotics the Right Way — 官方在线题 O24-08 至 O24-10
<!-- bilingual-en:start -->
*3.2.7 Asymptotics the Right Way — official online questions O24-08 through O24-10*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.7_asymptotics-the-right-way|3.2.7]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.7_asymptotics-the-right-way|3.2.7]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O24-08**：$O(\cdot)$、$o(\cdot)$、$\Theta(\cdot)$ 放在等号右侧，例如 $f=O(n^2)$。
> **O24-09**：“$f$ 至少是 $O(n^2)$”四方面都错：$O(n^2)$ 不是数量；Big O 表示上界；它是关系；“至少 $n^2$”应写 $n^2=O(f)$。
> **O24-10**：$\sum_{i=1}^ni=O(n)$ 错在逐项 $O(1)$ 没有统一常数；正确结果是 $\Theta(n^2)$，且 $O(1)$ 不能当作普通数字相加。
> <!-- bilingual-en:start -->
> **O24-08**:$O(\cdot)$, $o(\cdot)$, $\Theta(\cdot)$ to the right of the equal sign, for example, $f=O(n^2)$.
> **O24-09**: '$f$ is at least $O(n^2)$' is all wrong: $O(n^2)$ is not a quantity; Big O represents an upper bound; it is a relation; 'at least $n^2$' should be written as $n^2=O(f)$.
> The**O24-10**:$\sum_{i=1}^ni=O(n)$ error does not have a uniform constant in term-by-term $O(1)$; the correct result is $\Theta(n^2)$, and $O(1)$ cannot be added as an ordinary number.
> <!-- bilingual-en:end -->

### 3.2.8 Practice with Big O — 官方在线题 O24-11 至 O24-17
<!-- bilingual-en:start -->
*3.2.8 Practice with Big O — official online questions O24-11 through O24-17*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.8_practice-with-big-o|3.2.8]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.8_practice-with-big-o|3.2.8]]
<!-- bilingual-en:end -->

题目要求最小非负整数 $k$，使 $f(x)=O(x^k)$：
<!-- bilingual-en:start -->
The title requires a minimum non-negative integer $k$ such that $f(x)=O(x^k)$:
<!-- bilingual-en:end -->

| 编号 | $f(x)$ | 官方答案 | 核心化简 |
|---|---|---:|---|
| O24-11 | $2x^3+x^2\log x$ | $3$ | $x^2\log x=o(x^3)$ |
| O24-12 | $2x^2+x^3\log x$ | $4$ | $x^3\log x=o(x^4)$，但非 $O(x^3)$ |
| O24-13 | $1.1^x$ | none | 指数快于任意多项式 |
| O24-14 | $0.1^x$ | $0$ | 趋于 $0$，故 $O(1)$ |
| O24-15 | $(x^4+x^2+1)/(x^3+1)$ | $1$ | 比值主项为 $x$ |
| O24-16 | $(x^4+5\log x)/(x^4+1)$ | $0$ | 比值趋于 $1$ |
| O24-17 | $2^{3\log_2x^2}$ | $6$ | $2^{\log_2x^6}=x^6$ |
<!-- bilingual-en:start -->
|Encoding | $f(x)$ | Official Answers | Core Simplification |
|—|—|—:|—|
| O24-11 | $2x^3+x^2\log x$ | $3$ | $x^2\log x=o(x^3)$ |
| O24-12 | $2x^2+x^3\log x$ | $4$ | $x^3\log x=o(x^4)$, but not $O(x^3)$ |
| O24-13 | $1.1^x$ | none | Exponent faster than any polynomial |
| O24-14 | $0.1^x$ | $0$ | tends to $0$, so $O(1)$ |
| O24-15 | $(x^4+x^2+1)/(x^3+1)$ | $1$ | Ratio major is $x$ |
| O24-16 | $(x^4+5\log x)/(x^4+1)$ | $0$ | The ratio tends to $1$ |
| O24-17 | $2^{3\log_2x^2}$ | $6$ | $2^{\log_2x^6}=x^6$ |
<!-- bilingual-en:end -->

### 3.2.9 Practice with Order of Growth — 官方在线题 O24-18 至 O24-21
<!-- bilingual-en:start -->
*3.2.9 Practice with Order of Growth — official online questions O24-18 through O24-21*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.9_practice-with-order-of-growth|3.2.9]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.9_practice-with-order-of-growth|3.2.9]]
<!-- bilingual-en:end -->

| 编号 | $f,g$ | 官方答案 | 检查 |
|---|---|---|---|
| O24-18 | $\log_3n,\log_7n$ | $f=O(g)$ 且 $f=\Theta(g)$ | 比值 $\ln7/\ln3\ne1$，故不 $\sim$ |
| O24-19 | $0,33$ | $f=o(g)$ 且 $f=O(g)$ | 商恒为 $0$ |
| O24-20 | $1+\cos(\pi n/2),1+\sin(\pi n/2)$ | 所列均不成立 | 两函数交替取零，商无法统一控制 |
| O24-21 | $1.01^n,n^{100}$ | 所列均不成立 | 实际 $g=o(f)$，但选项只问 $f$ 相对 $g$ |
<!-- bilingual-en:start -->
|Encoding | $f,g$ | Official Answer | Check |
|—|—|—|—|
| O24-18 | $\log_3n,\log_7n$ | $f=O(g)$ and $f=\Theta(g)$ | ratio $\ln7/\ln3\ne1$, so not $\sim$ |
| O24-19 | $0,33$ | $f=o(g)$ and $f=O(g)$ | quotient constant is $0$ |
| O24-20 | $1+\cos(\pi n/2),1+\sin(\pi n/2)$ | None of the lists are true | The two functions are alternately zeroed and quotient cannot be uniformly controlled |
| O24-21 | $1.01^n,n^{100}$ | None of the lists are true | Actual $g=o(f)$, but the option is to ask $f$ only relative $g$ |
<!-- bilingual-en:end -->

### Session 24 易错点与反例
<!-- bilingual-en:start -->
*Session 24 Error-prone Points and Counterexamples*
<!-- bilingual-en:end -->

1. $f=O(g)$ 不意味着 $f$ “等于”某个具体函数；也不意味着这是最紧上界。
2. $f=\Theta(g)$ 允许常数倍，$f\sim g$ 要求常数恰为 $1$。
3. 证明 Big O 必须给**同一个** $c$ 与 $x_0$，它们不能随输入 $x$ 改变。
4. 若 $g$ 有无限多个零点，商式定义要谨慎；最好直接用最终不等式定义检查。
5. $4^n\ne O(2^n)$，因为比值 $2^n$ 无界；底数的常数差不能丢进 Big O。
<!-- bilingual-en:start -->
1. $f=O(g)$ does not mean that $f$ is "equal" to a specific function; nor does it mean that this is the tightest upper bound.
2. $f=\Theta(g)$ allows constant times, $f\sim g$ requires constant exactly $1$.
3. Prove that Big O must give**the same**$c$ and $x_0$, which cannot change with the input $x$.
4. If $g$ has infinite number of zeros, the definition of quotient should be carefully defined; it is best to check directly with the definition of final inequality.
5. $4^n\ne O(2^n)$, because the ratio $2^n$ is unbounded; the constant difference of the base number cannot be dropped into Big O.
<!-- bilingual-en:end -->

### Session 24 自检
<!-- bilingual-en:start -->
*Session 24 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 24-1
> 判断 $n\log n$ 与 $n^{1.1}$ 的关系。
> <!-- bilingual-en:start -->
> Determine the relationship between $n\log n$ and $n^{1.1}$.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $$
> \frac{n\log n}{n^{1.1}}=\frac{\log n}{n^{0.1}}\to0,
> $$
> 所以 $n\log n=o(n^{1.1})$。
> <!-- bilingual-en:start -->
> So, $n\log n=o(n^{1.1})$.
> <!-- bilingual-en:end -->

> [!question] 自检 24-2
> 给出 $f=O(g)$ 但既不 $f=o(g)$、也不 $f=\Theta(g)$ 的例子。
> <!-- bilingual-en:start -->
> An example of $f=O(g)$ but neither $f=o(g)$ nor $f=\Theta(g)$ is given.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 在正整数上令 $f(n)=n$（$n$ 为偶数）而 $f(n)=1$（$n$ 为奇数），令 $g(n)=n$。有 $f\le g$，故 $f=O(g)$；商在 $1$ 与 $1/n$ 间振荡，不趋零；反向 $g=O(f)$ 在奇数点失败，故不为 $\Theta$。
> <!-- bilingual-en:start -->
> On positive integers, let $f(n)=n$ ($n$ is even) and $f(n)=1$ ($n$ is odd), and let $g(n)=n$.  There is $f\le g$, so $f=O(g)$; the quotient oscillates between $1$ and $1/n$ and does not go to zero; the reverse $g=O(f)$ fails at the odd point and thus is not $\Theta$.
> <!-- bilingual-en:end -->

> [!question] 自检 24-3
> “每次循环是 $O(1)$，循环 $n$ 次，所以总计仍是 $O(1)$”错在哪里？
> <!-- bilingual-en:start -->
> "Every loop is $O(1)$, $n$ loops, so the total is still $O(1)$." Where is the error?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 即使每次都有统一常数 $C$，总成本也至多 $nC=O(n)$；不能把 $O(1)$ 当成相加后仍不变的数字。若每次隐藏常数还依赖循环下标，错误更严重。
> <!-- bilingual-en:start -->
> Even if there is a uniform constant $C$ at each time, the total cost is at most $nC=O(n)$; $O(1)$ cannot be considered a constant number after addition.  Errors are more serious if each hidden constant also depends on a cyclic subscript.
> <!-- bilingual-en:end -->

### Classroom Problems 24 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 24 — 5 complete independent questions*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|cp24 pp. 1–3]]
<!-- bilingual-en:start -->
Original Question: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|cp24 pp. 1–3]]
<!-- bilingual-en:end -->

> [!example]- C24-1 用定义找最小整数常数与阈值
> 课程定义 $f=O(g)$ 要求 $c,n_0\in\mathbb N$ 且 $n\ge n_0$ 时 $|f(n)|\le c g(n)$。
> **(a)** $f=n^2,g=3n$。$f/g=n/3$ 无界，故 $f\ne O(g)$。反向要求 $3n\le cn^2$。最小正整数 $c=1$；此时对所有 $n\ge3$ 成立，最小 $n_0=3$。
> **(b)** $f=(3n-7)/(n+4),g=4$。对所有 $n\ge0$ 有 $|f|\le4$，所以 $f=O(g)$ 的最小 $c=1,n_0=0$。反向 $4\le c|f|$；$c=1$ 永远不可能，因为 $f\to3$。取最小 $c=2$，在 $n\ge15$ 时
> $$
> \frac{3n-7}{n+4}\ge2,
> $$
> 因而最小 $n_0=15$。
> **(c)** $f=1+[n\sin(n\pi/2)]^2,g=3n$。偶数 $n$ 时 $f=1$；奇数 $n$ 时 $f=1+n^2$。奇数子列使 $f/g$ 无界，故 $f\ne O(g)$；偶数子列使 $g/f=3n$ 无界，故 $g\ne O(f)$。振荡可以让两个方向都失败。
> <!-- bilingual-en:start -->
> Course Definition $f=O(g)$ requires $c,n_0\in\mathbb N$ and $|f(n)|\le c g(n)$ at $n\ge n_0$.
> **(a)**$f=n^2,g=3n$.  $f/g=n/3$ is unbounded, so $f\ne O(g)$.  Reverse request $3n\le cn^2$.  The smallest positive integer, $c=1$; this is true for all $n\ge3$, and the smallest $n_0=3$.
> **(b)**$f=(3n-7)/(n+4),g=4$.  There is a $|f|\le4$ for all $n\ge0$, so the minimum $c=1,n_0=0$ for $f=O(g)$.  Reverse $4\le c|f|$; $c=1$ is never possible because $f\to3$.  Minimum $c=2$ at $n\ge15$
> So that a minimum $n_0=15$ is obtain.
> **(c)**$f=1+[n\sin(n\pi/2)]^2,g=3n$.  $f=1$ for even $n$ and $f=1+n^2$ for odd $n$.  The odd subcolumn makes $f/g$ unbounded, so $f\ne O(g)$; the even subcolumn makes $g/f=3n$ unbounded, so $g\ne O(f)$.  Oscillations can fail in both directions.
> <!-- bilingual-en:end -->

> [!example]- C24-2 把渐近关系分类
> **(a)**
>
> | 关系 | 分类 | 关键理由 |
> |---|---|---|
> | $f\sim g$ | 等价关系 E | 自反、对称、传递 |
> | $f=o(g)$ | 严格偏序 S | 非自反、传递，且不可能反向 Big O |
> | $f=O(g)$ | N | 是预序，但 $f=n,g=2n$ 破坏反对称性 |
> | $f=\Theta(g)$ | 等价关系 E | 互相 Big O |
> | $f=O(g)$ 且 $g\ne O(f)$ | 严格偏序 S | 非自反；传递可由 Big O 传递并排除反向关系 |
>
> **(b) 主要蕴含**：
> $$
> f\sim g\Rightarrow f=\Theta(g)
> \Rightarrow[f=O(g)\land g=O(f)],
> $$
> $$
> f=o(g)\Rightarrow[f=O(g)\land g\ne O(f)].
> $$
> 后一个严格 Big O 不推出 little o；可用自检 24-2 的振荡函数作为反例。
> <!-- bilingual-en:start -->
> **(a)**
> | Relationships | Categorizations | Key Reasons |
> |—|—|—|
> | $f\sim g$ | Equivalence relation E | Reflexive, Symmetric, Pass|
> | $f=o(g)$ | strictly partial order S | non-reflexive, transitive, and impossible to reverse Big O |
> | $f=O(g)$ | N | is a preorder, but $f=n,g=2n$ breaks the antisymmetry |
> | $f=\Theta(g)$ | equivalence relation E | mutual Big O |
> | $f=O(g)$ and $g\ne O(f)$ | strictly partial order S | non-reflexive; pass can be passed by Big O and exclude the reverse relationship |
> **(b) Major implication**:
> The latter strict Big O does not deduce little o; the oscillatory function of self-test 24-2 can be used as a counterexample.
> <!-- bilingual-en:end -->

> [!example]- C24-3 错误归纳“$2^n=O(1)$”
> **反证原命题**：若 $2^n=O(1)$，则存在单个 $c$、$n_0$，使所有 $n\ge n_0$ 有 $2^n\le c$；取 $n>\log_2c$ 即矛盾。
> **归纳错误**：Big O 中的 $n$ 是函数输入，不是一个可逐点归纳的命题参数。伪证明从控制 $2^n$ 的常数 $c_n$ 构造控制下一点的 $2c_n$；常数随 $n$ 变成 $2^n$，从未得到一个对所有充分大 $n$ 通用的常数。逐点“每个数都被某常数控制”是平凡事实，不等于整个函数有统一常数上界。
> <!-- bilingual-en:start -->
> **Disproof of the claim**: If $2^n=O(1)$, there would be fixed constants $c$ and $n_0$ such that $2^n\le c$ for every $n\ge n_0$. Choosing $n>\log_2c$ gives a contradiction.
> **Error in the induction**: In Big O notation, $n$ is the input to a function, not a proposition parameter that can be handled point by point. The bogus proof replaces a constant $c_n$ that bounds one value with $2c_n$ for the next value; the bound therefore grows with $n$ and never becomes one fixed constant valid for all sufficiently large inputs. Every individual number being bounded by some constant is trivial and does not imply a uniform bound on the whole function.
> <!-- bilingual-en:end -->

> [!example]- C24-4 四个真假命题
> 1. $n^2\sim n^2+n$：**真**，因为 $n^2/(n^2+n)=1/(1+1/n)\to1$。
> 2. $3^n=O(2^n)$：**假**，比值 $(3/2)^n\to\infty$。
> 3. $n^{\sin(n\pi/2)+1}=o(n^2)$：**假**。当 $n\equiv1\pmod4$ 时指数为 $2$，比值等于 $1$，不能趋零。
> 4. $n=\Theta\!\left(\dfrac{3n^3}{(n+1)(n-1)}\right)$：**真**。对 $n\ge2$，右式与 $3n$ 之比趋于 $1$，故与 $n$ 同阶；也可直接夹在正的常数倍 $n$ 之间。
> <!-- bilingual-en:start -->
> 1. $n^2\sim n^2+n$: **true**, because $n^2/(n^2+n)=1/(1+1/n)\to1$.
> 2. $3^n=O(2^n)$: **false**, because $(3/2)^n\to\infty$.
> 3. $n^{\sin(n\pi/2)+1}=o(n^2)$: **false**. When $n\equiv1\pmod4$, the exponent is $2$, so the ratio equals $1$ and cannot tend to zero.
> 4. $n=\Theta\!\left(\dfrac{3n^3}{(n+1)(n-1)}\right)$: **true**. For $n\ge2$, the expression on the right is asymptotic to $3n$, and it can also be bounded directly between positive constant multiples of $n$.
> <!-- bilingual-en:end -->

> [!example]- C24-5 不用 Stirling 证明 $\log(n!)=\Theta(n\log n)$
> 上界：每个因子 $i\le n$，故
> $$
> \log(n!)=\sum_{i=1}^n\log i\le n\log n.
> $$
> 下界：只保留后半段至少 $\lfloor n/2\rfloor$ 项，每项不少于 $\log(n/2)$：
> $$
> \log(n!)\ge\frac n2\log\frac n2
> =\frac n2(\log n-\log2).
> $$
> 对 $n\ge4$，$\log n-\log2\ge\frac12\log n$，所以
> $$
> \frac14n\log n\le\log(n!)\le n\log n.
> $$
> 因而 $\boxed{\log(n!)=\Theta(n\log n)}$。
> <!-- bilingual-en:start -->
> Upper bound: $i\le n$ per factor, so
> Lower bound: Retain the last $\lfloor n/2\rfloor$ terms of the sum; each is at least $\log(n/2)$:
> $n\ge4$ $\log n-\log2\ge\frac12\log n$, so..
> So $\boxed{\log(n!)=\Theta(n\log n)}$.
> <!-- bilingual-en:end -->

### Session 24 知识链小结
<!-- bilingual-en:start -->
*Session 24 Knowledge Chain Summary*
<!-- bilingual-en:end -->

$$
\text{极限比值}
\longrightarrow \sim,o
\longrightarrow O,\Omega,\Theta
\longrightarrow\text{关系结构}
\longrightarrow\text{增长层级}
\longrightarrow\text{算法规模的可证明比较}.
$$

---

## Problem Set 9 — after Session 24

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps9.pdf#page=1|Problem Set 9, pp. 1–2]]。以下为非官方独立题解。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps9.pdf#page=1|Problem Set 9, pp. 1–2]].  The following are unofficial independent questions.
<!-- bilingual-en:end -->

> [!example]- PS9-1 求 $\sum_{i=1}^{n}i^3$ 的多项式
> **目标**：找出闭式并验证。结论为
> $$
> \boxed{\sum_{i=1}^{n}i^3=\left[\frac{n(n+1)}2\right]^2}.
> $$
> 右边是前 $n$ 个整数之和的平方。用归纳验证：$n=1$ 时两边均为 $1$。假设对 $n$ 成立，则
> $$
> \begin{aligned}
> \sum_{i=1}^{n+1}i^3
> &=\frac{n^2(n+1)^2}{4}+(n+1)^3\\
> &=(n+1)^2\left(\frac{n^2}{4}+n+1\right)\\
> &=(n+1)^2\frac{(n+2)^2}{4}\\
> &=\left[\frac{(n+1)(n+2)}2\right]^2.
> \end{aligned}
> $$
> 所以对所有 $n\ge1$ 成立。
> <!-- bilingual-en:start -->
> **Target**: Locate closed and verify.  The conclusion is
> On the right is the square of the sum of the first $n$ integers.  By inductive verification, both sides are $1$ when $n=1$.  Assuming it is true for $n$, then
> So it's true for all $n\ge1$.
> <!-- bilingual-en:end -->

> [!example]- PS9-2 证明 $\ln((n^2)!)=\Theta(n^2\ln n)$
> 令 $N=n^2$。Stirling 公式给出
> $$
> \ln(N!)=N\ln N-N+\frac12\ln(2\pi N)+o(1).
> $$
> 代入 $N=n^2$：
> $$
> \ln((n^2)!)
> =2n^2\ln n-n^2+\ln n+\frac12\ln(2\pi)+o(1).
> $$
> 主项是 $2n^2\ln n$，其余项除以 $n^2\ln n$ 都趋于零，因此
> $$
> \boxed{\ln((n^2)!)\sim2n^2\ln n
> \quad\Longrightarrow\quad
> \ln((n^2)!)=\Theta(n^2\ln n)}.
> $$
> 定义域检查：$n$ 为正整数；对数底数换成任意固定 $>1$ 的底只差常数倍。
> <!-- bilingual-en:start -->
> Get $N=n^2$.  Stirling formula
> Introduce $N=n^2$:
> The dominant term is $2n^2\ln n$. After division by $n^2\ln n$, every remaining term tends to zero, so
> Domain checking: $n$ is a positive integer; the base of the logarithm is replaced by a constant multiple of the base of any fixed $>1$.
> <!-- bilingual-en:end -->

> [!example]- PS9-3 证明 $\sum_{k=1}^{n}k^6=\Theta(n^7)$
> $f(x)=x^6$ 为正且递增。积分法给出
> $$
> \frac{n^7-1}{7}+1
> \le\sum_{k=1}^{n}k^6
> \le\frac{n^7-1}{7}+n^6.
> $$
> 当 $n\ge2$，左边至少为 $n^7/7$；右边至多 $n^7/7+n^7=8n^7/7$。故可取 $c_1=1/7,c_2=8/7,n_0=2$，严格满足 Theta 定义：
> $$
> \boxed{\sum_{k=1}^{n}k^6=\Theta(n^7)}.
> $$
> <!-- bilingual-en:start -->
> $f(x)=x^6$ is positive and ascending.  integral method
> When $n\ge2$, the left is at least $n^7/7$; the right is at most $n^7/7+n^7=8n^7/7$.  So we can use $c_1=1/7,c_2=8/7,n_0=2$ to strictly meet the definition of Theta:
> <!-- bilingual-en:end -->

---

## Midterm 3 — after Session 24

原题：[[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm3.pdf#page=1|Midterm 3, pp. 1–8]]。这场考试安排在 Session 24 之后，但它是累计考试：Problem 1–5 回顾 Unit 1/2，Problem 6 检查本单元的和与积分。以下为非官方独立题解。
<!-- bilingual-en:start -->
Original paper: [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm3.pdf#page=1|Midterm 3, pp. 1–8]]. Although scheduled after Session 24, the exam is cumulative: Problems 1–5 review Units 1 and 2, while Problem 6 tests the sums-and-integrals material in this unit. The solutions below are independent and unofficial.
<!-- bilingual-en:end -->

### Problem 1 — Scheduling（15 分）
<!-- bilingual-en:start -->
*Problem 1 — Scheduling (15 points)*
<!-- bilingual-en:end -->

题图见 [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm3.pdf#page=2|Midterm 3 p. 2]]。[[无环图：树、生成树、DAG 与拓扑排序#DAG 与拓扑排序|DAG]] 的边为
<!-- bilingual-en:start -->
For the inscription, see [[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm3.pdf#page=2|Midterm 3 p. 2]].  The edge of the [[无环图：树、生成树、DAG 与拓扑排序#DAG 与拓扑排序|DAG]] is
<!-- bilingual-en:end -->

$$
A\to D,\quad D\to E,\quad B\to E,\quad B\to F,
\quad C\to F,\quad E\to G,\quad F\to H.
$$

> [!answer]- 完整解答
> **(a)** 传递可比关系中，$A<D<E<G$，$B<E<G$ 且 $B<F<H$，$C<F<H$。宽度为 $3$；两个最大反链是
> $$
> \boxed{\{A,B,C\},\qquad\{B,C,D\}}.
> $$
> 不能有 4 元反链：若选 $E$ 或 $F$，它分别排除多条前驱/后继；直接按层或用最小链覆盖
> $$
> A<D<E<G,\quad B,\quad C<F<H
> $$
> 可知任一反链至多从三条链各取一个元素。
> **(b)** 无限处理器下，最短时间等于最长链的任务数。链 $A\to D\to E\to G$ 有 4 个任务，故至少 4；按
> $$
> t_1:\{A,B,C\},\quad t_2:\{D,F\},\quad
> t_3:\{E,H\},\quad t_4:\{G\}
> $$
> 可在 4 完成，故答案 $\boxed{4}$。
> **(c)** 最多并行 2 个任务时，工作量下界为 $\lceil8/2\rceil=4$，关键路径下界仍为 4。安排
> $$
> t_1:\{A,B\},\quad t_2:\{C,D\},\quad
> t_3:\{E,F\},\quad t_4:\{G,H\}
> $$
> 满足全部先决条件，因此仍为 $\boxed{4}$。
> <!-- bilingual-en:start -->
> **(a)**In transitive comparability, $A<D<E<G$,$B<E<G$ and $B<F<H$,$C<F<H$.  Width is $3$; the two largest antichains are
> Cannot have 4-ary inverse chain: $E$ or $F$ to exclude multiple precursors/successors; directly by layer or covered with minimal chain
> It is known that at most one element of any inverse chain is taken from each of the three chains.
> **(b)**The minimum time equals the number of tasks in the longest chain for infinite processors.  Chain $A\to D\to E\to G$ has 4 tasks, so at least 4; press
> It can be done in 4, so answer $\boxed{4}$.
> **(c)**The lower bound for workload is $\lceil8/2\rceil=4$ and the lower bound for critical path is still 4 for up to 2 tasks in parallel.  arrange
> All prerequisites are met and therefore remain $\boxed{4}$.
> <!-- bilingual-en:end -->

### Problem 2 — Partial Orders & Equivalence（20 分）
<!-- bilingual-en:start -->
*Problem 2 — Partial Orders & Equivalence (20 points)*
<!-- bilingual-en:end -->

首次回链 [[02_Structures#Session 18 — Partial Orders and Equivalence|等价关系]] 与 [[02_Structures#Session 18 — Partial Orders and Equivalence|弱偏序]]。
<!-- bilingual-en:start -->
[[02_Structures#Session 18 — Partial Orders and Equivalence|equivalent relation]] and [[02_Structures#Session 18 — Partial Orders and Equivalence|weak partial order]].
<!-- bilingual-en:end -->

> [!answer]- 完整解答
> **(a)** 唯一候选是恒等关系
> $$
> I_A=\{(a,a):a\in A\},
> $$
> 即 $aI_Ab\iff a=b$。它自反、对称、传递，所以是等价关系；也自反、反对称、传递，所以是弱偏序。
> **(b) 唯一性证明**：设 $R$ 同时是等价关系和弱偏序。若 $aRb$，等价关系的对称性给 $bRa$；弱偏序的反对称性于是给 $a=b$。故 $R\subseteq I_A$。另一方面，$R$ 的自反性保证每个 $(a,a)\in R$，故 $I_A\subseteq R$。两边合并，$R=I_A$。
> <!-- bilingual-en:start -->
> **(a)**The only candidate is the identity relation
> $aI_Ab\iff a=b$.  It is reflexive, symmetric and transitive, so it is equivalent relation; it is also reflexive, anti-symmetric and transitive, so it is weak partial order.
> **(b) Uniqueness proof**: Let $R$ be both equivalence relation and weak partial order.  If $aRb$, the symmetry of the equivalence relation is given to $bRa$; the antisymmetry of the weak partial order is given to $a=b$.  So, $R\subseteq I_A$.  On the other hand, the reflexivity of $R$ guarantees each $(a,a)\in R$, so $I_A\subseteq R$.  $R=I_A$.
> <!-- bilingual-en:end -->

### Problem 3 — Simple Graphs（20 分）
<!-- bilingual-en:start -->
*Problem 3 — Simple Graphs (20 points)*
<!-- bilingual-en:end -->

> [!answer]- 完整解答
> **(a)** 取五个顶点 $u,v,a,b,c$，边集
> $$
> \{ua,ab,bv,ac,cb\}.
> $$
> 从 $u$ 到 $v$ 有两条不同路径 $u-a-b-v$ 与 $u-a-c-b-v$。唯一形成的环是 $a-b-c-a$，既不含 $u$ 也不含 $v$；且 $u,v$ 的度均为 $1$，不可能属于任何环。
> **(b)** 设 $P,Q$ 是 $u$ 到 $v$ 的两条不同简单路径。从 $u$ 出发，令 $x$ 为它们仍共同经过后第一次分叉的顶点。沿 $P$ 从 $x$ 前进，由于两路最终都到 $v$，必会再次碰到 $Q$；令 $y$ 为第一次重逢点。按 $y$ 的“第一次”定义，$P$ 的 $x$–$y$ 段与 $Q$ 的 $x$–$y$ 段除端点外没有公共顶点；两段也不同。沿第一段从 $x$ 到 $y$，再沿第二段反向回 $x$，得到一个简单环。因此不同路径必推出图中存在环，但环未必含 $u$ 或 $v$，这正是 (a) 的意义。
> <!-- bilingual-en:start -->
> **(a)** Take vertices $u,v,a,b,c$ and edges $\{ua,ab,bv,ac,cb\}$. There are two distinct $u$–$v$ paths, $u-a-b-v$ and $u-a-c-b-v$. The only cycle is $a-b-c-a$, which contains neither $u$ nor $v$; moreover, $u$ and $v$ each have degree $1$ and cannot lie on any cycle.
> **(b)** Let $P,Q$ be distinct simple paths from $u$ to $v$. Starting from $u$, let $x$ be their first divergence point, and let $y$ be the first later vertex at which they meet again. By the choice of $y$, the $x$–$y$ segments of $P$ and $Q$ share only their endpoints and are distinct. Traversing one segment from $x$ to $y$ and the other backward from $y$ to $x$ gives a simple cycle. Thus two distinct paths force a cycle somewhere in the graph, though not necessarily one containing $u$ or $v$, as part (a) shows.
> <!-- bilingual-en:end -->

### Problem 4 — Trees & Coloring（20 分）
<!-- bilingual-en:start -->
*Problem 4 — Trees & Coloring (20 points)*
<!-- bilingual-en:end -->

首次回链 [[无环图：树、生成树、DAG 与拓扑排序#Tree 的等价刻画|树]] 与 [[图着色与色数#如何证明 chromatic number|图着色]]。
<!-- bilingual-en:start -->
[[无环图：树、生成树、DAG 与拓扑排序#Tree 的等价刻画|tree]] and [[图着色与色数#如何证明 chromatic number|graph coloring]].
<!-- bilingual-en:end -->

> [!answer]- 完整归纳证明
> **命题**：固定 $n>1$ 种颜色，任何含 $m$ 个顶点的树都有恰好
> $$
> \boxed{n(n-1)^{m-1}}
> $$
> 个合法顶点着色。
> **基例 $m=1$**：唯一顶点可任选 $n$ 种颜色，数量 $n=n(n-1)^0$。
> **归纳步**：假设所有 $m-1$ 顶点的树均有 $n(n-1)^{m-2}$ 个合法着色。任取一棵 $m$ 顶点树 $T$。有限树至少有一个叶子 $v$；删去 $v$ 及其唯一关联边，所得 $T'$ 仍连通且无环，故是 $m-1$ 顶点树。按归纳假设，$T'$ 有 $n(n-1)^{m-2}$ 个着色。对每个着色，$v$ 只需避开其唯一邻点的颜色，所以恰有 $n-1$ 种扩展；不同 $T'$ 着色或不同扩展产生不同 $T$ 着色，且每个 $T$ 着色都唯一限制回 $T'$。由乘法法则，
> $$
> n(n-1)^{m-2}(n-1)=n(n-1)^{m-1}.
> $$
> 命题得证。
> <!-- bilingual-en:start -->
> **Claim.** For fixed $n>1$ colours, every tree with $m$ vertices has exactly $n(n-1)^{m-1}$ proper vertex colourings.
> **Base case $m=1$.** The single vertex may receive any of the $n$ colours, giving $n=n(n-1)^0$ colourings.
> **Inductive step.** Assume every tree with $m-1$ vertices has $n(n-1)^{m-2}$ proper colourings. Let $T$ be an $m$-vertex tree and remove a leaf $v$ and its incident edge. The remaining graph $T'$ is an $(m-1)$-vertex tree. By induction it has $n(n-1)^{m-2}$ colourings. For each one, $v$ may receive any colour except that of its unique neighbour, giving exactly $n-1$ extensions. Restriction back to $T'$ is unique, so the multiplication rule gives $n(n-1)^{m-1}$ colourings.
> <!-- bilingual-en:end -->

### Problem 5 — Stable Marriage（15 分）
<!-- bilingual-en:start -->
*Problem 5 — Stable Marriage (15 points)*
<!-- bilingual-en:end -->

首次回链 [[02_Structures#Session 22 — Stable Matching and Hall's Theorem|稳定匹配]]。题目把未婚男子与更偏爱他的已婚女子也视作 rogue couple；男子数可以多于女子数。
<!-- bilingual-en:start -->
[[02_Structures#Session 22 — Stable Matching and Hall's Theorem|stable matching]].  The topic also considered unmarried men and married women who preferred him as rogue couple; the number of men could be greater than that of women.
<!-- bilingual-en:end -->

> [!answer]- 完整判断
> 保持性不变量为 $\boxed{(a),(d),(g)}$。逐项检查状态转移：
>
> - **(a)** Alice 已有一个她比 Bob 更喜欢的追求者：女性只会保留目前最喜欢的追求者，之后保留者只可能变得更好，所以该性质保持。
> - **(b)** Bob 的名单只剩 Alice：若 Alice 拒绝 Bob，她会被删去，名单变空，故不保持。
> - **(c)** Alice 没有追求者：下一步可能有人向她求婚，故不保持。
> - **(d)** Bob 更喜欢 Alice，胜过他当前追求的女子：男子被拒后只会沿偏好表向下移动；若该关系为真，之后追求对象只会更不受偏爱，所以保持。
> - **(e)** Bob 正追求 Alice：Alice 可能拒绝他，故不保持。
> - **(f)** Bob 没有追求 Alice：他可能在被更高偏好女子拒绝后轮到 Alice，故不保持。
> - **(g)** Bob 的名单为空：名单只删不增，故保持。
> <!-- bilingual-en:start -->
> The retention invariant is $\boxed{(a),(d),(g)}$.  To check for status transitions on a case by case basis:
> -**(a)**Alice already has a suitor she prefers to Bob: women only retain their current favorite suitor, after which the retainer is only likely to become better, so that nature remains.
> -**(b)**Bob's list is Alice only: if Alice rejects Bob, she will be deleted and the list will be empty and therefore not maintained.
> -**(c)**Alice has no suitor: the next step may be a proposal from her and therefore not maintained.
> -**(d)**Bob prefers Alice to the woman he's currently pursuing: the man who is rejected will only move down the preference table; if the relationship is true, then the person he's after will be less preferred, so stay.
> -**(e)**Bob is pursuing the possibility that Alice:Alice may reject him and therefore not keep it.
> -**(f)**Bob did not pursue Alice: he may have been turned down by a woman with a higher preference, and therefore not held on to Alice.
> -**(g)**Bob's list is empty: the list is deleted but not added, so it remains.
> <!-- bilingual-en:end -->

### Problem 6 — Sums & Integrals（10 分）
<!-- bilingual-en:start -->
*Problem 6 — Sums & Integrals (10 points)*
<!-- bilingual-en:end -->

> [!answer]- 完整解答
> **(a)** 临界值为 $\boxed{a=-1}$。
> **(b)** 好方法是 $\boxed{\text{i 与 v}}$：计算 $\int_1^\infty x^pdx$，并与调和级数逐项比较。积分在 $p<-1$ 时有限，在 $p\ge-1$ 时发散；边界 $p=-1$ 正是调和级数。对 $p$ 归纳没有自然离散步长，也不能控制所有实数 $p$；对有限上限 $n$ 的部分和归纳不能自行决定无穷极限。
> <!-- bilingual-en:start -->
> **(a)** The critical value is $\boxed{a=-1}$.
> **(b)** The correct choices are $\boxed{\text{i and v}}$: compute $\int_1^\infty x^p\,dx$ and compare term by term with the harmonic series. The integral is finite when $p<-1$ and diverges when $p\ge-1$; the boundary case $p=-1$ is exactly the harmonic series. There is no natural induction step in the real parameter $p$, and induction on finite partial sums does not determine the infinite limit.
> <!-- bilingual-en:end -->

### Midterm 3 错误诊断
<!-- bilingual-en:start -->
*Midterm 3 Troubleshooting*
<!-- bilingual-en:end -->

- 调度时间看最长**链的顶点数**，不是边数；并行上限还要同时看总工作量。
- “对称 + 反对称”不是矛盾；它们合在一起把关系压缩到对角线。
- 两条路径保证某处有环，不保证端点在环上。
- 树着色归纳必须说明删叶后仍是树，以及扩展恰为 $n-1$ 对一。
- 不变量题问的是每次转移后的保持性，不是某个状态下偶然为真。
<!-- bilingual-en:start -->
- The longest scheduling time**the number of vertices in the chain**not the number of edges; the upper limit of parallelism also depends on the total workload.
- "Symmetry + antisymmetry" is not a contradiction; together, they compress the relationship diagonally.
- The two paths guarantee that there is a loop somewhere and that the endpoints are not on the loop.
- Tree coloring induction must show that leaves are still trees after deletion and that the extension is exactly $n-1$ to one.
- The invariant question is about retention after each transfer, not accidental truth in a state.
<!-- bilingual-en:end -->

---

## Session 25 — Counting with Bijections

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：一个对象能分成互斥情形时怎样相加？由连续选择构造时怎样相乘？当原对象难数时，怎样用一个可逆编码把它搬到熟悉集合？
<!-- bilingual-en:start -->
**Learning Problem**: How do objects add when they can be divided into mutually exclusive situations?  How are multiplications constructed from sequential selections?  When the original object is hard to be counted, how to transfer it to the familiar set with a reversible code?
<!-- bilingual-en:end -->

**前置知识**：有限集合、函数、单射/满射/双射、笛卡尔积。首次正式使用 [[组合计数原理#加法、乘法与双射|加法与乘法法则]]、[[组合计数原理#加法、乘法与双射|双射计数原理]] 与 [[组合计数原理#加法、乘法与双射|计数策略选择框架]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge**: Finite sets, functions, monojets/surjections/bijections, Cartesian products.  First live use of [[组合计数原理#加法、乘法与双射|addition and multiplication rules]], [[组合计数原理#加法、乘法与双射|double-fire counting principle]], and [[组合计数原理#加法、乘法与双射|Counting Policy Selection Framework]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session25.pdf#page=1|Session 25 reading, pp. 1–6]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp25.pdf#page=1|cp25, pp. 1–3]]

### 3.3.1 Sum and Product Rules — 先确认集合结构
<!-- bilingual-en:start -->
*3.3.1 Sum and Product Rules—First Confirm Collection Structure*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_SumProduct.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/yTrtVwKZkwU.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=yTrtVwKZkwU)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_SumProduct.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/yTrtVwKZkwU.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=yTrtVwKZkwU)
<!-- bilingual-en:end -->

#### 加法法则（Sum Rule）
<!-- bilingual-en:start -->
*Sum Rule*
<!-- bilingual-en:end -->

若有限集合 $A_1,\ldots,A_k$ **两两不交**，则
<!-- bilingual-en:start -->
If the finite set $A_1,\ldots,A_k$**two disjoint**
<!-- bilingual-en:end -->

$$
\boxed{\left|\bigcup_{i=1}^{k}A_i\right|
=\sum_{i=1}^{k}|A_i|}.
$$

证明很直接但条件不可省：并集中的每个对象恰属于一个 $A_i$，所以右边恰数一次。若集合重叠，右边会重复计数，需等到 Session 27 用容斥修正。
<!-- bilingual-en:start -->
The proof is straightforward, but the condition is inescapable: each object in the union belongs to exactly one $A_i$, so the right hand side is exactly once.  If the collections overlap, the right side will repeat the count, waiting until Session 27 is filled.
<!-- bilingual-en:end -->

#### 乘法法则（Product Rule）
<!-- bilingual-en:start -->
*Product Rule*
<!-- bilingual-en:end -->

对有限集合 $A_1,\ldots,A_k$，
<!-- bilingual-en:start -->
For a finite set $A_1,\ldots,A_k$,
<!-- bilingual-en:end -->

$$
\boxed{|A_1\times\cdots\times A_k|
=\prod_{i=1}^{k}|A_i|}.
$$

证明可对 $k$ 归纳。$k=2$ 时，对每个 $a_1\in A_1$ 恰有 $|A_2|$ 个有序对 $(a_1,a_2)$，共 $|A_1||A_2|$；归纳步把前 $k-1$ 个坐标视作一个整体。
<!-- bilingual-en:start -->
The proof can be generalized to $k$.  When $k=2$, there are exactly $|A_2|$ ordered pairs of $(a_1,a_2)$ for each $a_1\in A_1$, altogether $|A_1||A_2|$; the induction step treats the first $k-1$ coordinates as a whole.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-sum-product-rule.png|900]]

读图：互斥分支对应并集，所以把各分支数量相加；连续决策对应笛卡尔积，所以把每一步可选数量相乘。判断“分支还是步骤”比记公式更重要。
<!-- bilingual-en:start -->
Reading image: Mutually exclusive branches correspond to union, so the number of each branch is added; Continuous decision corresponds to Cartesian product, so the number of each step can be multiplied.  Determining whether a branch or step is more important than memorizing a formula.
<!-- bilingual-en:end -->

#### 例：密码
<!-- bilingual-en:start -->
*Example: Password*
<!-- bilingual-en:end -->

密码长度为 6、7 或 8；首字符必须是 52 个大小写字母之一，其余位置可用 62 个字母或数字。不同长度互斥，所以
<!-- bilingual-en:start -->
Passwords are 6, 7, or 8 in length; the first character must be one of 52 upper and lower case letters, and 62 letters or numbers are available in the remaining locations.  different length are mutually exclusive, so
<!-- bilingual-en:end -->

$$
52\cdot62^5+52\cdot62^6+52\cdot62^7
=\boxed{52(62^5+62^6+62^7)}.
$$

### 3.3.2 Counting Practice — 官方在线题 O25-01、O25-02
<!-- bilingual-en:start -->
*3.3.2 Counting Practice—official online questions O25-01, O25-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.2_counting-practice|3.3.2]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.2_counting-practice|3.3.2]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> **O25-01**：5 位 PIN 每位有 10 种选择，前导零允许，所以 $10^5=\boxed{100000}$。
> **O25-02**：上衣有 $3+2=5$ 种，下装有 $4+4=8$ 种，搭配数 $5\cdot8=\boxed{40}$。
> <!-- bilingual-en:start -->
> **O25-01**:5 bit PINs have 10 choices per pin, leading zeroes allow, so $10^5=\boxed{100000}$.
> **O25-02**: $3+2=5$ for tops, $4+4=8$ for underwear and $5\cdot8=\boxed{40}$ for number of pairs.
> <!-- bilingual-en:end -->

### 3.3.3 Counting with Bijections — 把“难数”变成“已知可数”
<!-- bilingual-en:start -->
*3.3.3 Counting with Bijections—Turning "hard" into "known to count"*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bijections.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/n0lce1dMAh8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=n0lce1dMAh8)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bijections.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/n0lce1dMAh8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=n0lce1dMAh8)
<!-- bilingual-en:end -->

#### 双射法则
<!-- bilingual-en:start -->
*bijective rule*
<!-- bilingual-en:end -->

若存在双射 $f:A\to B$，则
<!-- bilingual-en:start -->
If there is a bijection $f:A\to B$, then
<!-- bilingual-en:end -->

$$
\boxed{|A|=|B|}.
$$

**为什么**：单射保证不同的 $A$ 对象不会被合并，满射保证每个 $B$ 对象都有来源。因此配对一一对应，没有遗漏也没有重复。写双射题解时应包含三件事：正向编码、逆向解码、二者互逆。
<!-- bilingual-en:start -->
**Why:** Injectivity prevents distinct objects in $A$ from being merged, while surjectivity ensures that every object in $B$ is reached. Thus the correspondence has neither omissions nor duplicates. A bijective counting argument should give the forward encoding, the inverse decoding, and a check that the two maps are inverses.
<!-- bilingual-en:end -->

#### Stars and Bars：有重复选择
<!-- bilingual-en:start -->
*Stars and Bars: Duplicate Selection*
<!-- bilingual-en:end -->

从 $k$ 种甜甜圈中选 $n$ 个，令 $x_i$ 是第 $i$ 种数量，则需数
<!-- bilingual-en:start -->
Select $n$ from $k$ doughnuts, making $x_i$ the $i$, then count
<!-- bilingual-en:end -->

$$
x_1+\cdots+x_k=n,\qquad x_i\ge0.
$$

编码为 $n$ 个星与 $k-1$ 个隔板：
<!-- bilingual-en:start -->
Numbered $n$ stars and $k-1$ partitions:
<!-- bilingual-en:end -->

$$
\underbrace{\star\cdots\star}_{x_1}\mid
\underbrace{\star\cdots\star}_{x_2}\mid\cdots\mid
\underbrace{\star\cdots\star}_{x_k}.
$$

总位置 $n+k-1$，任选 $k-1$ 个放隔板（或任选 $n$ 个放星），故
<!-- bilingual-en:start -->
Total $n+k-1$, optional $k-1$ shelves (or optional $n$ shelves), so
<!-- bilingual-en:end -->

$$
\boxed{\#\{x_1+\cdots+x_k=n:x_i\ge0\}
=\binom{n+k-1}{k-1}}.
$$

这确为双射：给定向量能唯一写出字符串；给定字符串，数各隔板间星数能唯一恢复向量。
<!-- bilingual-en:start -->
This is a genuine bijection: a vector determines a unique stars-and-bars string, and counting the stars between successive bars uniquely recovers the vector.
<!-- bilingual-en:end -->

#### 计数全函数
<!-- bilingual-en:start -->
*Counting total functions*
<!-- bilingual-en:end -->

设 $A=\{a_1,\ldots,a_m\}$、$|B|=q$。全函数 $f:A\to B$ 与向量
<!-- bilingual-en:start -->
$A=\{a_1,\ldots,a_m\}$, $|B|=q$.  Full Function $f:A\to B$ and Vector
<!-- bilingual-en:end -->

$$
(f(a_1),\ldots,f(a_m))\in B^m
$$

双射，因此共有
<!-- bilingual-en:start -->
are in bijection, so there are
<!-- bilingual-en:end -->

$$
\boxed{q^m=|B|^{|A|}}
$$

个。每个定义域元素都是一个独立位置，每个位置有 $q$ 种选择。
<!-- bilingual-en:start -->
.  Each domain element is a separate location, with $q$ choices for each location.
<!-- bilingual-en:end -->

### 3.3.4 Selecting Donuts — 官方在线题 O25-03
<!-- bilingual-en:start -->
*3.3.4 Selecting Donuts — official online O25-03*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.4_selecting-donuts|3.3.4]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.4_selecting-donuts|3.3.4]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 选 13 个、4 种口味对应 13 个零（甜甜圈）与 3 个一（隔板），故与“长度 16、恰有 3 个 1 的二进制串”双射。答案为 $\binom{16}{3}$ 个。
> <!-- bilingual-en:start -->
> 13, 4 flavors correspond to 13 zeros (doughnuts) and 3 ones (separators), thus being biased to a "binary string of length 16 and exactly 3 ones".  The answer is $\binom{16}{3}$.
> <!-- bilingual-en:end -->

### 3.3.5 Counting Functions — 官方在线题 O25-04
<!-- bilingual-en:start -->
*3.3.5 Counting Functions — official online O25-04*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.5_counting-functions|3.3.5]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.5_counting-functions|3.3.5]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 若 $|A|=3,|B|=7$，全函数 $A\to B$ 与 $B^3$ 的向量双射，所以共有 $\boxed{7^3}$ 个。
> <!-- bilingual-en:start -->
> If $|A|=3$ and $|B|=7$, total functions $A\to B$ are in bijection with vectors in $B^3$, so there are $\boxed{7^3}$ such functions.
> <!-- bilingual-en:end -->

### Session 25 易错点与反例
<!-- bilingual-en:start -->
*Session 25 Error-prone Points and Counterexamples*
<!-- bilingual-en:end -->

1. 加法法则要求分类互斥；“衬衫”与“红色上衣”会重叠，不能直接相加。
2. 乘法法则数的是有序决策序列；若最终对象不记顺序，可能要用双射或除法消除重复。
3. 双射不能只说“显然对应”；必须说明逆映射，否则可能是多对一。
4. Stars and Bars 只处理非负整数解；若每类至少一个，先令 $y_i=x_i-1$。
5. PIN、字符串通常允许前导零；“五位整数”通常不允许，两个样本空间不同。
<!-- bilingual-en:start -->
1. The addition rule requires that the categories are mutually exclusive; "shirt" and "red jacket" overlap and cannot be added directly.
2. The number of multiplication rules is an ordered decision sequence; if the final object does not remember the order, it may need to be bijective or division to eliminate duplication.
3. Bijection cannot say only "apparent correspondence"; it must state the inverse mapping, or it may be many-to-one.
4. Stars and Bars directly handles nonnegative integer solutions. If each class must contain at least one object, first set $y_i=x_i-1$.
5. PINs, strings, usually allow leading zeros; "five-digit integers" are not allowed, and the two sample spaces are different.
<!-- bilingual-en:end -->

### Session 25 自检
<!-- bilingual-en:start -->
*Session 25 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 25-1
> 有多少长度 8 的二进制串恰含 3 个 1？
> <!-- bilingual-en:start -->
> How many binary strings of length 8 contain exactly 3 ones?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 选 3 个位置放 1：$\binom83=56$。
> <!-- bilingual-en:start -->
> Select 3 locations for 1:$\binom83=56$.
> <!-- bilingual-en:end -->

> [!question] 自检 25-2
> 非负整数解 $x_1+x_2+x_3=10$ 有多少个？若每个 $x_i\ge1$ 呢？
> <!-- bilingual-en:start -->
> How many non-negative integer solutions are $x_1+x_2+x_3=10$?  What about every $x_i\ge1$?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 非负解 $\binom{12}{2}=66$。正整数解令 $y_i=x_i-1$，则 $y_1+y_2+y_3=7$，有 $\binom92=36$。
> <!-- bilingual-en:start -->
> non-negative solution $\binom{12}{2}=66$.  A positive integer solution is $y_i=x_i-1$, then $y_1+y_2+y_3=7$, and has $\binom92=36$.
> <!-- bilingual-en:end -->

> [!question] 自检 25-3
> 为什么“从 5 人选主席和副主席”是 $5\cdot4$，不是 $\binom52$？
> <!-- bilingual-en:start -->
> Why is the "From 5 Chair and Vice chair" $5\cdot4$, not $\binom52$?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 两个职位有角色顺序；同一对人交换职位产生不同结果。先选主席 5 种，再从剩余人选副主席 4 种。
> <!-- bilingual-en:start -->
> The two offices are distinct, so swapping the same two people gives a different outcome. There are 5 choices for chair and then 4 choices for vice-chair.
> <!-- bilingual-en:end -->

### Classroom Problems 25 — 4 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 25 — 4 complete independent questions*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp25.pdf#page=1|cp25 pp. 1–3]]
<!-- bilingual-en:start -->
Original Question: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp25.pdf#page=1|cp25 pp. 1–3]]
<!-- bilingual-en:end -->

> [!example]- C25-1 含数字 1 与不相邻选书
> **(a)** 把 $1,\ldots,999{,}999{,}999$ 写成 9 位、允许前导零的串。每位避开 1 有 9 种，共 $9^9$ 个串，其中全零不在正整数范围，所以该区间不含 1 的数有 $9^9-1$ 个；$10^9$ 本身含 1。总共有 $10^9$ 个整数，因此
> $$
> \boxed{10^9-(9^9-1)=10^9-9^9+1=612{,}579{,}512}.
> $$
> 若把样本空间改为 $0$ 到 $10^9-1$，答案会少 1；这解释了常见的 $10^9-9^9$ 版本。
> **(b)** 设所选书位置为 $1\le i_1<\cdots<i_6\le20$，且 $i_{r+1}\ge i_r+2$。定义
> $$
> j_r=i_r-(r-1).
> $$
> 则 $1\le j_1<\cdots<j_6\le15$，在长度 15 的串中把第 $j_r$ 位设为 1。逆映射是 $i_r=j_r+(r-1)$，自动保证间隔至少 2。故所求与恰含 6 个 1 的 15 位串双射，数量 $\boxed{\binom{15}{6}}$。
> <!-- bilingual-en:start -->
> **(a)**Writes $1,\ldots,999{,}999{,}999$ as a 9-bit string that allows leading zeros.  There are 9 of $9^9$ strings for each bypass 1, where all zeros are not in the range of positive integers, so there are $9^9-1$ numbers for the interval without a 1; $10^9$ itself contains a 1.  There are a total of $10^9$ integers, so
> If you change the sample space to $0$ to $10^9-1$, the answer is less than 1; this explains the common $10^9-9^9$ versions.
> **(b)**Set the book location to $1\le i_1<\cdots<i_6\le20$ and $i_{r+1}\ge i_r+2$.  defined
> $1\le j_1<\cdots<j_6\le15$; put a 1 in positions $j_1,\ldots,j_6$ of a length-15 bit string. The inverse map is $i_r=j_r+(r-1)$, which automatically restores a gap of at least 2. Hence the desired selections are in bijection with length-15 bit strings containing six 1s, so their number is $\boxed{\binom{15}{6}}$.
> <!-- bilingual-en:end -->

> [!example]- C25-2 最大叶编码与 Cayley 公式
> 对编号树反复删除“当前最大叶子”，记录其相邻顶点（father），得到长度 $n-2$ 的码。
> **逆向算法**：给定当前剩余码 $c_1,\ldots,c_r$ 和尚未删除的标签集合，取“不出现在当前码中”的最大标签 $\ell$；把 $\ell$ 与 $c_1$ 连边，删除 $\ell$ 与首码 $c_1$，重复。码用尽后连接最后两个标签。
> **为什么选择唯一存在**：一棵树剩余时，未出现在余码中的标签正是会在成为 father 前作为叶删除的候选；取最大者严格复现编码规则。每一步正向删除与逆向加边互相抵消，最后恢复唯一树。因此编号树与 $\{1,\ldots,n\}^{n-2}$ 双射，
> $$
> \boxed{\#\text{编号树}=n^{n-2}}.
> $$
> <!-- bilingual-en:start -->
> Repeatedly delete the currently largest leaf and record its adjacent vertex, producing a code of length $n-2$.
> **Inverse algorithm:** Given the remaining code $c_1,\ldots,c_r$ and the set of labels not yet deleted, choose the largest label $\ell$ that does not appear in the remaining code. Add the edge $\{\ell,c_1\}$, remove $\ell$ and the first code entry $c_1$, and repeat. When the code is exhausted, connect the final two labels.
> **Why the choice exists and is unique:** In the remaining tree, the labels absent from the remaining code are exactly the vertices eligible to be deleted as leaves before ever appearing as a recorded neighbour. Choosing the largest one reproduces the encoding rule. Each forward deletion is undone by one inverse step, so the procedure reconstructs a unique tree. Thus labelled trees are in bijection with $\{1,\ldots,n\}^{n-2}$,
> <!-- bilingual-en:end -->

> [!example]- C25-3 两层双射
> **(a)** 对 $S_{n,k}=\{(x_1,\ldots,x_k)\in\mathbb N^k:\sum x_i\le n\}$，加入松弛量 $x_{k+1}=n-\sum_{i=1}^kx_i$。编码
> $$
> 0^{x_1}1,0^{x_2}1\cdots0^{x_k}1,0^{x_{k+1}}
> $$
> 恰含 $n$ 个 0、$k$ 个 1。按 1 分段数 0 即可逆向恢复，所以是双射。
> **(b)** 对弱递增序列 $0\le y_1\le\cdots\le y_k\le n$，令
> $$
> x_1=y_1,\qquad x_i=y_i-y_{i-1}\ (i\ge2).
> $$
> 则 $x_i\ge0$ 且 $\sum_{i=1}^kx_i=y_k\le n$，得到 $S_{n,k}$。逆映射为 $y_i=\sum_{j=1}^ix_j$。因此两集合双射，大小均为 $\binom{n+k}{k}$。
> <!-- bilingual-en:start -->
> **(a)** For $S_{n,k}=\{(x_1,\ldots,x_k)\in\mathbb N^k:\sum x_i\le n\}$, introduce the slack variable $x_{k+1}=n-\sum_{i=1}^kx_i$. Encode the vector as a string containing exactly $n$ zeros and $k$ ones. Splitting the string at its ones and counting the zeros in each segment recovers the vector, so the map is bijective.
> **(b)**For weakly increasing sequences $0\le y_1\le\cdots\le y_k\le n$,
> $x_i\ge0$ and $\sum_{i=1}^kx_i=y_k\le n$, resulting in $S_{n,k}$.  The inverse mapping is $y_i=\sum_{j=1}^ix_j$.  Therefore, the two sets are bijective, the size of which is $\binom{n+k}{k}$.
> <!-- bilingual-en:end -->

> [!example]- C25-4 关系、函数、部分函数、子集与置换
> 令 $|X|=m,|Y|=q$，并假设 $Y\ne\varnothing$。
> **(a)** 二元关系是 $X\times Y$ 的任意子集，共
> $$
> \boxed{2^{mq}}.
> $$
> **(b)** 固定 $X=(x_1,\ldots,x_m)$ 的次序，$f\mapsto(f(x_1),\ldots,f(x_m))$ 是从全函数集合到 $Y^m$ 的双射，所以有 $\boxed{q^m}$ 个。
> **(c)** 部分函数对每个 $x$ 有 $q$ 个像或“未定义”这一额外选择，共 $\boxed{(q+1)^m}$。它们与全函数数目之比为
> $$
> \left(1+\frac1q\right)^m,
> $$
> 对固定 $q$ 按指数增长，并且是 $O(2^m)$，不是 $O(1)$ 或 $O(m)$。
> **(d)** 子集 $S\subseteq X$ 映到特征函数 $\mathbf1_S:X\to\{0,1\}$；逆映射取 $f^{-1}(1)$，故 $|\operatorname{pow}(X)|=2^m$。
> **(e)** 双射 $X\to X$ 按固定标签次序写出像序列，恰是 $X$ 的一个排列；逆向由排列定义函数。因此有 $\boxed{m!}$ 个。
> <!-- bilingual-en:start -->
> Let $|X|=m,|Y|=q$ and assume $Y\ne\varnothing$.
> **(a)**The binary relation is any subset of $X\times Y$
> **(b)** Fix an ordering $X=(x_1,\ldots,x_m)$. The map $f\mapsto(f(x_1),\ldots,f(x_m))$ is a bijection from the set of total functions $X\to Y$ to $Y^m$, so there are $\boxed{q^m}$ such functions.
> **(c)**Some of the functions have an additional choice of $q$ images or "undefined" for each $x$ of $\boxed{(q+1)^m}$.  their ratio to the number of whole functions
> For a fixed $q$, the growth is exponential and is $O(2^m)$, not $O(1)$ or $O(m)$.
> **(d)**The subset $S\subseteq X$ is mapped to the characteristic function $\mathbf1_S:X\to\{0,1\}$; the inverse mapping is $f^{-1}(1)$, so $|\operatorname{pow}(X)|=2^m$.
> **(e)** For a bijection $X\to X$, list the images in a fixed order of the domain. This produces a permutation of $X$, and every permutation defines a unique bijection in reverse. Hence there are $\boxed{m!}$ bijections.
> <!-- bilingual-en:end -->

### Session 25 知识链小结
<!-- bilingual-en:start -->
*Session 25 Knowledge Chain Summary*
<!-- bilingual-en:end -->

$$
\text{互斥并集}\Rightarrow\text{加法}
\quad\text{与}\quad
\text{连续选择}\Rightarrow\text{乘法}
\longrightarrow\text{构造双射}
\longrightarrow\text{字符串/向量/子集标准模型}.
$$

---

## Session 26 — Repetitions & Binomial Theorem

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：多对一映射怎样安全地除掉重复？有重复字母的排列、无序分组、组合数与多项式系数为何是同一个结构？
<!-- bilingual-en:start -->
**Learning Questions**: How can many-to-one mappings safely eliminate duplicates?  Why is the arrangement of repeating letters, unordered grouping, number of combinations, and polynomial coefficients the same structure?
<!-- bilingual-en:end -->

**前置知识**：Session 25 的和、积、双射；阶乘；集合与序列。
<!-- bilingual-en:start -->
**Prerequisites**: Sum, Product, Bijective; Factorial; Set and Sequence for Session 25.
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session26.pdf#page=1|Session 26 reading, pp. 1–15]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp26.pdf#page=1|cp26, pp. 1–4]]

### 3.4.1 Generalized Counting Rules — 依赖选择与除法
<!-- bilingual-en:start -->
*3.4.1 Generalized Counting Rules—Dependent Selection and Division*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Generalized.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iDfyX8WRIyM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=iDfyX8WRIyM)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Generalized.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iDfyX8WRIyM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=iDfyX8WRIyM)
<!-- bilingual-en:end -->

#### 广义乘法法则
<!-- bilingual-en:start -->
*generalized multiplication rule*
<!-- bilingual-en:end -->

若一个合法对象由 $k$ 步构造，并且对每个合法的前 $i-1$ 步前缀，第 $i$ 步都**恰有** $m_i$ 种合法延伸，则对象数为
<!-- bilingual-en:start -->
If a valid object is built in $k$ steps and every valid prefix of length $i-1$ has **exactly** $m_i$ valid extensions at step $i$, then the number of objects is
<!-- bilingual-en:end -->

$$
\boxed{m_1m_2\cdots m_k}.
$$

证明对步骤数归纳：第一步分成 $m_1$ 个互斥类；每一类按归纳假设恰有 $m_2\cdots m_k$ 个完成方式，再用加法法则。注意延伸的具体选项可以依赖前缀，但数量必须统一为 $m_i$。
<!-- bilingual-en:start -->
Prove this by induction on the number of steps. The first choice partitions the objects into $m_1$ disjoint classes; by the induction hypothesis, each class has exactly $m_2\cdots m_k$ completions. The addition rule then gives the product. The available extensions may depend on the preceding choices, but their number must always be $m_i$ at step $i$.
<!-- bilingual-en:end -->

#### Division Rule

若映射 $f:A\to B$ 是满射，且每个 $b\in B$ 都**恰有 $k$ 个**原像，则称为 $k$-to-1，且
<!-- bilingual-en:start -->
If $f:A\to B$ is surjective and every $b\in B$ has **exactly $k$** preimages, then $f$ is called $k$-to-1, and
<!-- bilingual-en:end -->

$$
\boxed{|B|=\frac{|A|}{k}}.
$$

证明：各纤维 $f^{-1}(b)$ 两两不交、并集为 $A$，每个大小 $k$，故 $|A|=\sum_{b\in B}k=k|B|$。若原像数不恒定，不能除一个统一 $k$。
<!-- bilingual-en:start -->
The fibres $f^{-1}(b)$ are pairwise disjoint, their union is $A$, and each has size $k$. Therefore $|A|=\sum_{b\in B}k=k|B|$. If the number of preimages is not constant, division by a single common $k$ is invalid.
<!-- bilingual-en:end -->

#### 子集与组合数
<!-- bilingual-en:start -->
*Subsets and Combinations*
<!-- bilingual-en:end -->

从 $n$ 个不同元素中选 $k$ 个。先数有序选择：$n(n-1)\cdots(n-k+1)=n!/(n-k)!$。忘掉顺序后，每个 $k$ 元子集有 $k!$ 个排列作为原像，所以
<!-- bilingual-en:start -->
Choose $k$ elements from $n$ distinct elements. First count ordered choices: $n(n-1)\cdots(n-k+1)=n!/(n-k)!$. After order is forgotten, each $k$-element subset has $k!$ ordered preimages, so
<!-- bilingual-en:end -->

$$
\boxed{\binom nk=\frac{n!}{k!(n-k)!}}.
$$

#### 圆排列
<!-- bilingual-en:start -->
*Circle arrangement*
<!-- bilingual-en:end -->

$n$ 个不同对象排成圆，若旋转视为相同，则线性排列到圆排列是 $n$-to-1，数量
<!-- bilingual-en:start -->
$n$ different objects arranged in circles, if rotation is treated as the same, the linear arrangement to circles arrangement is $n$-to-1, quantity
<!-- bilingual-en:end -->

$$
\boxed{\frac{n!}{n}=(n-1)!}.
$$

### 3.4.2 Choosing Integers — 官方在线题 O26-01
<!-- bilingual-en:start -->
*3.4.2 Choosing Integers — official online O26-01*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S26_3.4.2_choosing-integers|3.4.2]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S26_3.4.2_choosing-integers|3.4.2]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 闭区间 $[3,15]$ 含 $15-3+1=13$ 个整数，任选 2 个有
> $$
> \boxed{\binom{13}{2}=78}
> $$
> 种。
> <!-- bilingual-en:start -->
> Closed interval $[3,15]$ contains $15-3+1=13$ integers, optionally 2 have
> Seed.
> <!-- bilingual-en:end -->

### 3.4.3 Two Pair Poker Hands — 先选结构，再选位置
<!-- bilingual-en:start -->
*3.4.3 Two Pair Poker Hands—structure first, then location*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_TwoPairPoker.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HswnmlLPGZ4.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HswnmlLPGZ4)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_TwoPairPoker.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HswnmlLPGZ4.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HswnmlLPGZ4)
<!-- bilingual-en:end -->

标准 52 张牌有 13 个点数、每点数 4 个花色。计数时把“点数角色”与“花色位置”分开：
<!-- bilingual-en:start -->
The standard 52 cards have 13 points and 4 colors per point.  Separate Point Role from Fancy Position when counting:
<!-- bilingual-en:end -->

- 四条：选四条点数 $13$；选边牌点数 $12$；选边牌花色 $4$：
  $$13\cdot12\cdot4=624.$$
- 葫芦：选三条点数与 3 个花色，再选对子点数与 2 个花色：
  $$13\binom43\cdot12\binom42.$$
- 两对：先选两个对子点数（无序），为各对选 2 花色，再选第五张点数与花色：
  $$\boxed{\binom{13}{2}\binom42^2\cdot11\cdot4}.$$
<!-- bilingual-en:start -->
- Four: select four points $13$; select side card points $12$; select side card suits $4$:
  $$13\cdot12\cdot4=624.$$
- Gourd: Choose 3 points and 3 suits, then choose a pair of sub points and 2 suits:
  $$13\binom43\cdot12\binom42.$$
- Two pairs: first select two pairs of sub-points (out of order), select 2 suits for each pair, and then select the fifth pair of points and suits:
  $$\boxed{\binom{13}{2}\binom42^2\cdot11\cdot4}.$$
<!-- bilingual-en:end -->

若先把两个对子点数按“第一对/第二对”有序选择，会把每手牌数两次，必须除以 $2!$。
<!-- bilingual-en:start -->
If the two pairs are selected in an ordered first-pair/second-pair procedure, each hand is counted twice, so divide by $2!$ once.
<!-- bilingual-en:end -->

### 3.4.4 Binomial Theorem — 系数就是选择位置
<!-- bilingual-en:start -->
*3.4.4 Binomial Theorem — Coefficient is the location of the selection*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BinomialTheo.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/jwjDj4GoSV0.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=jwjDj4GoSV0)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BinomialTheo.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/jwjDj4GoSV0.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=jwjDj4GoSV0)
<!-- bilingual-en:end -->

[[组合计数原理#二项式、鸽巢与容斥|二项式定理]]：对非负整数 $n$，
<!-- bilingual-en:start -->
[[组合计数原理#二项式、鸽巢与容斥|binomial theorem]]: non-negative integer $n$,
<!-- bilingual-en:end -->

$$
\boxed{(a+b)^n=\sum_{k=0}^{n}\binom nk a^{n-k}b^k}.
$$

**完整组合证明**：把左边写成 $n$ 个因子的积。展开中的每一项都从每个因子选一次 $a$ 或 $b$。要得到 $a^{n-k}b^k$，必须在恰好 $k$ 个因子位置选 $b$；位置集合有 $\binom nk$ 种。每种选择贡献同一个单项式，故其系数为 $\binom nk$。所有 $k=0,\ldots,n$ 情形互斥且穷尽，定理成立。
<!-- bilingual-en:start -->
**Complete combination proof**: Write the left side as a product of $n$ factors.  Each item in the expansion selects $a$ or $b$ from each factor once.  To get $a^{n-k}b^k$, you must select $b$ in exactly $k$ factor positions; the set of positions has $\binom nk$ types.  Each choice contributes to the same monomial, so its coefficient is $\binom nk$.  All $k=0,\ldots,n$ cases are mutually exclusive and exhaustive, and the theorem holds.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-binomial-paths.png|900]]

读图：从格点起点到终点的每条最短路径对应一个由“横步/竖步”组成的二进制串；固定竖步数就是选择这些步出现的位置，因此路径数与二项式系数是同一个计数对象。
<!-- bilingual-en:start -->
Read: Each shortest path from the start point to the end point of the grid corresponds to a binary string consisting of "horizontal/vertical steps"; the fixed number of vertical steps is to select the location where these steps occur, so the number of paths and the binomial coefficient are the same counting object.
<!-- bilingual-en:end -->

#### Pascal 恒等式
<!-- bilingual-en:start -->
*Pascal identities*
<!-- bilingual-en:end -->

从 $n$ 元集合选 $k$ 个，固定元素 $x$：不选 $x$ 有 $\binom{n-1}{k}$ 种，选 $x$ 后还需从其余选 $k-1$ 个，有 $\binom{n-1}{k-1}$ 种。两类互斥且穷尽，所以
<!-- bilingual-en:start -->
Choose $k$ elements from an $n$-element set and fix one element $x$. There are $\binom{n-1}{k}$ choices that omit $x$, and $\binom{n-1}{k-1}$ choices that include $x$ and choose the remaining $k-1$ elements. These cases are disjoint and exhaustive, so
<!-- bilingual-en:end -->

$$
\boxed{\binom nk=\binom{n-1}{k}+\binom{n-1}{k-1}}.
$$

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-stars-and-bars.png|900]]

读图：星代表被分配的相同对象，隔板把它们切成各变量的数量；选择隔板位置等价于选择一个组合，因此非负整数解数是 $\binom{n+k-1}{k-1}$。
<!-- bilingual-en:start -->
Read: stars represent the same objects that are assigned, and the clapboard cuts them into the number of variables; selecting the position of the clapboard is equivalent to selecting a combination, so the non-negative integer solution is $\binom{n+k-1}{k-1}$.
<!-- bilingual-en:end -->

### 3.4.5 Multinomial Theorem — 多种选择的统一形式
<!-- bilingual-en:start -->
*3.4.5 Multinomial Theorem—Unified Form of Multiple Choices*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bookkeeper.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/juGgfHsO-xM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=juGgfHsO-xM)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bookkeeper.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/juGgfHsO-xM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=juGgfHsO-xM)
<!-- bilingual-en:end -->

[[组合计数原理#排列、组合与重复|多项式定理（multinomial theorem）]]从多类位置分配出发。若 $k_1+\cdots+k_m=n$，把 $n$ 个不同位置分成大小分别为 $k_1,\ldots,k_m$ 的有标号组，方式数为多项式系数
<!-- bilingual-en:start -->
[[组合计数原理#排列、组合与重复|multinomial theorem]] starts with multi-class location assignment.  If $k_1+\cdots+k_m=n$, $n$ different locations are divided into labeled groups of $k_1,\ldots,k_m$ in size, and the mode number is polynomial coefficient
<!-- bilingual-en:end -->

$$
\boxed{\binom{n}{k_1,\ldots,k_m}
=\frac{n!}{k_1!\cdots k_m!}}.
$$

证明：先把所有位置线性排列有 $n!$ 种；同一分组中，第 $i$ 组内部的 $k_i!$ 个排列不改变分组，整体是 $k_1!\cdots k_m!$-to-1。
<!-- bilingual-en:start -->
There are $n!$ linear orderings of all positions. Within a fixed grouping, the $k_i!$ permutations inside group $i$ do not change the grouping, so the map from orderings to groupings is $k_1!\cdots k_m!$-to-1.
<!-- bilingual-en:end -->

多项式定理为
<!-- bilingual-en:start -->
Polynomial theorem is
<!-- bilingual-en:end -->

$$
\boxed{(x_1+\cdots+x_m)^n
=\sum_{k_1+\cdots+k_m=n}
\binom{n}{k_1,\ldots,k_m}
x_1^{k_1}\cdots x_m^{k_m}}.
$$

证明与二项式相同：要得到给定指数向量，需把 $n$ 个因子位置分配给 $m$ 种变量，分配数正是多项式系数。
<!-- bilingual-en:start -->
The proof is the same as for the binomial theorem: to obtain a given exponent vector, assign the $n$ factor positions to the $m$ variables. The number of such assignments is exactly the multinomial coefficient.
<!-- bilingual-en:end -->

#### Bookkeeper Rule：重复字母排列
<!-- bilingual-en:start -->
*Bookkeeper rule: permutations with repeated symbols*
<!-- bilingual-en:end -->

长度 $n$ 的词中，第 $i$ 类相同符号出现 $k_i$ 次，则不同排列数为
<!-- bilingual-en:start -->
If a word of length $n$ contains $k_i$ copies of the $i$th distinct symbol, then the number of distinct permutations is
<!-- bilingual-en:end -->

$$
\boxed{\frac{n!}{k_1!\cdots k_m!}}.
$$

给相同字母临时加下标后有 $n!$ 个全异排列；擦去下标，每个无下标词恰有 $k_1!\cdots k_m!$ 个原像。
<!-- bilingual-en:start -->
Temporarily distinguish repeated letters with subscripts, giving $n!$ permutations of distinct symbols. After erasing the subscripts, each unsubscripted word has exactly $k_1!\cdots k_m!$ preimages.
<!-- bilingual-en:end -->

### Session 26 易错点与反例
<!-- bilingual-en:start -->
*Session 26 Error-prone Points and Counterexamples*
<!-- bilingual-en:end -->

1. Division Rule 要求每个目标对象的原像数完全相同；“大多数有 $k$ 个”不够。
2. 圆排列只除旋转，不自动除反射；若镜像也视为相同，还需另作对称性分析。
3. 选相同对象用 Stars and Bars；选不同对象的子集用 $\binom nk$，不要混淆。
4. 两对扑克的两个对子无标签，若有序选点数就会重复两次。
5. 多项式系数的下标必须和为总次数，否则该单项式系数为 $0$。
<!-- bilingual-en:start -->
1. Division Rule requires the original number of pixels to be identical for each target object; "most have $k$" is not enough.
2. Circular alignments do not automatically eliminate reflections except for rotation; if the mirrors are also considered identical, a symmetry analysis is also required.
3. Choose Stars and Bars for the same object and $\binom nk$ for a subset of different objects, so there is no confusion.
4. Two pairs of poker are unlabeled and repeat twice if you select points in an orderly fashion.
5. The subscript of the polynomial coefficient must be summed up to the total number of times, otherwise the monomial coefficient is $0$.
<!-- bilingual-en:end -->

### Session 26 自检
<!-- bilingual-en:start -->
*Session 26 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 26-1
> `MISSISSIPPI` 有多少个不同排列？
> <!-- bilingual-en:start -->
> How many different permutations are there for `MISSISSIPPI`?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 共 11 个字母，$I$ 有 4 个、$S$ 有 4 个、$P$ 有 2 个、$M$ 有 1 个：
> $$\frac{11!}{4!4!2!}.$$
> <!-- bilingual-en:start -->
> There are 4 for $I$, 4 for $S$, 2 for $P$, 1 for $M$, 1 for 11 letters:
> $$\frac{11!}{4!4!2!}.$$
> <!-- bilingual-en:end -->

> [!question] 自检 26-2
> 求 $(2x-y)^5$ 中 $x^3y^2$ 的系数。
> <!-- bilingual-en:start -->
> The coefficients of $x^3y^2$ in $(2x-y)^5$ are calculated.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 选 2 个因子提供 $-y$，其余 3 个提供 $2x$：
> $$\binom52(2)^3(-1)^2=80.$$
> <!-- bilingual-en:start -->
> Select 2 factors to provide $-y$ and the remaining 3 to provide $2x$:
> $$\binom52(2)^3(-1)^2=80.$$
> <!-- bilingual-en:end -->

> [!question] 自检 26-3
> 10 个不同学生围圆桌就座，旋转相同、反射不同，有多少种？
> <!-- bilingual-en:start -->
> 10 different students sit around a round table, rotating the same, reflecting different, how many?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 每个圆排列对应 10 个线性旋转，故 $10!/10=9!$。
> <!-- bilingual-en:start -->
> Each circle arrangement corresponds to 10 linear rotations, so $10!/10=9!$.
> <!-- bilingual-en:end -->

### Classroom Problems 26 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 26 — 5 complete independent questions*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp26.pdf#page=1|cp26 pp. 1–4]]
<!-- bilingual-en:start -->
Original Question: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp26.pdf#page=1|cp26 pp. 1–4]]
<!-- bilingual-en:end -->

> [!example]- C26-1 把 12 人分成 4 个三人组
> **(a)** 线性名单按连续 3 人切成有序四组。每组内部 $3!$ 种排列不改变组序，所以映射为
> $$
> \boxed{(3!)^4\text{-to-1}}.
> $$
> **(b)** 忘掉四组的顺序，每个无序分组有 $4!$ 个有序组序列原像，故 $\boxed{4!\text{-to-1}}$。
> **(c)** 从 $12!$ 个名单连续应用两次 Division Rule：
> $$
> \boxed{\frac{12!}{(3!)^4,4!}}.
> $$
> **(d)** 一般地，$3n$ 人分成 $n$ 个无标号三人组：
> $$
> \boxed{\frac{(3n)!}{(3!)^n n!}}.
> $$
> <!-- bilingual-en:start -->
> **(a)** Split a linear list into four consecutive groups of three. Permuting the three people inside each group does not change the ordered grouping, so the map is $\boxed{(3!)^4\text{-to-1}}$.
> **(b)** Forgetting the order of the four groups gives $4!$ ordered preimages for every unordered grouping, so the map is $\boxed{4!\text{-to-1}}$.
> **(c)** Apply the division rule twice to the $12!$ linear lists: $\boxed{12!/((3!)^4 4!)}$.
> **(d)** In general, the number of ways to partition $3n$ people into $n$ unlabeled triples is $\boxed{(3n)!/((3!)^n n!)}$.
> <!-- bilingual-en:end -->

> [!example]- C26-2 间隔、整数解与弱递增序列
> **(a)** 设所选位置 $i_1<\cdots<i_8$ 且 $i_{r+1}-i_r\ge3$。令 $j_r=i_r-2(r-1)$，得到从 $1,\ldots,16$ 任选 8 个的双射，故 $\boxed{\binom{16}{8}}$。
> **(b)** $x_1+\cdots+x_m=k$ 的非负整数解：$k$ 星、$m-1$ 板，故
> $$\boxed{\binom{k+m-1}{m-1}}.$$
> **(c)** 加松弛变量 $x_{m+1}=k-\sum_{i=1}^mx_i$，变成 $m+1$ 个变量和为 $k$：
> $$\boxed{\binom{k+m}{m}}.$$
> **(d)** 弱递增序列 $0\le y_1\le\cdots\le y_m\le k$ 等价于从 $k+1$ 个值中有重复地选 $m$ 个，故
> $$\boxed{\binom{k+m}{m}}.$$
> <!-- bilingual-en:start -->
> **(a)** Let the selected positions be $i_1<\cdots<i_8$ with $i_{r+1}-i_r\ge3$. Setting $j_r=i_r-2(r-1)$ gives a bijection with choosing any 8 positions from $1,\ldots,16$, hence $\boxed{\binom{16}{8}}$.
> **(b)** Nonnegative integer solutions of $x_1+\cdots+x_m=k$ correspond to arrangements of $k$ stars and $m-1$ bars, so
> $$\boxed{\binom{k+m-1}{m-1}}.$$
> **(c)** Add the slack variable $x_{m+1}=k-\sum_{i=1}^mx_i$; the problem becomes one of $m+1$ nonnegative variables summing to $k$:
> $$\boxed{\binom{k+m}{m}}.$$
> **(d)** A weakly increasing sequence $0\le y_1\le\cdots\le y_m\le k$ is equivalent to choosing $m$ values with repetition from the $k+1$ possibilities.
> $$\boxed{\binom{k+m}{m}}.$$
> <!-- bilingual-en:end -->

> [!example]- C26-3 The Tao of BOOKKEEPER
> **(a)** `POKE` 全异：$4!=24$。
> **(b)** `BO₁O₂K` 全异：$4!=24$。
> **(c)** 擦去 $O$ 下标，例如 `O₂BO₁K` 与 `O₁BO₂K` 都映到 `OBOK`；`KO₂BO₁` 与 `KO₁BO₂` 都映到 `KOBO`；`BO₁O₂K` 与 `BO₂O₁K` 都映到 `BOOK`。
> **(d)** 每个无下标排列恰有 $2!$ 个原像，是 $2$-to-1。
> **(e)** `BOOK`：$4!/2!=12$。
> **(f)** `KE₁E₂PE₃R` 六符号全异：$6!=720$。
> **(g)** 映到 `REPEEK` 的六个原像是
> `RE₁PE₂E₃K`、`RE₁PE₃E₂K`、`RE₂PE₁E₃K`、`RE₂PE₃E₁K`、`RE₃PE₁E₂K`、`RE₃PE₂E₁K`。
> **(h)** 这是 $3!=6$-to-1。
> **(i)** `KEEPER`：$6!/3!=120$。
> **(j)** `BO₁O₂K₁K₂E₁E₂PE₃R` 十符号全异：$10!$。
> **(k)** 两个 O 相同：$10!/2!$。
> **(l)** 两个 O、两个 K 相同：$10!/(2!2!)$。
> **(m)** `BOOKKEEPER` 中 O 两个、K 两个、E 三个：
> $$\boxed{\frac{10!}{2!2!3!}=151200}.$$
> **(n)** `VOODOODOLL` 中 O 五个、D 两个、L 两个：
> $$\boxed{\frac{10!}{5!2!2!}=7560}.$$
> **(o)** 长 52 串恰有 17 个 2、23 个 5、12 个 9：
> $$\boxed{\frac{52!}{17!23!12!}}.$$
> <!-- bilingual-en:start -->
> The letters in `POKE` are distinct, so there are $4!=24$ permutations.
> **(b)** The symbols in `BO₁O₂K` are distinct, so there are $4!=24$ permutations.
> **(c)** Erase the subscripts on the two O's. For example, `O₂BO₁K` and `O₁BO₂K` both map to `OBOK`; `KO₂BO₁` and `KO₁BO₂` both map to `KOBO`; and `BO₁O₂K` and `BO₂O₁K` both map to `BOOK`.
> **(d)** Every unsubscripted arrangement has exactly $2!$ preimages, so the map is $2$-to-1.
> **(e)**`BOOK`:$4!/2!=12$.
> **(f)** The six symbols in `KE₁E₂PE₃R` are distinct, so there are $6!=720$ permutations.
> **(g)** The six preimages of `REPEEK` are
> `RE₁PE₂E₃K`,`RE₁PE₃E₂K`,`RE₂PE₁E₃K`,`RE₂PE₃E₁K`,`RE₃PE₁E₂K`,`RE₃PE₂E₁K`.
> **(h)**This is $3!=6$-to-1.
> **(i)**`KEEPER`:$6!/3!=120$.
> **(j)** The ten symbols in `BO₁O₂K₁K₂E₁E₂PE₃R` are distinct, so there are $10!$ permutations.
> **(k)**Two O's are the same: $10!/2!$.
> **(l)**Two O's, two K's the same: $10!/(2!2!)$.
> **(m)**Two O, Two K, Three E in `BOOKKEEPER`:
> $$\boxed{\frac{10!}{2!2!3!}=151200}.$$
> **(n)**Five O, two D, two L in `VOODOODOLL`:
> $$\boxed{\frac{10!}{5!2!2!}=7560}.$$
> **(o)**17 x2, 23 x5, 12 x9 for 52 long strings:
> $$\boxed{\frac{52!}{17!23!12!}}.$$
> <!-- bilingual-en:end -->

> [!example]- C26-4 三个系数
> **(a)** $(1+x)^{11}$ 中 $x^5$ 系数：$\boxed{\binom{11}{5}}$。
> **(b)** $(3x+2y)^{17}$ 中 $x^8y^9$ 系数：
> $$\boxed{\binom{17}{8}3^8 2^9}.$$
> **(c)** $(a^2+b^3)^5$ 中 $a^6b^6$：需选 3 个 $a^2$、2 个 $b^3$，系数 $\boxed{\binom53=10}$。
> <!-- bilingual-en:start -->
> **(a)**$x^5$ coefficient in $(1+x)^{11}$: $\boxed{\binom{11}{5}}$.
> **(b)**$x^8y^9$ coefficient in $(3x+2y)^{17}$:
> $$\boxed{\binom{17}{8}3^8 2^9}.$$
> **(c)**$a^6b^6$ in $(a^2+b^3)^5$: 3 $a^2$ required, 2 $b^3$ required, factor $\boxed{\binom53=10}$.
> <!-- bilingual-en:end -->

> [!example]- C26-5 分配任务与六位数字
> **(a)** 九人分别分到人数为 $1,2,3,1,2$ 的五个有标签任务：
> $$
> \boxed{\binom{9}{1,2,3,1,2}=\frac{9!}{1!2!3!1!2!}}.
> $$
> **(b)** 把小于 $10^6$ 的非负整数写成六位、允许前导零。先选唯一数字 9 的位置，有 6 种。其余五位数字和为 $8$；任何非负解的单个分量自动不超过 $8$，所以无额外数字上界问题。解数为 $\binom{12}{4}$，总计
> $$
> \boxed{6\binom{12}{4}=2970}.
> $$
> <!-- bilingual-en:start -->
> **(a)**Nine people are assigned five tagged tasks each with a $1,2,3,1,2$ population:
> **(b)**Writes a non-negative integer less than $10^6$ as a six-bit, allowing leading zeros.  Choose the location of the only number 9. There are six.  The sum of the remaining five digits is $8$; any single component of the non-negative solution is automatically no more than $8$, so there is no additional numerical upper bound problem.  The number of solutions is $\binom{12}{4}$, total
> <!-- bilingual-en:end -->

### Session 26 知识链小结
<!-- bilingual-en:start -->
*Session 26 Knowledge Chain Summary*
<!-- bilingual-en:end -->

$$
\text{广义乘法}
\longrightarrow k\text{-to-1 除法}
\longrightarrow\binom nk
\longrightarrow\text{二项式系数}
\longrightarrow\text{多项式系数}
\longrightarrow\text{重复排列与分组}.
$$

---

## Session 27 — Pigeonhole Principle & Inclusion–Exclusion

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：不知道对象具体分布时，怎样仅凭数量保证碰撞？多个“坏事件”重叠时，怎样既不漏数也不重复计数？
<!-- bilingual-en:start -->
**Learning Problem**: How to guarantee collision only by quantity when the object's distribution is unknown?  How do I count without missing or repeating when multiple "bad events" overlap?
<!-- bilingual-en:end -->

**前置知识**：函数与单射、组合数、二项式定理、加法法则。首次正式使用 [[组合计数原理#二项式、鸽巢与容斥|鸽巢原理]] 与 [[组合计数原理#二项式、鸽巢与容斥|容斥原理]]。
<!-- bilingual-en:start -->
**Prerequisite knowledge:** functions and injections, binomial coefficients, the binomial theorem, and the addition rule. This session gives the first formal use of the [[组合计数原理#二项式、鸽巢与容斥|pigeonhole principle]] and the [[组合计数原理#二项式、鸽巢与容斥|inclusion–exclusion principle]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session27.pdf#page=1|Session 27 reading, pp. 1–11]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp27.pdf#page=1|cp27, pp. 1–3]]

### 3.5.1 The Pigeonhole Principle — 数量迫使碰撞
<!-- bilingual-en:start -->
*3.5.1 The Pigeonhole Principle — Quantity Forced Collision*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ThePigeonhol.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/4Dz4vNUxnZM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=4Dz4vNUxnZM)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ThePigeonhol.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/4Dz4vNUxnZM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=4Dz4vNUxnZM)
<!-- bilingual-en:end -->

#### 基本形式
<!-- bilingual-en:start -->
*basic form*
<!-- bilingual-en:end -->

若有限集合 $A,B$ 满足 $|A|>|B|$，则任何全函数 $f:A\to B$ 都不是单射。
<!-- bilingual-en:start -->
If finite sets $A$ and $B$ satisfy $|A|>|B|$, then no total function $f:A\to B$ is injective.
<!-- bilingual-en:end -->

**反证**：若 $f$ 单射，则不同 $a$ 有不同像，$f(A)$ 含 $|A|$ 个元素；但 $f(A)\subseteq B$，于是 $|A|=|f(A)|\le|B|$，与 $|A|>|B|$ 矛盾。
<!-- bilingual-en:start -->
**Proof by contradiction:** If $f$ were injective, distinct elements of $A$ would have distinct images, so $f(A)$ would contain $|A|$ elements. But $f(A)\subseteq B$, giving $|A|=|f(A)|\le|B|$, contrary to $|A|>|B|$.
<!-- bilingual-en:end -->

在应用中必须明确三件事：
<!-- bilingual-en:start -->
There are three things to be clear about an app:
<!-- bilingual-en:end -->

1. pigeons 是哪些对象；
2. holes 是哪些类别；
3. 函数怎样把每个对象唯一分配到一个类别。
<!-- bilingual-en:start -->
1. What objects are pigeons;
2. What categories of holes are;
3. How the function assigns each object uniquely to a category.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-pigeonhole-principle.png|900]]

读图：每只鸽子被函数送进一个巢；当鸽子数超过巢数，单射不可能，至少一个巢接收两个；广义形式把“两个”替换为平均负载的向上取整。
<!-- bilingual-en:start -->
How to read the diagram: a function sends each pigeon to a hole. If there are more pigeons than holes, the function cannot be injective, so at least one hole receives two pigeons. The general form replaces “two” by the ceiling of the average load.
<!-- bilingual-en:end -->

#### 广义形式
<!-- bilingual-en:start -->
*generalized form*
<!-- bilingual-en:end -->

把 $N$ 个对象放入 $m$ 个类别，至少有一个类别含
<!-- bilingual-en:start -->
Put $N$ objects into $m$ categories, at least one of which contains
<!-- bilingual-en:end -->

$$
\boxed{\left\lceil\frac Nm\right\rceil}
$$

个对象。等价地，若 $N>km$，某类至少有 $k+1$ 个。
<!-- bilingual-en:start -->
objects.  Equivalently, if $N>km$, there are at least $k+1$ of a certain class.
<!-- bilingual-en:end -->

**证明**：反设每类至多 $k$ 个，则总对象数至多 $km$，与 $N>km$ 矛盾。取 $k=\lceil N/m\rceil-1$ 即得向上取整形式。
<!-- bilingual-en:start -->
**Proof**: If there are no more than $k$ objects in each class, the total number of objects is no more than $km$, which is in contradiction with $N>km$.  $k=\lceil N/m\rceil-1$ is rounded up.
<!-- bilingual-en:end -->

> [!tip] 反向设计阈值
> 要保证某巢至少有 $r$ 个，最坏情况可以让每巢先放 $r-1$ 个，所以最小充分总数是
> $$
> (r-1)m+1.
> $$
> <!-- bilingual-en:start -->
> To ensure that a nest has at least $r$, the worst-case scenario would be to put $r-1$ in each nest first, so the minimum sufficient total is
> <!-- bilingual-en:end -->

### 3.5.2 Rolling Dice — 官方在线题 O27-01
<!-- bilingual-en:start -->
*3.5.2 Rolling Dice — official online O27-01*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.2_rolling-dice|3.5.2]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.2_rolling-dice|3.5.2]]
<!-- bilingual-en:end -->

掷两枚骰子 25 次，记录和。可能的和是整数 $2,3,\ldots,12$，共
<!-- bilingual-en:start -->
25 dice tossed, and...  Possible sums are integers $2,3,\ldots,12$ of
<!-- bilingual-en:end -->

$$
12-2+1=11
$$

个巢，故至少一个和出现 $\lceil25/11\rceil=3$ 次。
<!-- bilingual-en:start -->
Nest, so at least one and appear $\lceil25/11\rceil=3$ times.
<!-- bilingual-en:end -->

> [!success]- 官方答案与勘误
> **O27-01**：$\boxed{3}$。官方反馈写作 $\lceil25/(12-2)\rceil=3$，分母少了端点计数的 $+1$；正确巢数是 11。这个笔误不改变最终答案，因为 $\lceil25/10\rceil$ 与 $\lceil25/11\rceil$ 都是 3。
> <!-- bilingual-en:start -->
> **O27-01**:$\boxed{3}$.  Official feedback is written as $\lceil25/(12-2)\rceil=3$, with the denominator minus the $+1$ of the endpoint count; the correct number of nests is 11.  This error does not change the final answer, because $\lceil25/10\rceil$ and $\lceil25/11\rceil$ are both 3.
> <!-- bilingual-en:end -->

### 3.5.3 Inclusion–Exclusion Example — 从 6042 模式看三集合
<!-- bilingual-en:start -->
*3.5.3 Inclusion-Exclusion Example—View three collections from 6042 mode*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_InculionExcl.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/51-b2mgZVNY.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=51-b2mgZVNY)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_InculionExcl.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/51-b2mgZVNY.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=51-b2mgZVNY)
<!-- bilingual-en:end -->

考虑数字 $0,1,\ldots,9$ 的排列，求至少含连续模式 `60`、`04`、`42` 之一的排列数。令 $P_{60},P_{04},P_{42}$ 分别表示含对应模式的集合。
<!-- bilingual-en:start -->
Consider the arrangement of the number $0,1,\ldots,9$, find the number of the arrangement containing at least one of the continuous modes `60`, `04`, `42`.  Let $P_{60},P_{04},P_{42}$ represent the set with corresponding patterns respectively.
<!-- bilingual-en:end -->

- 单个模式视作一个块，与其余 8 个数字共 9 个对象，所以每个 $|P_x|=9!$。
- 任意两个模式同时出现时都可压成 8 个对象：不重叠的 `60` 与 `42` 是两个块；相接的 `60` 与 `04` 压成 `604`。故每个二重交集为 $8!$。
- 三者同时出现必须含 `6042`，把它视为一个块后共 7 个对象，所以三重交集为 $7!$。
<!-- bilingual-en:start -->
- A single pattern is treated as a block with 9 objects for the remaining 8 numbers, so each $|P_x|=9!$.
- When any two patterns occur together, they can be compressed into 8 objects: the disjoint patterns `60` and `42` form two blocks, while the adjacent patterns `60` and `04` combine into the single block `604`. Therefore, each pairwise intersection has size $8!$.
- All three must contain `6042`, which is treated as a block and has a total of seven objects, so the triple intersection is $7!$.
<!-- bilingual-en:end -->

容斥给出
<!-- bilingual-en:start -->
inclusion-exclusion
<!-- bilingual-en:end -->

$$
\boxed{|P_{60}\cup P_{04}\cup P_{42}|
=3\cdot9!-3\cdot8!+7!}.
$$

先加单集合会把二重重叠数两次，所以减；但三重重叠先被加 3 次又被减 3 次，变成 0 次，必须再加一次。
<!-- bilingual-en:start -->
Adding the three single-set counts includes every pairwise overlap twice, so those overlaps must be subtracted. A triple-overlap element is then added three times and subtracted three times, leaving a count of zero, so it must be added back once.
<!-- bilingual-en:end -->

### 3.5.4 Inclusion–Exclusion: Two Sets and General Form

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_InclExclEx.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/nwpzBE9IwJQ.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=nwpzBE9IwJQ)
<!-- bilingual-en:start -->
Resources: [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_InclExclEx.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/nwpzBE9IwJQ.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=nwpzBE9IwJQ)
<!-- bilingual-en:end -->

#### 两集合公式的严格证明
<!-- bilingual-en:start -->
*Strict Proof of Two-Set Formula*
<!-- bilingual-en:end -->

集合 $A\cup B$ 可写成不交并
<!-- bilingual-en:start -->
The set $A\cup B$ can be written as a disjoint
<!-- bilingual-en:end -->

$$
A\mathbin{\dot\cup}(B\setminus A),
$$

所以 $|A\cup B|=|A|+|B\setminus A|$。另一方面，$B$ 是不交并
<!-- bilingual-en:start -->
So, $|A\cup B|=|A|+|B\setminus A|$.  On the other hand, $B$ is disjoint
<!-- bilingual-en:end -->

$$
(A\cap B)\mathbin{\dot\cup}(B\setminus A),
$$

所以 $|B\setminus A|=|B|-|A\cap B|$。代入即得
<!-- bilingual-en:start -->
So, $|B\setminus A|=|B|-|A\cap B|$.  be easily received
<!-- bilingual-en:end -->

$$
\boxed{|A\cup B|=|A|+|B|-|A\cap B|}.
$$

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-inclusion-exclusion.png|900]]

读图：直接相加 $|A|+|B|$ 会把交集数两次；减去一次交集后，每个并集元素恰被数一次。三集合及一般形式继续按“奇数层加、偶数层减”修正。
<!-- bilingual-en:start -->
Read: Add $|A|+|B|$ directly to count the number of intersections twice; subtract one intersection to count each union element exactly once.  The three sets and the general form are further modified by adding odd layers and subtracting even layers.
<!-- bilingual-en:end -->

#### 一般容斥公式及完整计数证明
<!-- bilingual-en:start -->
*General Inclusion Formula and Complete Count Proof*
<!-- bilingual-en:end -->

对有限集合 $A_1,\ldots,A_n$，
<!-- bilingual-en:start -->
For a finite set $A_1,\ldots,A_n$,
<!-- bilingual-en:end -->

$$
\boxed{
\left|\bigcup_{i=1}^{n}A_i\right|
=\sum_{\varnothing\ne S\subseteq[n]}
(-1)^{|S|+1}
\left|\bigcap_{i\in S}A_i\right|}.
$$

固定任一元素 $x$。若 $x$ 不在任何集合中，它对两边都贡献 0。若 $x$ 恰在 $r\ge1$ 个集合中，那么右侧所有包含 $x$ 的 $j$ 重交集共有 $\binom rj$ 个，它的总计数权重为
<!-- bilingual-en:start -->
Pin either element $x$.  If $x$ is not in any set, it contributes 0 to both sides.  If the $x$ is exactly in the $r\ge1$ set, then the right-hand side of all the $j$ containing the $x$ has a total of $\binom rj$, its total count weight is
<!-- bilingual-en:end -->

$$
\sum_{j=1}^{r}(-1)^{j+1}\binom rj
=1-\sum_{j=0}^{r}(-1)^j\binom rj
=1-(1-1)^r=1.
$$

所以每个并集元素恰被计一次，外部元素不计；两边逐元素相同，公式得证。
<!-- bilingual-en:start -->
So every union element is counted once, the exterior element is not counted; the two sides are the same element by element, the formula is proved.
<!-- bilingual-en:end -->

### 3.5.5 Pigeonhole Principle — 官方在线题 O27-02 至 O27-06
<!-- bilingual-en:start -->
*3.5.5 Pigeonhole Principle — official online questions O27-02 through O27-06*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.5_pigeonhole-principle|3.5.5]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.5_pigeonhole-principle|3.5.5]]
<!-- bilingual-en:end -->

| 编号 | 要保证的性质 | 官方答案 | 最坏分布 |
|---|---|---:|---|
| O27-02 | 至少 2 人同月同日生日 | $366$ | 365 天先各 1 人 |
| O27-03 | 至少 2 人都生于 1 月 1 日 | nh | 任意多人都可生于别日 |
| O27-04 | 至少 3 人同一星期几出生 | $15$ | 7 类先各放 2 人 |
| O27-05 | 至少 4 人同月出生 | $37$ | 12 月先各放 3 人 |
| O27-06 | 至少 2 人生日恰相隔一周 | nh | 所有人可同一天出生 |
<!-- bilingual-en:start -->
|Encoding|The nature to be guaranteed|Official Answers|Worst-case Distribution|
|—|—|—:|—|
| O27-02 | At least 2 people have birthdays on the same day of the same month | $366$ | 1 person each for 365 days |
| O27-03 | At least 2 people born on January 1 | nh | Any number of people can be born on another day |
| O27-04 | At least 3 people born in the same week | $15$ | 7 categories, 2 people each |
| O27-05 | At least 4 people born in the same month | $37$ | 3 people each in December |
| O27-06 | At least 2 births occur exactly one week apart | nh | All births occur on the same day |
<!-- bilingual-en:end -->

“nh”意为无论群体多大都不必成立。鸽巢原理能强制**同一类别碰撞**，不能强制某个指定类别非空，也不能强制两个类别之间具有固定距离。
<!-- bilingual-en:start -->
“nh” means that the claim need not become true merely because the set is large. The pigeonhole principle can force **two objects into the same class**, but it cannot force a specified class to be nonempty or force two classes to occur at a prescribed distance.
<!-- bilingual-en:end -->

### 3.5.6 6.042 TEAL Table — 官方在线题 O27-07 至 O27-10
<!-- bilingual-en:start -->
*3.5.6 6.042 TEAL Table —official online questions O27-07 through O27-10*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.6_6-042-teal-table|3.5.6]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.6_6-042-teal-table|3.5.6]]
<!-- bilingual-en:end -->

8 名学生围圆桌，旋转相同、反射不同：
<!-- bilingual-en:start -->
8 students round table, same rotation, different reflexes:
<!-- bilingual-en:end -->

| 编号 | 限制 | 官方答案 | 方法 |
|---|---|---:|---|
| O27-07 | 无限制 | $7!=5040$ | 固定一人消除旋转 |
| O27-08 | Alyssa 邻 Ben | $2\cdot6!=1440$ | 把二人视作块，内部 2 序 |
| O27-09 | Ben 同时邻 Alyssa、Carlos | $2\cdot5!=240$ | 三人块且 Ben 居中 |
| O27-10 | Ben 邻 Alyssa 或 Carlos | $4\cdot6!-2\cdot5!=2640$ | 两个事件容斥 |
<!-- bilingual-en:start -->
|Encoding|Restrictions|Official Answers|Methods|
|—|—|—:|—|
| O27-07 | Unlimited | $7!=5040$ | Pin One Eliminate Rotation|
| O27-08 | Ben | $2\cdot6!=1440$, Alyssa | Treat them as blocks, internal 2nd order |
| O27-09 | Ben Concurrent with Alyssa, Carlos | $2\cdot5!=240$ | Triple and Ben Centered |
| O27-10 | Alyssa or Carlos for Ben | $4\cdot6!-2\cdot5!=2640$ | Two events are mutually exclusive |
<!-- bilingual-en:end -->

### 3.5.7 Class Schedules — 官方在线题 O27-11
<!-- bilingual-en:start -->
*3.5.7 Class Schedules — official online O27-11*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.7_class-schedules|3.5.7]]
<!-- bilingual-en:start -->
Origin and official feedback: [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.7_class-schedules|3.5.7]]
<!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 11 门课中恰选 4 门，共 $\binom{11}{4}=330$ 种课表（巢）。要保证两名学生同课表，需要 $330+1=\boxed{331}$ 名学生。
> <!-- bilingual-en:start -->
> Out of 11 courses, exactly 4 of them were chosen, with $\binom{11}{4}=330$ class schedules (nests).  $330+1=\boxed{331}$ students are required to ensure two students have the same timetable.
> <!-- bilingual-en:end -->

### Session 27 易错点与反例
<!-- bilingual-en:start -->
*Session 27 Error-prone Points and Counterexamples*
<!-- bilingual-en:end -->

1. 闭区间整数个数是“末 − 首 + 1”；骰子和 $2$ 到 $12$ 共 11 种。
2. 鸽巢原理证明存在性，不告诉你碰撞具体发生在哪一巢。
3. “保证某指定日期有两人”不是碰撞问题；任意多人都能避开该日期。
4. 容斥的二重交集要减，三重交集要加；符号为 $(-1)^{|S|+1}$。
5. 把字符串模式视作块时，必须检查模式能否重叠、重叠方向是否唯一。
<!-- bilingual-en:start -->
1. The integer of the closed interval is "end-first + 1"; there are 11 kinds of dice and $2$ to $12$.
2. The pigeonhole principle proves existence; it does not identify exactly where the collision occurs.
3. "Guaranteeing that there are two persons on a given date" is not a collision problem; it can be avoided by any number of persons.
4. Inclusive double intersection to subtract, triple intersection to add; the symbol is $(-1)^{|S|+1}$.
5. When treating a string pattern as a block, you must check whether the patterns overlap and whether the direction of the overlap is unique.
<!-- bilingual-en:end -->

### Session 27 自检
<!-- bilingual-en:start -->
*Session 27 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 27-1
> 至少多少个整数可保证其中两个模 10 同余？
> <!-- bilingual-en:start -->
> At least how many integers can guarantee that two of the modules 10 are congruent?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 模 10 有 10 个余数类，需 $10+1=11$ 个整数。
> <!-- bilingual-en:start -->
> Module 10 has 10 remainder classes and requires $10+1=11$ integers.
> <!-- bilingual-en:end -->

> [!question] 自检 27-2
> 100 个对象放入 9 类，至少一类有多少个？
> <!-- bilingual-en:start -->
> 100 objects in 9 categories, how many in at least one category?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $\lceil100/9\rceil=12$。若每类至多 11 个，总数至多 99，矛盾。
> <!-- bilingual-en:start -->
> $\lceil100/9\rceil=12$.  If there are at most 11 in each category, the total is at most 99.
> <!-- bilingual-en:end -->

> [!question] 自检 27-3
> 1 到 100 中有多少整数能被 2 或 5 整除？
> <!-- bilingual-en:start -->
> How many integers from 1 to 100 are divisible by 2 or 5?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $\lfloor100/2\rfloor+\lfloor100/5\rfloor-\lfloor100/10\rfloor=50+20-10=60$。
> <!-- bilingual-en:start -->
> $\lfloor100/2\rfloor+\lfloor100/5\rfloor-\lfloor100/10\rfloor=50+20-10=60$.
> <!-- bilingual-en:end -->

### Classroom Problems 27 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 27 — 5 complete independent questions*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp27.pdf#page=1|cp27 pp. 1–3]]
<!-- bilingual-en:start -->
Original Question: [[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp27.pdf#page=1|cp27 pp. 1–3]]
<!-- bilingual-en:end -->

> [!example]- C27-1 四个鸽巢构造
> **(a)** 9 位 ID 首位固定为 9，其数字和最小 $9$、最大 $9+8\cdot9=81$，只有 $81-9+1=73$ 个可能和。75 名学生映到 73 个和，必有两人同和。
> **(b)** 把 100 个整数映到模 37 的余数 $0,\ldots,36$。两数同余，差即为 37 的倍数。
> **(c)** 把开单位正方形按两条中线分为四个边长 $1/2$ 的半开小正方形。5 个点中两点落在同一小正方形，距离至多其对角线 $1/\sqrt2$。达到等号必须取该小正方形的两个对角顶点；对四个角块，总有一个是原单位正方形边界点，而题目排除边界，因此实际距离严格小于 $1/\sqrt2$。
> **(d)** 把 $\{1,\ldots,2n\}$ 分成 $n$ 个巢
> $$
> \{1,2\},\{3,4\},\ldots,\{2n-1,2n\}.
> $$
> 选 $n+1$ 个数，两个落在同一对，正好是连续整数。
> <!-- bilingual-en:start -->
> **(a)** A 9-digit ID whose first digit is fixed at 9 has digit sum between $9$ and $9+8\cdot9=81$, so only $81-9+1=73$ sums are possible. Mapping 75 students to these 73 sums forces two students to have the same sum.
> **(b)** Map 100 integers to their residues $0,\ldots,36$ modulo 37. Two integers receive the same residue, so their difference is divisible by 37.
> **(c)** Divide the open unit square along its two midlines into four half-open squares of side $1/2$. Two of the five points lie in the same small square, so their distance is at most its diagonal, $1/\sqrt2$. Equality would require opposite corner points; in each corner block one such point lies on the excluded boundary of the original square. The distance is therefore strictly less than $1/\sqrt2$.
> **(d)** Partition $\{1,\ldots,2n\}$ into the $n$ pairs shown above.
> Choose $n+1$ numbers. Two must lie in the same pair, and those two numbers are consecutive integers.
> <!-- bilingual-en:end -->

> [!example]- C27-2 安全密码与容斥
> cword 是 `a,d,e,f,i,l,o,p,r,s` 的一个排列，总数 $10!$。令 $F,D,R$ 分别表示含子词 `fails`、`failed`、`drop`。
> **(a)** 把 `drop` 视作一块，加其余 6 个字符，共 7 个对象：$\boxed{|R|=7!}$。
> **(b)** `drop` 与 `fails` 字符互不重叠，视作两块，另有字符 `e`，共 3 个对象：$\boxed{|R\cap F|=3!=6}$。
> **(c)**
> $$
> |F|=6!,\qquad |D|=5!,\qquad |R|=7!.
> $$
> `fails` 与 `failed` 需要重复 `fail` 或在同一前缀后同时接 `s/e`，故 $F\cap D=\varnothing$。`drop` 与 `failed` 只能共享唯一的 `d`，并形成块 `failedrop`；再与 `s` 排列，故 $|D\cap R|=2!$。三重交集为空。于是坏 cword 数
> $$
> 6!+5!+7!-3!-2!,
> $$
> 密码数为
> $$
> \boxed{10!-7!-6!-5!+3!+2!=3{,}622{,}928}.
> $$
> <!-- bilingual-en:start -->
> cword is an arrangement of `a,d,e,f,i,l,o,p,r,s`, totaling $10!$.  Let $F,D,R$ denote the subwords `fails`, `failed`, and `drop`, respectively.
> **(a)**Treat `drop` as one block with the remaining 6 characters for a total of 7 objects: $\boxed{|R|=7!}$.
> **(b)**The `drop` and `fails` characters do not overlap and are treated as two blocks, with the character `e`, for a total of three objects: $\boxed{|R\cap F|=3!=6}$.
> **(c)**
> `fails` and `failed` need to repeat `fail` or `s/e` after the same prefix, so $F\cap D=\varnothing$.  `drop` and `failed` can only share a unique `d` and form a block `failedrop`; then they are arranged with `s`, so $|D\cap R|=2!$.  The triple intersection is empty.  So bad cword number
> Number of Passwords is
> <!-- bilingual-en:end -->

> [!example]- C27-3 避开两个障碍的格路径
> 从 $(0,0)$ 到 $(50,50)$ 的单调路径由 50 个横步与 50 个竖步组成，总数 $\binom{100}{50}$。令 $A=(10,11),B=(21,20)$。经过 $A$ 的路径数
> $$
> N_A=\binom{21}{10}\binom{79}{40};
> $$
> 经过 $B$ 的路径数
> $$
> N_B=\binom{41}{21}\binom{59}{29}.
> $$
> 同时经过二者只能按 $A\to B$，数量
> $$
> N_{A\cap B}=\binom{21}{10}\binom{20}{11}\binom{59}{29}.
> $$
> 因而避开两块巨石的路径数为
> $$
> \boxed{
> \binom{100}{50}
> -\binom{21}{10}\binom{79}{40}
> -\binom{41}{21}\binom{59}{29}
> +\binom{21}{10}\binom{20}{11}\binom{59}{29}}.
> $$
> <!-- bilingual-en:start -->
> The monotonic path from $(0,0)$ to $(50,50)$ consists of 50 horizontal steps and 50 vertical steps, totaling $\binom{100}{50}$.  Get $A=(10,11),B=(21,20)$.  Number of paths through $A$
> Number of paths through $B$
> After both, we can only press $A\to B$, quantity
> So the number of paths to avoid two boulders is
> <!-- bilingual-en:end -->

> [!example]- C27-4 只含 7 与尾随 0 的倍数
> 令 $R_j$ 是由 $j$ 个 7 组成的整数。固定正整数 $m$。考察 $R_1,\ldots,R_m$ 的模 $m$ 余数。若某个 $R_j\equiv0\pmod m$，则 $10R_j$ 是 $m$ 的倍数，且由若干 7 后接一个 0。否则这 $m$ 个数都落在仅 $m-1$ 个非零余数类中，故存在 $i<j$ 使 $R_i\equiv R_j\pmod m$。于是
> $$
> m\mid(R_j-R_i)=10^iR_{j-i},
> $$
> 右侧十进制表示正是若干 7 后接 $i\ge1$ 个 0。这证明 (a)。
> 若 $m$ 不被 2 或 5 整除，则 $\gcd(m,10)=1$，$10^i$ 模 $m$ 可逆。由 $m\mid10^iR_{j-i}$ 可消去 $10^i$，得到 $m\mid R_{j-i}$；故存在一个全由 7 构成的 $m$ 倍数，完成 (b)。
> <!-- bilingual-en:start -->
> Let $R_j$ be the integer consisting of $j$ copies of the digit $7$, and fix a positive integer $m$. Consider the residues of $R_1,\ldots,R_m$ modulo $m$. If some $R_j\equiv0\pmod m$, then $10R_j$ is a multiple of $m$ consisting of several $7$s followed by a zero. Otherwise all $m$ numbers occupy only the $m-1$ nonzero residue classes, so $R_i\equiv R_j\pmod m$ for some $i<j$. Hence $m\mid(R_j-R_i)=10^iR_{j-i}$, whose decimal expansion consists of $7$s followed by $i\ge1$ zeros. This proves (a).
> If $m$ is divisible by neither $2$ nor $5$, then $\gcd(m,10)=1$ and $10^i$ is invertible modulo $m$. Cancelling it from $m\mid10^iR_{j-i}$ gives $m\mid R_{j-i}$, so a multiple of $m$ consisting entirely of $7$s exists. This proves (b).
> <!-- bilingual-en:end -->

> [!example]- C27-5 201 个小于 300 的正整数
> 每个正整数唯一写成
> $$
> 3^a q,\qquad 3\nmid q.
> $$
> 把 $q$ 当作巢。小于 300 的可能 $q$ 是 $1,\ldots,299$ 中不被 3 整除者，共
> $$
> 299-\left\lfloor\frac{299}{3}\right\rfloor=299-99=200.
> $$
> 从任意 201 个数中，两个有同一 $q$，写成 $3^aq$ 与 $3^bq$。较大者除以较小者为 $3^{|a-b|}$，整除且商是 3 的幂。
> <!-- bilingual-en:start -->
> Every positive integer has a unique representation $3^a q$ with $3\nmid q$. Use $q$ as the pigeonhole. Among $1,\ldots,299$, exactly $299-\lfloor299/3\rfloor=200$ possible values of $q$ are not divisible by $3$.
> Among any 201 chosen numbers, two have the same value of $q$ and can be written as $3^aq$ and $3^bq$. Dividing the larger by the smaller gives $3^{|a-b|}$, a power of $3$.
> <!-- bilingual-en:end -->

### Session 27 知识链小结
<!-- bilingual-en:start -->
*Session 27 Knowledge Chain Summary*
<!-- bilingual-en:end -->

$$
\text{对象}\xrightarrow{\text{分类函数}}\text{有限个巢}
\longrightarrow\text{必然碰撞}
\quad\text{与}\quad
\text{重叠事件}
\xrightarrow{\text{交集层级}}\text{容斥修正}.
$$

---

## Problem Set 10 — after Session 27

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps10.pdf#page=1|Problem Set 10, pp. 1–3]]。以下为非官方独立题解。
<!-- bilingual-en:start -->
Original title: [[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps10.pdf#page=1|Problem Set 10, pp. 1–3]].  The following are unofficial independent questions.
<!-- bilingual-en:end -->

> [!example]- PS10-1 七枚彩色骰子
> 七枚骰子颜色不同，所以一次结果是按彩虹顺序排列的 7 元组。
> **(a) 恰有两枚为 6，其余五枚数值两两不同**：由于剩余面值只能来自 $1,\ldots,5$ 且需五个不同，恰好各用一次。双射数据为“显示 6 的两个颜色子集 + 其余五个颜色到 $1,\ldots,5$ 的一个双射”，故
> $$
> \boxed{\binom72 5!=2520}.
> $$
> **(b) 恰有一个对子，其余五枚数值两两不同**：六种面值中选对子值；选其两个颜色；把其余五个面值排列给剩余颜色：
> $$
> \boxed{6\binom72 5!=15120}.
> $$
> **(c) 重数型为 $(3,2,2)$**：选三条面值 6 种；从余下 5 值无序选两个对子值；选三条的 3 个颜色；在余下 4 色中选 2 个给较小的对子值，其余自动给另一个：
> $$
> \boxed{6\binom52\binom73\binom42=12600}.
> $$
> 每组数据都能唯一恢复骰子结果，反之也唯一抽取，故确为双射。
> <!-- bilingual-en:start -->
> The colors of the seven dice are different, so the result at a time is a seven-tuple in rainbow order.
> **(a) Exactly two of them are 6, and the remaining five have two different values**: Since the remaining par value can only come from $1,\ldots,5$ and needs five different values, it is used exactly once.  The bijection data is "showing two color subsets of 6 + one bijection of the remaining five colors to $1,\ldots,5$", so
> **(b) Exactly one pair of digits, the remaining five digits are two different**: select the pair of digits from the six values; select the two colors; arrange the remaining five digits into the remaining colors:
> **(c) Multiplicity type is $(3,2,2)$**: select three face values of six; select two pairs of sub-values out of order from the remaining five; select three colors; select two of the remaining four colors to give the smaller pair of sub-values, the remaining automatically give the other:
> Each set of data can recover the dice results uniquely, and vice versa, so it is indeed bijective.
> <!-- bilingual-en:end -->

> [!example]- PS10-2 四个计数问题
> **(a)** 先排列 21 个辅音，有 $21!$ 种。它们产生 21 个“紧接某辅音之前”的可用槽（不含末尾槽）；选 5 个槽并排列 5 个元音：
> $$
> \boxed{21!\binom{21}{5}5!}.
> $$
> 每槽至多一个元音保证不相邻，排除末尾槽保证最后不是元音。
> **(b)** 仍先排 21 个辅音。把槽编号为第 $i$ 个辅音之前。要让插入的元音后立即至少有两个辅音，槽只能在 $1,\ldots,20$，且两个所选槽不能相邻；从 20 个位置选 5 个不相邻位置有
> $$
> \binom{20-5+1}{5}=\binom{16}{5}
> $$
> 种。再排列元音，故总数
> $$
> \boxed{21!\binom{16}{5}5!}.
> $$
> **(c)** 先排列 $2n$ 名学生，再把相邻两人配对。组内交换产生 $2^n$ 重复，$n$ 对之间交换产生 $n!$ 重复：
> $$
> \boxed{\frac{(2n)!}{2^n n!}}.
> $$
> **(d)** 一个类型只记录数字 $0,\ldots,9$ 的出现次数 $x_0+\cdots+x_9=n$。非负解数为
> $$
> \boxed{\binom{n+9}{9}}.
> $$
> <!-- bilingual-en:start -->
> **(a)** First arrange the 21 consonants, giving $21!$ orders. They create 21 allowed slots immediately before consonants, excluding the final trailing slot. Choose 5 slots and permute the 5 vowels, giving $21!\binom{21}{5}5!$. Using at most one vowel per slot prevents adjacent vowels, and excluding the trailing slot prevents the word from ending in a vowel.
> **(b)** Again arrange the consonants first. Label by $i$ the slot immediately before consonant $i$. To leave at least two consonants after every inserted vowel, use only slots $1,\ldots,20$ and choose five nonadjacent slots. There are $\binom{16}{5}$ such choices; permuting the vowels gives $21!\binom{16}{5}5!$.
> **(c)** Arrange the $2n$ students, pair adjacent students, then divide by $2^n$ for swaps within pairs and by $n!$ for permutations of the pairs: $(2n)!/(2^n n!)$.
> **(d)** A type records only the counts $x_0,\ldots,x_9$ of the ten digits, with $x_0+\cdots+x_9=n$. The number of nonnegative solutions is $\binom{n+9}{9}$.
> <!-- bilingual-en:end -->

> [!example]- PS10-3 多项式定理推出 Fermat 小定理
> **(a)** 多项式展开为
> $$
> (x_1+\cdots+x_n)^p
> =\sum_{k_1+\cdots+k_n=p}
> \frac{p!}{k_1!\cdots k_n!}
> x_1^{k_1}\cdots x_n^{k_n}.
> $$
> 若指数向量不是某个 $k_i=p$、其余全零的纯项，则至少两个 $k_i$ 为正，所有正的 $k_i<p$。素数 $p$ 在分子 $p!$ 中出现一次，而每个 $k_i!$ 都不含因子 $p$，故多项式系数被 $p$ 整除。模 $p$ 后混合项全部消失，只剩纯项：
> $$
> \boxed{(x_1+\cdots+x_n)^p
> \equiv x_1^p+\cdots+x_n^p\pmod p}.
> $$
> **(b)** 令所有 $x_i=1$，得到
> $$
> n^p\equiv n\pmod p.
> $$
> 若 $p\nmid n$，则 $n$ 模 $p$ 有乘法逆元；两边消去 $n$：
> $$
> \boxed{n^{p-1}\equiv1\pmod p}.
> $$
> 上述“取 $n$ 个变量”先直接覆盖正整数 $n$；若允许负整数，取其模 $p$ 的非零代表元 $r\in\{1,\ldots,p-1\}$，由 $n\equiv r\pmod p$ 即得同一结论。
> 这条证明独立于 Fermat 小定理本身，没有循环论证。
> <!-- bilingual-en:start -->
> **(a)**Polynomial expansion as
> Unless the exponent vector is a pure term with one $k_i=p$ and all other entries zero, at least two $k_i$ are positive and every positive $k_i<p$. The prime $p$ occurs once in the numerator $p!$, while none of the denominator factors $k_i!$ contains $p$. Hence every mixed multinomial coefficient is divisible by $p$. Modulo $p$, all mixed terms vanish and only the pure terms remain:
> **(b)**Make all $x_i=1$, get
> If $p\nmid n$, then the $n$ module $p$ has the inverse multiplicative element;both sides eliminate $n$:
> Taking $n$ variables proves the claim directly for positive integers $n$. For a negative integer, choose its nonzero representative $r\in\{1,\ldots,p-1\}$ modulo $p$; since $n\equiv r\pmod p$, the same conclusion follows.
> This argument does not invoke Fermat's little theorem, so it is not circular.
> <!-- bilingual-en:end -->

---

## 全章方法地图
<!-- bilingual-en:start -->
*full chapter method map*
<!-- bilingual-en:end -->

| 题目特征 | 首选工具 | 必做检查 |
|---|---|---|
| 长和、正单调项 | 扰动、积分法 | 端点与单调方向 |
| 阶乘或大乘积 | 取对数、Stirling | 是 $\Theta$ 还是 $\sim$ |
| 只关心规模 | $O,o,\Theta,\sim$ | 统一常数、阈值、方向 |
| 互斥分类 | 加法法则 | 类别是否重叠 |
| 连续选择 | 广义乘法 | 每个前缀的延伸数是否固定 |
| 对象难数、编码容易 | 双射 | 正向、逆向、互逆 |
| 每个结果被数相同次数 | Division Rule | 原像数是否恒定 |
| 重复对象/指数分配 | Stars and Bars、多项式系数 | 对象同异、顺序是否重要 |
| 只需保证碰撞 | 鸽巢原理 | pigeons、holes、映射、端点 |
| 多个重叠事件的并 | 容斥 | 交集是否可行、奇加偶减 |
<!-- bilingual-en:start -->
|Subject Characteristics|Preferred Tools|Required Checks|
|—|—|—|
|Long Sum, Positive Monotone Term|Perturbation, Integral Method|Endpoint and Monotone Direction|
| Factorial or Large Product | Take Logarithm, Stirling | Is $\Theta$ or $\sim$ |
|Only care about size | $O,o,\Theta,\sim$ | Uniform constant, threshold, direction |
|Mutually Exclusive Classifications|Addition Rule|Whether Categories Overlapped|
|Sequential Selection|Generalized Multiplication|Is the number of extensions per prefix fixed|
|Difficult objects, easy encoding|Bijection|Forward map, inverse map, and proof that they are inverses|
|Same number of times each result is counted| Division Rule |Is the number of preimages constant?|
| Duplicate Objects/Exponential Allocations | Stars and Bars, Polynomial Coefficients | Object Similarities, Order Importance |
|Only a collision must be guaranteed|Pigeonhole principle|Pigeons, holes, mapping, and endpoint cases|
|Union of several overlapping events|Inclusion–exclusion|Feasible intersections and alternating signs|
<!-- bilingual-en:end -->

## 覆盖与资源核对
<!-- bilingual-en:start -->
*Coverage and Resource Reconciliation*
<!-- bilingual-en:end -->

- 官方在线题：Session 23 的 O23-01–O23-16（16 个）、Session 24 的 O24-01–O24-21（21 个）、Session 25 的 O25-01–O25-04（4 个）、Session 26 的 O26-01（1 个）、Session 27 的 O27-01–O27-11（11 个），合计 **53**。
- Classroom Problems：C23-1–5、C24-1–5、C25-1–4、C26-1–5、C27-1–5，合计 **24**；每题及全部子问均在对应 Session 末给出。
- 作业：PS9 3 题位于 Session 24 后；PS10 3 题及全部子问位于 Session 27 后。
- 考试：Midterm 3 共 6 题及全部子问位于 Session 24 后；第 1 题 DAG 已按原 PDF 图逐边读取。
- 视频顺序：17 个视频均按 3.1.1→3.5.4 官方 block 次序出现，并同时链接 slides、transcript 与在线 video。
<!-- bilingual-en:start -->
- Official online titles: O23-01-O23-16 (16) for Session 23, O24-01-O24-21 (21) for Session 24, O25-01-O25-04 (4) for Session 25, O26-01 (1) for Session 26, O27-01-O27-11 (11) for Session 27, total**53**.
- Classroom Problems:C23-1-5, C24-1-5, C25-1-4, C26-1-5, C27-1-5, totaling**24**; each question and all subquestions are given at the end of the corresponding Session.
- Job: PS9 3 questions are located after Session 24; PS10 3 questions and all subquestions are located after Session 27.
- Quiz: Midterm 3 with 6 questions and all questions behind Session 24; Question 1 DAG has been read side by side as per the original PDF.
- Video Order: Seventeen videos appear in the official block order of 3.1.1→3.5.4, linking slides, transcript, and online video.
<!-- bilingual-en:end -->

> [!summary] 一句话收束
> **和式**把局部贡献累积起来，**渐近**压缩其规模，**双射与除法**把对象搬到标准模型，**鸽巢**从数量推出必然性，**容斥**修复重叠；五者合起来就是离散计数的主干。
> <!-- bilingual-en:start -->
> summation accumulates local contributions, asymptotics compresses their scale, bijections and division move objects to standard counting models, the pigeonhole principle extracts unavoidable collisions from quantities, and inclusion–exclusion corrects overlap; together these form the backbone of discrete counting.
> <!-- bilingual-en:end -->
