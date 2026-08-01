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
This notebook follows the block/video order of MIT OCW Spring 2015 Unit 3: Session 23 → Session 24 → Problem Set 9 → Midterm 3 → Session 25 → Session 26 → Session 27 → Problem Set 10. For the course page, see [MIT OCW 6.042J](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/); for the local index, see [[MIT_OCW_6.042J_Materials/index|MIT 6.042J materials index]].
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
**Prerequisites**: finite sums, limits, derivatives, definite integrals, induction, and centres of mass. This session gives the first formal use of [[无穷级数与幂级数#数项级数与必要条件|series]], [[定积分与微积分基本定理#从黎曼和到定积分|integration]], and [[货币时间价值与贴现#年金、永续年金与增长永续年金|annuities]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session23.pdf#page=1|Session 23 reading, pp. 1–24]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp23.pdf#page=1|cp23, pp. 1–3]]

### 3.1.1 Arithmetic Sums — 配对扰动
<!-- bilingual-en:start -->
*3.1.1 Arithmetic Sums — Pairing as a Perturbation*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Arithmetic.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/v6axtBS6IF8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=v6axtBS6IF8)

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

> [!question] O23-01
> 年幼的 Gauss 用什么扰动快速求从 89 开始、公差 13 的 30 个整数之和？
> <!-- bilingual-en:start -->
> What rearrangement does young Gauss use to sum quickly the 30-term arithmetic progression with first term 89 and common difference 13?
> <!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 把和式与其逆序形式相加，使对应项的和相同。官方材料写成 $30(89+479)$ 后再除以 $2$；其要点是“正序 + 逆序”，而不是先调用现成闭式。注意：若严格按“首项 89、公差 13、共 30 项”，末项应为 $89+29\cdot13=466$；官方题面中的 479 与这三个数据存在一项偏差。本笔记保留官方答案，同时把这个边界不一致明确指出。
> <!-- bilingual-en:start -->
> Add the sum in forward order to the same sum in reverse order, so that every paired term has the same total. The official material writes the result as $30(89+479)/2$; the point is the “forward + reverse” pairing, not merely invoking a memorised formula. Strictly following “first term 89, common difference 13, 30 terms” gives the last term $89+29\cdot13=466$, so 479 is inconsistent with the other three data. This notebook retains the official answer while making that discrepancy explicit.
> <!-- bilingual-en:end -->

### 3.1.3 Geometric Sums — 移位相减
<!-- bilingual-en:start -->
*3.1.3 Geometric Sums — Shift and Subtract*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_GeometricSum.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ZDQk45NQbEo.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=ZDQk45NQbEo)

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
Subtracting gives $(1-x)S_n=1-x^{n+1}$. Thus, when $x\ne1$,
<!-- bilingual-en:end -->

$$
\boxed{S_n=\frac{1-x^{n+1}}{1-x}}.
$$

当 $x=1$ 时不能除以 $1-x$，必须回到原定义：$S_n=n+1$。这正是公式的适用边界。
<!-- bilingual-en:start -->
When $x=1$, division by $1-x$ is invalid, so return to the definition: $S_n=n+1$. This is precisely the exceptional case excluded by the closed form.
<!-- bilingual-en:end -->

#### 无限几何级数及其必要条件
<!-- bilingual-en:start -->
*Infinite Geometric Series and Its Necessary Conditions*
<!-- bilingual-en:end -->

所谓无限和，是部分和的极限：
<!-- bilingual-en:start -->
An infinite sum is defined as the limit of its partial sums:
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
*Differentiating a Geometric Sum: Weighted Sums*
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
Multiplying by $x$ gives:
<!-- bilingual-en:end -->

$$
\boxed{\sum_{i=1}^{n}i x^i
=\frac{x-(n+1)x^{n+1}+n x^{n+2}}{(1-x)^2}}.
$$

当 $|x|<1$，$n|x|^n\to0$，从而
<!-- bilingual-en:start -->
When $|x|<1$, we have $n|x|^n\to0$, and therefore
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{\infty}i x^i=\frac{x}{(1-x)^2}.
$$

### 3.1.4 Annuities — 官方在线题 O23-02
<!-- bilingual-en:start -->
*3.1.4 Annuities — official online O23-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.4_annuities|3.1.4]]

一笔在第 $i$ 年末收到的 $m$ 元，若年利率为 $p$，今天的现值是 $m/(1+p)^i$。从一年后开始、永久每年支付 $m$ 的永续年金现值为
<!-- bilingual-en:start -->
A payment of $m$ received at the end of year $i$ has present value $m/(1+p)^i$ when the annual interest rate is $p$. A perpetuity that pays $m$ once a year, beginning one year from today, therefore has present value
<!-- bilingual-en:end -->

$$
V=\sum_{i=1}^{\infty}\frac{m}{(1+p)^i}
=\frac{m}{1+p}\cdot\frac1{1-1/(1+p)}
=\boxed{\frac mp}.
$$

若第一次支付就在今天，则还要加 $m$，现值为 $m(1+p)/p$。时间点差一年，答案便差一个 $m$，这是金融题最常见的错误源。
<!-- bilingual-en:start -->
If the first payment is made today, add $m$, giving a present value of $m(1+p)/p$. Shifting the first payment by one year changes the answer by exactly $m$; this timing issue is a common source of errors in finance problems.
<!-- bilingual-en:end -->

> [!question] O23-02
> 年收益率恒为 $4\%$，从一年后起每年永久支付 $10{,}000$ 美元，今天应投入多少？
> <!-- bilingual-en:start -->
> If the annual return is fixed at $4\%$ and the perpetuity pays $10{,}000$ dollars each year beginning one year from now, how much must be invested today?
> <!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> $V=10{,}000/0.04=\boxed{250{,}000}$。等价直觉：$250{,}000$ 每年的 $4\%$ 利息恰为 $10{,}000$，不动本金即可永久支付。
> <!-- bilingual-en:start -->
> $V=10{,}000/0.04=\boxed{250{,}000}$. Equivalently, the annual interest on $250{,}000$ at $4\%$ is exactly $10{,}000$, so the payment can continue forever without drawing down the principal.
> <!-- bilingual-en:end -->

### 3.1.5 Book Stacking — 调和数从质心出现
<!-- bilingual-en:start -->
*3.1.5 Book Stacking — harmonic numbers appear from the center of mass*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BookStacking.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CdhuVhWTSMI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CdhuVhWTSMI)

每本书长度归一化为 $1$、质量相同。要使最上方 $n$ 本书在下一本书边缘上恰好稳定，它们的共同质心必须落在该边缘正上方。
<!-- bilingual-en:start -->
Each book has unit length and the same mass. For the top $n$ books to be just stable at the edge of the book beneath them, their common centre of mass must lie directly above that edge.
<!-- bilingual-en:end -->

设已有 $n$ 本书相对其支撑边缘的最大伸出量为 $B_n$。在下面加第 $n+1$ 本书并把上面 $n$ 本整体向右移动 $\Delta$。以新书中心为力矩原点：上面 $n$ 本的总质量为 $n$，其质心向右 $\Delta$；新书质量为 $1$，其中心距右边缘 $1/2$，向左的力臂是 $1/2-\Delta$。临界平衡要求
<!-- bilingual-en:start -->
Suppose the top $n$ books can overhang their supporting edge by at most $B_n$. Add book $n+1$ underneath and shift the block of $n$ books to the right by $\Delta$. Taking moments about the centre of the new book, the upper block has total mass $n$ and centre of mass $\Delta$ to the right, while the new book has mass $1$ and a leftward lever arm of $1/2-\Delta$. At the limiting equilibrium,
<!-- bilingual-en:end -->

$$
n\Delta=\frac12-\Delta,
\qquad
\Delta=\frac1{2(n+1)}.
$$

于是
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->

$$
B_{n+1}=B_n+\frac1{2(n+1)},\qquad B_1=\frac12,
$$

迭代得
<!-- bilingual-en:start -->
Iterating the recurrence gives
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
Because the harmonic series diverges, the stack can in principle overhang by any prescribed distance. However, $H_n$ grows only like $\ln n$, so the number of books required grows exponentially with the desired overhang.
<!-- bilingual-en:end -->

### 3.1.6 Harmonic Numbers — 官方在线题 O23-03
<!-- bilingual-en:start -->
*3.1.6 Harmonic Numbers — official online O23-03*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.6_harmonic-numbers|3.1.6]]

> [!question] O23-03
> 在四个陈述中选真命题：调和数有简单和式定义；$H_n=\sum_{i=0}^n1/i$；$n$ 本书伸出量为 $H_n$；新增一本时 $\Delta=\frac{1/2}{n+1}$ 来自 $n\Delta=1(1/2-\Delta)$。
> <!-- bilingual-en:start -->
> Select the true statements: harmonic numbers have a simple summation definition; $H_n=\sum_{i=0}^n1/i$; the overhang of $n$ books is $H_n$; and the increment $\Delta=\frac{1/2}{n+1}$ follows from $n\Delta=1(1/2-\Delta)$.
> <!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> 第 1、4 项正确。定义从 $i=1$ 开始，$H_n=\sum_{i=1}^n1/i$；最大伸出量是 $H_n/2$，不是 $H_n$。
> <!-- bilingual-en:start -->
> Statements 1 and 4 are correct. The sum starts at $i=1$, so $H_n=\sum_{i=1}^n1/i$; the maximum overhang is $H_n/2$, not $H_n$.
> <!-- bilingual-en:end -->

### 3.1.7 Integral Method — 用面积夹住离散和
<!-- bilingual-en:start -->
*3.1.7 Integral Method — Bounding a Discrete Sum by Areas*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_IntegralMeth.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/EegG5TPL29c.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=EegG5TPL29c)

令 $f:\mathbb R^+\to\mathbb R^+$，
<!-- bilingual-en:start -->
Let $f:\mathbb R^+\to\mathbb R^+$,
<!-- bilingual-en:end -->

$$
S=\sum_{i=1}^{n}f(i),\qquad I=\int_1^n f(x)\,dx.
$$

#### 单调递增情形的完整证明
<!-- bilingual-en:start -->
*Full Proof for the Nondecreasing Case*
<!-- bilingual-en:end -->

若 $f$ 弱递增，则对每个整数 $i=1,\ldots,n-1$ 以及 $x\in[i,i+1]$，
<!-- bilingual-en:start -->
If $f$ is nondecreasing, then for every integer $i=1,\ldots,n-1$ and every $x\in[i,i+1]$,
<!-- bilingual-en:end -->

$$
f(i)\le f(x)\le f(i+1).
$$

区间长度为 $1$，积分后得
<!-- bilingual-en:start -->
Because the interval has length $1$, integration gives
<!-- bilingual-en:end -->

$$
f(i)\le\int_i^{i+1}f(x)\,dx\le f(i+1).
$$

对 $i=1$ 到 $n-1$ 求和：
<!-- bilingual-en:start -->
Summing from $i=1$ to $n-1$ gives
<!-- bilingual-en:end -->

$$
\sum_{i=1}^{n-1}f(i)\le I\le\sum_{i=2}^{n}f(i).
$$

左式加 $f(n)$、右式加 $f(1)$ 并改写为 $S$：
<!-- bilingual-en:start -->
Add $f(n)$ to the left inequality and $f(1)$ to the right, then rewrite both sums in terms of $S$:
<!-- bilingual-en:end -->

$$
\boxed{I+f(1)\le S\le I+f(n)}.
$$

#### 单调递减情形的完整证明
<!-- bilingual-en:start -->
*Complete proof for the nonincreasing case*
<!-- bilingual-en:end -->

若 $f$ 弱递减，不等号方向反转：对 $x\in[i,i+1]$，
<!-- bilingual-en:start -->
If $f$ is nonincreasing, the inequalities reverse: for $x\in[i,i+1]$,
<!-- bilingual-en:end -->

$$
f(i+1)\le f(x)\le f(i).
$$

同样积分、求和得到
<!-- bilingual-en:start -->
Integrating and summing in the same way gives
<!-- bilingual-en:end -->

$$
\sum_{i=2}^{n}f(i)\le I\le\sum_{i=1}^{n-1}f(i),
$$

即
<!-- bilingual-en:start -->
that is,
<!-- bilingual-en:end -->

$$
\boxed{I+f(n)\le S\le I+f(1)}.
$$

严格单调只会让相应不等号变严格，不改变界的表达式。条件“正值”保证可用上下界判断收敛；“单调”保证每个单位区间上的矩形方向一致。
<!-- bilingual-en:start -->
Strict monotonicity merely makes the corresponding inequalities strict; it does not change the form of the bounds. Positivity lets these bounds be used in convergence arguments, while monotonicity fixes which endpoint rectangle lies above or below the curve on every unit interval.
<!-- bilingual-en:end -->

#### 例：$\sum_{i=1}^{n}\sqrt i$
<!-- bilingual-en:start -->
*Example: $\sum_{i=1}^{n}\sqrt i$*
<!-- bilingual-en:end -->

$f(x)=\sqrt x$ 递增，且
<!-- bilingual-en:start -->
$f(x)=\sqrt x$ is increasing, and
<!-- bilingual-en:end -->

$$
I=\int_1^n\sqrt x\,dx=\frac23(n^{3/2}-1).
$$

所以
<!-- bilingual-en:start -->
therefore,
<!-- bilingual-en:end -->

$$
\boxed{\frac23n^{3/2}+\frac13
\le\sum_{i=1}^{n}\sqrt i
\le\frac23n^{3/2}+\sqrt n-\frac23}.
$$

主导项两边相同，故 $\sum_{i=1}^n\sqrt i\sim\frac23n^{3/2}$。
<!-- bilingual-en:start -->
The two bounds have the same leading term, so $\sum_{i=1}^n\sqrt i\sim\frac23n^{3/2}$.
<!-- bilingual-en:end -->

#### 调和数的界与发散
<!-- bilingual-en:start -->
*Bounds and Divergence of the Harmonic Numbers*
<!-- bilingual-en:end -->

对递减函数 $f(x)=1/x$，直接应用定理得
<!-- bilingual-en:start -->
Applying the theorem to the decreasing function $f(x)=1/x$ gives
<!-- bilingual-en:end -->

$$
\ln n+\frac1n\le H_n\le1+\ln n.
$$

把每个 $1/i$ 看成区间 $[i,i+1]$ 上 $1/x$ 的左端矩形，还可得到更常用的下界
<!-- bilingual-en:start -->
Viewing each $1/i$ as the left-endpoint rectangle for $1/x$ on $[i,i+1]$ also gives the more familiar lower bound
<!-- bilingual-en:end -->

$$
\boxed{\ln(n+1)\le H_n\le1+\ln n}.
$$

两边除以 $\ln n$ 均趋于 $1$，由夹逼定理
<!-- bilingual-en:start -->
After division by $\ln n$, both bounds tend to $1$; the squeeze theorem therefore gives
<!-- bilingual-en:end -->

$$
H_n\sim\ln n.
$$

且下界 $\ln(n+1)\to\infty$，所以 [[无穷级数与幂级数#数项级数与必要条件|调和级数发散]]。
<!-- bilingual-en:start -->
Moreover, the lower bound $\ln(n+1)\to\infty$, so the [[无穷级数与幂级数#数项级数与必要条件|harmonic series diverges]].
<!-- bilingual-en:end -->

### 3.1.8 Integral Method Demystified — 官方在线题 O23-04 至 O23-11
<!-- bilingual-en:start -->
*3.1.8 Integral Method Demystified — Official Online Questions O23-04 through O23-11*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.8_integral-method-demystified|3.1.8]]

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
| Number | Question | Official answer | Reason |
|---|---|---|---|
| O23-04 | Upper bound for $S$ when $f$ is nondecreasing | $I+f(n)$ | Right-endpoint rectangles lie above the curve |
| O23-05 | Upper bound for $S$ when $f$ is nonincreasing | $I+f(1)$ | Left-endpoint rectangles lie above the curve |
| O23-06 | Lower bound for $S$ when $f$ is nondecreasing | $I+f(1)$ | After the last term is removed, the rectangles lie below the curve |
| O23-07 | Lower bound for $S$ when $f$ is nonincreasing | $I+f(n)$ | After the first term is removed, the rectangles lie below the curve |
| O23-08 | Does strict monotonicity change the form of the bounds? | No | It changes only $\le$ to $<$ |
| O23-09 | Upper bound for $H_n$ | $1+\ln n$ | Apply the decreasing case |
| O23-10 | Lower bound for $H_n$ | $\ln(n+1)$ | Sum the areas over $[i,i+1]$ |
| O23-11 | Asymptotic equivalent of $H_n$ | $\ln n$ | Both normalized bounds tend to $1$ |
<!-- bilingual-en:end -->

### 3.1.9 Stirling's Formula — 阶乘的规模
<!-- bilingual-en:start -->
*3.1.9 Stirling's Formula — The Scale of a Factorial*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_StirlingForm.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/lU_QT5GSuxI.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=lU_QT5GSuxI)

乘积先取对数：
<!-- bilingual-en:start -->
Take logarithms to turn the product into a sum:
<!-- bilingual-en:end -->

$$
\ln\prod_{i=1}^{n}f(i)=\sum_{i=1}^{n}\ln f(i).
$$

特别地，$\ln(n!)=\sum_{i=1}^n\ln i$。积分法先给出粗界
<!-- bilingual-en:start -->
In particular, $\ln(n!)=\sum_{i=1}^n\ln i$. The integral method first gives the coarse bounds
<!-- bilingual-en:end -->

$$
n\ln n-n+1\le\ln(n!)\le(n+1)\ln n-n+1,
$$

指数化后：
<!-- bilingual-en:start -->
Exponentiating gives:
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
**Step 1: Prove that $a_n$ has a finite positive limit.** Calculate the logarithmic difference between consecutive terms:
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

> [!question] O23-12
> 化简 $\dfrac{(2n)!}{2^{2n}(n!)^2}$ 的渐近等价式。
> <!-- bilingual-en:start -->
> Find a simplified asymptotic equivalent of $\dfrac{(2n)!}{2^{2n}(n!)^2}$.
> <!-- bilingual-en:end -->

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
> All exponential factors cancel, leaving only the square-root scale.
> <!-- bilingual-en:end -->

### 3.1.11 Convergence of Geometric Series — 官方在线题 O23-13
<!-- bilingual-en:start -->
*3.1.11 Convergence of Geometric Series — official online O23-13*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.11_convergence-of-geometric-series|3.1.11]]

> [!question] O23-13
> $r\in\{0,-0.5,0.5,1\}$ 时，哪个公比使几何级数不收敛？
> <!-- bilingual-en:start -->
> For $r\in\{0,-0.5,0.5,1\}$, which common ratio makes the geometric series diverge?
> <!-- bilingual-en:end -->

> [!success]- 官方答案与反馈
> $\boxed{r=1}$；几何级数收敛当且仅当 $|r|<1$。
> <!-- bilingual-en:start -->
> $\boxed{r=1}$. A geometric series converges if and only if $|r|<1$.
> <!-- bilingual-en:end -->

### 3.1.12 Summation — 官方在线题 O23-14、O23-15
<!-- bilingual-en:start -->
*3.1.12 Summation — Official Online Questions O23-14 and O23-15*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.12_summation|3.1.12]]

考察 $p$-级数 $\sum_{i=1}^{\infty}i^p$。对 $p\ne-1$，
<!-- bilingual-en:start -->
Examine the $p$-series $\sum_{i=1}^{\infty}i^p$. For $p\ne-1$,
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
*3.1.13 Upper and Lower Bounds for a Sum — official online O23-16*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S23_3.1.13_sum-s-upper-lower-bounds|3.1.13]]

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

&nbsp;
**1.** **An infinite sum is not merely formal algebra**: define the partial sums before taking a limit; the formula $1/(1-r)$ is invalid when $|r|\ge1$.<br>
**2.** **Off-by-one endpoints**: $\sum_{i=0}^{n}$ has $n+1$ terms; after differentiation, its highest-degree term is $nx^{n-1}$.<br>
**3.** **Timing of an annuity**: a first payment today versus one year from today changes the present value by exactly one payment $m$.<br>
**4.** **The integral method depends on monotonicity**: right-endpoint rectangles give an upper bound for increasing functions, while the direction reverses for decreasing functions.<br>
**5.** **$\sim$ cannot be inferred by comparing logarithms alone**: $\ln(n!)\sim n\ln n$ does not imply $n!\sim n^n$, because exponentiation magnifies smaller additive errors.<br>
<!-- bilingual-en:end -->

### Session 23 自检
<!-- bilingual-en:start -->
*Session 23 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 23-1
> 求 $\sum_{i=0}^{n}3\cdot2^i$，并说明何时能取 $n\to\infty$。
> <!-- bilingual-en:start -->
> Evaluate $\sum_{i=0}^{n}3\cdot2^i$, and state when the limit $n\to\infty$ may be taken.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 有限和为 $3(2^{n+1}-1)$。公比 $2$ 的绝对值不小于 $1$，故无限级数发散。
> <!-- bilingual-en:start -->
> The finite sum is $3(2^{n+1}-1)$. Since the common ratio has absolute value $2\ge1$, the corresponding infinite series diverges.
> <!-- bilingual-en:end -->

> [!question] 自检 23-2
> 用积分法给 $\sum_{i=1}^{n}i^2$ 一个足以证明 $\Theta(n^3)$ 的上下界。
> <!-- bilingual-en:start -->
> Use the integral method to bound $\sum_{i=1}^{n}i^2$ tightly enough to prove that it is $\Theta(n^3)$.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $f(x)=x^2$ 递增，$I=(n^3-1)/3$，所以
> $$
> \frac{n^3-1}{3}+1\le\sum_{i=1}^{n}i^2\le\frac{n^3-1}{3}+n^2.
> $$
> 两边均为 $\Theta(n^3)$。
> <!-- bilingual-en:start -->
> The function $f(x)=x^2$ is increasing and $I=(n^3-1)/3$, so the displayed bounds follow. Both bounds are $\Theta(n^3)$.
> <!-- bilingual-en:end -->

> [!question] 自检 23-3
> 为什么 $H_n$ 发散却增长得非常慢？
> <!-- bilingual-en:start -->
> Why does $H_n$ diverge even though it grows very slowly?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 因为 $\ln(n+1)\le H_n\le1+\ln n$。下界无界保证发散，但要让 $H_n$ 增加固定量 $c$，$n$ 大约要乘以 $e^c$。
> <!-- bilingual-en:start -->
> Because $\ln(n+1)\le H_n\le1+\ln n$. The unbounded lower bound proves divergence, but increasing $H_n$ by a fixed amount $c$ requires multiplying $n$ by roughly $e^c$.
> <!-- bilingual-en:end -->

### Classroom Problems 23 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 23 — complete independent solutions to five problems*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp23.pdf#page=1|cp23 pp. 1–3]]

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
> **Given:** Two cups each hold one pint. The first initially contains water and the second wine. In each round, pour $1/3$ pint from the first cup into the second, mix, and then pour $1/3$ pint back.
> **Goal:** Find the amount of wine in the first cup after $n$ rounds and the limiting amounts in both cups.
> Let $x_n$ be the amount of wine in the first cup after round $n$, with $x_0=0$. At the start of a round, the first cup contains $x_n$ pints of wine. The first pour leaves $2x_n/3$ there. The second cup then has volume $4/3$ and contains $1+x_n/3$ pints of wine. Pouring back $1/3$ pint transfers one quarter of the second cup, so the recurrence is $x_{n+1}=\frac34x_n+\frac14$.
> Its fixed point is $1$. Setting $y_n=1-x_n$ gives $y_{n+1}=\frac34y_n$ and hence $x_n=1-(3/4)^n$. The total amount of wine remains one pint, so the second cup contains $(3/4)^n$ pints. Thus the first cup tends to one pint of wine and the second to none. The directional order of the two pours explains the apparently surprising limit.
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
> **(c) Recursive proof.** Suppose a total of $n$ gallons permits a round-trip distance $D_n$. To move the base forward by $\delta$, the first $n-1$ loads cross the new segment in both directions and the last crosses it only outward. The water accumulated at the new base is $n-(2n-1)\delta$. This must equal the $n-1$ gallons needed for the recursive trip plus the $\delta$ gallons needed to return to the old base, so $\delta=1/(2n)$. Hence $D_n=D_{n-1}+1/(2n)$ and $D_n=H_n/2$. Since $H_n\to\infty$, every fixed distance is eventually reachable.
> **(d)** Reaching distance $10$ requires $H_n/2\gtrsim10$, so $\ln n\gtrsim20$ and $n\gtrsim e^{20}\approx4.85\times10^8$. At most one gallon can be collected from the oasis per day, so the collection trips alone require about $1.33\times10^6$ years—indeed more than a million years.
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
> For $x\in[i,i+1]$, monotonic decrease gives $f(i+1)\le f(x)\le f(i)$. Integrating and summing from $i=1$ to $n-1$ yields the displayed inequalities. With $S=\sum_{i=1}^nf(i)$ and $I=\int_1^nf(x)\,dx$, restoring the omitted endpoint terms gives $I+f(n)\le S\le I+f(1)$.
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
> Let $D_d$ be the debt at the end of day $d$, with $D_0=m$. Each day the service fee $f$ is added first, and the result is then multiplied by the interest factor $p$: $D_d=p(D_{d-1}+f)$.
> **(a)** $D_1=p(m+f)$.
> **(b)** $D_2=p[p(m+f)+f]=p^2m+fp(p+1)$.
> **(c)** Expanding the recurrence gives $D_d=p^dm+fp(1+p+\cdots+p^{d-1})$. Thus, for $p\ne1$, $D_d=p^dm+fp(p^d-1)/(p-1)$; for $p=1$, $D_d=m+df$.
> **(d)** With $m=10,f=0.1,p=1.01,d=365$, the displayed formula gives approximately $\boxed{749.35\text{ dollars}}$.
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
**Learning questions**: When do two functions have the same growth rate? What logical relationship does each of $O$, $o$, $\Theta$, and $\sim$ express? How can one prove such a claim with a single constant and threshold instead of guessing from the leading term?
<!-- bilingual-en:end -->

**前置知识**：函数极限、绝对值、关系的自反/对称/传递/反对称性质。首次正式使用 [[渐近记号与算法复杂度#O、Omega 与 Theta|渐近记号与 Θ 同阶关系]]。
<!-- bilingual-en:start -->
**Prerequisites**: limits of functions, absolute values, and the reflexive, symmetric, transitive, and antisymmetric properties of relations. This session gives the first formal use of [[渐近记号与算法复杂度#O、Omega 与 Theta|asymptotic notation and the Θ relation]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|Session 24 reading, pp. 1–8]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|cp24, pp. 1–3]]

### 3.2.1 Asymptotic Notation — 五种关系各说什么
<!-- bilingual-en:start -->
*3.2.1 Asymptotic Notation — What each of the five relationships says*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymNotation.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CWkh5kb4TGc.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=CWkh5kb4TGc)

以下默认 $f,g$ 在充分大的 $x$ 上有定义，且 $g(x)>0$；若 $f$ 可取负值，Big O 定义使用 $|f|$。
<!-- bilingual-en:start -->
Unless stated otherwise, assume that $f$ and $g$ are defined for all sufficiently large $x$ and that $g(x)>0$. If $f$ may be negative, the Big O definition uses $|f(x)|$.
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
It says that the relative error tends to zero: $f=g(1+o(1))$. For example, $n^2+n\sim n^2$, but $2n^2\not\sim n^2$.
<!-- bilingual-en:end -->

#### 严格低阶 $f=o(g)$
<!-- bilingual-en:start -->
*Strictly lower order: $f=o(g)$*
<!-- bilingual-en:end -->

$$
\boxed{f=o(g)\iff\lim_{x\to\infty}\frac{|f(x)|}{g(x)}=0}.
$$

它说无论给定多小的 $\varepsilon>0$，总存在 $x_0$，使 $x\ge x_0$ 时 $|f(x)|\le\varepsilon g(x)$。
<!-- bilingual-en:start -->
Equivalently, for every $\varepsilon>0$ there exists $x_0$ such that $|f(x)|\le\varepsilon g(x)$ whenever $x\ge x_0$.
<!-- bilingual-en:end -->

#### 上界关系 $f=O(g)$
<!-- bilingual-en:start -->
*Upper-Bound Relation $f=O(g)$*
<!-- bilingual-en:end -->

$$
\boxed{f=O(g)\iff
\exists c>0\ \exists x_0\ \forall x\ge x_0:
|f(x)|\le c g(x)}.
$$

若商不一定有普通极限，也可写成
<!-- bilingual-en:start -->
Even when the quotient has no ordinary limit, the condition can be expressed as
<!-- bilingual-en:end -->

$$
\limsup_{x\to\infty}\frac{|f(x)|}{g(x)}<\infty.
$$

Big O 只给**渐近上界**；它允许常数倍差异，也允许有界振荡。
<!-- bilingual-en:start -->
Big O gives only an **asymptotic upper bound**; it allows constant-factor differences and bounded oscillation.
<!-- bilingual-en:end -->

#### 同阶 $f=\Theta(g)$ 与下界 $\Omega$
<!-- bilingual-en:start -->
*Same order $f=\Theta(g)$ and lower bound $\Omega$*
<!-- bilingual-en:end -->

$$
\boxed{f=\Theta(g)\iff f=O(g)\text{ 且 }g=O(f)}.
$$

等价地，存在 $c_1,c_2>0$ 与 $x_0$，使充分大时
<!-- bilingual-en:start -->
Equivalently, there exist $c_1,c_2>0$ and $x_0$ such that, for all sufficiently large $x$,
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
> But neither converse holds: $2g=\Theta(g)$ is not asymptotically equivalent to $g$; $1=O(n)$ is not $\Theta(n)$.
> <!-- bilingual-en:end -->

### 3.2.2 Asymptotics as Relations — 官方在线题 O24-01、O24-02
<!-- bilingual-en:start -->
*3.2.2 Asymptotics as Relations — Official Online Questions O24-01 and O24-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.2_asymptotics-as-relations|3.2.2]]

> [!success]- 官方答案与反馈
> **O24-01（哪些关系对称）**：$\sim$ 与 $\Theta$。
> **O24-02（哪些关系表示同一增长阶）**：$\sim$ 与 $\Theta$。
> $f=o(g)$ 与 $f=O(g)$ 一般都不对称；$\sim$ 比 $\Theta$ 更强，因为它要求比值趋于 $1$。
> <!-- bilingual-en:start -->
> **O24-01 (which relations are symmetric?)**: $\sim$ and $\Theta$.
> **O24-02 (which relations express the same order of growth?)**: $\sim$ and $\Theta$.
> The relations $f=o(g)$ and $f=O(g)$ are generally not symmetric. The relation $\sim$ is stronger than $\Theta$ because it requires the ratio to tend to $1$.
> <!-- bilingual-en:end -->

### 3.2.3 Asymptotic Properties — 关系结构与增长层级
<!-- bilingual-en:start -->
*3.2.3 Asymptotic Properties — Relationship Structure and Growth Hierarchy*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymProperti.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HeyEK0TWiBw.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HeyEK0TWiBw)

#### 关系性质
<!-- bilingual-en:start -->
*Properties of the relations*
<!-- bilingual-en:end -->

- $\sim$ 是等价关系：自反来自 $f/f=1$；对称来自倒数；传递来自 $(f/g)(g/h)=f/h$。
- $\Theta$ 是等价关系：自反显然；定义本身对称；Big O 的传递性给传递。
- $o$ 是严格偏序式关系：非自反且传递。若 $f=o(g)$，不可能同时 $g=O(f)$，否则充分大时 $g\le C|f|$，而 $|f|/g\to0$ 与 $1\le C|f|/g$ 冲突。
- $O$ 是预序：自反、传递，但不反对称。不同函数 $f=n$ 与 $g=2n$ 互为 Big O。
- 关系“$f=O(g)$ 且 $g\ne O(f)$”是严格偏序，表达“严格不快于”。它比 $o$ 弱，因为商可能在 $0$ 与正常数量级之间振荡。
<!-- bilingual-en:start -->
- $\sim$ is an equivalence relation: reflexivity follows from $f/f=1$, symmetry from taking reciprocals, and transitivity from $(f/g)(g/h)=f/h$.
- $\Theta$ is also an equivalence relation: reflexivity is immediate, the definition is symmetric, and transitivity follows from the transitivity of Big O.
- $o$ behaves like a strict partial order: it is irreflexive and transitive. If $f=o(g)$, then $g=O(f)$ is impossible; otherwise $g\le C|f|$ eventually, contradicting $|f|/g\to0$.
- $O$ is a preorder: it is reflexive and transitive but not antisymmetric. For example, the distinct functions $f(n)=n$ and $g(n)=2n$ are Big O of each other.
- The relation "$f=O(g)$ and $g\ne O(f)$" is a strict partial order expressing that $f$ grows strictly no faster than $g$. It is weaker than $f=o(g)$ because the quotient may oscillate between values near zero and values of ordinary size.
<!-- bilingual-en:end -->

#### 对数慢于任意正幂
<!-- bilingual-en:start -->
*Logarithms grow more slowly than every positive power*
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
Thus $\ln x=o(x^\varepsilon)$; “logarithms grow slowly” is a proved comparison, not merely an intuition.
<!-- bilingual-en:end -->

#### 指数快于任意固定多项式
<!-- bilingual-en:start -->
*Exponentials grow faster than every fixed polynomial*
<!-- bilingual-en:end -->

设 $a>1$、$c\ge0$，考察 $u_n=n^c/a^n$：
<!-- bilingual-en:start -->
Let $a>1$ and $c\ge0$, and consider $u_n=n^c/a^n$:
<!-- bilingual-en:end -->

$$
\frac{u_{n+1}}{u_n}=\frac{(1+1/n)^c}{a}\longrightarrow\frac1a<1.
$$

取 $q$ 满足 $1/a<q<1$，则充分大时 $u_{n+1}\le q u_n$，故尾部被收敛几何数列控制，$u_n\to0$。所以
<!-- bilingual-en:start -->
Choose $q$ with $1/a<q<1$. For all sufficiently large $n$, $u_{n+1}\le q u_n$, so the tail is dominated by a convergent geometric sequence and $u_n\to0$. Therefore,
<!-- bilingual-en:end -->

$$
n^c=o(a^n).
$$

典型增长层级为
<!-- bilingual-en:start -->
A typical growth hierarchy is
<!-- bilingual-en:end -->

$$
1\prec\log n\prec n^\varepsilon\prec n^c\prec a^n\prec n!\prec n^n,
$$

其中 $\prec$ 表示左边是右边的 little o；幂指数的具体顺序需满足 $0<\varepsilon<c$。
<!-- bilingual-en:start -->
where $f\prec g$ means $f=o(g)$. The displayed ordering of powers assumes $0<\varepsilon<c$.
<!-- bilingual-en:end -->

### 3.2.4 Little oh / Big Oh — 官方在线题 O24-03 至 O24-05
<!-- bilingual-en:start -->
*3.2.4 Little oh / Big Oh — Official Online Questions O24-03 through O24-05*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.4_little-oh-big-oh|3.2.4]]

> [!success]- 官方答案与反馈
> **O24-03**：以下四项为真：$f=o(g)\Rightarrow f=O(g)$；$f=o(g)\Rightarrow g\ne O(f)$；$f\sim g$ 时 $f=O(g)$；即使 $f\not\sim g$，只要 $f=o(g)$ 仍有 $f=O(g)$。`O ⇒ o` 为假。
> **O24-04**：错误陈述是“对所有整数 $a,b$ 都有 $x^a=o(x^b)$”；正确条件是 $a<b$。其余核心事实是 $\log x=o(x^\varepsilon)$ 与任意固定幂 $x^c=o(a^x)$（$a>1$）。
> **O24-05**：Big O 用 $\limsup$ 是为了容纳商没有普通极限但始终被常数控制的振荡情形；$\limsup$ 不是“更严格的普通极限”。
> <!-- bilingual-en:start -->
> **O24-03**: All four listed implications are true: $f=o(g)\Rightarrow f=O(g)$; $f=o(g)\Rightarrow g\ne O(f)$; $f\sim g\Rightarrow f=O(g)$; and $f=o(g)$ can imply $f=O(g)$ even when $f\not\sim g$. The converse `O ⇒ o` is false.
> **O24-04**: The false statement is "$x^a=o(x^b)$ for all integers $a,b$"; the correct condition is $a<b$. The other key facts are $\log x=o(x^\varepsilon)$ and $x^c=o(a^x)$ for every fixed $c$ when $a>1$.
> **O24-05**: Big O uses $\limsup$ to allow bounded oscillation when the quotient has no ordinary limit. A limit superior is not simply a “stricter ordinary limit.”
> <!-- bilingual-en:end -->

### 3.2.5 Theta — 官方在线题 O24-06、O24-07
<!-- bilingual-en:start -->
*3.2.5 Theta — Official Online Questions O24-06 and O24-07*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.5_theta|3.2.5]]

> [!success]- 官方答案与反馈
> **O24-06（若 $f=\Theta(g)$，哪些必须真）**：$g=\Theta(f)$、$f=O(g)$、$g=O(f)$。
> **O24-07（哪些可能真）**：除上述三项外，$f\sim g$ 也可能真；但 $f=o(g)$ 或 $g=o(f)$ 都不可能与互相 Big O 并存。
> <!-- bilingual-en:start -->
> **O24-06 (what must be true if $f=\Theta(g)$?)**: $g=\Theta(f)$, $f=O(g)$, and $g=O(f)$.
> **O24-07 (what may also be true?)**: In addition to those three statements, $f\sim g$ may hold. Neither $f=o(g)$ nor $g=o(f)$ can coexist with mutual Big O bounds.
> <!-- bilingual-en:end -->

### 3.2.6 Asymptotic Blunders — 语法错往往对应逻辑错
<!-- bilingual-en:start -->
*3.2.6 Asymptotic Blunders — Syntactic Errors Often Reveal Logical Errors*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_AsymBlunders.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Y9Blo_G-Mvg.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=Y9Blo_G-Mvg)

1. **把关系写成数量**：$O(n^2)$ 不是一个可比较大小的数，而是一类函数或二元关系的右侧。
2. **说“至少 $O(n^2)$”**：Big O 是上界。“$f$ 至少像 $n^2$”应写 $n^2=O(f)$ 或 $f=\Omega(n^2)$。
3. **把逐项常数当统一常数**：对固定 $i$，$i=O(1)$，但这里隐藏常数依赖 $i$。求和到 $n$ 时不能用同一个常数控制所有 $i$，实际 $\sum_{i=1}^ni=\Theta(n^2)$。
4. **随意对渐近关系做非线性运算**：$f\sim g$ 不保证 $3^f=\Theta(3^g)$。例：$f=n+\sqrt n$、$g=n$，虽 $f/g\to1$，但 $3^f/3^g=3^{\sqrt n}\to\infty$。
5. **忽略常数与交叉点**：渐近阶只描述充分大输入；实际系统中 $1000n$ 可能在很长区间内慢于 $n^2$，也可能快于带巨大常数的低阶算法。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Treating a relation as a number**: $O(n^2)$ is not a numerical quantity that can be compared by size; it denotes a class of functions, or the right-hand side of an asymptotic relation.<br>
**2.** **Saying “at least $O(n^2)$”**: Big O is an upper bound. “$f$ grows at least as fast as $n^2$” should be written $n^2=O(f)$ or $f=\Omega(n^2)$.<br>
**3.** **Mistaking pointwise constants for one uniform constant**: For fixed $i$, we may write $i=O(1)$, but the hidden constant depends on $i$. No single constant bounds every term up to $n$; in fact, $\sum_{i=1}^ni=\Theta(n^2)$.<br>
**4.** **Applying nonlinear operations indiscriminately**: $f\sim g$ does not imply $3^f=\Theta(3^g)$. For $f=n+\sqrt n$ and $g=n$, the ratio $f/g\to1$, but $3^f/3^g=3^{\sqrt n}\to\infty$.<br>
**5.** **Ignoring constants and crossover points**: Asymptotic order describes only sufficiently large inputs. In practice, $1000n$ may exceed $n^2$ over a long range, while a lower-order algorithm with a huge constant may still be slower.<br>
<!-- bilingual-en:end -->

### 3.2.7 Asymptotics the Right Way — 官方在线题 O24-08 至 O24-10
<!-- bilingual-en:start -->
*3.2.7 Asymptotics the Right Way — Official Online Questions O24-08 through O24-10*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.7_asymptotics-the-right-way|3.2.7]]

> [!success]- 官方答案与反馈
> **O24-08**：$O(\cdot)$、$o(\cdot)$、$\Theta(\cdot)$ 放在等号右侧，例如 $f=O(n^2)$。
> **O24-09**：“$f$ 至少是 $O(n^2)$”四方面都错：$O(n^2)$ 不是数量；Big O 表示上界；它是关系；“至少 $n^2$”应写 $n^2=O(f)$。
> **O24-10**：$\sum_{i=1}^ni=O(n)$ 错在逐项 $O(1)$ 没有统一常数；正确结果是 $\Theta(n^2)$，且 $O(1)$ 不能当作普通数字相加。
> <!-- bilingual-en:start -->
> **O24-08**: Put $O(\cdot)$, $o(\cdot)$, and $\Theta(\cdot)$ on the right-hand side of an equality, for example $f=O(n^2)$.
> **O24-09**: “$f$ is at least $O(n^2)$” is wrong in four ways: $O(n^2)$ is not a quantity; Big O denotes an upper bound; it is a relation; and “at least $n^2$” should be written $n^2=O(f)$.
> **O24-10**: The claim $\sum_{i=1}^ni=O(n)$ wrongly assumes that the termwise $O(1)$ bounds share one uniform constant. The correct order is $\Theta(n^2)$, and $O(1)$ cannot be added as though it were an ordinary number.
> <!-- bilingual-en:end -->

### 3.2.8 Practice with Big O — 官方在线题 O24-11 至 O24-17
<!-- bilingual-en:start -->
*3.2.8 Practice with Big O — Official Online Questions O24-11 through O24-17*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.8_practice-with-big-o|3.2.8]]

题目要求最小非负整数 $k$，使 $f(x)=O(x^k)$：
<!-- bilingual-en:start -->
Each question asks for the smallest nonnegative integer $k$ such that $f(x)=O(x^k)$:
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
| Number | $f(x)$ | Official answer | Key simplification |
|---|---|---:|---|
| O24-11 | $2x^3+x^2\log x$ | $3$ | $x^2\log x=o(x^3)$ |
| O24-12 | $2x^2+x^3\log x$ | $4$ | $x^3\log x=o(x^4)$, but not $O(x^3)$ |
| O24-13 | $1.1^x$ | none | An exponential grows faster than every polynomial |
| O24-14 | $0.1^x$ | $0$ | tends to $0$, so $O(1)$ |
| O24-15 | $(x^4+x^2+1)/(x^3+1)$ | $1$ | The leading ratio is $x$ |
| O24-16 | $(x^4+5\log x)/(x^4+1)$ | $0$ | The ratio tends to $1$ |
| O24-17 | $2^{3\log_2x^2}$ | $6$ | $2^{\log_2x^6}=x^6$ |
<!-- bilingual-en:end -->

### 3.2.9 Practice with Order of Growth — 官方在线题 O24-18 至 O24-21
<!-- bilingual-en:start -->
*3.2.9 Practice with Order of Growth — Official Online Questions O24-18 through O24-21*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S24_3.2.9_practice-with-order-of-growth|3.2.9]]

| 编号 | $f,g$ | 官方答案 | 检查 |
|---|---|---|---|
| O24-18 | $\log_3n,\log_7n$ | $f=O(g)$ 且 $f=\Theta(g)$ | 比值 $\ln7/\ln3\ne1$，故不 $\sim$ |
| O24-19 | $0,33$ | $f=o(g)$ 且 $f=O(g)$ | 商恒为 $0$ |
| O24-20 | $1+\cos(\pi n/2),1+\sin(\pi n/2)$ | 所列均不成立 | 两函数交替取零，商无法统一控制 |
| O24-21 | $1.01^n,n^{100}$ | 所列均不成立 | 实际 $g=o(f)$，但选项只问 $f$ 相对 $g$ |
<!-- bilingual-en:start -->
| Number | $f,g$ | Official answer | Check |
|---|---|---|---|
| O24-18 | $\log_3n,\log_7n$ | $f=O(g)$ and $f=\Theta(g)$ | The ratio is $\ln7/\ln3\ne1$, so the functions are not asymptotically equivalent |
| O24-19 | $0,33$ | $f=o(g)$ and $f=O(g)$ | The quotient is identically $0$ |
| O24-20 | $1+\cos(\pi n/2),1+\sin(\pi n/2)$ | None of the listed relations holds | The functions alternate between zero and positive values, preventing either quotient from being uniformly controlled |
| O24-21 | $1.01^n,n^{100}$ | None of the listed relations holds | In fact $g=o(f)$, whereas the choices compare only $f$ with $g$ |
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

&nbsp;
**1.** $f=O(g)$ does not mean that $f$ is "equal" to a specific function; nor does it mean that this is the tightest upper bound.<br>
**2.** $f=\Theta(g)$ allows a constant-factor difference, whereas $f\sim g$ requires the limiting factor to be exactly $1$.<br>
**3.** A Big O proof must provide **one fixed** pair $c,x_0$; neither may vary with the input $x$.<br>
**4.** If $g$ has infinitely many zeros, quotient-based formulations require care. It is safer to check the eventual inequality definition directly.<br>
**5.** $4^n\ne O(2^n)$ because their ratio $2^n$ is unbounded; a constant difference between exponential bases cannot be absorbed into Big O.<br>
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
> Thus, $n\log n=o(n^{1.1})$.
> <!-- bilingual-en:end -->

> [!question] 自检 24-2
> 给出 $f=O(g)$ 但既不 $f=o(g)$、也不 $f=\Theta(g)$ 的例子。
> <!-- bilingual-en:start -->
> Give an example in which $f=O(g)$ but neither $f=o(g)$ nor $f=\Theta(g)$.
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 在正整数上令 $f(n)=n$（$n$ 为偶数）而 $f(n)=1$（$n$ 为奇数），令 $g(n)=n$。有 $f\le g$，故 $f=O(g)$；商在 $1$ 与 $1/n$ 间振荡，不趋零；反向 $g=O(f)$ 在奇数点失败，故不为 $\Theta$。
> <!-- bilingual-en:start -->
> On the positive integers, let $f(n)=n$ for even $n$ and $f(n)=1$ for odd $n$, and set $g(n)=n$. Since $f\le g$, we have $f=O(g)$. The ratio $f/g$ alternates between $1$ and $1/n$, so it does not tend to zero. Conversely, $g=O(f)$ fails along the odd integers, so $f$ is not $\Theta(g)$.
> <!-- bilingual-en:end -->

> [!question] 自检 24-3
> “每次循环是 $O(1)$，循环 $n$ 次，所以总计仍是 $O(1)$”错在哪里？
> <!-- bilingual-en:start -->
> "Every loop is $O(1)$, $n$ loops, so the total is still $O(1)$." Where is the error?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 即使每次都有统一常数 $C$，总成本也至多 $nC=O(n)$；不能把 $O(1)$ 当成相加后仍不变的数字。若每次隐藏常数还依赖循环下标，错误更严重。
> <!-- bilingual-en:start -->
> Even if every iteration is bounded by the same constant $C$, the total cost is at most $nC=O(n)$. The notation $O(1)$ cannot be added as though it were a number that remains unchanged. The error is worse if the hidden constant also depends on the loop index.
> <!-- bilingual-en:end -->

### Classroom Problems 24 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 24 — complete independent solutions to five problems*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp24.pdf#page=1|cp24 pp. 1–3]]

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
> The course definition of $f=O(g)$ requires $c,n_0\in\mathbb N$ such that $|f(n)|\le c g(n)$ for every $n\ge n_0$.
> **(a)** For $f=n^2$ and $g=3n$, the ratio $f/g=n/3$ is unbounded, so $f\ne O(g)$. In the reverse direction we need $3n\le cn^2$. The smallest positive integer is $c=1$, and then the least valid threshold is $n_0=3$.
> **(b)** For $f=(3n-7)/(n+4)$ and $g=4$, we have $|f|\le4$ for all $n\ge0$, so the smallest values for $f=O(g)$ are $c=1,n_0=0$. Conversely, $4\le c|f|$ cannot hold eventually with $c=1$ because $f\to3$. With the smallest possible $c=2$, the inequality holds exactly from $n=15$ onward, so the least threshold is $n_0=15$.
> **(c)** For $f=1+[n\sin(n\pi/2)]^2$ and $g=3n$, we have $f=1$ on even $n$ and $f=1+n^2$ on odd $n$. Along the odd subsequence $f/g$ is unbounded, so $f\ne O(g)$; along the even subsequence $g/f=3n$ is unbounded, so $g\ne O(f)$. Oscillation can make both directions fail.
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
> | Relation | Classification | Key reason |
> |---|---|---|
> | $f\sim g$ | Equivalence relation (E) | Reflexive, symmetric, and transitive |
> | $f=o(g)$ | Strict partial order (S) | Irreflexive and transitive; reverse Big O is impossible |
> | $f=O(g)$ | Neither (N) | A preorder, but $f=n,g=2n$ violates antisymmetry |
> | $f=\Theta(g)$ | Equivalence relation (E) | Defined by mutual Big O bounds |
> | $f=O(g)$ and $g\ne O(f)$ | Strict partial order (S) | Irreflexive; transitivity follows from Big O while the reverse relation is excluded |
> **(b) Main implications:** $f\sim g\Rightarrow f=\Theta(g)\Rightarrow[f=O(g)\land g=O(f)]$, and $f=o(g)\Rightarrow[f=O(g)\land g\ne O(f)]$. The latter strict Big O relation does not imply little o; the oscillating function from Self-test 24-2 is a counterexample.
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
>
> &nbsp;
> **1.** $n^2\sim n^2+n$: **true**, because $n^2/(n^2+n)=1/(1+1/n)\to1$.<br>
> **2.** $3^n=O(2^n)$: **false**, because $(3/2)^n\to\infty$.<br>
> **3.** $n^{\sin(n\pi/2)+1}=o(n^2)$: **false**. When $n\equiv1\pmod4$, the exponent is $2$, so the ratio equals $1$ and cannot tend to zero.<br>
> **4.** $n=\Theta\!\left(\dfrac{3n^3}{(n+1)(n-1)}\right)$: **true**. For $n\ge2$, the expression on the right is asymptotic to $3n$, and it can also be bounded directly between positive constant multiples of $n$.<br>
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
> For the upper bound, every factor satisfies $i\le n$, so $\log(n!)=\sum_{i=1}^n\log i\le n\log n$.
> For the lower bound, keep only the last $\lfloor n/2\rfloor$ terms; each is at least $\log(n/2)$. For $n\ge4$, $\log n-\log2\ge\frac12\log n$, giving $\frac14n\log n\le\log(n!)\le n\log n$. Hence $\boxed{\log(n!)=\Theta(n\log n)}$.
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
> **Goal:** Find and verify a closed form. The result is the displayed square of the sum of the first $n$ positive integers. For $n=1$, both sides equal $1$. Assuming the formula holds for $n$, adding $(n+1)^3$ and factoring gives the corresponding formula for $n+1$. Hence it holds for every $n\ge1$.
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
> Let $N=n^2$. Stirling's formula gives the displayed expansion for $\ln(N!)$. Substituting $N=n^2$ makes the leading term $2n^2\ln n$; every remaining term tends to zero after division by $n^2\ln n$. Therefore $\ln((n^2)!)\sim2n^2\ln n$ and hence $\ln((n^2)!)=\Theta(n^2\ln n)$. Here $n$ is a positive integer; changing to any fixed logarithm base greater than $1$ only multiplies the expression by a constant.
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
> The function $f(x)=x^6$ is positive and increasing, so the integral method gives the displayed bounds. When $n\ge2$, the lower bound is at least $n^7/7$ and the upper bound is at most $n^7/7+n^7=8n^7/7$. Thus $c_1=1/7,c_2=8/7,n_0=2$ satisfy the definition of Theta exactly.
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
The [[无环图：树、生成树、DAG 与拓扑排序#DAG 与拓扑排序|DAG]] has edges
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
> **(a)** The transitive comparabilities include $A<D<E<G$, $B<E<G$, $B<F<H$, and $C<F<H$. The width is $3$, and the two maximum antichains are $\{A,B,C\}$ and $\{B,C,D\}$. A four-element antichain is impossible: the chain cover $A<D<E<G$, $B$, and $C<F<H$ shows that any antichain contains at most one element from each of three chains.
> **(b)** With unlimited processors, the minimum completion time equals the number of tasks on a longest chain. The chain $A\to D\to E\to G$ gives a lower bound of $4$, and the displayed four-step schedule achieves it. Thus the answer is $\boxed{4}$.
> **(c)** With at most two parallel tasks, both the workload bound $\lceil8/2\rceil=4$ and the critical-path bound equal $4$. The displayed schedule satisfies every prerequisite in four steps, so the answer remains $\boxed{4}$.
> <!-- bilingual-en:end -->

### Problem 2 — Partial Orders & Equivalence（20 分）
<!-- bilingual-en:start -->
*Problem 2 — Partial Orders & Equivalence (20 points)*
<!-- bilingual-en:end -->

首次回链 [[02_Structures#Session 18 — Partial Orders and Equivalence|等价关系]] 与 [[02_Structures#Session 18 — Partial Orders and Equivalence|弱偏序]]。
<!-- bilingual-en:start -->
This problem first reconnects to [[02_Structures#Session 18 — Partial Orders and Equivalence|equivalence relations]] and [[02_Structures#Session 18 — Partial Orders and Equivalence|weak partial orders]].
<!-- bilingual-en:end -->

> [!answer]- 完整解答
> **(a)** 唯一候选是恒等关系
> $$
> I_A=\{(a,a):a\in A\},
> $$
> 即 $aI_Ab\iff a=b$。它自反、对称、传递，所以是等价关系；也自反、反对称、传递，所以是弱偏序。
> **(b) 唯一性证明**：设 $R$ 同时是等价关系和弱偏序。若 $aRb$，等价关系的对称性给 $bRa$；弱偏序的反对称性于是给 $a=b$。故 $R\subseteq I_A$。另一方面，$R$ 的自反性保证每个 $(a,a)\in R$，故 $I_A\subseteq R$。两边合并，$R=I_A$。
> <!-- bilingual-en:start -->
> **(a)** The only candidate is the identity relation $I_A$, defined by $aI_Ab\iff a=b$. It is reflexive, symmetric, and transitive, so it is an equivalence relation; it is also reflexive, antisymmetric, and transitive, so it is a weak partial order.
> **(b) Uniqueness proof:** Suppose $R$ is both an equivalence relation and a weak partial order. If $aRb$, symmetry gives $bRa$, and antisymmetry then gives $a=b$. Hence $R\subseteq I_A$. Conversely, reflexivity places every $(a,a)$ in $R$, so $I_A\subseteq R$. Therefore $R=I_A$.
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
This problem reconnects to [[无环图：树、生成树、DAG 与拓扑排序#Tree 的等价刻画|trees]] and [[图着色与色数#如何证明 chromatic number|graph colouring]].
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
This problem reconnects to [[02_Structures#Session 22 — Stable Matching and Hall's Theorem|stable matching]]. Here an unmarried man and a married woman who prefers him to her current partner also count as a rogue couple, and there may be more men than women.
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
> The invariants are $\boxed{(a),(d),(g)}$. Check each property under one transition:
> - **(a)** Alice already has a suitor she prefers to Bob. A woman keeps only her favourite proposal so far, and that retained suitor can only improve, so the property persists.
> - **(b)** Bob's list contains only Alice. If Alice rejects him, she is deleted and the list becomes empty, so the property need not persist.
> - **(c)** Alice has no suitor. Someone may propose to her next, so the property need not persist.
> - **(d)** Bob prefers Alice to the woman he is currently pursuing. After rejection, a man moves only downward through his preference list, so every later target is still less preferred than Alice; the property persists.
> - **(e)** Bob is currently proposing to Alice. She may reject him, so the property need not persist.
> - **(f)** Bob is not proposing to Alice. Rejections from women he ranks above Alice may eventually bring him to her, so the property need not persist.
> - **(g)** Bob's list is empty. Entries are deleted but never restored, so the property persists.
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
*Midterm 3 Error Diagnosis*
<!-- bilingual-en:end -->

- 调度时间看最长**链的顶点数**，不是边数；并行上限还要同时看总工作量。
- “对称 + 反对称”不是矛盾；它们合在一起把关系压缩到对角线。
- 两条路径保证某处有环，不保证端点在环上。
- 树着色归纳必须说明删叶后仍是树，以及扩展恰为 $n-1$ 对一。
- 不变量题问的是每次转移后的保持性，不是某个状态下偶然为真。
<!-- bilingual-en:start -->
- Scheduling time is governed by the **number of vertices on a longest chain**, not the number of edges; a bound on parallelism also introduces a total-work lower bound.
- Symmetry and antisymmetry are not contradictory. Together they force every related pair onto the diagonal.
- Two distinct paths guarantee a cycle somewhere in the graph, but not necessarily a cycle through the endpoints.
- An induction for tree colouring must show that deleting a leaf leaves a tree and that each colouring extends in exactly $n-1$ ways.
- An invariant question asks whether a property survives every transition, not whether it happens to hold in one state.
<!-- bilingual-en:end -->

---

## Session 25 — Counting with Bijections

### 本节问题、前置知识与资源
<!-- bilingual-en:start -->
*Questions, Prerequisites, and Resources for this Section*
<!-- bilingual-en:end -->

**学习问题**：一个对象能分成互斥情形时怎样相加？由连续选择构造时怎样相乘？当原对象难数时，怎样用一个可逆编码把它搬到熟悉集合？
<!-- bilingual-en:start -->
**Learning questions**: How should counts be added when objects split into mutually exclusive cases? How should choices be multiplied when an object is built through successive decisions? When the original objects are hard to count, how can a reversible encoding transfer the problem to a familiar set?
<!-- bilingual-en:end -->

**前置知识**：有限集合、函数、单射/满射/双射、笛卡尔积。首次正式使用 [[组合计数原理#加法、乘法与双射|加法与乘法法则]]、[[组合计数原理#加法、乘法与双射|双射计数原理]] 与 [[组合计数原理#加法、乘法与双射|计数策略选择框架]]。
<!-- bilingual-en:start -->
**Prerequisites**: finite sets, functions, injections, surjections, bijections, and Cartesian products. This session gives the first formal use of the [[组合计数原理#加法、乘法与双射|sum and product rules]], the [[组合计数原理#加法、乘法与双射|bijection rule]], and the [[组合计数原理#加法、乘法与双射|framework for choosing a counting strategy]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session25.pdf#page=1|Session 25 reading, pp. 1–6]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp25.pdf#page=1|cp25, pp. 1–3]]

### 3.3.1 Sum and Product Rules — 先确认集合结构
<!-- bilingual-en:start -->
*3.3.1 Sum and Product Rules — Identify the Set Structure First*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_SumProduct.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/yTrtVwKZkwU.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=yTrtVwKZkwU)

#### 加法法则（Sum Rule）
<!-- bilingual-en:start -->
*Sum Rule*
<!-- bilingual-en:end -->

若有限集合 $A_1,\ldots,A_k$ **两两不交**，则
<!-- bilingual-en:start -->
If the finite sets $A_1,\ldots,A_k$ are **pairwise disjoint**, then
<!-- bilingual-en:end -->

$$
\boxed{\left|\bigcup_{i=1}^{k}A_i\right|
=\sum_{i=1}^{k}|A_i|}.
$$

证明很直接但条件不可省：并集中的每个对象恰属于一个 $A_i$，所以右边恰数一次。若集合重叠，右边会重复计数，需等到 Session 27 用容斥修正。
<!-- bilingual-en:start -->
The proof is simple, but the condition is essential: every object in the union belongs to exactly one $A_i$, so the right-hand side counts it once. If the sets overlap, the sum double-counts some objects and must be corrected by inclusion–exclusion in Session 27.
<!-- bilingual-en:end -->

#### 乘法法则（Product Rule）
<!-- bilingual-en:start -->
*Product Rule*
<!-- bilingual-en:end -->

对有限集合 $A_1,\ldots,A_k$，
<!-- bilingual-en:start -->
For finite sets $A_1,\ldots,A_k$,
<!-- bilingual-en:end -->

$$
\boxed{|A_1\times\cdots\times A_k|
=\prod_{i=1}^{k}|A_i|}.
$$

证明可对 $k$ 归纳。$k=2$ 时，对每个 $a_1\in A_1$ 恰有 $|A_2|$ 个有序对 $(a_1,a_2)$，共 $|A_1||A_2|$；归纳步把前 $k-1$ 个坐标视作一个整体。
<!-- bilingual-en:start -->
The proof proceeds by induction on $k$. When $k=2$, each $a_1\in A_1$ can be paired with exactly $|A_2|$ choices of $a_2$, giving $|A_1||A_2|$ ordered pairs. For the induction step, treat the first $k-1$ coordinates as one combined choice.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-sum-product-rule.png|900]]

读图：互斥分支对应并集，所以把各分支数量相加；连续决策对应笛卡尔积，所以把每一步可选数量相乘。判断“分支还是步骤”比记公式更重要。
<!-- bilingual-en:start -->
How to read the diagram: mutually exclusive branches form a union, so their counts are added; successive decisions form a Cartesian product, so the number of choices at each step is multiplied. Deciding whether the structure is “cases” or “steps” matters more than memorising a formula.
<!-- bilingual-en:end -->

#### 例：密码
<!-- bilingual-en:start -->
*Example: Password*
<!-- bilingual-en:end -->

密码长度为 6、7 或 8；首字符必须是 52 个大小写字母之一，其余位置可用 62 个字母或数字。不同长度互斥，所以
<!-- bilingual-en:start -->
A password has length 6, 7, or 8. Its first character must be one of 52 uppercase or lowercase letters, while each remaining position may contain any of 62 letters or digits. The three lengths are mutually exclusive, so
<!-- bilingual-en:end -->

$$
52\cdot62^5+52\cdot62^6+52\cdot62^7
=\boxed{52(62^5+62^6+62^7)}.
$$

### 3.3.2 Counting Practice — 官方在线题 O25-01、O25-02
<!-- bilingual-en:start -->
*3.3.2 Counting Practice — Official Online Questions O25-01 and O25-02*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.2_counting-practice|3.3.2]]

> [!success]- 官方答案与反馈
> **O25-01**：5 位 PIN 每位有 10 种选择，前导零允许，所以 $10^5=\boxed{100000}$。
> **O25-02**：上衣有 $3+2=5$ 种，下装有 $4+4=8$ 种，搭配数 $5\cdot8=\boxed{40}$。
> <!-- bilingual-en:start -->
> **O25-01**: A five-digit PIN has 10 choices in each position, and leading zeros are allowed, so there are $10^5=\boxed{100000}$ PINs.
> **O25-02**: There are $3+2=5$ choices of top and $4+4=8$ choices of bottoms, giving $5\cdot8=\boxed{40}$ outfits.
> <!-- bilingual-en:end -->

### 3.3.3 Counting with Bijections — 把“难数”变成“已知可数”
<!-- bilingual-en:start -->
*3.3.3 Counting with Bijections — Transforming a Difficult Set into a Familiar One*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bijections.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/n0lce1dMAh8.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=n0lce1dMAh8)

#### 双射法则
<!-- bilingual-en:start -->
*The bijection rule*
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
*Stars and Bars: Choosing with Repetition*
<!-- bilingual-en:end -->

从 $k$ 种甜甜圈中选 $n$ 个，令 $x_i$ 是第 $i$ 种数量，则需数
<!-- bilingual-en:start -->
To choose $n$ doughnuts from $k$ flavours, let $x_i$ be the number chosen of flavour $i$. The problem is to count the solutions of
<!-- bilingual-en:end -->

$$
x_1+\cdots+x_k=n,\qquad x_i\ge0.
$$

编码为 $n$ 个星与 $k-1$ 个隔板：
<!-- bilingual-en:start -->
Encode a solution using $n$ stars and $k-1$ bars:
<!-- bilingual-en:end -->

$$
\underbrace{\star\cdots\star}_{x_1}\mid
\underbrace{\star\cdots\star}_{x_2}\mid\cdots\mid
\underbrace{\star\cdots\star}_{x_k}.
$$

总位置 $n+k-1$，任选 $k-1$ 个放隔板（或任选 $n$ 个放星），故
<!-- bilingual-en:start -->
There are $n+k-1$ positions in total. Choose $k-1$ of them for the bars, or equivalently choose $n$ for the stars, so
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
Let $A=\{a_1,\ldots,a_m\}$ and $|B|=q$. A total function $f:A\to B$ is represented by the vector
<!-- bilingual-en:end -->

$$
(f(a_1),\ldots,f(a_m))\in B^m
$$

双射，因此共有
<!-- bilingual-en:start -->
This representation is a bijection, so there are
<!-- bilingual-en:end -->

$$
\boxed{q^m=|B|^{|A|}}
$$

个。每个定义域元素都是一个独立位置，每个位置有 $q$ 种选择。
<!-- bilingual-en:start -->
total functions. Each element of the domain is an independent position with $q$ possible images.
<!-- bilingual-en:end -->

### 3.3.4 Selecting Donuts — 官方在线题 O25-03
<!-- bilingual-en:start -->
*3.3.4 Selecting Donuts — official online O25-03*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.4_selecting-donuts|3.3.4]]

> [!success]- 官方答案与反馈
> 选 13 个、4 种口味对应 13 个零（甜甜圈）与 3 个一（隔板），故与“长度 16、恰有 3 个 1 的二进制串”双射。答案为 $\binom{16}{3}$ 个。
> <!-- bilingual-en:start -->
> Choosing 13 doughnuts from four flavours corresponds to a string of 13 zeros (doughnuts) and three ones (bars). This is a bijection with length-16 bit strings containing exactly three ones, so the answer is $\binom{16}{3}$.
> <!-- bilingual-en:end -->

### 3.3.5 Counting Functions — 官方在线题 O25-04
<!-- bilingual-en:start -->
*3.3.5 Counting Functions — official online O25-04*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S25_3.3.5_counting-functions|3.3.5]]

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

&nbsp;
**1.** The addition rule requires that the categories are mutually exclusive; "shirt" and "red jacket" overlap and cannot be added directly.<br>
**2.** The product rule counts ordered sequences of decisions. If the final object forgets that order, use a bijection or the division rule to remove duplicates.<br>
**3.** A bijection argument cannot stop at “the correspondence is obvious”; it must give the inverse map, or the proposed map may be many-to-one.<br>
**4.** Stars and Bars directly handles nonnegative integer solutions. If each class must contain at least one object, first set $y_i=x_i-1$.<br>
**5.** PINs and strings usually allow leading zeros, while five-digit integers do not. These are different sample spaces.<br>
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
> Choose the three positions occupied by 1s: $\binom83=56$.
> <!-- bilingual-en:end -->

> [!question] 自检 25-2
> 非负整数解 $x_1+x_2+x_3=10$ 有多少个？若每个 $x_i\ge1$ 呢？
> <!-- bilingual-en:start -->
> How many nonnegative integer solutions does $x_1+x_2+x_3=10$ have? What if every $x_i\ge1$?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 非负解 $\binom{12}{2}=66$。正整数解令 $y_i=x_i-1$，则 $y_1+y_2+y_3=7$，有 $\binom92=36$。
> <!-- bilingual-en:start -->
> There are $\binom{12}{2}=66$ nonnegative solutions. For positive solutions, set $y_i=x_i-1$; then $y_1+y_2+y_3=7$, giving $\binom92=36$ solutions.
> <!-- bilingual-en:end -->

> [!question] 自检 25-3
> 为什么“从 5 人选主席和副主席”是 $5\cdot4$，不是 $\binom52$？
> <!-- bilingual-en:start -->
> Why are there $5\cdot4$ ways to choose a chair and vice-chair from five people, rather than $\binom52$?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 两个职位有角色顺序；同一对人交换职位产生不同结果。先选主席 5 种，再从剩余人选副主席 4 种。
> <!-- bilingual-en:start -->
> The two offices are distinct, so swapping the same two people gives a different outcome. There are 5 choices for chair and then 4 choices for vice-chair.
> <!-- bilingual-en:end -->

### Classroom Problems 25 — 4 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 25 — complete independent solutions to four problems*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp25.pdf#page=1|cp25 pp. 1–3]]

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
> **(a)** Write each integer from $1$ to $999{,}999{,}999$ as a nine-digit string with leading zeros. Each position has nine choices if the digit 1 is forbidden, giving $9^9$ strings, but the all-zero string is not a positive integer. Thus $9^9-1$ integers in that range avoid 1, while $10^9$ itself contains 1. Among the $10^9$ integers in total, the answer is $10^9-9^9+1=612{,}579{,}512$. If the sample space were instead $0$ through $10^9-1$, the answer would be smaller by one; this explains the common $10^9-9^9$ version.
> **(b)** Let the selected book positions be $1\le i_1<\cdots<i_6\le20$ with $i_{r+1}\ge i_r+2$, and define $j_r=i_r-(r-1)$. Then $1\le j_1<\cdots<j_6\le15$. Marking positions $j_1,\ldots,j_6$ with 1s gives a length-15 bit string containing six 1s. The inverse $i_r=j_r+(r-1)$ restores gaps of at least two, so this is a bijection and the count is $\boxed{\binom{15}{6}}$.
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
> **Why the choice exists and is unique:** In the remaining tree, the labels absent from the remaining code are exactly the vertices eligible to be deleted as leaves before ever appearing as a recorded neighbour. Choosing the largest one reproduces the encoding rule. Each forward deletion is undone by one inverse step, so the procedure reconstructs a unique tree. Thus labelled trees are in bijection with $\{1,\ldots,n\}^{n-2}$ and their number is $n^{n-2}$.
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
> **(b)** For a weakly increasing sequence $0\le y_1\le\cdots\le y_k\le n$, set $x_1=y_1$ and $x_i=y_i-y_{i-1}$ for $i\ge2$. Then every $x_i\ge0$ and $\sum_{i=1}^kx_i=y_k\le n$, giving an element of $S_{n,k}$. The inverse map is $y_i=\sum_{j=1}^ix_j$. Hence the two sets are bijective and both have size $\binom{n+k}{k}$.
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
> **(a)** A binary relation is an arbitrary subset of $X\times Y$, so there are $\boxed{2^{mq}}$ relations.
> **(b)** Fix an ordering $X=(x_1,\ldots,x_m)$. The map $f\mapsto(f(x_1),\ldots,f(x_m))$ is a bijection from the set of total functions $X\to Y$ to $Y^m$, so there are $\boxed{q^m}$ such functions.
> **(c)** For each $x$, a partial function has $q$ possible images plus the additional choice “undefined,” giving $\boxed{(q+1)^m}$ functions. The ratio to the number of total functions is $(1+1/q)^m$. For fixed $q$, it grows exponentially and is $O(2^m)$, not $O(1)$ or $O(m)$.
> **(d)** Map a subset $S\subseteq X$ to its characteristic function $\mathbf1_S:X\to\{0,1\}$. The inverse sends $f$ to $f^{-1}(1)$, so $|\operatorname{pow}(X)|=2^m$.
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
**Learning questions**: How can a many-to-one map remove overcounting safely? Why do permutations with repeated letters, unlabeled groupings, binomial coefficients, and multinomial coefficients all share the same structure?
<!-- bilingual-en:end -->

**前置知识**：Session 25 的和、积、双射；阶乘；集合与序列。
<!-- bilingual-en:start -->
**Prerequisites**: the sum rule, product rule, and bijections from Session 25; factorials; sets; and sequences.
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session26.pdf#page=1|Session 26 reading, pp. 1–15]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp26.pdf#page=1|cp26, pp. 1–4]]

### 3.4.1 Generalized Counting Rules — 依赖选择与除法
<!-- bilingual-en:start -->
*3.4.1 Generalized Counting Rules — Dependent Choices and Division*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Generalized.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/iDfyX8WRIyM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=iDfyX8WRIyM)

#### 广义乘法法则
<!-- bilingual-en:start -->
*The generalized product rule*
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
*Circular permutations*
<!-- bilingual-en:end -->

$n$ 个不同对象排成圆，若旋转视为相同，则线性排列到圆排列是 $n$-to-1，数量
<!-- bilingual-en:start -->
Arrange $n$ distinct objects on a circle and regard rotations as identical. The map from linear permutations to circular arrangements is $n$-to-1, so the number of arrangements is
<!-- bilingual-en:end -->

$$
\boxed{\frac{n!}{n}=(n-1)!}.
$$

### 3.4.2 Choosing Integers — 官方在线题 O26-01
<!-- bilingual-en:start -->
*3.4.2 Choosing Integers — official online O26-01*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S26_3.4.2_choosing-integers|3.4.2]]

> [!success]- 官方答案与反馈
> 闭区间 $[3,15]$ 含 $15-3+1=13$ 个整数，任选 2 个有
> $$
> \boxed{\binom{13}{2}=78}
> $$
> 种。
> <!-- bilingual-en:start -->
> The closed interval $[3,15]$ contains $15-3+1=13$ integers. Choosing any two gives $\boxed{\binom{13}{2}=78}$ possibilities.
> <!-- bilingual-en:end -->

### 3.4.3 Two Pair Poker Hands — 先选结构，再选位置
<!-- bilingual-en:start -->
*3.4.3 Two-Pair Poker Hands — Choose the Structure, Then the Suits*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_TwoPairPoker.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HswnmlLPGZ4.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=HswnmlLPGZ4)

标准 52 张牌有 13 个点数、每点数 4 个花色。计数时把“点数角色”与“花色位置”分开：
<!-- bilingual-en:start -->
A standard 52-card deck has 13 ranks and four suits for each rank. Count the roles played by the ranks separately from the choices of suits:
<!-- bilingual-en:end -->

- 四条：选四条点数 $13$；选边牌点数 $12$；选边牌花色 $4$：
  $$13\cdot12\cdot4=624.$$
- 葫芦：选三条点数与 3 个花色，再选对子点数与 2 个花色：
  $$13\binom43\cdot12\binom42.$$
- 两对：先选两个对子点数（无序），为各对选 2 花色，再选第五张点数与花色：
  $$\boxed{\binom{13}{2}\binom42^2\cdot11\cdot4}.$$
<!-- bilingual-en:start -->
- Four of a kind: choose its rank in $13$ ways, the kicker's rank in $12$ ways, and the kicker's suit in $4$ ways:
  $$13\cdot12\cdot4=624.$$
- Full house: choose the rank and three suits for the triple, then the rank and two suits for the pair:
  $$13\binom43\cdot12\binom42.$$
- Two pair: choose the two pair ranks as an unordered set, choose two suits for each pair, then choose the fifth card's rank and suit:
  $$\boxed{\binom{13}{2}\binom42^2\cdot11\cdot4}.$$
<!-- bilingual-en:end -->

若先把两个对子点数按“第一对/第二对”有序选择，会把每手牌数两次，必须除以 $2!$。
<!-- bilingual-en:start -->
If the two pairs are selected in an ordered first-pair/second-pair procedure, each hand is counted twice, so divide by $2!$ once.
<!-- bilingual-en:end -->

### 3.4.4 Binomial Theorem — 系数就是选择位置
<!-- bilingual-en:start -->
*3.4.4 Binomial Theorem — Coefficients Count Choices of Positions*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_BinomialTheo.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/jwjDj4GoSV0.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=jwjDj4GoSV0)

[[组合计数原理#二项式、鸽巢与容斥|二项式定理]]：对非负整数 $n$，
<!-- bilingual-en:start -->
The [[组合计数原理#二项式、鸽巢与容斥|binomial theorem]] states that, for every nonnegative integer $n$,
<!-- bilingual-en:end -->

$$
\boxed{(a+b)^n=\sum_{k=0}^{n}\binom nk a^{n-k}b^k}.
$$

**完整组合证明**：把左边写成 $n$ 个因子的积。展开中的每一项都从每个因子选一次 $a$ 或 $b$。要得到 $a^{n-k}b^k$，必须在恰好 $k$ 个因子位置选 $b$；位置集合有 $\binom nk$ 种。每种选择贡献同一个单项式，故其系数为 $\binom nk$。所有 $k=0,\ldots,n$ 情形互斥且穷尽，定理成立。
<!-- bilingual-en:start -->
**Complete combinatorial proof:** Write the left-hand side as a product of $n$ factors. Each term in the expansion chooses either $a$ or $b$ from every factor. To obtain $a^{n-k}b^k$, choose $b$ from exactly $k$ factor positions; there are $\binom nk$ such position sets. Every choice contributes the same monomial, so its coefficient is $\binom nk$. The cases $k=0,\ldots,n$ are disjoint and exhaustive.
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-binomial-paths.png|900]]

读图：从格点起点到终点的每条最短路径对应一个由“横步/竖步”组成的二进制串；固定竖步数就是选择这些步出现的位置，因此路径数与二项式系数是同一个计数对象。
<!-- bilingual-en:start -->
How to read the diagram: every shortest lattice path corresponds to a binary string of horizontal and vertical steps. Fixing the number of vertical steps amounts to choosing their positions, so counting paths and computing a binomial coefficient are the same counting problem.
<!-- bilingual-en:end -->

#### Pascal 恒等式
<!-- bilingual-en:start -->
*Pascal's identity*
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
How to read the diagram: stars represent identical objects being distributed, and bars divide them into the amounts assigned to each variable. Choosing the bar positions is an ordinary subset choice, so the number of nonnegative integer solutions is $\binom{n+k-1}{k-1}$.
<!-- bilingual-en:end -->

### 3.4.5 Multinomial Theorem — 多种选择的统一形式
<!-- bilingual-en:start -->
*3.4.5 Multinomial Theorem — A Unified Form for Several Choices*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_Bookkeeper.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/juGgfHsO-xM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=juGgfHsO-xM)

[[组合计数原理#排列、组合与重复|多项式定理（multinomial theorem）]]从多类位置分配出发。若 $k_1+\cdots+k_m=n$，把 $n$ 个不同位置分成大小分别为 $k_1,\ldots,k_m$ 的有标号组，方式数为多项式系数
<!-- bilingual-en:start -->
The [[组合计数原理#排列、组合与重复|multinomial theorem]] begins with assigning positions to several labelled classes. If $k_1+\cdots+k_m=n$, the number of ways to divide $n$ distinct positions into labelled groups of sizes $k_1,\ldots,k_m$ is the multinomial coefficient
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
The multinomial theorem is
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

&nbsp;
**1.** The division rule requires every target object to have exactly the same number of preimages; “most have $k$” is not enough.<br>
**2.** Circular permutations identify rotations, not reflections. If mirror images are also identical, analyse that additional symmetry separately.<br>
**3.** Use Stars and Bars for identical objects chosen with repetition; use $\binom nk$ for subsets of distinct objects.<br>
**4.** The two pair ranks in a two-pair poker hand are unlabeled. Choosing them in order counts every hand twice.<br>
**5.** The lower indices of a multinomial coefficient must sum to the total degree; otherwise the corresponding monomial has coefficient $0$.<br>
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
> There are 11 letters: four $I$s, four $S$s, two $P$s, and one $M$. Therefore the number of distinct permutations is
> $$\frac{11!}{4!4!2!}.$$
> <!-- bilingual-en:end -->

> [!question] 自检 26-2
> 求 $(2x-y)^5$ 中 $x^3y^2$ 的系数。
> <!-- bilingual-en:start -->
> Find the coefficient of $x^3y^2$ in $(2x-y)^5$.
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
> How many seatings are there for 10 distinct students around a round table if rotations are identical but reflections are distinct?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 每个圆排列对应 10 个线性旋转，故 $10!/10=9!$。
> <!-- bilingual-en:start -->
> Each circle arrangement corresponds to 10 linear rotations, so $10!/10=9!$.
> <!-- bilingual-en:end -->

### Classroom Problems 26 — 5 题完整独立题解
<!-- bilingual-en:start -->
*Classroom Problems 26 — complete independent solutions to five problems*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp26.pdf#page=1|cp26 pp. 1–4]]

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
> **(a)** The letters in `POKE` are distinct, so there are $4!=24$ permutations.
> **(b)** The symbols in `BO₁O₂K` are distinct, so there are $4!=24$ permutations.
> **(c)** Erase the subscripts on the two O's. For example, `O₂BO₁K` and `O₁BO₂K` both map to `OBOK`; `KO₂BO₁` and `KO₁BO₂` both map to `KOBO`; and `BO₁O₂K` and `BO₂O₁K` both map to `BOOK`.
> **(d)** Every unsubscripted arrangement has exactly $2!$ preimages, so the map is $2$-to-1.
> **(e)** `BOOK`: $4!/2!=12$.
> **(f)** The six symbols in `KE₁E₂PE₃R` are distinct, so there are $6!=720$ permutations.
> **(g)** The six preimages of `REPEEK` are
> `RE₁PE₂E₃K`,`RE₁PE₃E₂K`,`RE₂PE₁E₃K`,`RE₂PE₃E₁K`,`RE₃PE₁E₂K`,`RE₃PE₂E₁K`.
> **(h)** This map is $3!=6$-to-1.
> **(i)** `KEEPER`: $6!/3!=120$.
> **(j)** The ten symbols in `BO₁O₂K₁K₂E₁E₂PE₃R` are distinct, so there are $10!$ permutations.
> **(k)** Identifying the two O's gives $10!/2!$.
> **(l)** Identifying both O's and both K's gives $10!/(2!2!)$.
> **(m)** `BOOKKEEPER` has two O's, two K's, and three E's:
> $$\boxed{\frac{10!}{2!2!3!}=151200}.$$
> **(n)** `VOODOODOLL` has five O's, two D's, and two L's:
> $$\boxed{\frac{10!}{5!2!2!}=7560}.$$
> **(o)** A length-52 string with seventeen 2s, twenty-three 5s, and twelve 9s has
> $$\boxed{\frac{52!}{17!23!12!}}.$$
> <!-- bilingual-en:end -->

> [!example]- C26-4 三个系数
> **(a)** $(1+x)^{11}$ 中 $x^5$ 系数：$\boxed{\binom{11}{5}}$。
> **(b)** $(3x+2y)^{17}$ 中 $x^8y^9$ 系数：
> $$\boxed{\binom{17}{8}3^8 2^9}.$$
> **(c)** $(a^2+b^3)^5$ 中 $a^6b^6$：需选 3 个 $a^2$、2 个 $b^3$，系数 $\boxed{\binom53=10}$。
> <!-- bilingual-en:start -->
> **(a)** The coefficient of $x^5$ in $(1+x)^{11}$ is $\boxed{\binom{11}{5}}$.
> **(b)** The coefficient of $x^8y^9$ in $(3x+2y)^{17}$ is
> $$\boxed{\binom{17}{8}3^8 2^9}.$$
> **(c)** To obtain $a^6b^6$ in $(a^2+b^3)^5$, choose $a^2$ from three factors and $b^3$ from two. The coefficient is $\boxed{\binom53=10}$.
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
> **(a)** Assign nine people to five labelled tasks whose group sizes are $1,2,3,1,2$. The number of assignments is $\binom{9}{1,2,3,1,2}=9!/(1!2!3!1!2!)$.
> **(b)** Write each nonnegative integer below $10^6$ as a six-digit string with leading zeros. Choose the position of the unique digit 9 in six ways. The other five digits must sum to $8$; every component of a nonnegative solution is automatically at most $8$, so no extra digit bound is needed. There are $\binom{12}{4}$ such solutions, giving $\boxed{6\binom{12}{4}=2970}$ in total.
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
**Learning questions**: When the exact distribution is unknown, how can sheer quantity force a collision? When several “bad events” overlap, how can they be counted without omissions or duplicates?
<!-- bilingual-en:end -->

**前置知识**：函数与单射、组合数、二项式定理、加法法则。首次正式使用 [[组合计数原理#二项式、鸽巢与容斥|鸽巢原理]] 与 [[组合计数原理#二项式、鸽巢与容斥|容斥原理]]。
<!-- bilingual-en:start -->
**Prerequisites:** functions and injections, binomial coefficients, the binomial theorem, and the addition rule. This session gives the first formal use of the [[组合计数原理#二项式、鸽巢与容斥|pigeonhole principle]] and the [[组合计数原理#二项式、鸽巢与容斥|inclusion–exclusion principle]].
<!-- bilingual-en:end -->

- Reading：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session27.pdf#page=1|Session 27 reading, pp. 1–11]]
- Classroom Problems：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp27.pdf#page=1|cp27, pp. 1–3]]

### 3.5.1 The Pigeonhole Principle — 数量迫使碰撞
<!-- bilingual-en:start -->
*3.5.1 The Pigeonhole Principle — Quantity Forces a Collision*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS16_ThePigeonhol.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/4Dz4vNUxnZM.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=4Dz4vNUxnZM)

#### 基本形式
<!-- bilingual-en:start -->
*Basic form*
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
Every application must identify three things clearly:
<!-- bilingual-en:end -->

1. pigeons 是哪些对象；
2. holes 是哪些类别；
3. 函数怎样把每个对象唯一分配到一个类别。
<!-- bilingual-en:start -->

&nbsp;
**1.** which objects are the pigeons;<br>
**2.** which categories are the holes;<br>
**3.** how the function assigns each object to exactly one category.<br>
<!-- bilingual-en:end -->

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-pigeonhole-principle.png|900]]

读图：每只鸽子被函数送进一个巢；当鸽子数超过巢数，单射不可能，至少一个巢接收两个；广义形式把“两个”替换为平均负载的向上取整。
<!-- bilingual-en:start -->
How to read the diagram: a function sends each pigeon to a hole. If there are more pigeons than holes, the function cannot be injective, so at least one hole receives two pigeons. The general form replaces “two” by the ceiling of the average load.
<!-- bilingual-en:end -->

#### 广义形式
<!-- bilingual-en:start -->
*Generalised form*
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
objects. Equivalently, if $N>km$, some category contains at least $k+1$ objects.
<!-- bilingual-en:end -->

**证明**：反设每类至多 $k$ 个，则总对象数至多 $km$，与 $N>km$ 矛盾。取 $k=\lceil N/m\rceil-1$ 即得向上取整形式。
<!-- bilingual-en:start -->
**Proof:** If every category contained at most $k$ objects, the total would be at most $km$, contradicting $N>km$. Taking $k=\lceil N/m\rceil-1$ gives the ceiling form.
<!-- bilingual-en:end -->

> [!tip] 反向设计阈值
> 要保证某巢至少有 $r$ 个，最坏情况可以让每巢先放 $r-1$ 个，所以最小充分总数是
> $$
> (r-1)m+1.
> $$
> <!-- bilingual-en:start -->
> To guarantee that some hole contains at least $r$ objects, the worst case fills every hole with $r-1$ objects first. The smallest sufficient total is therefore
> <!-- bilingual-en:end -->

### 3.5.2 Rolling Dice — 官方在线题 O27-01
<!-- bilingual-en:start -->
*3.5.2 Rolling Dice — official online O27-01*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.2_rolling-dice|3.5.2]]

掷两枚骰子 25 次，记录和。可能的和是整数 $2,3,\ldots,12$，共
<!-- bilingual-en:start -->
Roll two dice 25 times and record each sum. The possible sums are the integers $2,3,\ldots,12$, giving
<!-- bilingual-en:end -->

$$
12-2+1=11
$$

个巢，故至少一个和出现 $\lceil25/11\rceil=3$ 次。
<!-- bilingual-en:start -->
holes. Therefore at least one sum appears $\lceil25/11\rceil=3$ times.
<!-- bilingual-en:end -->

> [!success]- 官方答案与勘误
> **O27-01**：$\boxed{3}$。官方反馈写作 $\lceil25/(12-2)\rceil=3$，分母少了端点计数的 $+1$；正确巢数是 11。这个笔误不改变最终答案，因为 $\lceil25/10\rceil$ 与 $\lceil25/11\rceil$ 都是 3。
> <!-- bilingual-en:start -->
> **O27-01:** $\boxed{3}$. The official feedback writes $\lceil25/(12-2)\rceil=3$, omitting the $+1$ required when counting both endpoints. The correct number of holes is 11. This typo does not change the final answer because both $\lceil25/10\rceil$ and $\lceil25/11\rceil$ equal 3.
> <!-- bilingual-en:end -->

### 3.5.3 Inclusion–Exclusion Example — 从 6042 模式看三集合
<!-- bilingual-en:start -->
*3.5.3 Inclusion–Exclusion Example — Three Sets from the Pattern 6042*
<!-- bilingual-en:end -->

资源：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_InculionExcl.pdf#page=1|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/51-b2mgZVNY.pdf#page=1|transcript]] · [video](https://www.youtube.com/watch?v=51-b2mgZVNY)

考虑数字 $0,1,\ldots,9$ 的排列，求至少含连续模式 `60`、`04`、`42` 之一的排列数。令 $P_{60},P_{04},P_{42}$ 分别表示含对应模式的集合。
<!-- bilingual-en:start -->
Consider permutations of the digits $0,1,\ldots,9$. Count those containing at least one of the consecutive patterns `60`, `04`, or `42`. Let $P_{60},P_{04},P_{42}$ denote the sets containing the corresponding patterns.
<!-- bilingual-en:end -->

- 单个模式视作一个块，与其余 8 个数字共 9 个对象，所以每个 $|P_x|=9!$。
- 任意两个模式同时出现时都可压成 8 个对象：不重叠的 `60` 与 `42` 是两个块；相接的 `60` 与 `04` 压成 `604`。故每个二重交集为 $8!$。
- 三者同时出现必须含 `6042`，把它视为一个块后共 7 个对象，所以三重交集为 $7!$。
<!-- bilingual-en:start -->
- Treating one pattern as a block leaves nine objects in total, so each $|P_x|=9!$.
- When any two patterns occur together, they can be compressed into 8 objects: the disjoint patterns `60` and `42` form two blocks, while the adjacent patterns `60` and `04` combine into the single block `604`. Therefore, each pairwise intersection has size $8!$.
- If all three patterns occur, they form `6042`. Treating this as one block leaves seven objects, so the triple intersection has size $7!$.
<!-- bilingual-en:end -->

容斥给出
<!-- bilingual-en:start -->
Inclusion–exclusion gives
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

#### 两集合公式的严格证明
<!-- bilingual-en:start -->
*A rigorous proof of the two-set formula*
<!-- bilingual-en:end -->

集合 $A\cup B$ 可写成不交并
<!-- bilingual-en:start -->
The set $A\cup B$ can be written as the disjoint union
<!-- bilingual-en:end -->

$$
A\mathbin{\dot\cup}(B\setminus A),
$$

所以 $|A\cup B|=|A|+|B\setminus A|$。另一方面，$B$ 是不交并
<!-- bilingual-en:start -->
Hence $|A\cup B|=|A|+|B\setminus A|$. On the other hand, $B$ is the disjoint union
<!-- bilingual-en:end -->

$$
(A\cap B)\mathbin{\dot\cup}(B\setminus A),
$$

所以 $|B\setminus A|=|B|-|A\cap B|$。代入即得
<!-- bilingual-en:start -->
Thus $|B\setminus A|=|B|-|A\cap B|$. Substitution gives
<!-- bilingual-en:end -->

$$
\boxed{|A\cup B|=|A|+|B|-|A\cap B|}.
$$

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit03-inclusion-exclusion.png|900]]

读图：直接相加 $|A|+|B|$ 会把交集数两次；减去一次交集后，每个并集元素恰被数一次。三集合及一般形式继续按“奇数层加、偶数层减”修正。
<!-- bilingual-en:start -->
How to read the diagram: adding $|A|+|B|$ counts every element of the intersection twice. Subtracting $|A\cap B|$ once makes every element of the union count exactly once. For three or more sets, continue the correction by adding odd-order intersections and subtracting even-order intersections.
<!-- bilingual-en:end -->

#### 一般容斥公式及完整计数证明
<!-- bilingual-en:start -->
*The general inclusion–exclusion formula and an elementwise proof*
<!-- bilingual-en:end -->

对有限集合 $A_1,\ldots,A_n$，
<!-- bilingual-en:start -->
For finite sets $A_1,\ldots,A_n$,
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
Fix an element $x$. If it belongs to none of the sets, it contributes zero to both sides. If it belongs to exactly $r\ge1$ sets, then it appears in $\binom rj$ of the $j$-fold intersections on the right. Its total weight is therefore
<!-- bilingual-en:end -->

$$
\sum_{j=1}^{r}(-1)^{j+1}\binom rj
=1-\sum_{j=0}^{r}(-1)^j\binom rj
=1-(1-1)^r=1.
$$

所以每个并集元素恰被计一次，外部元素不计；两边逐元素相同，公式得证。
<!-- bilingual-en:start -->
Thus every element of the union is counted exactly once, while every outside element is counted zero times. The two sides agree element by element, proving the formula.
<!-- bilingual-en:end -->

### 3.5.5 Pigeonhole Principle — 官方在线题 O27-02 至 O27-06
<!-- bilingual-en:start -->
*3.5.5 Pigeonhole Principle — Official Online Questions O27-02 through O27-06*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.5_pigeonhole-principle|3.5.5]]

| 编号 | 要保证的性质 | 官方答案 | 最坏分布 |
|---|---|---:|---|
| O27-02 | 至少 2 人同月同日生日 | $366$ | 365 天先各 1 人 |
| O27-03 | 至少 2 人都生于 1 月 1 日 | nh | 任意多人都可生于别日 |
| O27-04 | 至少 3 人同一星期几出生 | $15$ | 7 类先各放 2 人 |
| O27-05 | 至少 4 人同月出生 | $37$ | 12 月先各放 3 人 |
| O27-06 | 至少 2 人生日恰相隔一周 | nh | 所有人可同一天出生 |
<!-- bilingual-en:start -->
| Number | Property to guarantee | Official answer | Worst-case distribution |
|---|---|---:|---|
| O27-02 | At least 2 people share the same birthday | $366$ | First place one person on each of 365 days |
| O27-03 | At least 2 people born on January 1 | nh | Any number of people can be born on another day |
| O27-04 | At least 3 people were born on the same day of the week | $15$ | First place 2 people in each of 7 weekday classes |
| O27-05 | At least 4 people were born in the same month | $37$ | First place 3 people in each of 12 months |
| O27-06 | At least 2 births occur exactly one week apart | nh | All births occur on the same day |
<!-- bilingual-en:end -->

“nh”意为无论群体多大都不必成立。鸽巢原理能强制**同一类别碰撞**，不能强制某个指定类别非空，也不能强制两个类别之间具有固定距离。
<!-- bilingual-en:start -->
“nh” means that the claim need not become true merely because the set is large. The pigeonhole principle can force **two objects into the same class**, but it cannot force a specified class to be nonempty or force two classes to occur at a prescribed distance.
<!-- bilingual-en:end -->

### 3.5.6 6.042 TEAL Table — 官方在线题 O27-07 至 O27-10
<!-- bilingual-en:start -->
*3.5.6 6.042 TEAL Table — Official Online Questions O27-07 through O27-10*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.6_6-042-teal-table|3.5.6]]

8 名学生围圆桌，旋转相同、反射不同：
<!-- bilingual-en:start -->
Eight students sit around a round table, with rotations identified but reflections treated as distinct:
<!-- bilingual-en:end -->

| 编号 | 限制 | 官方答案 | 方法 |
|---|---|---:|---|
| O27-07 | 无限制 | $7!=5040$ | 固定一人消除旋转 |
| O27-08 | Alyssa 邻 Ben | $2\cdot6!=1440$ | 把二人视作块，内部 2 序 |
| O27-09 | Ben 同时邻 Alyssa、Carlos | $2\cdot5!=240$ | 三人块且 Ben 居中 |
| O27-10 | Ben 邻 Alyssa 或 Carlos | $4\cdot6!-2\cdot5!=2640$ | 两个事件容斥 |
<!-- bilingual-en:start -->
| Number | Restriction | Official answer | Method |
|---|---|---:|---|
| O27-07 | No restriction | $7!=5040$ | Fix one person to remove rotational symmetry |
| O27-08 | Alyssa sits next to Ben | $2\cdot6!=1440$ | Treat them as a block with two internal orders |
| O27-09 | Ben sits next to both Alyssa and Carlos | $2\cdot5!=240$ | Treat the three as a block with Ben in the middle |
| O27-10 | Ben sits next to Alyssa or Carlos | $4\cdot6!-2\cdot5!=2640$ | Apply inclusion–exclusion to the two adjacency events |
<!-- bilingual-en:end -->

### 3.5.7 Class Schedules — 官方在线题 O27-11
<!-- bilingual-en:start -->
*3.5.7 Class Schedules — official online O27-11*
<!-- bilingual-en:end -->

原题与官方反馈：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S27_3.5.7_class-schedules|3.5.7]]

> [!success]- 官方答案与反馈
> 11 门课中恰选 4 门，共 $\binom{11}{4}=330$ 种课表（巢）。要保证两名学生同课表，需要 $330+1=\boxed{331}$ 名学生。
> <!-- bilingual-en:start -->
> Choosing exactly four of 11 courses gives $\binom{11}{4}=330$ possible schedules, which serve as the holes. Therefore $330+1=\boxed{331}$ students guarantee that two students have the same schedule.
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

&nbsp;
**1.** The number of integers in a closed interval is “last minus first plus one”; the possible sums from $2$ through $12$ number 11.<br>
**2.** The pigeonhole principle proves existence; it does not identify exactly where the collision occurs.<br>
**3.** “Guarantee that two people were born on one specified date” is not a collision problem; arbitrarily many people may all avoid that date.<br>
**4.** Inclusion–exclusion subtracts pairwise intersections and adds triple intersections; the sign is $(-1)^{|S|+1}$.<br>
**5.** When treating a string pattern as a block, you must check whether the patterns overlap and whether the direction of the overlap is unique.<br>
<!-- bilingual-en:end -->

### Session 27 自检
<!-- bilingual-en:start -->
*Session 27 Self Test*
<!-- bilingual-en:end -->

> [!question] 自检 27-1
> 至少多少个整数可保证其中两个模 10 同余？
> <!-- bilingual-en:start -->
> How many integers are needed to guarantee that two are congruent modulo 10?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> 模 10 有 10 个余数类，需 $10+1=11$ 个整数。
> <!-- bilingual-en:start -->
> There are 10 residue classes modulo 10, so $10+1=11$ integers are required.
> <!-- bilingual-en:end -->

> [!question] 自检 27-2
> 100 个对象放入 9 类，至少一类有多少个？
> <!-- bilingual-en:start -->
> If 100 objects are placed into nine categories, how many objects must some category contain?
> <!-- bilingual-en:end -->

> [!answer]- 答案
> $\lceil100/9\rceil=12$。若每类至多 11 个，总数至多 99，矛盾。
> <!-- bilingual-en:start -->
> $\lceil100/9\rceil=12$. If every category contained at most 11 objects, the total would be at most 99, a contradiction.
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
*Classroom Problems 27 — complete independent solutions to five problems*
<!-- bilingual-en:end -->

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp27.pdf#page=1|cp27 pp. 1–3]]

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
> A cword is a permutation of `a,d,e,f,i,l,o,p,r,s`, so there are $10!$ in total. Let $F,D,R$ denote the sets containing the substrings `fails`, `failed`, and `drop`, respectively.
> **(a)** Treat `drop` as one block together with the other six characters, giving seven objects and $\boxed{|R|=7!}$.
> **(b)** The characters of `drop` and `fails` are disjoint. Treating them as two blocks together with `e` leaves three objects, so $\boxed{|R\cap F|=3!=6}$.
> **(c)** We have $|F|=6!$, $|D|=5!$, and $|R|=7!$. The substrings `fails` and `failed` cannot both occur without repeating `fail` or placing both `s` and `e` immediately after the same prefix, so $F\cap D=\varnothing$. The substrings `drop` and `failed` can share only their single `d`, forming the block `failedrop`; arranging that block with `s` gives $|D\cap R|=2!$. The triple intersection is empty. Thus the number of bad cwords is $6!+5!+7!-3!-2!$, and the number of valid passwords is $\boxed{10!-7!-6!-5!+3!+2!=3{,}622{,}928}$.
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
> A monotone path from $(0,0)$ to $(50,50)$ consists of 50 horizontal and 50 vertical steps, so there are $\binom{100}{50}$ paths in total. Let $A=(10,11)$ and $B=(21,20)$. The numbers passing through $A$ and $B$ are $N_A=\binom{21}{10}\binom{79}{40}$ and $N_B=\binom{41}{21}\binom{59}{29}$. A path passing through both must visit them in the order $A\to B$, giving $N_{A\cap B}=\binom{21}{10}\binom{20}{11}\binom{59}{29}$. Inclusion–exclusion therefore gives the displayed count of paths avoiding both boulders.
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
> The seven dice have distinct colours, so an outcome is a seven-tuple listed in rainbow order.
> **(a) Exactly two dice show 6 and the other five values are all distinct:** The remaining faces must be $1,\ldots,5$, each used once. Choose the two colours showing 6 and then biject the other five colours with $1,\ldots,5$, giving $\boxed{\binom72 5!=2520}$.
> **(b) Exactly one pair and all other values distinct:** Choose the repeated face value, choose the two colours that show it, and permute the other five face values among the remaining colours. The count is $\boxed{6\binom72 5!=15120}$.
> **(c) Multiplicity pattern $(3,2,2)$:** Choose the face value appearing three times, choose an unordered pair of values for the two pairs, choose the three colours in the triple, and choose two of the remaining four colours for the smaller pair value. This gives $\boxed{6\binom52\binom73\binom42=12600}$.
> In each case the chosen data uniquely determine the dice outcome and can be uniquely recovered from it, so the constructions are genuine bijections.
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
> **(a)** Expand by the multinomial theorem.
> Unless the exponent vector is a pure term with one $k_i=p$ and all other entries zero, at least two $k_i$ are positive and every positive $k_i<p$. The prime $p$ occurs once in the numerator $p!$, while none of the denominator factors $k_i!$ contains $p$. Hence every mixed multinomial coefficient is divisible by $p$. Modulo $p$, all mixed terms vanish and only the pure terms remain:
> **(b)** Setting every $x_i=1$ gives $n^p\equiv n\pmod p$. If $p\nmid n$, then $n$ has a multiplicative inverse modulo $p$, so cancelling $n$ yields $n^{p-1}\equiv1\pmod p$.
> Taking $n$ variables proves the claim directly for positive integers $n$. For a negative integer, choose its nonzero representative $r\in\{1,\ldots,p-1\}$ modulo $p$; since $n\equiv r\pmod p$, the same conclusion follows.
> This argument does not invoke Fermat's little theorem, so it is not circular.
> <!-- bilingual-en:end -->

---

## 全章方法地图
<!-- bilingual-en:start -->
*Method Map for the Entire Chapter*
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
| Problem structure | Preferred tool | Required check |
|---|---|---|
| Long sum with positive monotone terms | Perturbation or the integral method | Endpoints and direction of monotonicity |
| Factorial or large product | Logarithms and Stirling's formula | Whether the claim is $\Theta$ or $\sim$ |
| Only the scale matters | $O,o,\Theta,\sim$ | One uniform constant, a threshold, and the direction of the relation |
| Mutually exclusive cases | Sum rule | Whether the cases overlap |
| Sequential choices | Generalised product rule | Whether every valid prefix has the same number of extensions |
| Objects are hard to count but easy to encode | Bijection | Forward map, inverse map, and proof that they are inverses |
| Every result is counted the same number of times | Division rule | Whether the number of preimages is constant |
| Repeated objects or allocation of exponents | Stars and Bars or multinomial coefficients | Whether objects are identical and whether order matters |
| Only a collision must be guaranteed | Pigeonhole principle | Pigeons, holes, mapping, and endpoint cases |
| Union of overlapping events | Inclusion–exclusion | Feasible intersections and alternating signs |
<!-- bilingual-en:end -->

## 覆盖与资源核对

- 官方在线题：Session 23 的 O23-01–O23-16（16 个）、Session 24 的 O24-01–O24-21（21 个）、Session 25 的 O25-01–O25-04（4 个）、Session 26 的 O26-01（1 个）、Session 27 的 O27-01–O27-11（11 个），合计 **53**。
- Classroom Problems：C23-1–5、C24-1–5、C25-1–4、C26-1–5、C27-1–5，合计 **24**；每题及全部子问均在对应 Session 末给出。
- 作业：PS9 3 题位于 Session 24 后；PS10 3 题及全部子问位于 Session 27 后。
- 考试：Midterm 3 共 6 题及全部子问位于 Session 24 后；第 1 题 DAG 已按原 PDF 图逐边读取。
- 视频顺序：17 个视频均按 3.1.1→3.5.4 官方 block 次序出现，并同时链接 slides、transcript 与在线 video。

> [!summary] 一句话收束
> **和式**把局部贡献累积起来，**渐近**压缩其规模，**双射与除法**把对象搬到标准模型，**鸽巢**从数量推出必然性，**容斥**修复重叠；五者合起来就是离散计数的主干。
> <!-- bilingual-en:start -->
> **Sums** accumulate local contributions, **asymptotics** compress their scale, **bijections and division** transfer objects to standard counting models, the **pigeonhole principle** extracts unavoidable collisions from quantities, and **inclusion–exclusion** corrects overlap. Together, these form the backbone of discrete counting.
> <!-- bilingual-en:end -->
