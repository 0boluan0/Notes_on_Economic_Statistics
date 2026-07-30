---
aliases:
  - MIT 6.042J Unit 2 Structures
  - 离散结构
tags:
  - discrete-mathematics
  - mit-ocw
  - course-note
  - structures
course: MIT 6.042J Mathematics for Computer Science
unit: 2
sessions: 12-22
---

# Unit 2: Structures

> [!info] 这篇笔记解决什么问题
> 本单元把第一单元的证明工具用于三类真正的计算结构：整数的整除结构、任务的先后结构、网络的连接结构。顺序严格对应 MIT 6.042J Spring 2015：Session 12–22；PS5 位于 Session 14 后，PS6 与 Midterm 2 位于 Session 16 后，PS7 位于 Session 19 后，PS8 位于 Session 22 后。

> [!warning] 答案来源
> “在线反馈题”中的答案和反馈来自官方离线 courseware；课堂题、Problem Set 与 Midterm 2 没有公开官方解答，以下均为**非官方独立题解**。每个正式题解都给出足够的推导，以便自行复算。

## 导航

- [[#Session 12 — GCDs|12 GCDs]]
- [[#Session 13 — Congruences|13 Congruences]]
- [[#Session 14 — Euler's Theorem|14 Euler's Theorem]] → [[#Problem Set 5]]
- [[#Session 15 — RSA Encryption|15 RSA]]
- [[#Session 16 — Digraphs Walks and Paths|16 Digraphs]] → [[#Problem Set 6]] → [[#Midterm 2]]
- [[#Session 17 — Directed Acyclic Graphs|17 DAGs]]
- [[#Session 18 — Partial Orders and Equivalence|18 Partial Orders]]
- [[#Session 19 — Degrees and Isomorphism|19 Degrees & Isomorphism]] → [[#Problem Set 7]]
- [[#Session 20 — Coloring and Connectivity|20 Coloring & Connectivity]]
- [[#Session 21 — Trees and Minimum Spanning Trees|21 Trees]]
- [[#Session 22 — Stable Matching and Hall's Theorem|22 Stable Matching]] → [[#Problem Set 8]]

---

## Session 12 — GCDs

### 本节问题与前置知识

**问题。** 怎样在不做质因数分解的前提下求最大公因数？为什么 Euclidean algorithm 正确？怎样把 gcd 写成两个整数的线性组合？

**前置。** 直接证明、良序原理、状态机不变量、结构归纳。默认变量在 $\mathbb Z$ 中；谈 gcd 时若无特别说明取非负值。

### 12.1 整除与最大公因数

[[Divisibility|整除]]定义为

$$
a\mid b\quad\Longleftrightarrow\quad \exists k\in\mathbb Z, b=ak.
$$

这里 $a\mid b$ 是一个命题，不是分数。由定义可逐步得到：

- $a\mid 0$，因为 $0=a\cdot0$；
- $1\mid a$ 与 $-1\mid a$；
- 若 $a\mid b$ 且 $b\mid c$，则 $b=ar,c=bs$，故 $c=a(rs)$，即 $a\mid c$；
- 若 $a\mid b$ 且 $a\mid c$，则对任意 $x,y\in\mathbb Z$，$a\mid xb+yc$。

最后一条说明：**共同因子必定整除任意整数线性组合**。它是本节所有算法正确性的入口。

> [!definition] 最大公因数
> 对不全为零的整数 $a,b$，[[Greatest Common Divisor|最大公因数]] $\gcd(a,b)$ 是同时整除 $a,b$ 的最大正整数。约定 $\gcd(0,0)=0$，且 $\gcd(a,b)=\gcd(|a|,|b|)$。

### 12.2 Division Algorithm 与 Euclidean Algorithm

**除法定理。** 若 $n>0$，则对每个整数 $a$，存在唯一整数 $q,r$ 使

$$
a=qn+r,\qquad 0\le r<n.
$$

记 $r=\operatorname{rem}(a,n)$。唯一性证明不能跳步：若又有 $a=q'n+r'$ 且 $0\le r,r'<n$，相减得 $(q-q')n=r'-r$。右侧严格位于 $(-n,n)$，其中唯一的 $n$ 的倍数是 $0$，所以 $r=r'$，继而 $q=q'$。

[[Euclidean Algorithm|Euclidean algorithm]]依赖等式

$$
\gcd(a,b)=\gcd\bigl(b,\operatorname{rem}(a,b)\bigr),\qquad b>0.
$$

**完整证明。** 写 $a=qb+r$。若 $d\mid a$ 且 $d\mid b$，则 $d\mid(a-qb)=r$；反过来若 $d\mid b$ 且 $d\mid r$，则 $d\mid(qb+r)=a$。因此两对数的共同因子集合完全相同，最大正元素也相同。

每一步把第二个正参数换成更小的余数，故余数构成严格下降的非负整数列；良序原理保证它最终到达 $0$。若最后一行是 $r_{k-1}=q_kr_k+0$，则答案为 $r_k$。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-euclidean-algorithm.png|900]]

读图：每一行都把上一行的除数变成新被除数、余数变成新除数，最后一个非零余数就是 gcd。

**例：**

$$
\begin{aligned}
1944&=2(874)+196,\\
874&=4(196)+90,\\
196&=2(90)+16,\\
90&=5(16)+10,\\
16&=1(10)+6,\\
10&=1(6)+4,\\
6&=1(4)+2,\\
4&=2(2)+0.
\end{aligned}
$$

所以 $\gcd(1944,874)=2$。

### 12.3 Bézout identity 与 Pulverizer

[[Bezout Identity|Bézout identity]]断言：若 $(a,b)\ne(0,0)$，则存在 $s,t\in\mathbb Z$ 使

$$
sa+tb=\gcd(a,b).
$$

**为什么反向代入一定成功？** Euclidean algorithm 的每个余数都是前两个余数的整数线性组合。初始的 $a,b$ 显然是 $a,b$ 的线性组合；若 $r_{i-2}$ 与 $r_{i-1}$ 都是线性组合，则

$$
r_i=r_{i-2}-q_ir_{i-1}
$$

仍是线性组合。最后一个非零余数就是 gcd，故结论成立。这也是扩展 Euclidean algorithm（课中称 **Pulverizer**）的循环不变量。

例如

$$
30=1(22)+8,\quad22=2(8)+6,\quad8=1(6)+2,
$$

反代得到

$$
2=8-6=3\cdot8-22=3\cdot30-4\cdot22.
$$

### 12.4 素数分解为什么唯一

关键引理是 Euclid's lemma：若素数 $p\mid ab$，则 $p\mid a$ 或 $p\mid b$。

**证明。** 若 $p\nmid a$，素数 $p$ 与 $a$ 互素。由 Bézout identity，存在 $x,y$ 使 $xp+ya=1$。两边乘 $b$：$xpb+yab=b$。左侧两项都被 $p$ 整除，因此 $p\mid b$。

**唯一分解证明。** 假设

$$
n=p_1p_2\cdots p_r=q_1q_2\cdots q_s
$$

都是素数分解。由 $p_1\mid q_1\cdots q_s$ 及 Euclid's lemma，$p_1$ 等于某个 $q_j$；交换顺序并约去这个正素数。重复后，左右素因子逐一匹配，且不可能一边先耗尽，否则剩余素数乘积会等于 $1$。因此多重集合唯一。

### 官方 block 与视频顺序

| 顺序 | 内容 | 本地入口 |
|---:|---|---|
| 2.1.1 | GCDs & Linear Combinations | [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/et3FOZdI6pk.pdf]] · [[MIT_OCW_6.042J_Materials/04_Captions/et3FOZdI6pk.srt]] |
| 2.1.2 | Euclidean Algorithm | [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/dW0f62lcCLE.pdf]] |
| 2.1.3 | Run Euclid Run | 在线题 |
| 2.1.4 | Pulverizer | [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/yzKPotFLfsc.pdf]] |
| 2.1.5 | GCDs I | 在线题 |
| 2.1.6 | Die Hard Primes | [[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Hard_Primes.pdf|slides]] · [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/c3qNBNl1h8g.pdf|transcript]] |
| 2.1.7 | Unique Factorization | [[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QsKtEuUyIdw.pdf]] |
| 2.1.8–10 | Unique Primes；Divisors；GCDs II | 在线题 |

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session12.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_GCDsandLinear.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_EuclidnAlgori.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Pulverizer.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_UniqueFactor.pdf]]。

### 在线反馈题（9 prompts，官方答案）

| block | 题意 | 官方答案与检查 |
|---|---|---|
| 2.1.3 Q1 | 用 Euclid algorithm 求 $\gcd(874,1944)$ | $2$；完整余数链见上文 |
| 2.1.5 Q1–Q2 | 求 $\gcd(21212121,12121212)$；用了几次 gcd 递推 | $3030303$；$3$ 次 |
| 2.1.8 Q1–Q2 | $40500$ 的素因子总数（计重数）与不同素因子数 | $40500=2^2 3^4 5^3$，答案 $9$ 与 $3$ |
| 2.1.9 Q1–Q3 | $12$ 的不同素因子数、正因子数、整数因子数 | $2,6,12$ |
| 2.1.10 Q1 | 两个给定素因数分解的 gcd | $37^2\cdot59^{29}$；共同素数取较小指数 |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S12_2.1.3_run-euclid-run.md|2.1.3]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S12_2.1.5_gcds-i.md|2.1.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S12_2.1.8_unique-primes.md|2.1.8]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S12_2.1.9_divisors.md|2.1.9]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S12_2.1.10_gcds-ii.md|2.1.10]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp12.pdf]]。

> [!example]- CP12 Problem 1：Pulverizer
> (a) 上文已得 $3(30)-4(22)=2$，故 $(x,y)=(3,-4)$。
>
> (b) 方程除以 $2$ 为 $15x+11y=1$。通解为 $x=3+11t,y=-4-15t$。取 $t=-1$，得 $(x',y')=(-8,11)$，且 $0\le11<30$。代回：$-8(30)+11(22)=2$。

> [!example]- CP12 Problem 2：从素因数分解求 gcd 与 lcm
> 对每个素数分别比较指数：gcd 取最小指数，lcm 取最大指数。因此题中
> $$\gcd(m,n)=2^3 11^7 17^9,$$
> $$\operatorname{lcm}(m,n)=2^9 5^{24}7^{22}11^{21}13\,17^{12}19^2.$$
> 对任意素数 $p$，两式乘积中 $p$ 的指数为 $\min(\alpha_p,\beta_p)+\max(\alpha_p,\beta_p)=\alpha_p+\beta_p$，正好等于 $mn$ 中的指数，故 $\gcd(m,n)\operatorname{lcm}(m,n)=mn$。

> [!example]- CP12 Problem 3：Binary GCD 状态机
> (a) 取不变量 $e\gcd(x,y)=\gcd(a,b)$。初态 $(a,b,1)$ 成立。两数同为偶数时，$\gcd(x,y)=2\gcd(x/2,y/2)$；仅一数为偶数而另一数为奇数时，除去偶数的因子 $2$ 不改变 gcd；两奇数相减也不改变 gcd。规则 (7) 把 $(x,x,e)$ 变成 $(1,0,ex)$，仍保持不变量。终态只能形如 $(1,0,g)$，于是 $g=\gcd(a,b)$。
>
> (b) 一旦执行规则 (3)–(6)，新状态至少有一个坐标为奇数；规则 (7) 则终止。之后不可能再次“两坐标均为偶数”，故规则 (2) 只能出现在开头。
>
> (c) 在规则 (7) 之前始终有 $x,y>0$，考察正整数势函数 $P=xy$。把转移分块：规则 (2) 一步使 $P$ 变为 $P/4$，规则 (3),(4) 一步使 $P$ 变为 $P/2$。若下一步是规则 (5) 且 $x>y$，则此时 $x,y$ 都为奇数；减法后 $x-y$ 为偶数、$y$ 为奇数，所以紧接着必执行规则 (3)。这两步把乘积变为 $((x-y)/2)y<xy/2$；规则 (6) 完全对称。因此每个至多含两步的块都使 $P$ 至少减半。由初值 $P=ab$ 且终止前 $P\ge1$，这样的块不超过 $\log_2(ab)=\log_2a+\log_2b$个。再加最后的规则 (7)，转移数至多 $1+2(\log_2a+\log_2b)$，当然也满足题目要求的较粗上界 $1+3(\log_2a+\log_2b)$。

> [!example]- CP12 Problem 4：四个 gcd 性质
> (a) Bézout 给出 $g=sa+tb$；任何共同因子 $d$ 整除右侧，所以 $d\mid g$。
>
> (b) 若 $a\mid bc$ 且 $\gcd(a,b)=1$，取 $xa+yb=1$，乘 $c$ 得 $xac+ybc=c$；两项都被 $a$ 整除，故 $a\mid c$。
>
> (c) 令 $a=p$。若 $p\nmid b$，则 $\gcd(p,b)=1$，由 (b) 得 $p\mid c$。
>
> (d) 设 $m=xa+yb$ 是最小正线性组合。$g=\gcd(a,b)$ 整除 $m$。再用除法定理写 $a=qm+r$，则 $r=a-q(xa+yb)$ 也是非负线性组合且 $0\le r<m$；最小性迫使 $r=0$，故 $m\mid a$。同理 $m\mid b$，于是 $m\le g$；而 $g\mid m$ 又给 $g\le m$，故 $m=g$。

### 易错点与三道自检

- $0\mid b$ 只在 $b=0$ 时成立；但任意非零整数都整除 $0$。
- gcd 按约定非负；Bézout 系数通常不唯一。
- “有共同因子”不能推出“能约去”；模运算中的约分还需要互素条件。

> [!question]- 自检 1
> 用 Pulverizer 写出 $\gcd(252,198)$ 的 Bézout 线性组合。
>
> [!success]- 答案
> $252=198+54,198=3\cdot54+36,54=36+18$，故 $18=54-36=4\cdot54-198=4\cdot252-5\cdot198$。

> [!question]- 自检 2
> 为什么 Euclidean algorithm 不需要预先知道素因数分解？
>
> [!success]- 答案
> 每步只用除法定理和“共同因子集合不变”的证明；终止性来自严格下降余数，而不是唯一分解。

> [!question]- 自检 3
> 若 $d\mid a,b$，为什么 $d\mid\gcd(a,b)$ 比“$d\le\gcd(a,b)$”更强？
>
> [!success]- 答案
> 后者只比较大小；前者还给出代数结构，可继续推出 $d$ 整除任何 Bézout 线性组合。

**知识链：**整除 → 共同因子 → Euclidean algorithm → Bézout → Euclid's lemma → 唯一素因数分解。

---

## Session 13 — Congruences

### 本节问题与前置知识

**问题。** 怎样把“余数相同”变成可代数运算的等价关系？什么时候模方程可以约分、除法或求逆？

### 13.1 同余是等价关系

对模数 $n>1$，[[Modular Arithmetic|同余]]定义为

$$
a\equiv b\pmod n\quad\Longleftrightarrow\quad n\mid(a-b).
$$

它等价于 $a,b$ 除以 $n$ 的余数相同。证明见除法定理：写 $a=q_1n+r_1,b=q_2n+r_2$，则 $a-b=(q_1-q_2)n+(r_1-r_2)$；而 $r_1-r_2\in(-n,n)$，只有在 $r_1=r_2$ 时才可能被 $n$ 整除。

由定义立即验证自反、对称、传递，所以模 $n$ 同余把 $\mathbb Z$ 分成 $n$ 个余数类。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-modular-clock.png|900]]

读图：时钟上相差整数圈的数落在同一余数位置，因而模 $n$ 运算可看成沿圆周前进后回绕。

### 13.2 同余运算规则

若 $a\equiv b\pmod n$ 且 $c\equiv d\pmod n$，则

$$
a+c\equiv b+d,\qquad ac\equiv bd\pmod n.
$$

乘法证明：$ac-bd=a(c-d)+d(a-b)$，右侧两项都被 $n$ 整除。由结构归纳，任意整数系数多项式 $p$ 都满足 $a\equiv b\Rightarrow p(a)\equiv p(b)$。

**不能随意约分。** $8\cdot2\equiv3\cdot2\pmod{10}$ 成立，但 $8\not\equiv3\pmod{10}$。若 $ak\equiv bk\pmod n$ 且 $\gcd(k,n)=1$，Bézout 给出 $uk+vn=1$；由 $n\mid k(a-b)$ 可推出 $n\mid(a-b)$，此时才能约去 $k$。

### 13.3 模逆元

[[Modular Inverse|模逆元]] $k^{-1}$ 满足 $kk^{-1}\equiv1\pmod n$。

**定理。** $k$ 模 $n$ 可逆，当且仅当 $\gcd(k,n)=1$。

**证明。** 若可逆，则 $kk^{-1}-1=qn$，即 $kk^{-1}+n(-q)=1$，所以任何共同因子只能是 $1$。反过来，若 gcd 为 $1$，Bézout 给出 $sk+tn=1$，取模 $n$ 即 $sk\equiv1$，所以 $s$ 是逆元。

### 13.4 Chinese Remainder Theorem

[[Chinese Remainder Theorem|Chinese Remainder Theorem]]：若 $\gcd(a,b)=1$，对任意 $m,n$，方程

$$
x\equiv m\pmod a,\qquad x\equiv n\pmod b
$$

在模 $ab$ 意义下有唯一解。

**构造。** 令 $b^{-1}$ 是 $b$ 模 $a$ 的逆元，$e_a=b^{-1}b$；同理 $e_b=a^{-1}a$。则

$$
e_a\equiv1\pmod a, e_a\equiv0\pmod b;\qquad
e_b\equiv0\pmod a, e_b\equiv1\pmod b.
$$

所以 $x=me_a+ne_b$ 同时满足两式。

**唯一性。** 若 $x,x'$ 都满足，则 $a\mid(x-x')$ 且 $b\mid(x-x')$。写 $x-x'=ak$。因为 $b\mid ak$ 且 $\gcd(a,b)=1$，Euclid's lemma 的互素版本给出 $b\mid k$，故 $ab\mid(x-x')$。

### 官方顺序、资源与在线题

顺序为 Congruence mod $n$ → Divisibility and Congruence → Inverses mod $n$ → 约分条件 → Multiplicative Inverses → Inverses With Linear Combinations。

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session13.pdf]]。视频讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/KvtLWgCTwn4.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/CAKSh3M0y8k.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_congruence.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_inverses_mod.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.2.2 Q1 | 从七个说法中选择与 $a\equiv b\pmod n$ 等价者 | 同余；余数相同；$n\mid(a-b)$；$a=b+nk$；$a-b$ 是 $n$ 的倍数 |
| 2.2.4 Q1 | 为什么不能在 $8\cdot2\equiv3\cdot2\pmod{10}$ 中约去 $2$ | $2$ 与 $10$ 有共同因子 |
| 2.2.4 Q2 | $k,n$ 满足什么条件才能约去 $k$ | relatively prime |
| 2.2.5 Q1 | $2$ 模 $7$ 的逆元 | $4$ |
| 2.2.6 Q1 | 由 $1=9(25)-7(32)$ 求 $32^{-1}\pmod{25}$ | $-7\equiv18$ |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S13_2.2.2_divisibility-and-congruence.md|2.2.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S13_2.2.4_inverses-mod-n.md|2.2.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S13_2.2.5_multiplicative-inverses.md|2.2.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S13_2.2.6_inverses-with-linear-combinations.md|2.2.6]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp13.pdf]]。

> [!example]- CP13 Problem 1：巨指数只看余数周期
> 题式为 $9876^{3456789}(9^{99})^{5555}-6789^{3414259}$ 模 $14$。有 $9876\equiv6$，而 $6$ 的正幂在 $6,8$ 间交替；指数为奇数，故第一因子为 $6$。$9^3\equiv1$ 且 $99$ 被 $3$ 整除，故 $(9^{99})^{5555}\equiv1$。$6789\equiv-1$ 且指数为奇数，故末项为 $-1$。总和 $6-(-1)=7\pmod{14}$。

> [!example]- CP13 Problem 2：CRT 的存在与唯一
> (a)–(d) 正是上文 $e_a,e_b$ 的构造与唯一性证明。
>
> (e) 逆命题成立且不需要互素：若 $x\equiv x'\pmod{ab}$，则 $ab\mid(x-x')$，自然有 $a\mid(x-x')$ 与 $b\mid(x-x')$。

> [!example]- CP13 Problem 3：整数多项式保持同余并产生倍数
> (a) 对递归定义的多项式作结构归纳。恒等函数与常函数显然保持同余。若 $r,s$ 保持同余，则加法兼容性给 $(r+s)(j)\equiv(r+s)(k)$，乘法兼容性给 $(rs)(j)\equiv(rs)(k)$。所有构造均覆盖，结论成立。
>
> (b) 设 $v=q(k)>1$。由 (a)，$k+tv\equiv k\pmod v$ 推出 $q(k+tv)\equiv q(k)=v\equiv0\pmod v$。正次数、正首项系数保证 $t$ 足够大时 $q(k+tv)$ 严格增长，故得到无限多个互异的 $v$ 的倍数。

### 易错点、自检与知识链

- “$a\equiv b\pmod n$”是关于两数和模数的三元关系；$(\bmod n)$ 不是只修饰右边。
- CRT 的模数互素保证模乘积唯一；不互素时需先检查余数在 gcd 模下相容。

> [!question]- 自检 1
> 求 $17^{-1}\pmod{43}$。
>
> [!success]- 答案
> $43=2(17)+9,17=9+8,9=8+1$，反代 $1=2(43)-5(17)$，故逆元为 $-5\equiv38$。

> [!question]- 自检 2
> 解 $x\equiv2\pmod3, x\equiv3\pmod5$。
>
> [!success]- 答案
> $5^{-1}\equiv2\pmod3$，$3^{-1}\equiv2\pmod5$，故 $x\equiv2(2\cdot5)+3(2\cdot3)=38\equiv8\pmod{15}$。

> [!question]- 自检 3
> $6x\equiv6y\pmod{15}$ 能否约去 $6$？
>
> [!success]- 答案
> 不能直接约，因为 $\gcd(6,15)=3$。只能推出 $x\equiv y\pmod5$。

**知识链：**余数 → 同余等价类 → 运算兼容 → Bézout → 逆元 → CRT。

---

## Session 14 — Euler's Theorem

### 14.1 Euler function 与单位群

[[Euler Totient Function|Euler's totient function]]定义为

$$
\varphi(n)=\bigl|\{k\in\{0,1,\ldots,n-1\}:\gcd(k,n)=1\}\bigr|.
$$

这些可逆余数类构成 $\mathbb Z_n^*$。若 $p$ 为素数，$\varphi(p)=p-1$；若 $p$ 为素数且 $k\ge1$，在 $0,\ldots,p^k-1$ 中恰有 $p^{k-1}$ 个 $p$ 的倍数，所以

$$
\varphi(p^k)=p^k-p^{k-1}.
$$

若 $\gcd(a,b)=1$，CRT 给出 $\mathbb Z_{ab}^*\leftrightarrow\mathbb Z_a^*\times\mathbb Z_b^*$ 的双射，因此 $\varphi(ab)=\varphi(a)\varphi(b)$。由此若 $n=\prod p_i^{\alpha_i}$，则

$$
\varphi(n)=n\prod_{p\mid n}\left(1-\frac1p\right).
$$

### 14.2 Euler's Theorem 的完整证明

[[Euler's Theorem|Euler's theorem]]：若 $\gcd(k,n)=1$，则

$$
k^{\varphi(n)}\equiv1\pmod n.
$$

**构造。** 将 $\mathbb Z_n^*$ 中元素列为 $r_1,\ldots,r_{\varphi(n)}$。乘以 $k$ 后，$kr_i$ 仍可逆；若 $kr_i\equiv kr_j$，因 $k$ 可逆可约去，得 $r_i\equiv r_j$。有限集合上的单射是双射，因此 $kr_1,\ldots,kr_\varphi$ 只是原列表的重排。

于是

$$
k^{\varphi(n)}\prod_i r_i\equiv\prod_i kr_i\equiv\prod_i r_i\pmod n.
$$

$\prod_i r_i$ 仍可逆，约去后即得结论。若 $n=p$ 为素数，得到 [[Fermat's Little Theorem|Fermat's little theorem]]：$p\nmid k\Rightarrow k^{p-1}\equiv1\pmod p$。

> [!warning] 假设不可删除
> 若 $\gcd(k,n)\ne1$，Euler theorem 不保证成立。例如 $2^{\varphi(4)}=4\not\equiv1\pmod4$。

### 14.3 模指数的算法直觉

不要先计算 $k^N$ 再取余。重复平方法将 $N$ 写成二进制；每次平方或乘法后立即取模。由于同余与乘法兼容，中间缩小数值不改变最终余数，复杂度只需 $O(\log N)$ 次模乘。

### 官方顺序、资源与在线题（9 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session14.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TeRYL7kkhqs.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/dZgI16nMuqE.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ModularEuler.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_TheRingZn.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.3.2 Q1 | 哪些式子等于 $\varphi(300)$ | $\varphi(3)\varphi(100)$；$\varphi(4)\varphi(3)\varphi(25)$ |
| 2.3.4 Q1 | $\mathbb Z_n$ 中恒成立的环律 | 加法交换、乘法结合、分配律；一般不能约去因子 |
| 2.3.4 Q2 | $\mathbb Z_7^*$ | $1,2,3,4,5,6$ |
| 2.3.5 Q1–Q2 | $9+12$ 在 $\mathbb Z_{13}$；$7\cdot5$ 在 $\mathbb Z_8$ | $8$；$3$ |
| 2.3.5 Q3 | 用一个单位乘遍 $\mathbb Z_n^*$ 的效果 | 元素集合不变，顺序可能改变 |
| 2.3.6 Q1 | $\operatorname{rem}(24^{78},79)$ | $1$ |
| 2.3.7 Q1–Q2 | $\varphi(175)$；$22^{12001}\bmod175$ | $120$；$22$ |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S14_2.3.2_euler-s-totient-function.md|2.3.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S14_2.3.4_the-ring.md|2.3.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S14_2.3.5_z-mod-n.md|2.3.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S14_2.3.6_fermat-s-little-theorem.md|2.3.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S14_2.3.7_euler-s-theorem.md|2.3.7]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp14.pdf]]。

> [!example]- CP14 Problem 1
> $297=3^3\cdot11$，故 $\varphi(297)=18\cdot10=180$。又 $\gcd(26,297)=1$，且 $1818181=180(10101)+1$，所以
> $$26^{1818181}\equiv(26^{180})^{10101}26\equiv26\pmod{297}.$$

> [!example]- CP14 Problem 2
> $2012\equiv10\pmod{77}$，而 $\gcd(10,77)=1$，所以 $2012$ 及其任何正幂都有逆元。(b) $77=7\cdot11$，$\varphi(77)=6\cdot10=60$。(c) $1200=20\varphi(77)$，故 $2012^{1200}\equiv1\pmod{77}$。

> [!example]- CP14 Problem 3
> $0,1,\ldots,p^k-1$ 中不与 $p^k$ 互素的数恰是 $p$ 的倍数：$0,p,2p,\ldots,(p^{k-1}-1)p$，共 $p^{k-1}$ 个。因此 $\varphi(p^k)=p^k-p^{k-1}$。

> [!example]- CP14 Problem 4
> (a) Euler theorem 只直接处理与 $10$ 互素的 $n$，偶数和 $5$ 的倍数不满足假设。
>
> (b) 对 $d=0,\ldots,9$ 直接检查；更结构化地，模 $2$ 有 $d^{13}\equiv d$，模 $5$ 时若 $5\nmid d$，$d^{12}\equiv1$，若 $5\mid d$ 两边均为 $0$。CRT 得 $d^{13}\equiv d\pmod{10}$。
>
> (c) 任意 $n\equiv d\pmod{10}$，多项式保持同余给 $n^{13}\equiv d^{13}\equiv d\equiv n\pmod{10}$。

### 自检与知识链

> [!question]- 自检 1
> 求 $\varphi(360)$。
>
> [!success]- 答案
> $360=2^3 3^2 5$，故 $360(1-1/2)(1-1/3)(1-1/5)=96$。

> [!question]- 自检 2
> 求 $7^{222}\bmod20$。
>
> [!success]- 答案
> $\varphi(20)=8$，$222=27\cdot8+6$，而 $7^2\equiv9,7^4\equiv1$，故 $7^6\equiv9$。

> [!question]- 自检 3
> Euler theorem 的“乘法重排”证明在哪一步使用了 $\gcd(k,n)=1$？
>
> [!success]- 答案
> 两次：保证 $kr_i$ 仍在 $\mathbb Z_n^*$；保证从 $kr_i\equiv kr_j$ 约去 $k$，从而得到置换。

**知识链：**CRT → 单位集合 → $\varphi$ 乘法性 → 乘法置换 → Euler/FLT → 快速模指数。

---

## Problem Set 5

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps5.pdf]]。

> [!example]- PS5 Problem 1：Binary Pulverizer
> **预处理。** 若原输入 $A,B$ 同时为偶数，先重复提取共同因子 $2$，写成 $A=2^ra,B=2^rb$，使 $a,b$ 至少一个为奇数。以下先求 $\gcd(a,b)$ 的系数；最后因 $\gcd(A,B)=2^r\gcd(a,b)$，同一对系数也立即给出 $A,B$ 的 Bézout 组合。
>
> **目标。** 在 binary gcd 的同时保持
> $$x=u_xa+v_xb,\qquad y=u_ya+v_yb.$$
> 初始系数为 $(u_x,v_x)=(1,0),(u_y,v_y)=(0,1)$。减法规则 $x\leftarrow x-y$ 同时更新 $(u_x,v_x)\leftarrow(u_x-u_y,v_x-v_y)$；交换亦交换系数。
>
> 若 $z=ua+vb$ 为偶数：若 $u,v$ 都为偶数，直接各除以 $2$；否则需证 $u-b$ 与 $v+a$ 均为偶数。分三种奇偶性：若 $a$ 奇、$b$ 偶，则 $ua+vb$ 偶迫使 $u$ 偶，而“$u,v$ 不全偶”迫使 $v$ 奇；若 $a$ 偶、$b$ 奇则对称；若 $a,b$ 均奇，则 $u+v$ 偶，故不全偶时 $u,v$ 均奇。三种情形都给出 $u-b,v+a$ 为偶数，且
> $$(u-b)a+(v+a)b=ua+vb=z.$$
> 因而 $z/2=((u-b)/2)a+((v+a)/2)b$。这给出所有除以 $2$ 的合法系数更新。算法终止时 $x$ 或 $y$ 为 gcd，相应系数就是 Bézout 系数。

> [!example]- PS5 Problem 2：Wilson's Theorem
> (a) $k^2\equiv1\pmod p$ 等价于 $p\mid(k-1)(k+1)$。因 $p$ 为素数，$p\mid k-1$ 或 $p\mid k+1$；在 $0<k<p$ 内即 $k=1$ 或 $p-1$。逆向直接代入。
>
> (b) 在 $1,\ldots,p-1$ 中，每个元素有唯一逆元。除自逆元 $1,p-1$ 外，其余元素可成对配成 $a,a^{-1}$，每对乘积为 $1$。所以
> $$(p-1)!\equiv1\cdot(p-1)\equiv-1\pmod p.$$

> [!example]- PS5 Problem 3：$\varphi$ 的乘法性
> (a) CRT 恰好说明 $f(x)=(x\bmod a,x\bmod b)$ 从 $[0,ab)$ 到 $[0,a)\times[0,b)$ 既存在逆映射又唯一，故为双射。
>
> (b) $\gcd(x,ab)=1$ 当且仅当同时 $\gcd(x,a)=\gcd(x,b)=1$；所以 $f$ 限制为 $\mathbb Z_{ab}^*\to\mathbb Z_a^*\times\mathbb Z_b^*$ 的双射。
>
> (c) 取基数：$\varphi(ab)=\varphi(a)\varphi(b)$。
>
> (d) 将 $n=\prod_i p_i^{\alpha_i}$，反复应用乘法性与 $\varphi(p^\alpha)=p^\alpha-p^{\alpha-1}$：
> $$\varphi(n)=\prod_i p_i^{\alpha_i}(1-1/p_i)=n\prod_i(1-1/p_i).$$

---

## Session 15 — RSA Encryption

### 本节问题与前置知识

**问题。** 不预先共享秘密的两个人怎样安全通信？RSA 的解密为什么对所有消息正确？哪些计算必须容易，哪个计算必须困难？

### 15.1 公钥与私钥的构造

[[RSA|RSA public-key cryptosystem]]的数学核心如下。

1. 选取不同大素数 $p,q$，令 $n=pq$，因而 $\varphi(n)=(p-1)(q-1)$。
2. 选 $e$ 使 $\gcd(e,\varphi(n))=1$。
3. 用 Pulverizer 求 $d$，使 $ed\equiv1\pmod{\varphi(n)}$。
4. 公开 $(e,n)$，保密 $(d,n)$ 及 $p,q$。
5. 消息 $m\in[0,n)$ 加密为 $c\equiv m^e\pmod n$；解密为 $c^d\pmod n$。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-rsa-flow.png|900]]

读图：公钥 $(e,n)$ 只承担模幂加密，私钥指数 $d$ 把 ciphertext 送回原消息，正确性依赖 $ed$ 的指数同余。

这里公开钥允许任何人加密，却没有直接暴露逆指数 $d$。实际系统还必须使用随机化 padding；“裸 RSA”具有确定性和可塑性，不能直接作为安全协议。

### 15.2 正确性：不能偷用互素假设

由 $ed\equiv1\pmod{(p-1)(q-1)}$，存在整数 $t$ 使

$$
ed=1+t(p-1)(q-1).
$$

若 $\gcd(m,n)=1$，Euler theorem 立即给 $m^{ed}\equiv m\pmod n$。但消息可能被 $p$ 或 $q$ 整除；完整证明应分别模 $p,q$：

- 若 $p\mid m$，则 $m^{ed}\equiv0\equiv m\pmod p$；
- 若 $p\nmid m$，则 $ed\equiv1\pmod{p-1}$，Fermat 给 $m^{ed}=m(m^{p-1})^k\equiv m\pmod p$。

所以无论哪种情形都有 $m^{ed}\equiv m\pmod p$。同理模 $q$ 成立。$p,q$ 互素，CRT 推出

$$
m^{ed}\equiv m\pmod{pq}.
$$

这一步明确覆盖了 $m=0$、$p\mid m$、$q\mid m$ 等边界情况。

### 15.3 可行性与安全假设

合法用户必须能高效完成：生成随机大素数、素性测试、gcd、模逆元、重复平方法。攻击者若能从 $n$ 得到 $p,q$，即可算 $\varphi(n)=(p-1)(q-1)$ 并恢复 $d$。

反之，若已知 $n$ 与 $\varphi(n)$，则

$$
p+q=n-\varphi(n)+1.
$$

$p,q$ 是方程 $X^2-(p+q)X+n=0$ 的两根，故也能因式分解 $n$。课程中的 “Reducing Factoring to SAT” 用来说明：一个具体计算问题可编码为布尔可满足性问题；它不等于证明 SAT 或 factoring 在经典计算上容易。

### 官方顺序、资源与在线题（4 prompts）

顺序：RSA Public Key Encryption → RSA Encryption → Reducing Factoring to SAT → Relative Primality → RSA computations。

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session15.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ZUZ8VbX1YNQ.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/yWIQCewgfwY.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_RSA_Encytion.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_FactoringSAT.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.4.2 Q1 | 私钥指数 $d$ 如何得到 | 求某个与 $(p-1)(q-1)$ 互素的数 $e$ 在模 $(p-1)(q-1)$ 下的逆元 |
| 2.4.4 Q1 | $1$ 到 $3780$ 中与 $3780$ 互素者数量 | $3780=2^2 3^3 5\cdot7$，答案 $864$ |
| 2.4.5 Q1 | RSA 必须容易的计算 | 素性测试；算 $pq$ 与 $\varphi(pq)$（已知 $p,q$）；随机数字/大素数；gcd、逆元；模快速幂 |
| 2.4.5 Q2 | 为保证安全而应困难的计算 | 只知 $n$ 求 $p,q$，即分解 600 位合数 |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S15_2.4.2_rsa-encryption.md|2.4.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S15_2.4.4_relative-primality.md|2.4.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S15_2.4.5_rsa-computations.md|2.4.5]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp15.pdf]]。

> [!example]- CP15 Problem 1：完整的小型 RSA 演算
> 选 $p=11,q=17$，则 $n=187,\varphi(n)=160$。选 $e=3$，因 $\gcd(3,160)=1$；$3\cdot107=321\equiv1\pmod{160}$，所以 $d=107$。公钥 $(3,187)$，私钥 $(107,187)$。
>
> 发送代码 $m=5$：$c=5^3\bmod187=125$。接收方计算 $125^{107}\bmod187=5$。最后一步应用重复平方法；结果回到区间 $[0,187)$ 中的原消息，完成验算。

> [!example]- CP15 Problem 2：知道 $\varphi(n)$ 就能破解
> (a) 由公开 $e$ 与 $\varphi(n)$ 求 $d=e^{-1}\pmod{\varphi(n)}$，即可解密。
>
> (b) 若 $n=pq$，则 $\varphi(n)=pq-p-q+1=n-(p+q)+1$，故 $S=p+q=n-\varphi(n)+1$。判别式 $S^2-4n=(p-q)^2$，于是
> $$p,q=\frac{S\pm\sqrt{S^2-4n}}2.$$

> [!example]- CP15 Problem 3：去掉 $\gcd(m,n)=1$ 限制
> (a) 互素时 $m^{ed}=m(m^{\varphi(n)})^t\equiv m$。
>
> (b) 若 $a\equiv1\pmod{p-1}$：当 $p\mid m$ 时两边均为 $0$；否则 Fermat 给 $m^a=m(m^{p-1})^t\equiv m\pmod p$。
>
> (c) 若不同素数 $p_i$ 都整除 $a-b$，其乘积也整除 $a-b$；这是反复使用“互素因子分别整除则乘积整除”。
>
> (d) $a\equiv1\pmod{\varphi(n)}$ 蕴含对每个 $p_i\mid n$ 都有 $a\equiv1\pmod{p_i-1}$。由 (b) 得 $m^a\equiv m\pmod{p_i}$，由 (c) 合并为模 $n$ 同余。取 $a=ed$ 即为 RSA 正确性。

### 易错点、自检与知识链

- RSA 正确性是定理；“分解大整数在经典计算上足够难”是安全假设，不是本课证明的数学定理。
- 指数逆元是模 $(p-1)(q-1)$ 求，不是模 $pq$ 求。
- 现代协议不直接加密任意长文本；先编码、分块并使用经过标准化的 padding。

> [!question]- 自检 1
> 对 $p=5,q=11,e=3$，求 $d$。
>
> [!success]- 答案
> $(p-1)(q-1)=40$，$3^{-1}\equiv27\pmod{40}$。

> [!question]- 自检 2
> 为什么消息恰好被 $p$ 整除不会破坏解密？
>
> [!success]- 答案
> 模 $p$ 时原消息与任意正幂均为 $0$；模 $q$ 使用 Fermat，最后由 CRT 合并。

> [!question]- 自检 3
> 已知 $n=55,\varphi(n)=40$，恢复 $p,q$。
>
> [!success]- 答案
> $p+q=55-40+1=16$，根为 $(16\pm\sqrt{256-220})/2=(16\pm6)/2$，即 $5,11$。

**知识链：**逆元 + Euler/FLT + CRT → RSA 正确性；快速模指数 → 可行性；因式分解 → 安全边界。

---

## Session 16 — Digraphs Walks and Paths

### 16.1 定义必须分清

[[Directed Graph|有向图]] $G=(V,E)$ 中，边是有序对 $(u,v)$，记作 $u\to v$。允许自环与否、是否允许平行边必须由模型声明；本课程默认边集合，因此同方向平行边不重复。

- **walk**：$v_0,e_1,v_1,\ldots,e_k,v_k$，允许重复顶点和边；长度为 $k$。
- **path**：不重复顶点的 walk；长度至多 $|V|-1$。
- **closed walk**：$v_0=v_k$；正长度时才提供实际运动。
- **cycle**：除首尾相同外不重复顶点的正长度 closed walk。
- $u$ 到 $v$ 的距离 $\operatorname{dist}(u,v)$ 是最短 path 长度；不可达时记 $\infty$。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-directed-walk.png|900]]

读图：箭头限定可走方向，walk 可重复顶点，而 path 会把绕圈的重复段删去。

删除 walk 中两个相同顶点之间的片段可以消去“绕圈”，所以任何从 $u$ 到 $v$ 的 walk 都包含一条不更长的 path。正 closed walk 至少包含一个 cycle，但那个 cycle 不一定包含原 walk 的每个顶点。

### 16.2 距离三角不等式

若 $u\leadsto x$ 与 $x\leadsto v$，连接两条最短 path 得一个 walk，删环后不变长，所以

$$
\operatorname{dist}(u,v)\le\operatorname{dist}(u,x)+\operatorname{dist}(x,v).
$$

等号成立当且仅当 $x$ 位于某条 $u$ 到 $v$ 的最短 path 上。注意这不是说 $x$ 位于**所有**最短 path 上。

### 16.3 邻接矩阵与路径计数

给顶点编号 $1,\ldots,n$，[[Adjacency Matrix|邻接矩阵]] $A$ 定义为 $A_{ij}=1$ 当且仅当 $i\to j$。矩阵乘法给

$$
(A^k)_{ij}=\text{从 }i\text{ 到 }j\text{ 的长度恰为 }k\text{ 的 walks 数}.
$$

**归纳证明。** $k=1$ 即定义。若对 $k$ 成立，则每条长度 $k+1$ walk 可按倒数第二个顶点 $r$ 唯一拆成长度 $k$ 的 $i\leadsto r$ walk 加边 $r\to j$；计数为 $\sum_r(A^k)_{ir}A_{rj}=(A^{k+1})_{ij}$。

> [!warning] walk 不是 path
> $A^k$ 计数允许重复顶点，不能直接计数简单 paths；后者通常困难得多。

### 16.4 Reachability 与 strongly connected

正 walk 关系 $E^+$ 表示存在正长度 walk；$E^*$ 还加入长度 $0$，所以是自反传递闭包。在有限图中，两点互相可达形成强连通分量。Web hyperlinks、控制流、依赖传播和状态迁移都自然产生 digraph。

### 官方顺序、资源与在线题（8 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session16.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/MX-mBxt6huU.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/QORX1OUabio.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_WalksPaths.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Concted_Vrtics.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.5.2 Q1 | 给定四点 digraph 的最长 path | $3$ |
| 2.5.2 Q2 | 邻接矩阵所有元素之和为 $6$ | 图有 $6$ 条有向边 |
| 2.5.4 Q1–Q2 | 11 点、10 边图的 longest path 最大/最小可能值 | $10$；$1$ |
| 2.5.5 Q1 | 邻接矩阵的必真命题 | $(A^2)_{ij}\ne0$ 当且仅当存在长度 2 walk；无向图才必有 $A=A^T$ |
| 2.5.6 Q1–Q3 | 五点无自环 complete digraph：边数、最长无环 path、这类 Hamilton paths 数 | $20,4,120$ |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S16_2.5.2_walks-and-paths.md|2.5.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S16_2.5.4_longest-path.md|2.5.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S16_2.5.5_adjacency-matrix.md|2.5.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S16_2.5.6_counting-paths.md|2.5.6]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp16.pdf]]。

> [!example]- CP16 Problem 1：closed walk 与 cycle
> (a) 取一个有向 3-cycle，沿它走两圈得到长度 $6$ 的 closed walk，但图中仅有长度 $3$ 的 cycle。
>
> (b) 取边 $v\to u,u\to v$，并在 $u$ 加自环。$v\to u\to u\to v$ 长度 $3$，但含 $v$ 的唯一 cycle 长度 $2$；奇 cycle 只在 $u$。
>
> (c) 反复从 closed walk 中剪去两个相同内部顶点之间的 closed 子段，最终分解为 cycles。总长度为奇数，若每个 cycle 都为偶数则总和为偶数，矛盾；所以至少有一个奇 cycle，其顶点当然位于原 walk 上。

> [!example]- CP16 Problem 2：距离等号
> 若 $\operatorname{dist}(u,v)=\operatorname{dist}(u,x)+\operatorname{dist}(x,v)$，连接两段最短 path 得到恰为最短距离的 walk；删环不可能变短，否则会小于定义的最短距离，所以存在经过 $x$ 的最短 path。反过来，若某最短 path 经过 $x$，其两段必须各自最短；若一段可缩短，替换后会得到更短的 $u$–$v$ path。因此等号成立。

> [!example]- CP16 Problem 3：de Bruijn string
> (a) `0001011100` 的八个连续三位窗口依次覆盖 $000,001,010,101,011,111,110,100$。八种窗口至少需要 $8+3-1=10$ 位，故最短。
>
> (b) 图的顶点是两位串；边 $x_1x_2\to x_2b$ 标记追加位 $b$。每走一条边便产生一个新的三位窗口，所以遍历全部八条边得到 3-good string。
>
> (c) 每条边恰一次产生每个三位串恰一次，长度达到下界 $10$。
>
> (d) $B_k$ 顶点为 $\{0,1\}^k$，边 $x_1\cdots x_k\to x_2\cdots x_kb$。每点入度、出度均为 $2$；通过依次追加目标串的位，任意点在至多 $k$ 步到任意点。遍历 $2^{k+1}$ 条边产生最短 $(k+1)$-good string，长度为 $2^{k+1}+k$。

> [!example]- CP16 Supplemental Problem 4：tournament ranking
> (a) 三点传递 tournament $a\to b,a\to c,b\to c$ 至少有路径 $a,b,c$；若另加适当方向可给多个排名，三-cycle 本身已有三个循环移位的 Hamilton paths。
>
> (b) 对顶点数归纳。删去新顶点 $v$，旧 tournament 有排名 $x_1\to\cdots\to x_n$。从左向右找第一个满足 $v\to x_i$ 的位置；把 $v$ 插在 $x_{i-1}$ 与 $x_i$ 间。此前若存在则有 $x_{i-1}\to v$，之后有 $v\to x_i$；若没有这样的 $i$，把 $v$ 放末尾。得到包含全部顶点的 path。

### 自检与知识链

> [!question]- 自检 1
> 为什么“存在正 closed walk”不能直接说“这条 walk 是 cycle”？
>
> [!success]- 答案
> walk 可重复内部顶点和边；必须删去重复段才提取 cycle。

> [!question]- 自检 2
> 若 $A$ 是邻接矩阵，$(A^3)_{ii}>0$ 说明什么？
>
> [!success]- 答案
> 存在从 $i$ 出发、三条边后回到 $i$ 的 closed walk；不必是三角形，因为可含自环或重复。

> [!question]- 自检 3
> 有向距离为什么一般不对称？
>
> [!success]- 答案
> $u\to v$ 的边不自动提供 $v\to u$；甚至一向可达而反向距离为 $\infty$。

**知识链：**关系 → 有向边 → walk/path/cycle → 距离 → 邻接矩阵幂 → 可达闭包。

---

## Problem Set 6

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps6.pdf]]。

> [!example]- PS6 Problem 1：危险 RSA 消息比例
> $[0,pq)$ 中 gcd 恰为 $p$ 的消息是 $p,2p,\ldots,(q-1)p$，共 $q-1$ 个；gcd 恰为 $q$ 的消息有 $p-1$ 个，两集合不交。因此比例
> $$\frac{p+q-2}{pq}\approx\frac1p+\frac1q.$$
> 当 $p,q$ 都约为 200 位时，量级为 $10^{-200}$（常数因子约 $2$ 不改变最近数量级）。

> [!example]- PS6 Problem 2：互相可达与 cycle
> (a) 取三点 $u,x,v$，边为 $u\to x,x\to v,v\to x,x\to u$。$u,v$ 互相可达，但任何同时经过二者的 closed walk 必须重复瓶颈 $x$，所以没有包含二者的 cycle。
>
> (b) 从任一由 $v$ 出发并回到 $v$ 的正 walk 开始。若内部顶点重复，删去两次出现之间的 closed 子段；持续操作直到内部无重复。首尾仍为 $v$，于是得到包含 $v$ 的 cycle。

> [!example]- PS6 Problem 3：King Chicken Theorem
> (a) 令 $v$ 只击败 $w$，令 $w$ 击败其余八只；其余边任意定向。于是 $v$ 出度 $1$，却能经 $w$ 在两步内到达所有鸡。
>
> (b) 在 $\mathbb Z_5$ 上令 $i$ 击败 $i+1,i+2$。每点直接击败两点，并通过其中一点两步到达剩余两点，故全是 king。
>
> (c) 设 $v$ 出度最大。对任意击败 $v$ 的 $u$，若不存在 $v\to w\to u$，则 $u$ 必击败 $v$ 的每个出邻居，再加上 $u\to v$，得到 $\deg^+(u)\ge\deg^+(v)+1$，与最大性矛盾。因此 $v$ 对每点均可在至多两步到达。

---

## Midterm 2

原题：[[MIT_OCW_6.042J_Materials/07_Exams/MIT6_042JS15_midterm2.pdf]]。

> [!example]- Midterm 2 Problem 1：Structural Induction
> 对 RAF 表达式 $h$ 定义 $P(h):\forall g\in RAF, h\circ g\in RAF$。
>
> **恒等基例：** $\operatorname{id}\circ g=g\in RAF$。**常函数基例：** $c\circ g=c$，仍是 RAF 常函数。
>
> **构造步：** 若 $h=f\operatorname{op}k$，其中 $\operatorname{op}\in\{+,\cdot,/\}$，归纳假设给 $f\circ g,k\circ g\in RAF$。逐点计算
> $$(h\circ g)(x)=f(g(x))\operatorname{op}k(g(x))=((f\circ g)\operatorname{op}(k\circ g))(x),$$
> 由 RAF 构造规则得 $h\circ g\in RAF$。除法情形在分母非零的共同定义域上理解。

> [!example]- Midterm 2 Problem 2：Buckets invariant
> (a) 清空小桶：$(b,l)\to(b,0)$。大桶倒入小桶：若 $b+l\le10$，转到 $(0,b+l)$；若 $b+l>10$，转到 $(b+l-10,10)$。
>
> (b) 取不变量“$b,l$ 都是 $5$ 的倍数”。初态 $(0,0)$ 成立。填满加入容量 $25$ 或 $10$；清空变 $0$；倒水只进行和、差 $10$，均保持 $5$ 的倍数。其余对称操作同理。因此所有可达状态的 $b$ 都被 $5$ 整除，而 $13$ 不是，$(13,x)$ 不可达。

> [!example]- Midterm 2 Problem 3：无限集加可数集
> 题设还给出 $A\cap B=\varnothing$。枚举 $B=\{b_0,b_1,\ldots\}$，并从无限集 $A$ 中取互异的 $a_0,a_1,\ldots$。定义 $H:A\cup B\to A$：
> $$
> H(x)=
> \begin{cases}
> a_{2i},&x=b_i,\\
> a_{2i+1},&x=a_i,\\
> x,&x\in A\setminus\{a_0,a_1,\ldots\}.
> \end{cases}
> $$
> 由 $A\cap B=\varnothing$，三个分支的定义域两两不交；它们的像分别是 $\{a_{2i}\mid i\in\mathbb N\}$、$\{a_{2i+1}\mid i\in\mathbb N\}$ 与 $A\setminus\{a_i\mid i\in\mathbb N\}$，也两两不交且合并为 $A$。所以 $H$ 既单射又满射。取逆映射即得 $A\leftrightarrow A\cup B$ 的双射。

> [!example]- Midterm 2 Problem 4：GCDs
> 逐素数取指数最小/最大：
> $$\gcd(m,n,p)=2^3 7^4,$$
> $$\operatorname{lcm}(m,n,p)=2^9 3^4 5^{24}7^{6042}11^7 19^{30}.$$
> 对非空 $A$，$\nu_k(\gcd A)=\min\nu_k(A)$；当 $p$ 为素数，$\nu_p(\operatorname{lcm}A)=\max\nu_p(A)$。复合底数不能把 lcm 的估值写成各估值最大值：$a=2,b=3$ 时 $\nu_6(a)=\nu_6(b)=0$，但 $\nu_6(\operatorname{lcm}(2,3))=1$。

> [!example]- Midterm 2 Problem 5：合并同余
> $14\mid(a-b)$ 且 $5\mid(a-b)$。因 $\gcd(14,5)=1$，互素因子乘积整除同一个数，故 $70\mid(a-b)$，即 $a\equiv b\pmod{70}$。

> [!example]- Midterm 2 Problem 6：$\varphi(k)$ 的奇偶
> $\varphi(2)=1$；满足 $\varphi(k)=2$ 的三个例子是 $k=3,4,6$。
>
> 对 $k>2$，在 $\mathbb Z_k^*$ 中将 $r$ 与 $-r$ 配对。两者都可逆；若它们相同，则 $2r\equiv0\pmod k$，因 $r$ 可逆可约去，得 $k\mid2$，与 $k>2$ 矛盾。因此没有固定点，单位集合被分成二元组，$\varphi(k)$ 为偶数。

---

## Session 17 — Directed Acyclic Graphs

### 17.1 DAG 与拓扑排序

[[Directed Acyclic Graph|directed acyclic graph, DAG]] 是没有有向 cycle 的 digraph。课程先用 prerequisites 建模：边 $u\to v$ 表示 $u$ 必须先完成；正 walk 关系给出间接先修关系。

**有限 DAG 必有入度为 $0$ 的顶点。** 反证：若每点都有入边，从任一点不断沿入边逆行。有限性保证某顶点重复，重复段形成有向 cycle，矛盾。对出度为 $0$ 的顶点同理。

[[Topological Sort|拓扑排序]]可由此递归构造：反复删除一个入度为 $0$ 的点并输出。每条边的起点必在终点之前。反过来，若图存在拓扑顺序，就不可能有 cycle，因为沿 cycle 每一步都要求位置严格增加，最后却回到起点。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-dag-topological-order.png|900]]

读图：拓扑序把 DAG 的所有箭头统一排成从早到晚，每个任务都位于其后继任务之前。

> [!theorem] 等价条件
> 对有限 digraph，以下等价：无有向 cycle；存在拓扑排序；每个非空诱导子图都有 source；正 walk 关系是严格偏序。

### 17.2 调度、chain 与 antichain

若每项任务用一单位时间，依赖关系形成 DAG：

- 一条 [[Chain|chain]] 中的任务必须分期执行，所以最长 chain 长度是工期下界；
- 一个 [[Antichain|antichain]] 中任务两两不可比，可以并行；
- 处理器数为 $P$ 时，总工作量还给下界 $\lceil n/P\rceil$。

无限处理器下，按“最长前驱链长度”分层：第 $i$ 层放所有最长前驱链长度为 $i$ 的任务。边必从低层指向高层，故每层可并行；层数恰等于最长 chain。这证明该下界可达。

有任务时长 $w(v)$ 时，critical path 的权重和仍是下界，但还要同时检查总工作量/处理器数；两者都只是下界，非抢占调度可能因空闲缝隙而更长。

### 17.3 覆盖边与 transitive reduction

若 $a\to b$ 且所有 $a$ 到 $b$ 的 path 都必须用这条边，称它为 covering edge。有限 DAG 删除所有可由更长 path 替代的边后，仍保留同一正可达关系；所得 covering subgraph 是唯一的最小 transitive reduction。无环性很关键：有 cycle 时不同最小表示可能不唯一。

### 官方顺序、资源与在线题（13 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session17.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Sdw8_0RDZuw.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/1TpzSCMLg08.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/cUYTlKA8jaw.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_DAGs.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Scheduling.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_TimeProcsors.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.6.2 Q1 | digraph 不允许的特征（边为集合的本课约定） | 同一方向重复边；空顶点集 |
| 2.6.2 Q2 | 从 $V$ 删边得 $U$ 后 shortest path | 不会更短 |
| 2.6.2 Q3 | 只保留 covering edges 后的 longest path | 长度与原 DAG **相同**；一条最长 path 不可含可被更长路径替代的边 |
| 2.6.4 Q1 | `6.042, 6.046, 6.01` | neither |
| 2.6.4 Q2 | `18.01, 6.02, 6.004` | chain |
| 2.6.4 Q3 | `6.042, 6.02, 6.034` | antichain |
| 2.6.4 Q4–Q5 | 最大 antichain；按拓扑序一学期一门需几学期 | $5$；$12$ |
| 2.6.6 Q1 | 无限处理器的最短学期数 | maximum chain size |
| 2.6.6 Q2 | 达到最短工期所需每期最大课数的上界 | maximum antichain size |
| 2.6.6 Q3 | 同一数量的下界 | $\lceil n/\text{max-chain}\rceil$ |
| 2.6.6 Q4 | max chain 的下界 | $n/\text{max-antichain}$ |
| 2.6.7 Q1 | 在 $1,\ldots,12$ 的 divisibility DAG 加点 $24$ 最少加几条 cover edges | $2$，从 $8,12$ 指向 $24$ |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S17_2.6.2_dags.md|2.6.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S17_2.6.4_scheduling-prerequisites.md|2.6.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S17_2.6.6_processor-time-bounds.md|2.6.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S17_2.6.7_the-divisibility-dag.md|2.6.7]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp17.pdf]]。

> [!example]- CP17 Problem 1：课程 DAG 调度
> (a) 最长 chain $18.01\to18.03\to6.002\to6.004\to6.033\to6.857$ 有六门，故至少六学期。按每期所有当前 sources 的 greedy schedule：
>
> 1. $18.01,8.01,6.001$；
> 2. $6.042,18.02,18.03,8.02,6.034$；
> 3. $6.046,6.002$；
> 4. $6.840,6.003,6.004$；
> 5. $6.033$；
> 6. $6.857$。
>
> (b) 一个不含 $18.03$ 的五门 antichain 是 $\{6.042,18.02,6.034,6.003,6.004\}$。完整枚举有 $9$ 个：固定 $18.02,6.034,6.003$，再从 $\{6.042,6.046,6.840\}$ 与 $\{6.004,6.033,6.857\}$ 各选一个。
>
> (c) 任一拓扑序即可，例如按上述六层从左到右逐门修，共 $15$ 学期。
>
> (d) 每期至多两门，工作量下界 $\lceil15/2\rceil=8$。可行的八期安排为：$(18.01,8.01)$，$(6.042,18.02)$，$(18.03,6.046)$，$(6.840,8.02)$，$(6.001,6.002)$，$(6.034,6.004)$，$(6.003,6.033)$，$(6.857)$，故恰为 $8$。
>
> (e) 三门上限时最长 chain 仍给 $6$。六期可行安排：$(18.01,8.01,6.001)$，$(6.042,18.03,8.02)$，$(18.02,6.046,6.002)$，$(6.840,6.034,6.004)$，$(6.003,6.033)$，$(6.857)$。

> [!example]- CP17 Problem 2：征服银河系
> (a) 任一拓扑顺序，例如 logo → fleet → seize UN → shots → Starbucks → train → launch → Microsoft。
>
> (b) 总工作量 $74$ 人日，双人给下界 $37$ 天；依赖可能造成一人空闲，所以未必可达。
>
> (c) critical path `logo–UN–Starbucks–train–Microsoft` 长 $8+9+10+4+8=39$ 天；即使路径外工作总量太大或排布不合，也可能更长。
>
> (d) 最优为 $41$ 天：一人做 logo $0$–$8$、UN $8$–$17$、Starbucks $17$–$27$；另一人 fleet $0$–$18$、shots $18$–$29$；随后 train $29$–$33$，两人并行 launch $33$–$39$ 与 Microsoft $33$–$41$。若想在 $40$ 天内结束，关键链最多只容许一日延迟，但 fleet 与 shots 共 $29$ 人日不能在另一处理器截止 train 前的可用窗口中完成；故 $41$ 最小。

> [!example]- CP17 Problem 3：$n$ 个任务、最长 chain 为 $t$
> (a) 同时可做的一组 = antichain（4）；单处理器可行次序 = topological sort（2）；无限处理器最短周数 = longest chain length（7）。
>
> (b) 工作量下界给至少 $\lceil n/t\rceil$ 个 Ringwraith。构造 $\lceil n/t\rceil$ 条顶点不交、彼此之间无边的 chains，将 $n$ 个任务尽量平均分配进去；每条 chain 长度至多 $t$。一个 Ringwraith 每周顺序执行自己 chain 中的下一项，便可在 $t$ 周内完成。因而 $\lceil n/t\rceil$ 确实是“最幸运的 DAG”可能需要的最小人数。
>
> (c) 最大永远不超过 $n-t+1$：按“以该任务结尾的最长 chain 长度”分成 $t$ 层调度。固定一条 $t$ 元最长 chain，它在每层恰贡献一个任务；任一层除这个任务外，至多再放入链外的 $n-t$ 个任务，故每层大小至多 $n-t+1$。这证明这么多人总是足够。为证紧性：$t=1$ 时取 $n$ 个互不相关任务；$t\ge2$ 时取 chain $c_1<\cdots<c_t$，并令每个额外任务 $x$ 都满足 $x<c_2$。要在恰好 $t$ 周内完成，chain 的 $c_i$ 被迫放在第 $i$ 周，而所有额外任务与 $c_1$ 都必须在第一周执行，恰需 $n-t+1$ 人。

> [!example]- CP17 Problem 4：covering edges
> (a) 图的边为 $1\to2,1\to3,1\to4,1\to5,1\to6,2\to4,2\to6,3\to6$。其中 $1\to4$ 可由 $1\to2\to4$ 代替，$1\to6$ 可由 $1\to2\to6$ 或 $1\to3\to6$ 代替；其余六条都是 covering edges。因此答案为 $1\to2,1\to3,1\to5,2\to4,2\to6,3\to6$。
>
> (b) 若 $a\leadsto b$，取一条最长的 $a$–$b$ path。若其中某边不是 covering edge，可用绕开该边的 path 替代，得到更长 walk；在 DAG 中 walk 不重复顶点，故成为更长 path，与最长性矛盾。所以 covering subgraph 仍含 $a$–$b$ path。
>
> (c) covering edge 可由正 walk 关系本身刻画为：$aE^+b$，且不存在 $c\ne a,b$ 使 $aE^+cE^+b$。故同一正 walk 关系给同一 covering edges。
>
> (d) 任何保持同一正可达关系的图都必须含所有 covering edges；(b) 又说明这些边已经足够，故它是唯一最小 DAG。
>
> (e) 在顶点 $\{1,2\}$ 上，令一图只有自环 $1\to1$，另一图只有自环 $2\to2$。cover 的定义只针对不同顶点，所以两图 covering edge 集都为空；正 walk 关系却分别含 $(1,1)$ 与 $(2,2)$，并不相同。
>
> (f) 三点 complete digraph（无自环）没有 covering edge，因为每条 $a\to b$ 都可绕经第三点。三-cycle 的三条边却全是 covering edges：从一端到下一端的任何 path 都必须经过该唯一离开方向。两图的正 walk 关系仍都为 $V\times V$。所以有环时，即使正可达关系相同，covering edges 也可能不同。

### 自检与知识链

> [!question]- 自检 1
> 为什么“每点入度至少 1”在有限 DAG 中不可能？
>
> [!success]- 答案
> 不断逆着入边走，有限点集必重复，重复段构成 cycle。

> [!question]- 自检 2
> 12 个单位任务、最长 chain 为 5、3 个处理器，最短工期至少多少？
>
> [!success]- 答案
> $\max(5,\lceil12/3\rceil)=5$；这是下界，仍需具体 DAG 才能判断能否达到。

> [!question]- 自检 3
> 为什么 transitive reduction 在有环图中可能不唯一？
>
> [!success]- 答案
> cycle 内可用不同方向边组合互相替代；可达闭包不再唯一指出哪条原边不可替代。

**知识链：**无 cycle → source → topological sort → chain/antichain → 调度界 → transitive reduction。

---

## Session 18 — Partial Orders and Equivalence

### 18.1 四组易混关系性质

对 $R\subseteq A\times A$：

- reflexive：$\forall a,\ aRa$；irreflexive：$\forall a,\ \lnot(aRa)$；
- symmetric：$aRb\Rightarrow bRa$；antisymmetric：$aRb\land bRa\Rightarrow a=b$；
- asymmetric：$aRb\Rightarrow\lnot(bRa)$；它蕴含 irreflexive；
- transitive：$aRb\land bRc\Rightarrow aRc$。

[[Partial Order|弱偏序]]是 reflexive + antisymmetric + transitive；严格偏序是 irreflexive + transitive。由弱偏序 $\preceq$ 可定义 $a\prec b\iff a\preceq b\land a\ne b$；反向可用 $a\preceq b\iff a\prec b\lor a=b$。

[[Equivalence Relation|等价关系]]是 reflexive + symmetric + transitive。每个 $a$ 的等价类 $[a]=\{x:xRa\}$；两个等价类要么相同，要么不交，因此等价关系与集合划分一一对应。

### 18.2 linear order、chains 与 Hasse diagram

偏序中两元素可比较若 $a\preceq b$ 或 $b\preceq a$。所有元素两两可比的偏序称 linear/total order。chain 是两两可比子集，antichain 是任意不同元素都不可比的子集。

[[Hasse Diagram|Hasse diagram]]只画 cover relation，并省略自环与由传递性可恢复的边。读图时“向上存在 path”才代表偏序，不应误认为只有画出的相邻边才相关。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-partial-order-hasse.png|900]]

读图：Hasse 图只留不能由中间元素推出的 cover edges，两点间只要存在向上路径就表示可比。

- maximal element：没有严格更大的元素；可能有多个。
- maximum：大于等于所有元素；至多一个，且一定 maximal。
- minimal/minimum 对偶。

### 18.3 product partial order 与 subset representation

若 $(A,\preceq_A),(B,\preceq_B)$ 为偏序，积偏序定义

$$
(a,b)\preceq(c,d)\iff a\preceq_Ac\land b\preceq_Bd.
$$

即使两个因子都是 linear order，积偏序通常不 linear：$(1,2)$ 与 $(2,1)$ 不可比。

任何有限偏序都可表示为集合包含：映射 $x\mapsto D(x)=\{y:y\preceq x\}$。若 $x\preceq z$，传递性给 $D(x)\subseteq D(z)$；反过来 $x\in D(x)\subseteq D(z)$，所以 $x\preceq z$。这解释了偏序为何可以画成“逐步增加信息”的结构。

### 官方顺序、资源与在线题（11 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session18.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/0w9luYcxHrw.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/bHvMYZvZp7Y.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/s-E5T3igntw.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_PartialOrder.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_ReprsentPrtal.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_EquivRelations.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.7.2 Q1 | 页面既定“按年龄”关系何时 linear | no two people are the same age |
| 2.7.2 Q2 | 两个 linear orders 的严格 product relation 性质 | antisymmetric、asymmetric、transitive、acyclic；不保证 linear |
| 2.7.5 Q1 | “同龄” | reflexive、transitive（也 symmetric） |
| 2.7.5 Q2 | “更年轻” | irreflexive、antisymmetric、transitive |
| 2.7.5 Q3 | “父母相同” | reflexive、transitive（也 symmetric） |
| 2.7.5 Q4 | “是后代” | irreflexive、antisymmetric、transitive |
| 2.7.5 Q5 | “至少一个共同父母” | reflexive；一般不传递 |
| 2.7.6 Q1 | 补集/交集关系性质 | symmetric 的补仍 symmetric；两个 reflexive 关系的交仍 reflexive |
| 2.7.7 Q1 | “同岁” | equivalence relation |
| 2.7.7 Q2 | “年龄不小于”作为年龄数值关系 | total order |
| 2.7.7 Q3 | “年龄整除”作为人的关系 | none of the above |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S18_2.7.2_population-partial-order.md|2.7.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S18_2.7.5_relational-properties.md|2.7.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S18_2.7.6_properties-of-relations.md|2.7.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S18_2.7.7_equivalence-relations-and-partial-orders.md|2.7.7]]

> [!warning] 定义域决定 antisymmetry
> 几道 population 题在“人”与“年龄数值”之间切换。年龄数值上的 $\ge$ 是 total order；搬到人身上后，两位同龄者会双向相关却不相等，从而破坏 antisymmetry。做题时先写清元素究竟是什么。

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp18.pdf]]。

> [!example]- CP18 Problem 1：逐项分类
> (a) $\supseteq$：弱偏序，不 linear（如 $\{1\},\{2\}$ 不可比）。
>
> (b) 模 $8$ 同余：等价关系。
>
> (c) 公式间有效蕴含：reflexive、transitive，但语法不同的逻辑等价公式相互蕴含，破坏 antisymmetry；也不 symmetric，故 none。
>
> (d) 有效 IFF：等价关系。
>
> (e) Rock–Paper–Scissors 的 beats：irreflexive、asymmetric，却有三-cycle，非 transitive，故 none。
>
> (f) 实数上的空关系：严格偏序，非 linear。
>
> (g) 整数上的 identity：既是等价关系，也是弱偏序；非 linear（不同整数不可比）。
>
> (h) $\mathbb Z$ 上整除：reflexive、transitive，但 $1\mid-1$ 且 $-1\mid1$ 而 $1\ne-1$，破坏 antisymmetry；也非 symmetric，故 none。

> [!example]- CP18 Problem 2：$\operatorname{pow}([1..6])$ 上的 $\subset$
> (a) maximal chain 有 $7$ 个集合，例如 $\varnothing\subset\{1\}\subset\{1,2\}\subset\cdots\subset[1..6]$。
>
> (b) 所有三元子集组成大小 $\binom63=20$ 的 antichain；Sperner theorem 说明它最大。
>
> (c) 唯一 minimal/minimum 为 $\varnothing$，唯一 maximal/maximum 为 $[1..6]$。
>
> (d) 删除 $\varnothing$ 后，六个单元素集合都是 minimal，但没有 minimum；$[1..6]$ 仍是唯一 maximum。

> [!example]- CP18 Problem 3：subsequence 与 product order
> (a) 最大递增子序列为 $1238,1258$；最大递减子序列为 $641,642,643,653,753,953$。
>
> (b) 定义 $a\prec a'$ 当且仅当数值更小且在序列中更早。题例的 minimal 为 $6,4,1$；maximal 为 $9,8$。逐点检查：minimal 没有更早且更小者，maximal 没有更晚且更大者。
>
> (c) chain 正是递增子序列；antichain 按出现顺序读时数值必递减，正是递减子序列。
>
> (d) 对每个元素记录以它结尾的最长递增长度 $I$ 与最长递减长度 $D$。两个不同元素若在序列中有先后，则数值必一大一小，因而对应的 $I$ 或 $D$ 严格增加，所以 $(I,D)$ 对彼此不同。若最长递增不超过 $\sqrt n$ 且最长递减严格小于 $\sqrt n$，可用的整数对少于 $n$，与 $n$ 个互异元素矛盾。故递增长度 $>\sqrt n$ 或递减长度 $\ge\sqrt n$。

> [!example]- CP18 Problem 4：函数的 kernel relation
> (a) $a\equiv_fa'$ 定义为 $f(a)=f(a')$。等号的自反、对称、传递直接给出等价关系三性质。
>
> (b) 给定等价关系 $R$，令 $f(a)=R(a)=[a]$。若 $aRa'$，等价类相同；若 $[a]=[a']$，因 $a\in[a]=[a']$，得 $aRa'$。所以 $aRa'\iff f(a)=f(a')$，即 $R=\equiv_f$。

### 自检与知识链

> [!question]- 自检 1
> antisymmetric 与 asymmetric 有何根本区别？
>
> [!success]- 答案
> antisymmetric 允许 $aRa$，只禁止不同元素双向相关；asymmetric 连自环也禁止，并要求一向成立时反向必不成立。

> [!question]- 自检 2
> 整除在正整数上与整数上为何分类不同？
>
> [!success]- 答案
> 正整数中 $a\mid b,b\mid a$ 推出 $a=b$；整数中还可能 $a=-b$，所以 antisymmetry 失败。

> [!question]- 自检 3
> 等价类为何不能部分重叠？
>
> [!success]- 答案
> 若 $x\in[a]\cap[b]$，则 $aRx$ 且 $xRb$，传递得 $aRb$；再用对称与传递可证 $[a]=[b]$。

**知识链：**关系性质 → strict/weak partial order → chains/Hasse → equivalence classes → quotient/representation。

---

## Session 19 — Degrees and Isomorphism

### 19.1 simple graph 与 degree

[[Simple Graph|简单图]] $G=(V,E)$ 的边是二元无序集合 $\{u,v\}$，无自环、无平行边。顶点 $v$ 的 degree $\deg(v)$ 是以它为端点的边数。

[[Handshaking Lemma|Handshaking lemma]]：

$$
\sum_{v\in V}\deg(v)=2|E|.
$$

**证明。** 对 incidence pairs $(v,e)$ 双重计数。按顶点先选得到左侧；按边先选，每条边有两个不同端点，得到右侧。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-handshake-lemma.png|900]]

读图：同一批“顶点–边关联对”按顶点数得 degree sum，按边数则每边恰贡献两次。

推论：奇度顶点个数为偶数。因为总度数为偶数，偶度顶点贡献偶数；若有奇数个奇度顶点，它们的 degree 之和为奇数，矛盾。

### 19.2 图同构

[[Graph Isomorphism|图同构]]是双射 $f:V(G)\to V(H)$，满足

$$
\{u,v\}\in E(G)\iff\{f(u),f(v)\}\in E(H).
$$

同构保留所有只依赖邻接结构的性质：顶点/边数、degree multiset、walk/cycle 数、连通性、着色数。图纸上的坐标、边长、顶点名字不是图结构。

**degree 被保留的证明。** $f$ 将 $u$ 的邻居集合双射到 $f(u)$ 的邻居集合；取基数得 $\deg_G(u)=\deg_H(f(u))$。因此 degree sequence 不同足以证明不同构，但相同 degree sequence 不保证同构。

### 官方顺序、资源与在线题（8 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session19.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TIpnudrzvgg.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/hVerxuP4cFg.pdf]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Degrees.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Isomorphism.pdf]]。

| block | prompt | 官方答案 |
|---|---|---|
| 2.8.2 Q1 | degrees 为 $4,3,3,2,2$ 的图有几条边 | $(4+3+3+2+2)/2=7$ |
| 2.8.2 Q2 | 所有 simple graphs 必真 | degree sum 为偶数；奇度顶点数为偶数 |
| 2.8.4 Q1 | 三对图中哪些同构 | Pair 2 与 Pair 3 |
| 2.8.4 Q2 | 不被同构保留的性质 | “顶点的数值标签均非偶数” |
| 2.8.5 Q1–Q2 | connected graph 总 degree $44$ 的最少/最多顶点 | $8$；$23$ |
| 2.8.6 Q1 | 题图两图间同构数 | $10$ |
| 2.8.7 Q1 | 可证明两图不同构的断言 | $H$ 有 degree-4 点而 $G$ 无；$G$ 有四个 degree-3 点而 $H$ 仅两个 |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S19_2.8.2_counting-degrees-and-edges.md|2.8.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S19_2.8.4_isomorphism.md|2.8.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S19_2.8.5_extreme-graphs.md|2.8.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S19_2.8.6_isomorphic-graphs.md|2.8.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S19_2.8.7_non-isomorphic-graphs.md|2.8.7]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp19.pdf]]。

> [!example]- CP19 Problem 1：异性伴侣平均数
> 设组内关系边数为 $E$。男性平均 $E/m$，女性平均 $E/f$，前者高 $10\%$ 给 $E/m=1.1E/f$，故 $m=(10/11)f$。两个平均数的分母不同，所以“每条边有一男一女”不推出两个平均相等。
>
> 排除 virgin 后，男性人数 $0.95m$，女性 $0.8f$，平均数比为
> $$\frac{E/(0.95m)}{E/(0.8f)}=\frac{16}{19}\frac fm,$$
> 所以题设 $x=16/19$。又 $m<f$，不可能把每位女性单射配给不同男性。

> [!example]- CP19 Problem 2：奇度点与 handshake sequence
> (a) handshaking lemma 使奇度点数为偶数。(b) 握手图直接应用。(c) George 所在 connected component 内的 degree sum 也为偶数，因此该分量内奇度顶点数为偶数；既有 George，就还至少有另一奇度点。连通分量定义保证二者间有 handshake path。

> [!example]- CP19 Problem 3：列出题图同构
> 两图结构均把点分成三类：$\{1,2\}$（或 $\{a,b\}$）是可交换的两个上方点；$\{3,4\}$（或 $\{c,d\}$）是左右接口；$\{5,6\}$（或 $\{e,f\}$）是 degree-2 底边端点。所有同构为：任选 $1\mapsto a,2\mapsto b$ 或交换；再任选 $(3,4,5,6)\mapsto(c,d,e,f)$ 或同时左右反转为 $(d,c,f,e)$。共 $2\cdot2=4$ 个。degree 与“是否位于唯一的 degree-2–degree-2 边”固定了三类点，所以没有其他同构。

> [!example]- CP19 Problem 4：哪些是同构不变量
> (a) 是，本质为顶点数 $7$；(b) 是，Hamilton cycle；(c) 是，degree multiset；(d) 否，当前图纸上的边长；(e) 是，无桥/删任一边仍连通；(f) 是，两点不交 cycles；(g) 否，顶点标签的集合含义；(h) 是，若命题是“存在某种所有边等长的画法”，可沿同构搬运该画法；(i) 是，两个不变量的 OR 仍不变；(j) 是，不变量的否定仍不变。

> [!example]- CP19 Supplemental Problem 5
> 对任意 $h\in V(H)$：
> $$h\in H(f(v))\iff(f(v),h)\in E(H)\iff(v,f^{-1}(h))\in E(G)$$
> $$\iff f^{-1}(h)\in G(v)\iff h\in f(G(v)).$$
> 故 $f(G(v))=H(f(v))$。$f$ 因而把 outdegree-$k$ 顶点双射到 outdegree-$k$ 顶点，两图此类顶点数相同。

### 自检与知识链

> [!question]- 自检 1
> degree sequence 相同为何不能证明同构？
>
> [!success]- 答案
> degree 只记录局部计数，不记录这些邻居如何彼此连接；可有相同 degrees 而 cycle/连通分量不同。

> [!question]- 自检 2
> 是否存在恰有三个奇度顶点的 simple graph？
>
> [!success]- 答案
> 不存在，handshaking lemma 的奇度推论排除。

> [!question]- 自检 3
> 顶点重命名后 adjacency matrix 为什么可能变化但图仍同构？
>
> [!success]- 答案
> 重命名同时置换矩阵的行和列；数表位置变了，但邻接关系未变。

**知识链：**无向边 → degree → double counting → handshaking → 邻接保持双射 → 图不变量。

---

## Problem Set 7

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps7.pdf]]。

> [!example]- PS7 Problem 1：transitive relations 的运算
> (a) $R^{-1}$ 必 transitive：$aR^{-1}b,bR^{-1}c$ 等价于 $bRa,cRb$；由 $cRa$ 得 $aR^{-1}c$。
>
> (b) $R\cap S$ 必 transitive，因为两条前提同时在 $R,S$ 中，两个传递性分别给结论。
>
> (c) $R\circ R$ 必 transitive。若 $a(R\circ R)b$ 与 $b(R\circ R)c$，存在 $x,y$ 使 $aRxRb$ 与 $bRyRc$；传递性给 $xRy$、再给 $xRc$，故 $aRxRc$，即 $a(R\circ R)c$。
>
> (d) $R\circ S$ 不一定。集合 $\{0,1,2\}$ 上取 $R=\{(1,1),(2,0)\}$、$S=\{(0,1),(2,2)\}$；两者各自 transitive，但按 $(a,c)\in R\circ S\iff\exists b(aSb\land bRc)$，复合含 $(2,0),(0,1)$ 而不含 $(2,1)$。

> [!example]- PS7 Problem 2：equivalence relations 的交与并
> (a) $R_1\cap R_2$ 仍 reflexive、symmetric、transitive，所以是等价关系。
>
> (b) 并不一定 transitive。令 $R_1$ 的非单点类为 $\{1,2\}$，$R_2$ 的非单点类为 $\{2,3\}$。并关系有 $1R2,2R3$，却无 $1R3$。

> [!example]- PS7 Problem 3：四张图的同构分类
> 先用不变量排除，再给出真正的同构，不能凭“画得像”判断。$G_3$ 的顶点 $8,10$ 均为 degree $4$，其余三图所有顶点均为 degree $3$，所以 $G_3$ 自成一类。$G_2$ 含 4-cycle，例如 $2-3-8-7-2$；$G_1,G_4$ 均不含 4-cycle，所以 $G_2$ 也不与它们同构。
>
> 这两个排除标准确为同构不变量：上文已证同构把每个顶点的邻居集双射到像顶点的邻居集，因而保持 degree；若 $v_1-v_2-v_3-v_4-v_1$ 是 4-cycle，邻接保持性使 $f(v_1)-f(v_2)-f(v_3)-f(v_4)-f(v_1)$ 仍是 4-cycle，对 $f^{-1}$ 同理，故“存在 4-cycle”也被保持。
>
> $G_1\to G_4$ 的一个完整同构是
> $$
> \begin{array}{c|cccccccccc}
> v&1&2&3&4&5&6&7&8&9&10\\ \hline
> f(v)&1&2&6&7&10&9&5&4&3&8
> \end{array}
> $$
> 直接检查 $G_1$ 的外五环、内五环及五条配对边，它们的端点像在 $G_4$ 中仍然相邻；又因 $f$ 是十个顶点上的双射，所以所有边且仅有边被保持。结论：唯一的非平凡同构类是 $\{G_1,G_4\}$，$G_2$ 与 $G_3$ 各自成类。

> [!example]- PS7 Problem 4：two-ended 不推出 line graph
> (a) 取“不相交的一个三角形与一条单边”。单边两端 degree $1$，三角形三点 degree $2$，所以恰 two-ended；但图不连通，不是 line graph。
>
> (b) bogus proof 的第一处错误是声称所有 $G_{n+1}$ 都能由某个 two-ended $G_n$ 加一条边得到。对上述反例，无论删三角形边还是孤立单边，所得图都不再恰有两个 degree-1 点。归纳步只证明了某种构造保持性质，没有覆盖所有 $n+1$ 阶对象。

---

## Session 20 — Coloring and Connectivity

### 学习问题与前置知识

本节回答三个常在算法中出现的问题：如何把“相互冲突的对象不能共用资源”翻译成图着色？如何用 odd cycle 完全判定二着色可能性？一张图在删掉多少顶点或边后仍能连通？前置知识是 Session 19 的 simple graph、path、cycle 和 degree。

### 20.1 proper coloring 与 chromatic number

给定 simple graph $G=(V,E)$，一个 **proper $k$-coloring** 是函数

$$
c:V\to\{1,2,\ldots,k\},
$$

满足每条边 $\{u,v\}\in E$ 的两端颜色不同，即 $c(u)\ne c(v)$。图的 [[Chromatic Number|着色数]]

$$
\chi(G)=\min\{k:G\text{ has a proper }k\text{-coloring}\}
$$

是所需颜色数的最小值。它衡量的不是“图有多大”，而是冲突约束能够被多少类兼容资源承担。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-graph-coloring.png|900]]

读图：相邻顶点必用不同颜色，着色数就是满足全部冲突边所需的最少资源类别。

三个立即可用的边界：

1. 若 $G$ 含 $K_r$ 作为子图，则 $\chi(G)\ge r$，因为这 $r$ 个顶点两两相邻。
2. 若 $G$ 有至少一条边，则 $\chi(G)\ge2$；若无边且非空，则 $\chi(G)=1$。
3. 贪心地按任意顶点顺序着色，每次选邻居未用的最小颜色，至多用 $\Delta(G)+1$ 色，因为当前顶点至多有 $\Delta(G)$ 种被邻居占用的颜色。

> [!warning]
> $\Delta(G)+1$ 是普适上界，不是下界。有 $n-1$ 最大度的 star graph 仍只需 $2$ 色。“某个点邻居多”并不意味着邻居之间也冲突。

#### 从寄存器分配看建模

编译器中，每个变量是顶点；若两个变量的 **live ranges** 重叠，它们在某个时刻必须同时保存，就在两点间连边。一种颜色对应一个 register；同色顶点没有重叠 live range，可以安全重用该 register。因此最小寄存器数正是冲突图的 $\chi(G)$。

### 20.2 bipartite iff no odd cycle

图 $G$ 是 [[Bipartite Graph|二分图]]，若 $V$ 可分成不交集 $L,R$，且每条边都一端在 $L$、一端在 $R$。这等价于 $G$ 可 $2$-color。

> [!theorem] Odd-cycle characterization
> 一张 finite simple graph 是 bipartite，当且仅当它不含奇数长度 cycle。

**必要性。** 假设 $G$ 已用两色着色。沿 cycle $v_0,v_1,\ldots,v_{m-1},v_0$ 前进时，每过一条边颜色必切换一次。回到 $v_0$ 时必须回到原颜色，所以切换次数 $m$ 为偶数。因此 odd cycle 不可能存在。

**充分性。** 对每个 connected component 选根 $r$ 并建立一棵 BFS tree，把 BFS 距离 $d(r,v)$ 为偶数的点染蓝，为奇数的点染橙。需证每条边 $\{u,v\}$ 的两端奇偶性不同。反设 $d(r,u),d(r,v)$ 同奇偶，在 BFS tree 中取根到 $u,v$ 两条唯一 tree paths 的最后公共顶点 $z$。$z$ 之后的两段 tree path 内部顶点不交，再加边 $\{u,v\}$ 确实形成 simple cycle；其长度为

$$
\bigl(d(r,u)-d(r,z)\bigr)+\bigl(d(r,v)-d(r,z)\bigr)+1,
$$

前两项之和为偶数，故 cycle 为奇长，与假设矛盾。所以该着色 proper，$G$ bipartite。

这个证明同时给出线性时间算法：用 BFS 时可直接以最短距离奇偶性着色；用 DFS 时则沿搜索树的深度交替着色，不再把 DFS 深度解读为最短距离。若遇到同色边，搜索树中的两条路与该边就给出 odd-cycle certificate。

### 20.3 connectivity、components 与 cut

两顶点之间有 path 时称其 **connected**。“存在 path”在顶点集上是等价关系：

- reflexive：长度 $0$ 的 path 连自己；
- symmetric：反向读一条无向 path；
- transitive：串接两条 walk，删掉重复段得 path。

它的等价类就是 [[Connected Component|连通分量]]。图 connected 当且仅当只有一个 component。

一条边 $e$ 是 **bridge/cut edge**，若删除 $e$ 后 component 数增加。一个顶点 $v$ 是 **cut vertex/articulation point**，若删除 $v$ 及其 incident edges 后 component 数增加（对原本 connected 的图，这就是删点后不再 connected）。边在某个 cycle 上当且仅当它不是 bridge：若在 cycle 上，删边后可绕行；若删边后两端仍有 path，该 path 加原边形成 cycle。

### 20.4 $k$-vertex-connectivity 与 $k$-edge-connectivity

对至少 $k+1$ 个顶点的图：

- $G$ 是 **$k$-vertex-connected**，若删掉任意少于 $k$ 个顶点后仍 connected；
- $G$ 是 **$k$-edge-connected**，若删掉任意少于 $k$ 条边后仍 connected。

其最大 $k$ 分别记为 vertex connectivity $\kappa(G)$ 与 edge connectivity $\lambda(G)$。对非平凡 connected graph，有

$$
\kappa(G)\le\lambda(G)\le\delta(G),
$$

其中 $\delta(G)$ 是最小 degree。右不等式因为删掉某个最小度顶点的所有 incident edges 会孤立它。左不等式的直观是：“删点”还会同时删掉该点的所有边，通常比单独删边更有破坏力。课程中用切集转换证明 $k$-vertex-connected $\Rightarrow k$-edge-connected。

#### 为什么 $K_n$ 恰是 $(n-1)$-connected

删掉至多 $n-2$ 个顶点后，所余顶点仍两两相邻，因而 connected；删掉 $n-1$ 个顶点已超出通常连通度定义的非平凡范围。因此 $\kappa(K_n)=n-1$，同样 $\lambda(K_n)=n-1$。

### 官方顺序、资源与在线题（8 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session20.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/Penh4mv5gAg.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/TIQ3xN38jgM.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/5wCZqdCDafc.pdf]]。字幕：[[MIT_OCW_6.042J_Materials/04_Captions/Penh4mv5gAg.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/TIQ3xN38jgM.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/5wCZqdCDafc.srt]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Coloring.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_graphconnectivity.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_k-connectivity.pdf]]。

| block | prompt | 官方答案与核验 |
|---|---|---|
| 2.9.2 Q1 | maximum degree $k$ 是否强迫 $\chi(G)\ge k$ | False；$n$-vertex star 的 $\Delta=n-1$，但 $\chi=2$ |
| 2.9.2 Q2 | 偶数顶点 wheel graph 的着色数 | $3$；课件约定下 rim 为偶 cycle，用两色交替，hub 用第三色 |
| 2.9.2 Q3 | $K_n$ 需多少色 | $n$；每两点相邻 |
| 2.9.5 Q1 | $k$-edge/顶点连通的有效推论 | $k$-vertex-connected $\Rightarrow k$-edge-connected |
| 2.9.5 Q2 | $K_n$ 的连通度 | $(n-1)$-connected |
| 2.9.6 Q1 | 题图的 chromatic number | $3$；图含 triangle，而外圈可两色交替、中心用第三色 |
| 2.9.7 Q1 | acyclic graph 的 chromatic number | 有边时为 $2$；无边非空图为 $1$ |
| 2.9.8 Q1 | 整数为顶点、$|i-j|=6$ 时连边，有几个 components | $6$；路径保持模 $6$ 余数，每个余数类内又可通过 $\pm6$ 到达 |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S20_2.9.2_chromatic-number.md|2.9.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S20_2.9.5_k-connected.md|2.9.5]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S20_2.9.6_graph-coloring-i.md|2.9.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S20_2.9.7_graph-coloring-ii.md|2.9.7]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S20_2.9.8_connected-components-in-integers.md|2.9.8]]

> [!note]
> Wheel 的符号约定在不同教材中不一：有的把 $W_n$ 的 $n$ 指总顶点数，有的指 rim 顶点数。上表保留本 OCW 题库的官方答案 $3$；真正判断规则是：rim 为偶 cycle 时 $\chi=3$，rim 为奇 cycle 时 $\chi=4$。

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp20.pdf]]。

> [!example]- CP20 Problem 1：register allocation
> **(a) 建图。** 每个变量 $a,b,c,d,e,f,g,h$ 是一个顶点。从某变量获得值到它最后一次被使用（若为 output，则到程序末）是它的 live interval；两个值必须同时保留时，对应顶点间连冲突边。例如 $a,b$ 同为 inputs，故相邻；$c,d$ 在 Step 2 之后都仍 live，也相邻；$b$ 在 Step 1 后已 dead，因而它的 register 可用来存新算出的 $c$。
>
> **(b) 最小着色。** 先注意 $a$ 的最后一次使用是 Step 2，所以它在 Step 3 结束时已经 dead，不能把 $a,c,d,e$ 当作 $K_4$。Step 5 结束后，$d,g$ 都是必须保留到程序末的 outputs，$f$ 还要在 Step 6 中使用；故 $\{d,f,g\}$ 三者同时 live，形成 $K_3$，至少需 $3$ 个 registers。下列分配恰好使用 $3$ 个：
>
> | register | 按时间重用的变量 |
> |---|---|
> | $R_1$ | $d$ |
> | $R_2$ | $a\rightarrow e\rightarrow g$ |
> | $R_3$ | $b\rightarrow c\rightarrow f\rightarrow h$ |
>
> 箭头表示前者最后一次被读后，register 才被后者覆写：Step 1 用 $c$ 覆写已经用完的 $b$，Step 3 用 $e$ 覆写已用完的 $a$，Step 4 用 $f$ 覆写 $c$，Step 5 用 $g$ 覆写 $e$，Step 6 用 $h$ 覆写 $f$。这给出三着色上界；结合 $K_3$ 下界，得 $\chi(G)=3$，最少需 $3$ 个 registers。
>
> **(c) 变量被多次赋值。** 先做 SSA-style renaming：把两次 $t$ 分成 $t_1=r+s$ 与 $t_2=m-k$，并把各次读取改为对应版本，如 $u=t_1-3,v=t_2+u$。每个版本只有一个起点，live range 才不会被错误合并。

> [!example]- CP20 Problem 2：positive degree 不推出 connected
> **(a) 反例。** 取两条不相交的单边，四个顶点 degree 均为 $1>0$，但图有两个 components。
>
> **(b) 第一个非法步骤。** 归纳步从“任意一张满足前件的 $(n+1)$-顶点图”出发，应先删去一点得到仍满足前件的 $n$-顶点图。bogus proof 却从一张已知 connected 的 $n$-顶点图出发，再特别地添加 $x$。它只覆盖了“删掉 $x$ 后仍满足 positive degree”的图；若某顶点唯一邻居是 $x$，删 $x$ 后该点 degree 变 $0$，不能套 $P(n)$。

> [!example]- CP20 Problem 3：3-color OR/AND gadget
> 记左侧 triangle 的三种颜色为 $N,T,F$，点线表示与 $N$ 相邻。
>
> **(a)** 因 $P,Q$ 都与 $N$ 相邻，在有效 $3$-coloring 中它们只能取 $T$ 或 $F$，因而“存在着色”必要地推出二者都非 $N$。反向，对 $(P,Q)$ 的四种 $T/F$ 输入逐一传播颜色，内部顶点均能完成；因为只有四行，这一穷尽检查构成充分性证明。
>
> **(b) 真值表。** 沿两个 triangle 和菱形的边依次排除已被邻点占用的颜色，得
>
> | $P$ | $Q$ | $P\lor Q$ |
> |---|---|---|
> | $F$ | $F$ | $F$ |
> | $F$ | $T$ | $T$ |
> | $T$ | $F$ | $T$ |
> | $T$ | $T$ | $T$ |
>
> 每行不仅“可以”给输出该色，边约束还排除了另一种 $T/F$ 输出，所以 gadget 真正强制 OR 语义。
>
> **(c)** 把图中从颜色顶点 $T$ 指向左中间顶点的边，改成从 $F$ 指向该顶点；其余边不变。同样做四行排除，输出依次为 $F,F,F,T$，恰是 $P\land Q$。本题的关键是边的端点从 $T$ 改到 $F$，不是只换图上标签。

> [!example]- CP20 Problem 4：hypercube 的 connectivity
> $H_n$ 的顶点是 $n$-bit strings，两点恰差一位时相邻。由 XOR 对称，证 $H_3$ 时可令起点为 $000$，终点只需按 Hamming distance 分类。
>
> **distance $1$，以 $100$ 为终点：**
> $$000-100,$$
> $$000-010-110-100,$$
> $$000-001-101-100.$$
>
> **distance $2$，以 $110$ 为终点：**
> $$000-100-110,$$
> $$000-010-110,$$
> $$000-001-101-111-110.$$
>
> **distance $3$，以 $111$ 为终点：**
> $$000-100-110-111,$$
> $$000-010-011-111,$$
> $$000-001-101-111.$$
>
> 每组三条 path 除共用端点外内部顶点互不相交。因而删任意两个顶点都无法同时截断三条 path，有 $\kappa(H_3)\ge3$。反之，删掉某顶点的三个邻居会孤立该顶点，所以 $\kappa(H_3)\le3$。结论是 $\kappa(H_3)=3$。
>
> 对 $H_n$，每点 degree $n$，故 $\kappa(H_n)\le n$。下界可将上述“用不同首次翻转位绕行”的构造推广成 $n$ 条内部顶点不交 path，再用 Menger 定理；因此 $\kappa(H_n)=n$。

### 边界情况与易错点

- “有边的 forest 需两色”与“所有 acyclic graph 恰需两色”不完全相同；无边图只需一色。
- $k$-connected 中的“删少于 $k$”不是“删至少 $k$”；后者是破坏连通所需的最小 cut 大小。
- 证明 $\chi(G)=k$ 必须同时给上界（一个 $k$-着色）和下界（例如 clique/odd cycle）。
- register allocation 中边代表“生存期重叠”，不是“两个变量出现在同一条语句”这一更粗的条件。

### 自检与知识链

> [!question]- 自检 1
> 为什么一个含 triangle 的图不可能 bipartite？
>
> [!success]- 答案
> triangle 是长度 $3$ 的 odd cycle；二着色沿环交替三次后回到起点会要求起点同时取两色，矛盾。

> [!question]- 自检 2
> 已知 $\Delta(G)=20$，能否推出 $\chi(G)\ge20$？能推出什么？
>
> [!success]- 答案
> 不能，star 是反例。贪心着色只给出 $\chi(G)\le\Delta(G)+1=21$。

> [!question]- 自检 3
> 为什么删掉 $H_n$ 某点的 $n$ 个邻居证明 $\kappa(H_n)\le n$？
>
> [!success]- 答案
> 该点本身未被删，但它的所有 incident edges 都随邻居消失，于是它成为孤立 component，图不连通。

**知识链：**冲突关系 → graph coloring → chromatic number → bipartite/odd cycle → components → vertex/edge cuts → $k$-connectivity。

---

## Session 21 — Trees and Minimum Spanning Trees

### 学习问题与前置知识

本节把“连通”与“无环”压缩成一种最小骨架：tree。核心问题是：为什么 tree 有多种完全等价的定义？为什么 connected graph 一定含 spanning tree？带权时，为什么 Kruskal、Prim 和并行合并都可安全选取某条最轻边？前置知识是 path、cycle、component 与 coloring。

### 21.1 tree 的等价刻画

一张 finite simple graph $T=(V,E)$ 是 [[Tree|树]]，若它 **connected 且 acyclic**。不要把这两个形容词当成互不相干的条件：它们联合后迫使 tree 恰好处在“保持全部顶点连通”的边数下界。

> [!theorem] Tree 的常用等价定义
> 对 finite simple graph $T$，以下条件等价：
>
> 1. $T$ connected 且无 cycle；
> 2. 任意两顶点间恰有一条 simple path；
> 3. $T$ connected，且每条边都是 cut edge；
> 4. $T$ 对“connected”是 edge-minimal；
> 5. $T$ acyclic，且加任意一条新边都产生 cycle；
> 6. 若 $|V|=n$，则 $T$ connected 且 $|E|=n-1$。

#### $1\Longleftrightarrow2$：唯一 path

**$1\Rightarrow2$。** connected 保证任意 $u,v$ 间至少有一条 path。若有两条不同 simple paths，从 $u$ 同时沿它们前进，取第一个分叉点 $x$ 和分叉后第一个重会点 $y$。两条 $x$ 到 $y$ 的内部不交子路合成 cycle，与 acyclic 矛盾。因此 path 唯一。

**$2\Rightarrow1$。** “每两点有 path”直接给 connected。若有 cycle，取 cycle 上两点 $u,v$，沿 cycle 的顺、逆两个方向得两条不同 simple paths，违反唯一性。

#### $1\Longleftrightarrow3\Longleftrightarrow4$：每条边都必不可少

若 tree 中某边 $e=\{u,v\}$ 不是 cut edge，删它后 $u,v$ 仍有 path，该 path 加 $e$ 形成 cycle，矛盾。所以每边皆为 cut edge，即删任意边都破坏 connected，这正是 edge-minimal connected。

反向，若 connected graph 含 cycle，删掉 cycle 上任意一边后，两端可沿 cycle 剩余部分绕行，图仍 connected。这同时违反“每边都是 cut edge”和“edge-minimal”。

#### $1\Longleftrightarrow5$：无环性的极大对象

在 tree 的两个不相邻顶点 $u,v$ 之间加新边，原图中唯一的 $u$-$v$ path 与新边构成 cycle。反向，若 acyclic graph 不 connected，可取两个不同 components 中的点加边；新边两端原先无 path，所以不可能新增 cycle，与 edge-maximal acyclic 矛盾。

#### $1\Longleftrightarrow6$：$n-1$ 条边

先证 leaf lemma：至少两个顶点的 finite tree 有至少两片 [[Leaf Vertex|叶子]]（degree $1$ 顶点）。取一条最长 simple path $v_0,\ldots,v_m$。若 $v_0$ 还有不是 $v_1$ 的邻居 $w$，则 $w$ 若在路上会造成 cycle，若不在路上则 $w,v_0,\ldots,v_m$ 比原 path 更长；两者皆矛盾。故 $\deg(v_0)=1$，同理 $\deg(v_m)=1$。

现对 $n$ 归纳证 tree 有 $n-1$ 边。$n=1$ 时无边。$n\ge2$ 时取 leaf $v$，删去 $v$ 及其唯一 incident edge；所得图仍 connected 且 acyclic，是 $n-1$ 顶点 tree。归纳假设给它 $n-2$ 边，加回一边得 $n-1$。

反向，从任意 finite connected graph 出发，只要有 cycle 就删 cycle 上一边；连通性不变，边数严格减少，所以过程必停于某个 spanning tree。该 spanning tree 已有 $n-1$ 边。如果原 connected graph 也恰有 $n-1$ 边，整个删边过程不能真删任何边，故原图本身无 cycle，是 tree。

> [!warning]
> “$n$ 顶点、$n-1$ 边”单独不足以证明 tree。例如一个 triangle 加一个孤立点有 $4$ 顶点、$3$ 边，但它不 connected 且有 cycle。必须再知 connected 或 acyclic 其一。

### 21.2 forest、leaves 与二着色

[[Forest Graph|森林]]是 acyclic graph，每个 connected component 都是 tree。若 forest 有 $n$ 个顶点、$c$ 个 components，则

$$
|E|=n-c,
$$

因为第 $i$ 个 component 有 $n_i-1$ 边，求和得 $\sum_i(n_i-1)=n-c$。

所有 tree 都 $2$-colorable。一种证法是 tree 无 odd cycle，应用 Session 20 定理；更构造性的证法是选根 $r$，按唯一 $r$-$v$ path 的长度奇偶染色。每条 tree edge 使深度变 $1$，因而两端颜色不同。

按本课“leaf = degree-$1$ 顶点”的定义，边界必须分开：$n=1$ 时唯一顶点 degree $0$，没有 leaf；$n=2$ 时唯一的 tree 是 $K_2$，两个顶点都是 leaves。对 $n\ge3$，leaves 至少 $2$、至多 $n-1$：下界由 leaf lemma；若所有 $n\ge3$ 个顶点都 degree $1$，图只能是若干不交的单边，不可能 connected；star $K_{1,n-1}$ 达到上界 $n-1$。

### 21.3 spanning tree 与 MST

图 $G$ 的 **spanning subgraph** 保留 $G$ 的全部顶点，只可能删边。若该子图还是 tree，就是 [[Spanning Tree|生成树]]。一张 finite graph 有 spanning tree 当且仅当它 connected：

- 有 spanning tree 则其中的 path 也是原图 path，原图 connected；
- connected 则反复删 cycle edge，有限步后得到 connected acyclic spanning subgraph。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-spanning-tree.png|900]]

读图：spanning tree 保留原图全部顶点，只删去多余环边，最终用 $n-1$ 条边留下唯一路径骨架。

现令每条边 $e$ 有实数权重 $w(e)$。spanning tree $T$ 的总权重为

$$
w(T)=\sum_{e\in E(T)}w(e).
$$

[[Minimum Spanning Tree|最小生成树]]（MST）是总权重最小的 spanning tree。因 spanning trees 数量有限，connected finite weighted graph 一定存在 MST；权重相同时它可能不唯一。

### 21.4 cut/gray-edge lemma

将顶点分成两个非空部分 $S$ 与 $V\setminus S$称为 cut；一端在 $S$、一端在补集的边称为 crossing/gray edge。

> [!theorem] Cut property（gray-edge lemma）
> 给定任意 cut，其上权重最小的 crossing edge $e$ 属于某个 MST。若 $e$ 是该 cut 上唯一最轻边，则它属于每个 MST。

**目标。** 把一个未必包含 $e$ 的 MST 交换成包含 $e$ 且不更重的 tree。

**构造。** 取 MST $T$。若 $e\in T$ 已完成。否则在 $T$ 中加 $e$；由 tree 的唯一 path 性，恰产生一个 cycle $C$。$e$ 跨越 cut，cycle 每次进入 $S$ 最终必须离开 $S$，所以 $C$ 上还有另一条 crossing edge $f\in T$。

**逐步依据。** 删去 $f$，得 $T'=T+e-f$。删 cycle 上一边后仍 connected，且边数恢复 $n-1$，故 $T'$ 是 spanning tree。因 $e$ 是 cut 上最轻边，$w(e)\le w(f)$，于是

$$
w(T')=w(T)+w(e)-w(f)\le w(T).
$$

$T$ 已最小，所以必取等，$T'$ 也是包含 $e$ 的 MST。若 $e$ 唯一最轻，则 $w(e)<w(f)$ 会使不包含 $e$ 的 $T$ 被严格改善，矛盾；故每个 MST 都含 $e$。

### 21.5 Kruskal、Prim 与并行合并

**Kruskal.** 从空边集 $F$ 开始，按权重从小到大查看边；若加该边不产生 cycle就接受，否则跳过。当前 $F$ 始终是 forest。被接受的边连接两个不同 components，并是此时跨越对应 cut 的最轻可用边，因而由 cut property 是 safe edge。接受 $n-1$ 条后得 MST。

**Prim.** 从任意根 $r$ 的单点 tree $S$ 开始；每次选恰有一端在 $S$中的最轻边，把另一端加入。该边就是 cut $(S,V\setminus S)$ 的最轻边，故 safe。每次增一顶点且无 cycle，$n-1$ 次后得 MST。

**Component-growth / Borůvka 观点。** 严格的安全版可写成“一次选一个当前 component，加入离开它的最轻边”：该 component 与其余顶点形成 cut，故每次加入的边都由 cut property 保证 safe。不同 components 可并行**计算**候选边，但接受候选边时必须保持当前已选边仍为 forest，并在每次合并后按新 component 重新解释 cut。若边权可并列，不经协调地同时接受所有候选边可能形成等权 cycle；需固定 tie-breaking 或只接受其中的无环子集。

### 官方顺序、资源与在线题（19 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session21.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/ZEsk64C0fJg.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/g2mOvmC1TKc.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/RqqzyWDVMA.pdf]]。字幕：[[MIT_OCW_6.042J_Materials/04_Captions/ZEsk64C0fJg.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/g2mOvmC1TKc.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/RqqzyWDVMA.srt]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_trees.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_treescoloring.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_SpaingTrees.pdf]]。

| block | prompt | 官方答案与核验 |
|---|---|---|
| 2.10.2 Q1 | 哪些可作 tree 的定义 | 除“connected and vertex-minimal”外的六项：connected acyclic；每边为 cut edge；edge-minimal connected；edge-maximal acyclic；connected 且 $n-1$ 边；两点间唯一 path |
| 2.10.4 Q1 | “_____ trees are 2-colorable” | All |
| 2.10.6 Q1 | 保证存在 spanning tree 的条件 | connected |
| 2.10.6 Q2 | 什么子图保留 $G$ 全部顶点 | exact copy、spanning tree、spanning subgraph 均是，故 all of the above |
| 2.10.6 Q3 | 任意最小_____ connected spanning _____ 是 spanning _____ | edge, graph, tree |
| 2.10.6 Q4 | gray-edge method 的两个空 | coloring **components** black/white，选 **minimum-weight** gray edge |
| 2.10.7 Q1 | 四张图哪些是 trees | Graphs $2,4$；$1$ disconnected，$3$ 含 cycle |
| 2.10.8 Q1 | 所有顶点都是 leaves 的 tree 最多几个顶点 | $2$；再多就不能 connected |
| 2.10.8 Q2 | $99$-顶点 tree 最少 leaves | $2$；path 达到 |
| 2.10.8 Q3 | $99$-顶点 tree 最多 leaves | $98$；star 达到 |
| 2.10.8 Q4 | $1000$-顶点 forest 最多 leaves | $1000$；取 $500$ 个互不相交的 $K_2$ components，每个顶点 degree $1$，故全部 $1000$ 点都是 leaves |
| 2.10.9 Q1 | MST 是否总唯一 | False；等权 triangle 有三个 MST |
| 2.10.9 Q2 | 边权互异时 MST 必真的断言 | MST 唯一；全局最小边在 MST 中 |
| 2.10.10 Q1 | 选边算法从初始就成立的 preserved invariant | marked edges 始终 acyclic |
| 2.10.10 Q2 | unmarked edges 的数量 | strictly decreasing |
| 2.10.10 Q3 | marked edges 的数量 | strictly increasing |
| 2.10.10 Q4 | marked + unmarked 边数 | constant |
| 2.10.10 Q5 | marked $-$ unmarked 边数 | strictly increasing，每次增 $2$ |
| 2.10.10 Q6 | marked-edge subgraph 的 component 数 | strictly decreasing，每条新 marked edge 合并两分量 |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.2_trees-many-definitions.md|2.10.2]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.4_2-colorable-trees.md|2.10.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.6_span-all-the-graphs.md|2.10.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.7_tree-or-not-tree.md|2.10.7]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.8_leaves.md|2.10.8]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.9_minimum-spanning-trees.md|2.10.9]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S21_2.10.10_graph-algorithm.md|2.10.10]]

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp21.pdf]]。

> [!example]- CP21 Problem 1：$4\times4$ grid 的 MST
> 水平边权为 $w(h_{i,j})=(4i+j)/100$，全部落在 $0$ 到 $0.11$；竖直边权为 $w(v_{j,i})=1+(i+4j)/100$，全部至少 $1$。因此 Kruskal 先选全部 $12$ 条水平边；它们把 $4$ 行分别连成 path，不产生 cycle。然后只需最轻的三条竖直边
> $$v_{0,0},\quad v_{0,1},\quad v_{0,2}$$
> 把四行串起来。总边数 $12+3=15=16-1$，图 connected，故这是 spanning tree。总重量为
> $$
> \sum_{i=0}^{2}\sum_{j=0}^{3}\frac{4i+j}{100}
> +\sum_{i=0}^{2}\left(1+\frac{i}{100}\right)
> =\frac{66}{100}+3.03=3.69.
> $$
>
> **Kruskal 正确性。** 每次选不闭环的全局最轻边，该边连接当前 forest 的两个 components；对其中一个 component 构造 cut，gray-edge lemma 证它 safe。
>
> **Prim 正确性。** 从 $(1,2)$ 生长时，每次所选边是当前 tree 与外部之间的最轻 gray edge，同一 lemma 保证 safe。
>
> **并行法正确性。** 从 $(0,0),(0,3),(2,3)$ 三个单点 trees 出发，各 component 可并行计算离开它的最轻边。本题所有边权互异，并且题面明确要求：若候选边直接连接两棵当前 trees，先回退并合并它们，再按新 component 重新选边。因此每条**实际接受**的边都是当时某个 component cut 上的 gray minimum，且已选边始终无环；逐步应用 gray-edge lemma 即保证最终得到 MST。
>
> 本题所有边权互异，所以 MST 唯一；三种算法虽然选边顺序不同，最终都必得上述 $15$ 条边。

> [!example]- CP21 Problem 2：tree iff 两点间唯一 path
> **$\Rightarrow$：** tree connected，故至少有一条 path。若有两条不同 path，取第一分叉与下一重会点，两段合成 cycle，违反 acyclic。
>
> **$\Leftarrow$：** 每两点有 path 给 connected。如果含 cycle，cycle 上两点可沿两个方向相连，得两条不同 path，违反唯一性。故无 cycle，是 tree。

> [!example]- CP21 Problem 3：唯一全局最小边必在 MST
> 设 $e$ 是全图唯一最小权边。反设某 MST $T$ 不含 $e$。向 $T$ 加 $e$ 恰得一个 cycle $C$；在 $C\setminus\{e\}$ 上选任意边 $f$。由全局唯一最小，$w(e)<w(f)$。$T'=T+e-f$ 仍是 spanning tree，但
> $$w(T')=w(T)+w(e)-w(f)<w(T),$$
> 与 $T$ 最小矛盾。故任意 MST 均含 $e$。

> [!example]- CP21 Problem 4：width one iff forest
> 图的 width 至多 $1$，意味顶点可排成 $v_1,\ldots,v_n$，每个 $v_i$ 至多与一个更早顶点相邻。
>
> **(a) width one $\Rightarrow$ forest。** 反设有 cycle，在该 cycle 中选排序最晚的顶点 $v$。cycle 上 $v$ 的两个不同邻居都比 $v$ 早，使 $v$ 至少有两个 earlier neighbors，矛盾。所以无 cycle，是 forest。
>
> **(b) finite tree $\Rightarrow$ width one。** 对顶点数归纳。单点情况显然。对 $n\ge2$，取 leaf $\ell$，删它后仍是 tree；由归纳假设，剩余顶点有 width-one 排序。把 $\ell$ 放到列表最后：它只有一个邻居，故至多有一个 earlier neighbor，而旧顶点的 earlier-neighbor 情况不变。归纳完成。对 forest 的每个 component 分别这样排，再串接列表；components 之间无边，仍 width one。

### 边界情况与易错点

- MST 只对 connected weighted graph 是一棵 spanning tree；不 connected 时对应对象是 minimum spanning forest。
- 权重互异 $\Rightarrow$ MST 唯一；逆命题不成立，有重复权重的图也可恰有一个 MST。
- Prim 选的是“跨越当前 tree cut 的最轻边”，不是全图尚未使用的最轻边；后者是 Kruskal 的视角。
- “加边产生 cycle”在 tree 中恰产生一个 cycle，因为新边两端原来只有一条 path。

### 自检与知识链

> [!question]- 自检 1
> 一张 $12$ 顶点、$11$ 边的 simple graph 一定是 tree 吗？
>
> [!success]- 答案
> 不一定；还需 connected 或 acyclic。可用一个含 cycle 的 component 加其余树分量构造反例。

> [!question]- 自检 2
> 为什么 MST 的交换证明中，$T+e$ 一定含 cycle？
>
> [!success]- 答案
> $T$ 中 $e$ 的两端已有唯一 path，新边 $e$ 与该 path 合成 cycle。

> [!question]- 自检 3
> forest 有 $40$ 个顶点、$7$ 个 components，有多少边？
>
> [!success]- 答案
> $40-7=33$，因每个 tree component 贡献 $n_i-1$ 边。

**知识链：**connected + acyclic → unique paths/leaves → $n-1$ edges → spanning tree → weighted spanning tree → cut property → Kruskal/Prim/Borůvka。

---

## Session 22 — Stable Matching and Hall's Theorem

### 学习问题与前置知识

一个“每人都有严格偏好”的双边分配问题，往往不是找总分最高的配对，而是排除任何一对双方都想背离当前结果的对象。本节问：Gale–Shapley/Mating Ritual 为什么必停、为什么稳定？“男方最优”究竟是定义还是定理？只给一张 bipartite graph 而没有偏好时，什么条件恰保证能完美匹配？前置知识是 invariant、bipartite graph、injection 与 path。

### 22.1 matching、rogue pair 与 stability

令两侧集合 $B$ 与 $G$ 各有 $n$ 人，每人对另侧所有人给出一个严格全序偏好。一个 **perfect matching** 是双射 $M:B\to G$；记 $M(b)$ 为 $b$ 的配偶，$M^{-1}(g)$ 为 $g$ 的配偶。

一对未在当前 matching 中的 $(b,g)$ 是 **rogue pair/blocking pair**，若

$$
g\succ_b M(b)
\quad\text{and}\quad
b\succ_g M^{-1}(g),
$$

即 $b$ 更喜欢 $g$ 而不是当前配偶，$g$ 也更喜欢 $b$ 而不是当前配偶。一个 matching [[Stable Matching|稳定]]，当且仅当它没有 rogue pair。

![[98_attachment/mathematics_for_computer_science/mit6_042j/unit02-stable-matching.png|900]]

读图：稳定匹配不要求每人获得首选，只要求不存在一条未选边使两端都想抛弃当前配偶。

> [!note]
> Stability 是“没有局部双边背离”，不是总幸福最大、每人都拿第一选择，也不保证 stable matching 在该实例中唯一。

### 22.2 Mating Ritual / deferred acceptance

以 $B$ 为 proposer side，$G$ 为 receiver side。算法可写成：

1. 每个 $b$ 的列表初始包含全部 $g$，按偏好由高到低排列。
2. 只要存在未暂定配对且列表非空的 $b$，就向列表顶部的 $g$ 求婚。
3. $g$ 在新求婚者与当前暂留者中保留自己最喜欢的一人，拒绝其余人。
4. 被 $g$ 拒绝的 $b$ 永久把 $g$ 从列表划去，下次改向更低的选择求婚。
5. 无未暂定 proposer 时结束，输出所有暂定 pairs。

“deferred acceptance”的意思是 receiver 对当前求婚只是暂留，以后可用更好 proposer 替换；一旦拒绝却不会反悔。

#### 终止性与运行时间

每一次 proposal 使某一对 $(b,g)$ 首次被考察；若拒绝，$g$ 被永久删除，同一对不再重复。总共只有 $n^2$ 对，所以 proposal 数至多 $n^2$，算法必停。

结束时不会有 proposer 空列表且未匹配。若 $b$ 已被全部 $n$ 位 receivers 拒绝，拒绝 invariant 表明每位 receiver 都已有一个比 $b$ 更喜欢的暂留者；$n$ 位 receivers 需要 $n$ 个互异的 proposers，但除 $b$ 外只有 $n-1$ 人，矛盾。因此输出是 perfect matching。

### 22.3 拒绝 invariant 与 stability 证明

> [!theorem] Basic rejection invariant
> 若 receiver $g$ 已从 proposer $b$ 的列表上被划去，则 $g$ 当前有一位自己严格更喜欢的暂留者，并且以后的暂留者只会更好。

**初始。** 初始没有任何人被划去，命题真值是 vacuous true。

**保持。** $g$ 只会在自己已有更喜欢的求婚者 $b'$ 时拒绝 $b$。以后 $g$ 只在新人比当前人更好时替换，所以“比 $b$ 好”永远保留。

> [!theorem] 算法输出 stable matching
> 反设最终 matching $M$ 有 rogue pair $(b,g)$。由 $b$ 偏好 $g$ 甚于 $M(b)$，$b$ 必在向 $M(b)$ 求婚之前先向 $g$ 求婚，然后被 $g$ 拒绝。由 rejection invariant，$g$ 最终配偶 $M^{-1}(g)$ 是她比 $b$ 更喜欢的人，这与 rogue pair 要求 $g$ 更喜欢 $b$ 矛盾。故无 rogue pair，$M$ stable。

### 22.4 proposer-optimal 与 receiver-pessimal

对一个人 $x$，若存在某个 stable matching 将 $x$ 与 $y$ 配对，称 $y$ 是 $x$ 的 **feasible spouse**。$x$ 的 optimal feasible spouse 是其所有 feasible spouses 中偏好最高者，pessimal feasible spouse 则是最低者。

> [!theorem] Strong rejection lemma
> 一旦 $g$ 拒绝 $b$，$(b,g)$ 就不可能出现在任何 stable matching 中。

**证明。** 假设不然，在算法的所有“拒绝了某个 feasible spouse”事件中取最早的一次：$g$ 拒绝 $b$，改留自己更喜欢的 $b'$。取一个把 $b$ 与 $g$ 配对的 stable matching $S$，记 $b'$ 在 $S$ 中的配偶为 $g'$。

$b'$ 已向 $g$ 求婚，所以在 $b'$ 的列表上，$g$ 高于它以后才会求婚的人。若 $b'$ 不偏好 $g$ 甚于 $g'$，那么 $g'$ 必早已拒绝 $b'$；但 $(b',g')$ 出现在 stable matching $S$，这就是比所选事件更早的 feasible-spouse rejection，矛盾。所以 $b'$ 更喜欢 $g$ 而不是 $g'$。同时 $g$ 在拒绝时已明确更喜欢 $b'$ 而不是 $b$。因而 $(b',g)$ 是 $S$ 的 rogue pair，与 $S$ stable 矛盾。强拒绝引理得证。

**Proposer-optimal.** 算法结束时，$b$ 最终配到的是其列表中未拒绝的最高者；所有比她更高的人都已拒绝 $b$，由强拒绝引理都不 feasible。所以每个 proposer 同时得到自己的 optimal feasible spouse。

**Receiver-pessimal.** 设算法把 $b$ 配给 $g$，但某 stable matching $S$ 把 $g$ 配给她更不喜欢的 $b'$。在 $S$ 中 $b$ 配某 $g'$。由 proposer-optimal，$b$ 在算法中得到的 $g$ 不低于任何 feasible spouse，特别地 $b$ 更喜欢 $g$ 而不是 $g'$。而 $g$ 也更喜欢 $b$ 而不是 $b'$，于是 $(b,g)$ 阻断 $S$，矛盾。故 $g$ 的算法配偶是其 pessimal feasible spouse。

因而 proposer 方与 receiver 方谁主动会影响所选的 stable matching。在严格完整偏好下，固定 proposer side 后的 proposer-optimal outcome 是确定的；但问题本身仍可有其他 stable matchings。

### 22.5 bipartite matching 与 Hall's condition

现在忘掉偏好，只保留一张 bipartite graph $G=(L\cup R,E)$；边表示“允许配对”。一个 [[Graph Matching|匹配]] 是顶点互不重复的边集。若它覆盖 $L$ 的每个顶点，就给出一个从 $L$ 到 $R$ 的 total injection；若 $|L|=|R|$，则这也是 perfect matching。

对 $S\subseteq L$，定义邻居集

$$
N(S)=\{r\in R:\exists \ell\in S,\{\ell,r\}\in E\}.
$$

若 $|N(S)|<|S|$，称 $S$ 是 **bottleneck**：$|S|$ 个左点只能选少于 $|S|$ 个右点，由 pigeonhole principle 不可能被一对一完全匹配。

> [!theorem] [[Hall's Marriage Theorem|Hall's Marriage Theorem]]
> finite bipartite graph $G=(L\cup R,E)$ 有一个覆盖全部 $L$ 的 matching，当且仅当
> $$
> \forall S\subseteq L,\qquad |N(S)|\ge |S|.
> $$

**必要性。** 若 matching $M$ 覆盖 $L$，则对每个 $\ell\in S$，其 matching partner $M(\ell)$ 均在 $N(S)$ 中；且 matching 使这些 partners 互异。因而 $S\to N(S)$ 有 injection，$|S|\le|N(S)|$。

**充分性（augmenting-path 证明）。** 假设 Hall condition 成立，取一个边数最多的 matching $M$。反设它未覆盖 $L$，令 $S_0\ne\varnothing$ 是未匹配左点集。从 $S_0$ 出发做 alternating search：

- 从左向右只沿不在 $M$ 中的边；
- 从右向左只沿 $M$ 中的匹配边。

记搜到的左点为 $S$，右点为 $T$。若搜到某个未匹配右点，则从 $S_0$ 到它的 alternating path 两端未匹配；把路上“非 matching/matching”边互换，matching 边数增 $1$，与 $M$ 最大矛盾。故 $T$ 中每个右点均已匹配。

每个 $t\in T$ 的 matching partner 都被搜到并落在 $S$；反之，$S\setminus S_0$ 中每个左点都是通过某个 $t\in T$ 的 matching edge 到达。因 matching 一对一，

$$
|S|=|T|+|S_0|>|T|.
$$

另一方面，$S$ 的所有邻居都会被 alternating search 到达，所以 $N(S)=T$。于是 $|N(S)|=|T|<|S|$，违反 Hall condition。反设不成立，$M$ 必覆盖所有 $L$。

### 官方顺序、资源与在线题（11 prompts）

阅读：[[MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session22.pdf]]。讲稿：[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/RE5PmdGNgj0.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/6vgHIImFwHo.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/n4KKgKpp--0.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/HZLKDC9OSaQ.pdf]]、[[MIT_OCW_6.042J_Materials/03_Video_Transcripts/i5AWE-OoOsY.pdf]]。字幕：[[MIT_OCW_6.042J_Materials/04_Captions/RE5PmdGNgj0.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/6vgHIImFwHo.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/n4KKgKpp--0.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/HZLKDC9OSaQ.srt]]、[[MIT_OCW_6.042J_Materials/04_Captions/i5AWE-OoOsY.srt]]。Slides：[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_stablematchg.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Mating_ritual.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_Optimal.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_bip_mtchig.pdf]]、[[MIT_OCW_6.042J_Materials/02_Lecture_Slides/MIT6_042JS15_halls_thorem.pdf]]。

| block | prompt | 官方答案与核验 |
|---|---|---|
| 2.11.3 Q1 | receiver 当前最佳选择的 rank | weakly increasing |
| 2.11.3 Q2 | proposer 当前最佳选择的 rank | weakly decreasing |
| 2.11.4 Q1 | Mating Ritual 的性质 | 官方选项：确定性地产生同一 proposer-optimal stable matching；这不表示问题只有一个 stable matching |
| 2.11.4 Q2 | proposer 列表上剩余名字总数 | weakly decreasing |
| 2.11.6 Q1 | boy-optimal marriage 的定义 | 该 boy 配到他在某个 stable matching 中可能配到的最高排名 girl |
| 2.11.8 Q1 | girls 比 boys 少，要让每个 girl 匹配 | 从 girls 到 boys 的 total injection |
| 2.11.10 Q1 | bottleneck 定义 | $|S|>|N(S)|$（题面记邻居集为 $E(S)$） |
| 2.11.11 Q1 | 题图 a,b,c 中哪些 bipartite | a；b,c 各含 odd cycle |
| 2.11.12 Q1 | 给定边集的 perfect matching | $(a,3),(b,2),(c,4),(d,1)$，且它恰唯一 |
| 2.11.12 Q2 | 无 perfect matching 图中真正的 bottleneck 性质 | $3,5$：$\{b,c,d\}$ 只有两邻居；换边后 $\{3,4\}$ 只有一邻居 |
| 2.11.13 Q1 | 八个断言中的 preserved invariants | $1,4,6,7,8$ |

原始练习：[[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.3_derived-variables.md|2.11.3]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.4_mating-ritual.md|2.11.4]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.6_boy-optimal.md|2.11.6]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.8_bipartite-equivalence-relation.md|2.11.8]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.10_bottleneck.md|2.11.10]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.11_bipartite-graphs.md|2.11.11]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.12_matching.md|2.11.12]] · [[MIT_OCW_6.042J_Materials/08_Courseware_Exercises/S22_2.11.13_stable-matching-invariants.md|2.11.13]]

> [!note]
> 上表 rank 沿用本课题库的“数值越大表示越喜欢”约定，所以 receiver 最佳 rank 弱增、proposer 当前选项 rank 弱减。若另一本书用“$1$ 是第一选择”，单调方向会反过来，但偏好改善/恶化的实质不变。

### 课堂题（非官方独立题解）

原题：[[MIT_OCW_6.042J_Materials/05_In_Class_Questions/MIT6_042JS15_cp22.pdf]]。

> [!example]- CP22 Problem 1：Students 与 Companies
> **(a) Students proposing。** 第一轮 Albert、Tasha 都向 HP，Sarah 向 AT&T，Elizabeth 向 Draper。HP 在 Albert/Tasha 中留 Tasha，Albert 被拒后改向 Bellcore，其余人保持。得
> $$
> \boxed{\text{Tasha--HP, Sarah--AT\&T, Elizabeth--Draper, Albert--Bellcore}}.
> $$
>
> **Companies proposing。** AT&T 与 HP 先都向 Elizabeth，Bellcore 向 Tasha，Draper 向 Sarah。按 receivers 的偏好逐次替换：Elizabeth 先留 AT&T，HP 转向 Tasha；Tasha 改留 HP，Bellcore 转向 Sarah；Sarah 改留 Bellcore，Draper 转向 Elizabeth；Elizabeth 改留 Draper，AT&T 最后转向 Albert。得
> $$
> \boxed{\text{Elizabeth--Draper, Tasha--HP, Sarah--Bellcore, Albert--AT\&T}}.
> $$
>
> 两个结果分别是 students-optimal 与 companies-optimal，并且都 stable。
>
> **(b) 唯一性判定。** 分别让两侧 proposing，求得 $M_L,M_R$。若不同，已有两个 stable matchings，故不唯一。若相同，任何 stable matching 中每个左方人的配偶不能比左方最优结果更好，也不能比左方最差（即右方最优）结果更差；两个界相同时每人配偶被强制，故结果唯一。

> [!example]- CP22 Problem 2：preserved invariants
> **(a) 是。** 一旦 Alice 成为 Harry 列表上唯一名字，以后不会添回其他人；Alice 若再拒绝 Harry 则 Harry 列表变空，但在两侧等大、完整偏好的 Ritual 中不可能出现空 proposer 列表。故真后保持为真。
>
> **(b) 否。** 某 girl 当前没有 suitor，下一轮可收到被别人拒绝的 proposer，命题由真变假。
>
> **(c) 是。** “Alice 不在 Harry 列表”意味 Alice 已拒绝 Harry；基本 rejection invariant 保证她当前且以后都有一位比 Harry 更喜欢的 suitor。
>
> **(d) 是。** Alice 被 Harry 划去后不会被加回；Harry 按偏好从高到低前进，所以他以后求婚的人都排在 Alice 之后。合取命题的两部分都被保持。
>
> **(e) 否。** Alice 留在 Harry 列表上只说她还没拒绝 Harry，不说明她对 Harry 的偏好。她可已有一位更喜欢的 suitor。
>
> 因此 preserved invariants 是 **(a), (c), (d)**。注意“preserved”只要求一旦为真就不再变假，不要求它在初始状态就为真。

> [!example]- CP22 Problem 3：由 invariant 证 stability
> 设输出存在 rogue pair $(B,G)$。$B$ 比较喜欢 $G$ 而不是最终 wife，所以按列表顺序，$B$ 曾先向 $G$ 求婚，后来 $G$ 被从 $B$ 的列表划去。由题给 invariant，从那时起 $G$ 有一位她比 $B$ 更喜欢的 favorite suitor，而她以后的 favorite 只会改善。因而她最终 husband 也排在 $B$ 之上，不可能更喜欢 $B$。这与 rogue pair 的第二个条件矛盾，故输出 stable。

> [!example]- CP22 Problem 4：有容量的 hospital–student matching
> 对每个容量为 $q_h$ 的 hospital $h$，拆成 $q_h$ 个 slots $h^{(1)},\ldots,h^{(q_h)}$。所有 slots 沿用 $h$ 对 students 的偏好；每个 student 把同一 hospital 的 slots 连续排放，并用任意固定顺序打破 slots 间的平局。现在 hospital slots 与 students 都是 capacity $1$。
>
> 若两侧总数不等，在较少一侧加 dummy participants：dummy student 表示空床位，dummy slot 表示 student 未匹配。真实参与者可把 dummy/outside option 排在所有可接受对象之后。对扩充后的等大实例运行 deferred acceptance，最后合并同一 hospital 的 slots 并删 dummy pairs。
>
> 稳定性改为：不存在 student $s$ 与 hospital $h$，使 $s$ 更喜欢 $h$ 而不是当前归宿，且 $h$ 有空 slot 或愿意用 $s$ 替换当前某位更低排名 student。若这样一对存在，它恰对应 cloned one-to-one 实例中的 rogue pair；所以算法输出稳定。

### 边界情况与易错点

- proposer-optimal 是“在所有 stable matchings 中可能得到的最好配偶”，不是“每个 proposer 都得到偏好表第一名”。
- Hall condition 必须检查所有 $S\subseteq L$，只检查单个顶点的 degree 不足够；多个点可能挤在过小的共同邻居集里。
- Stable matching 与 graph perfect matching 不是同一层问题：前者使用完整偏好并排除 blocking pair；后者只问允许边中是否有不冲突的全覆盖。
- 存在 ties 或 incomplete lists 时，“stable”可分 weak/strong/super stability；本节定理默认严格、完整偏好。

### 自检与知识链

> [!question]- 自检 1
> receiver 为什么可以安全地永久拒绝某 proposer？
>
> [!success]- 答案
> 拒绝时她已有更喜欢的暂留者，以后只会换成更好的人；强拒绝引理进一步证明该被拒绝 pair 不在任何 stable matching 中。

> [!question]- 自检 2
> Hall condition 中若 $|S|=5,|N(S)|=4$，为什么必无覆盖 $L$ 的 matching？
>
> [!success]- 答案
> $S$ 的五个点只能匹配给四个右点；matching 要求不同左点用不同右点，与 pigeonhole principle 矛盾。

> [!question]- 自检 3
> 把 proposer 与 receiver 交换后，结果一定相同吗？
>
> [!success]- 答案
> 不一定。原算法给原 proposer-optimal/receiver-pessimal，交换后给原 receiver-optimal/proposer-pessimal；只在极值 matching 恰重合等情况下相同。

**知识链：**preferences → proposals/rejections → preserved invariant → stable matching → proposer-optimal/receiver-pessimal → bipartite matching → Hall condition → augmenting paths。

---

## Problem Set 8

原题：[[MIT_OCW_6.042J_Materials/06_Problem_Sets/MIT6_042JS15_ps8.pdf]]。以下均为对原题的非官方独立完整解答。

> [!example]- PS8 Problem 1：边权互异则 MST 唯一
> **已知。** finite connected graph 的所有边权互异。
>
> **目标。** 证明不可能存在两个不同 MST $M,N$。
>
> **交换构造。** 反设 $M\ne N$，在对称差 $M\triangle N$ 中取权重最小边 $e$；不失一般设 $e\in M\setminus N$。向 tree $N$ 加入 $e$，恰产生一个 cycle $C$。$M$ 无 cycle，所以 $C$ 不可能全部边都属于 $M$；因而存在 $f\in C\cap(N\setminus M)$。
>
> $e,f$ 都在 $M\triangle N$ 中，且 $e$ 是其最轻边；又因权重互异，
> $$w(e)<w(f).$$
> 令 $N'=N+e-f$。删 cycle 上边 $f$ 后 $N'$ 仍 connected，且有 $n-1$ 边，所以是 spanning tree。但
> $$w(N')=w(N)+w(e)-w(f)<w(N),$$
> 与 $N$ 为 MST 矛盾。故 MST 唯一。
>
> **边界检查。** 证明只用“互异”得到严格不等式；若允许等权，只能得 $w(e)\le w(f)$，交换后可能只是另一个同权 MST，无法推出唯一。

> [!example]- PS8 Problem 2：triangle-free 但 $\chi=4$ 的图
> 题图是五环 $C_5$ 的 Mycielski construction。为避免依赖图纸坐标，将五环顶点顺次记为 $v_1,\ldots,v_5$，下标按模 $5$ 计；对应的五个 shadow vertices 记为 $u_1,\ldots,u_5$，$u_i$ 与 $v_i$ 的两个 cycle neighbors $v_{i-1},v_{i+1}$ 相邻；第十一点 $w$ 与所有 $u_i$ 相邻。这正是题图的抽象邻接关系。
>
> **(a) 4-coloring。** $C_5$ 可用三色 proper color，记为 $c(v_i)\in\{1,2,3\}$。定义
> $$c(u_i)=c(v_i),\qquad c(w)=4.$$
> 需检查的新边有两类：$u_i$ 只与 $v_{i-1},v_{i+1}$ 相邻，而 $v_i$ 在 $C_5$ 的 proper coloring 中与这两点颜色不同，所以 $u_i$ 复制 $v_i$ 颜色安全；$w$ 用新颜色 $4$，与所有 $u_i$ 不同。故 $\chi(G)\le4$。
>
> **(b) 排除 3-coloring。** 反设 $G$ 有三着色。不失一般设 $c(w)=3$，则所有与 $w$ 相邻的 $u_i$ 只能用颜色 $1,2$。现从该着色构造 $C_5$ 的二着色 $c'$：
> $$
> c'(v_i)=
> \begin{cases}
> c(v_i),&c(v_i)\ne3,\\
> c(u_i),&c(v_i)=3.
> \end{cases}
> $$
> 所有 $c'(v_i)$ 都在 $\{1,2\}$。检查 cycle edge $v_iv_{i+1}$：原着色不会让两端同为 $3$。若两端均不是 $3$，颜色未变且原本不同；若 $v_i$ 原为 $3$，则它改成 $c(u_i)$，而 $u_i$ 与 $v_{i+1}$ 相邻，所以新颜色仍不同。故 $c'$ 是 $C_5$ 的 proper 2-coloring，但 odd cycle 不可 2-color，矛盾。因此 $\chi(G)\ge4$，结合 (a) 得
> $$\boxed{\chi(G)=4}.$$
>
> 题面已指出该图 triangle-free，所以这也明确反驳“高着色数必来自同阶 clique”的逆命题。

> [!example]- PS8 Problem 3：多个 stable assignments
> **(a) 对所有未指定偏好都 stable。** 考察
> $$M=\{(B_1,G_1),(B_2,G_2),(B_3,G_3),(B_4,G_4)\}.$$
> 在第一个 $2\times2$ block 中，$B_1,B_2$ 都得到自己第一选择，所以他们不可能参与 rogue pair。在第二个 block 中，$B_3$ 更喜欢 $G_4$，但 $G_4$ 当前得到自己更喜欢的 $B_4$；$B_4$ 更喜欢 $G_3$，但 $G_3$ 当前更喜欢 $B_3$。所以这两个候选 cross pairs 都不 rogue。
>
> 对跨 blocks 的 pair 需分方向检查，不能假定所有 dash 都排在已写名字之后。$B_1,B_2$ 已分别得到整张表的第一选择，所以不会想转向 $G_3,G_4$。$B_3,B_4$ 虽可能更喜欢排在前两位的 $G_1,G_2$，但 $G_1,G_2$ 的前两位恰是 $B_1,B_2$，所以她们都会拒绝 $B_3,B_4$。这一结论与每个 dash 内部如何排序无关。故 $M$ 对任意补全方式都 stable。
>
> **(b) 既非 boy-optimal 也非 boy-pessimal。** 在第一 block，$B_1,B_2$ 都得第一选择；在第二 block，$B_3,B_4$ 都得第二选择。交换每个 block 内两条配对仍可得 stable matching，所以前两人还有更差 feasible spouses，后两人还有更好 feasible spouses。因此整体结果不是所有 boys 同时 optimal，也不是他们同时 pessimal，故不会是任一方作 proposer 的 Ritual 输出。
>
> **(c) 构造至少 $2^{n/2}$ 个 stable matchings。** 假设 $n$ 为偶数，把 boys 分成 $n/2$ 对 $(b_{i,1},b_{i,2})$，girls 也分成对应的 $n/2$ 对 $(g_{i,1},g_{i,2})$。每人把自己 block 的两人都排在所有外部对象之前，并在第 $i$ 个 block 内设
> $$
> b_{i,1}:g_{i,1}\succ g_{i,2},\qquad
> b_{i,2}:g_{i,2}\succ g_{i,1},
> $$
> $$
> g_{i,1}:b_{i,2}\succ b_{i,1},\qquad
> g_{i,2}:b_{i,1}\succ b_{i,2}.
> $$
> 每个 block 有两个 stable 选择：diagonal
> $$D_i=\{(b_{i,1},g_{i,1}),(b_{i,2},g_{i,2})\}$$
> 与 crossed
> $$C_i=\{(b_{i,1},g_{i,2}),(b_{i,2},g_{i,1})\}.$$
> 对 $D_i$，两个 boys 都已得第一选择；对 $C_i$，每个 boy 虽更喜欢另一 girl，但该 girl 正配着自己更喜欢的 boy，故无 rogue pair。跨 block 时，每人都更喜欢当前 block 内配偶，也不会 rogue。
>
> 因此每个 block 的 $D_i/C_i$ 可独立二选一，共得
> $$2\cdot2\cdots2=\boxed{2^{n/2}}$$
> 个互不相同的 stable matchings。

---

## Unit 2 结构回顾

本单元从整数结构走向关系与图，再落到可验证的算法：

1. **Number theory：**gcd/Bezout 使模逆可计算；congruence 把无穷整数折叠成有限余数类；Euler/Fermat 给出指数周期；RSA 把这些结构组成公钥系统。
2. **Relations and directed structure：**digraph 区分 walk/path/cycle；DAG 用 topological order 表达依赖；equivalence relation 分割集合，partial order/Hasse diagram 表达层级。
3. **Undirected structure：**degree 与 handshaking 来自 double counting；isomorphism 区分结构与画法；coloring 处理冲突，connectivity 衡量网络的韧性。
4. **Minimal skeleton and allocation：**tree 是 connected 的最小骨架，cut property 证明 MST 贪心选边安全；deferred acceptance 用拒绝 invariant 保证稳定，Hall 定理用 bottleneck/augmenting path 刻画 perfect matching 存在性。

### 单元级综合自检

> [!question]- 综合题 1
> 为什么“存在性证明”在 MST 与 Hall 定理中都出现了交换 path/cycle？
>
> [!success]- 答案
> 两者都先取一个极值对象：MST 证明取已有最小 tree，Hall 证明取最大 matching。加一条边后，tree 中出现 cycle，matching 中出现 alternating path；沿该结构交换可保持可行性并改善目标，与极值性矛盾。

> [!question]- 综合题 2
> 某 dependency digraph 是 DAG，其 underlying undirected graph 是 tree。这两个性质分别告诉你什么？
>
> [!success]- 答案
> DAG 保证存在 topological order，可按依赖顺序执行；underlying tree 保证忽略方向后任意两任务间恰有一条 path，且有 $n-1$ 条依赖边。前者是方向/时序结构，后者是无向连接骨架，不可混为同一命题。

> [!question]- 综合题 3
> RSA 正确性与 Mating Ritual 稳定性的证明风格有何共同点？
>
> [!success]- 答案
> 两者都不是只检查若干例子，而是先找全程保持的结构。RSA 分别在模 $p,q$ 下使用指数同余，再由 CRT 合并；Mating Ritual 使用“被划去的 receiver 永远持有更好 suitor”的 invariant，再排除 rogue pair。关键都是把最终目标转成可在每一步维持的性质。

### 本单元本地材料入口

统一从 [[MIT_OCW_6.042J_Materials/index|课程材料索引]] 进入；索引内已按 Session readings、Lecture slides、Video transcripts、Captions、In-class questions、Problem sets 与 Exams 分类。
