---
aliases:
  - 裴蜀等式
  - Bézout identity
student_os: knowledge-atom
atom_id: MCS-NT-005
atom_type: theorem
status: source-checked
part_of:
  - "[[数论与RSA.canvas]]"
requires:
  - "[[最大公因数]]"
  - "[[欧几里得算法]]"
related:
  - "[[扩展欧几里得算法]]"
leads_to:
  - "[[模逆元存在条件]]"
  - "[[素数整除乘积]]"
---

<!-- bilingual-en:start -->
# Bézout’s identity
<!-- bilingual-en:end -->

*Bézout 等式（裴蜀等式）*

<!-- bilingual-en:start -->
Bézout’s identity says that the [[最大公因数|greatest common divisor]] can be built from the two original integers using integer coefficients, which may be negative. For $30$ and $22$, three copies of $30$ minus four copies of $22$ give $2$, exactly their gcd. The identity guarantees that such a construction always exists; the [[扩展欧几里得算法|extended Euclidean algorithm]] is the separate procedure that finds its coefficients.
<!-- bilingual-en:end -->

Bézout 等式说：两个原整数的[[最大公因数]]可以用它们的整数倍组合出来，系数允许为负。对 $30$ 与 $22$，三份 $30$ 减去四份 $22$ 正好得到它们的 gcd——$2$。等式保证这样的构造总存在；具体怎样找到系数，由独立的[[扩展欧几里得算法]]处理。

<!-- bilingual-en:start -->
An integer linear combination uses integer coefficients to add and subtract the two inputs. The important point is not merely that common divisors divide every such combination: **the gcd itself is attainable as a combination**. For example, $\gcd(30,22)=2$ and $2=3\cdot30-4\cdot22$.
<!-- bilingual-en:end -->

“整数线性组合”就是只允许用整数系数加减这两个数。这里最重要的是：不仅它们的共同因子能整除所有这样的组合，**gcd 本身也确实能组合出来**。例如 $\gcd(30,22)=2$，且 $2=3\cdot30-4\cdot22$。

<!-- bilingual-en:start -->
Here the integers $s,t$ are the coefficients in $sa+tb$; “linear” means that the inputs are multiplied by coefficients and added, without squaring them or multiplying them together. The restriction $s,t\in\mathbb Z$ matters. Fractions would make a different, much easier problem. Coefficients need not be positive: restricting both to nonnegative integers would already make $30s+22t=2$ impossible.
<!-- bilingual-en:end -->

这里 $sa+tb$ 中的整数 $s,t$ 称为系数；“线性”指将输入乘以系数再相加，不把输入平方，也不把两个输入互相相乘。限制 $s,t\in\mathbb Z$ 很重要；若允许分数，那是另一个容易得多的问题。系数无须为正：若要求两个系数都非负，$30s+22t=2$ 就已经不可能成立。

<!-- bilingual-en:start -->
## The statement
<!-- bilingual-en:end -->

*等式的断言*

<!-- bilingual-en:start -->
For integers $a,b$ not both zero, there are integers $s,t$ such that
<!-- bilingual-en:end -->

若整数 $a,b$ 不全为零，则存在整数 $s,t$ 使

$$sa+tb=\gcd(a,b).$$

<!-- bilingual-en:start -->
The [[最大公因数|gcd convention]] makes this value positive whenever the inputs are not both zero. The quantifiers are: for every allowed pair $a,b$, there exist integer coefficients $s,t$, which may depend on that pair. There is no requirement that one coefficient pair work for every input.
<!-- bilingual-en:end -->

依照[[最大公因数]]的约定，输入不全为零时，这个值为正。量词的意思是：对每一对允许的输入 $a,b$，都存在整数系数 $s,t$，系数可以依赖于这一对输入；不是要求同一对系数适用于所有输入。

<!-- bilingual-en:start -->
## Why the gcd is attainable
<!-- bilingual-en:end -->

## 为什么成立

<!-- bilingual-en:start -->
In the [[欧几里得算法|Euclidean algorithm]], the initial values are $1a+0b$ and $0a+1b$. Each later remainder subtracts an integer multiple of one previous value from the other, preserving representation as an integer linear combination of $a,b$. The last nonzero value is the gcd, so it has that representation. For negative inputs, take absolute values and absorb signs into the coefficients; a single zero input is immediate.
<!-- bilingual-en:end -->

在[[欧几里得算法]]中，初始两个数已经分别是 $1a+0b$、$0a+1b$。以后每个余数都是前两个数的“一个减去另一个的整数倍”，所以仍是 $a,b$ 的整数线性组合。算法结束时最后一个非零数就是 gcd，因此它也有这种表示。负输入先取绝对值，最后把符号吸收到系数中；一个输入为零时结论直接成立。

<!-- bilingual-en:start -->
To make the preservation step explicit, first work with $A=|a|$ and $B=|b|$. If two current values have representations $x=uA+vB$ and $y=wA+zB$, then, while $y>0$, [[带余除法|division with remainder]] gives $x=qy+r$, with integer quotient $q$. Subtracting the representations gives:
<!-- bilingual-en:end -->

把保持性的步骤展开：先处理 $A=|a|$、$B=|b|$。若当前两个数已有表示 $x=uA+vB$、$y=wA+zB$，当 $y>0$ 时，[[带余除法]]给出 $x=qy+r$，商 $q$ 是整数。将这两个表示相减，便得到：

$$r=x-qy=(u-qw)A+(v-qz)B.$$

<!-- bilingual-en:start -->
The new coefficients are integers because sums and products of integers remain integers. Initially $A=1A+0B$ and $B=0A+1B$, so induction over the finitely many Euclidean steps gives a representation for every remainder. The Euclidean algorithm preserves the gcd and terminates at $(g,0)$; hence its first entry $g$ has the required representation. This proof uses the ordinary algorithm’s already-proved gcd preservation and termination, without assuming Bézout’s identity to justify that algorithm.
<!-- bilingual-en:end -->

新系数仍是整数，因为整数的和与积还是整数。初始有 $A=1A+0B$、$B=0A+1B$，沿有限次欧几里得更新归纳，就得到每个余数的表示。欧几里得算法保持 gcd，并在 $(g,0)$ 终止，所以第一项 $g$ 具有所需表示。这个证明使用普通算法已经证明的 gcd 保持性和终止性，没有先假设 Bézout 等式来证明算法。

<!-- bilingual-en:start -->
Common divisibility alone would not prove attainability. If $d\mid a$ and $d\mid b$, write $a=du$, $b=dv$; then $sa+tb=d(su+tv)$, so $d$ [[整除|divides]] every combination. That only constrains possible outputs. Tracking the Euclidean remainders supplies the missing existence argument for the particular output $g$.
<!-- bilingual-en:end -->

仅有共同整除性，还不能证明 gcd 能组合出来。若 $d\mid a$、$d\mid b$，写 $a=du$、$b=dv$，则 $sa+tb=d(su+tv)$，所以 $d$ [[整除]]每个组合。这只限制了可能的结果；追踪欧几里得余数才补上了“特定结果 $g$ 确实存在”的论证。

<!-- bilingual-en:start -->
## Signs, zeros and different coefficient pairs
<!-- bilingual-en:end -->

*符号、零与不同系数*

<!-- bilingual-en:start -->
If $u|a|+v|b|=g$, change the sign of the coefficient attached to each negative input. For instance, $3\cdot30-4\cdot22=2$ becomes $(-3)(-30)-4\cdot22=2$. With one zero input, choose the coefficient of the nonzero input to be its sign: $0\cdot0+(-1)(-7)=7$, and $(-1)(-7)+0\cdot0=7$.
<!-- bilingual-en:end -->

若 $u|a|+v|b|=g$，把对应负输入的系数变号即可。例如 $3\cdot30-4\cdot22=2$ 变成 $(-3)(-30)-4\cdot22=2$。只有一个输入为零时，给非零输入取与它符号相同的系数：$0\cdot0+(-1)(-7)=7$，以及 $(-1)(-7)+0\cdot0=7$。

<!-- bilingual-en:start -->
The theorem above excludes $(0,0)$. Under the vault’s separate convention $\gcd(0,0)=0$, the equation extends trivially: every pair $s,t$ gives $s0+t0=0$. This is a convention-compatible extension, not a claim that zero has a greatest positive common divisor.
<!-- bilingual-en:end -->

上面的定理排除了 $(0,0)$。若使用本库单独的约定 $\gcd(0,0)=0$，等式就平凡地延伸到这里：任意 $s,t$ 都有 $s0+t0=0$。这只是与约定相容的延拓，没有声称双零输入有最大的正公因数。

<!-- bilingual-en:start -->
Existence does not mean uniqueness. With $g>0$, suppose $(s,t)$ is one coefficient pair. Because $g$ divides both inputs, $a/g$ and $b/g$ are integers, and for every integer $k$ the following is another valid pair:
<!-- bilingual-en:end -->

存在不等于唯一。设 $g>0$，且 $(s,t)$ 是一对系数。由于 $g$ 整除两个输入，$a/g$、$b/g$ 都是整数；对每个整数 $k$，下面仍是一对有效系数：

$$s_k=s+k\frac{b}{g},\qquad t_k=t-k\frac{a}{g},\qquad s_ka+t_kb=sa+tb=g.$$

<!-- bilingual-en:start -->
The two extra terms cancel. In the $30,22$ example, $(3,-4)$ and $(14,-19)$ both give $2$. For a nonzero input pair, at least one of $a/g,b/g$ is nonzero, so varying $k$ really produces different coefficient pairs. No division by $g$ is used in the separate two-zero case.
<!-- bilingual-en:end -->

新增的两项恰好抵消。在 $30,22$ 的例子中，$(3,-4)$ 与 $(14,-19)$ 都给出 $2$。输入不全为零时，$a/g,b/g$ 至少有一个非零，因此改变 $k$ 确实得到不同系数对；单独的双零情形不使用除以 $g$ 的公式。

<!-- bilingual-en:start -->
## From the identity to modular inverses
<!-- bilingual-en:end -->

*从等式到模逆元*

<!-- bilingual-en:start -->
For this application, take an integer modulus $n\ge2$. Congruence modulo $n$ means that the difference is divisible by $n$; an [[模逆元|inverse]] of $a$ is represented by an integer whose product with $a$ is congruent to $1$.
<!-- bilingual-en:end -->

在这个应用中，约定整数模数 $n\ge2$。模 $n$ 同余表示差能被 $n$ 整除；$a$ 的[[模逆元]]由一个整数代表，它与 $a$ 的乘积同余于 $1$。

<!-- bilingual-en:start -->
The identity guarantees existence, not uniqueness, of the coefficients; the [[扩展欧几里得算法|extended Euclidean algorithm]] computes them. When $\gcd(a,n)=1$, reducing $sa+tn=1$ modulo $n$ leaves $sa\equiv1$, which is why coprimality suffices in the [[模逆元存在条件|criterion for an inverse]].
<!-- bilingual-en:end -->

等式保证系数存在，不保证唯一；具体计算交给[[扩展欧几里得算法]]。当 $\gcd(a,n)=1$ 时，$sa+tn=1$ 在模 $n$ 下留下 $sa\equiv1$，这就是[[模逆元存在条件]]中“互素足以求逆”的理由。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
Section 8.2.2 verifies Theorem 8.2.2 and its constructive proof through remainder coefficients. The text separates existence from the coefficient-computation procedure and checks the example identity directly.
<!-- bilingual-en:end -->

- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/99_Books/MIT6_042JS15_textbook.pdf#page=259|MIT Mathematics for Computer Science §8.2.2，PDF pp.259–261]]：核验定理 8.2.2 及余数系数保持的构造证明；正文把存在性与求系数的操作分开，例式直接复算。

<!-- bilingual-en:start -->
The [official MIT Session12 reading](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_session12.pdf), PDF3 and 6–10, supplies integer linear combinations, gcd scope, Theorem8.2.2 and its constructive remainder proof. PDF8–10 reproduce textbook PDF259–261. The coefficient family and signed/zero cases above are verified algebraically. [Session13](https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/mit6_042js15_session13.pdf), PDF9–10, supports the inverse application under a modulus greater than one.
<!-- bilingual-en:end -->

官方 MIT Session12 PDF3、6–10 支持整数线性组合、gcd 论域、Theorem8.2.2 及余数构造证明，其中 PDF8–10 对应教材 PDF259–261。正文的系数族与符号、零边界由代数直接验证；Session13 PDF9–10 支持模数大于一时的逆元应用。

