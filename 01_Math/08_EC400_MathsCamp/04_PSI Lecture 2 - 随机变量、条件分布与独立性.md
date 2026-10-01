---
aliases:
  - EC400 PSI Lecture 2 Statistics I
---

# PSI Lecture 2：随机变量、条件分布与独立性

<!-- bilingual-en:start -->
*Random variables, conditional distributions, and independence*
<!-- bilingual-en:end -->

[[Lecture 2 - Statistics I.pdf|课程 slides（90 页）]] · [[Joint, Marginal and Conditional Distributions - Summary.pdf|分布补充讲义（8 页）]] · [[PSI Lecture 2 - Statistics I - 手写笔记 - 2026-09-30.pdf|手写原稿（7 页）]] · [[PSI Lecture 2 - Statistics I - 课堂记录 - 2026-09-30|课堂记录]] · [[01_Math/08_EC400_MathsCamp/00_课程总览|课程阅读路径]]

上一讲问的是：要估计什么，怎样判断估计量好不好？这一讲先把这些问题需要的概率工具铺好。我们从一个随机试验出发，把结果变成数值，用分布描述数值的不确定性；接着问，看到另一项信息以后，分布、均值和方差会怎样改变。最后再回到 OLS，说明“独立”“均值独立”“不相关”为什么不能混用。

<!-- bilingual-en:start -->
Lecture 1 asked what to estimate and how to evaluate an estimator. This lecture develops the probability tools behind those questions: turn experimental outcomes into numbers, describe their distributions, and examine how information changes distributions, means, and variances. The final section returns to OLS and distinguishes independence, mean independence, and zero covariance.
<!-- bilingual-en:end -->

正文按 slides 顺序展开，页码 1–88 与 PDF 前 88 页一致；PDF 第 89–90 页是标作 non-examinable 的附录。手写要求补入的内容已放回相应段落。特征函数（slide 36）和积分的严格构造不在本次已讲范围；本笔记只解释 $dF$ 的用法，并把全方差证明作为可折叠补充。课程原例、补充讲义原例与课堂辅助例均在出现处注明。

<!-- bilingual-en:start -->
The main text follows the slides. Printed pages 1–88 match PDF pages 1–88; PDF pages 89–90 are non-examinable appendices. Requested material from the handwritten notes is integrated in context. Characteristic functions and formal integral construction lie outside the recorded teaching scope. The notation dF is explained, and the total-variance proof is included as an optional foldout. Examples are identified as coming from the slides, official supplement, or classroom illustration.
<!-- bilingual-en:end -->

> [!abstract] 阅读路线
> [[#1. 从结果到事件：概率空间与三条规则（slides 4–6）|事件与概率]] → [[#2. 把结果变成数值：随机变量、CDF 与 PDF（slides 7–10）|随机变量与分布]] → [[#3. 用少量数字描述分布（slides 12–24）|矩与分位数]] → [[#4. 两个分布与密度的准确含义（slides 26–35）|均匀、正态与混合分布]] → [[#5. 变换随机变量：先改写事件，再求分布（slides 38–42）|变换与 Jacobian]] → [[#6. 一张格子图串起联合、边际、条件与 Bayes（slides 44–59）|联合与条件分布]] → [[#7. 条件矩：在每一组里重新计算（slides 61–64）|条件均值与方差]] → [[#8. 分组以后怎样回到总体（slides 66–72）|全期望与全方差]] → [[#9. 三种不同强度的关系（slides 74–85）|独立性的层次]] → [[#10. 为什么这些区别会影响 OLS（slide 86）|OLS 的条件]]

## 1. 从结果到事件：概率空间与三条规则（slides 4–6）

<!-- bilingual-en:start -->
*From outcomes to events: the probability space and its axioms*
<!-- bilingual-en:end -->

### 1.1 先分清四个对象

<!-- bilingual-en:start -->
*Four different objects*
<!-- bilingual-en:end -->

[[概率空间]]写成 $(\Omega,\mathcal F,P)$。$\Omega$（大写 omega）是样本空间；$\omega$（小写 omega）是其中一个基本结果；事件 $A$ 是一组结果；$\mathcal F$ 是允许讨论概率的事件组成的集合。最后，$P$ 给每个合法事件分配一个 $[0,1]$ 内的数。注意层级：$\omega\in\Omega$，$A\subseteq\Omega$，$A\in\mathcal F$，而 $P(A)$ 是数。

<!-- bilingual-en:start -->
A [[概率空间|probability space]] is a triple. Omega is the sample space; lowercase omega is one outcome; an event A is a set of outcomes; and the calligraphic F is the collection of measurable events. P maps each such event to a number between zero and one. Outcome membership, event inclusion, event measurability, and numerical probability are different levels.
<!-- bilingual-en:end -->

例如掷一次骰子，$\Omega=\{1,2,3,4,5,6\}$；$\omega=4$ 是一个结果；$A=\{2,4,6\}$ 是“出现偶数”。若只记录奇偶，可以采用 $\mathcal F=\{\varnothing,\Omega,\{2,4,6\},\{1,3,5\}\}$。这已经是一个 [[σ-代数]]，却没有把 $\{1\}$ 当作可区分事件。有限空间常用所有子集组成的 $2^\Omega$，但这是一种选择；σ-代数并不是只在不可数空间里才需要。

<!-- bilingual-en:start -->
For a die, an outcome might be four and the even event is the set containing two, four, and six. If only parity is recorded, the empty set, full space, even outcomes, and odd outcomes form a valid [[σ-代数|sigma-algebra]]. It does not distinguish the singleton containing one. The full power set is a common choice on a finite space, not the only possible choice.
<!-- bilingual-en:end -->

σ-代数要求：包含 $\Omega$；若包含 $A$，也包含补事件 $A^c=\Omega\setminus A$；若包含 $A_1,A_2,\ldots$，也包含可数并 $\bigcup_{j=1}^{\infty}A_j$。它保证“没有发生”“至少一个发生”等操作仍产生合法事件。由补集关系，也能得到空集和可数交。

<!-- bilingual-en:start -->
A sigma-algebra contains the whole space and is closed under complements and countable unions. Thus negation and countable alternatives remain measurable events. The empty set and countable intersections follow from these rules.
<!-- bilingual-en:end -->

### 1.2 三条规则怎样推出常用公式

<!-- bilingual-en:start -->
*Deriving familiar rules from three axioms*
<!-- bilingual-en:end -->

[[概率三公理]]分别是非负性 $P(A)\ge0$、归一化 $P(\Omega)=1$、可数可加性。第三条只有在各事件互斥时才直接相加；“互斥”指任意 $i\ne j$ 都有 $A_i\cap A_j=\varnothing$。

<!-- bilingual-en:start -->
The [[概率三公理|probability axioms]] require nonnegative probabilities, total probability one, and countable additivity. Direct addition applies to disjoint events: no outcome belongs to two different events in the collection.
<!-- bilingual-en:end -->

$$P\!\left(\bigcup_{j=1}^{\infty}A_j\right)=\sum_{j=1}^{\infty}P(A_j)\quad\text{for pairwise disjoint }A_j.$$

先用 $\Omega=\Omega\cup\varnothing$ 且两者互斥，得到 $1=1+P(\varnothing)$，所以 $P(\varnothing)=0$。再把 $\Omega$ 分成互斥的 $A$ 与 $A^c$，得到 $1=P(A)+P(A^c)$，移项就是 [[补事件法]]。若 $A\subseteq B$，把 $B$ 写成互斥并 $A\cup(B\setminus A)$，便得到 $P(B)=P(A)+P(B\setminus A)\ge P(A)$。这就是 [[概率的单调性]]：事件包含更多可能结果，其概率不会更小。

<!-- bilingual-en:start -->
Writing the whole space as its disjoint union with the empty set gives zero probability for the empty event. Splitting the whole space into A and its complement gives the [[补事件法|complement rule]]. If A is contained in B, decomposing B into A and the remaining outcomes proves monotonicity by nonnegativity.
<!-- bilingual-en:end -->

对可以重叠的 $A,B$，不能直接把概率相加。按照 [[互斥事件加法]]，先保留整个 $A$，再只加 $B$ 中没有计入 $A$ 的部分：

<!-- bilingual-en:start -->
For overlapping events, [[互斥事件加法|the addition rule]] counts all of A and then only the part of B not already counted:
<!-- bilingual-en:end -->

$$\begin{aligned}P(A\cup B)&=P(A)+P(B\setminus A)\\&=P(A)+P(B)-P(A\cap B)\\&\le P(A)+P(B).\end{aligned}$$

最后一步只用了交集概率非负，完全不需要独立性。把这个论证逐次用于更多事件，得到 [[并事件上界]]（union bound）：$P(\bigcup_{i=1}^{n}A_i)\le\sum_{i=1}^{n}P(A_i)$。上界可能超过 1；这不表示真实概率超过 1，只表示这个上界比较松。

<!-- bilingual-en:start -->
The inequality uses only the nonnegativity of the intersection probability, not independence. Repeating the argument gives the [[并事件上界|union bound]]. Its right-hand side may exceed one, which simply makes the bound loose.
<!-- bilingual-en:end -->

**课堂辅助例：两枚公平、相互独立的骰子至少有一枚是 6。** 一次结果写作有顺序的 $(d_1,d_2)$，共有 $6\times6=36$ 个等可能结果。令 $A=\{d_1=6\}$、$B=\{d_2=6\}$。$A$ 有 6 个结果，$B$ 也有 6 个，但 $(6,6)$ 被计算了两次。因此 $P(A\cup B)=(6+6-1)/36=11/36$。也可用补事件得到 $1-(5/6)^2=11/36$。这里独立性负责保证 36 个有序结果等可能及补事件乘法；并事件公式本身不需要它。

<!-- bilingual-en:start -->
**Classroom illustration: at least one six with two fair independent dice.** There are 36 equiprobable ordered outcomes. Each die contributes six favourable outcomes, but the double-six is counted twice, so the probability is 11/36. The complement calculation gives the same result. Independence justifies the equal weighting and product in this example; it is unnecessary for the union identity itself.
<!-- bilingual-en:end -->

## 2. 把结果变成数值：随机变量、CDF 与 PDF（slides 7–10）

<!-- bilingual-en:start -->
*Random variables, CDFs, and densities*
<!-- bilingual-en:end -->

### 2.1 随机变量是函数，分布是它搬运出来的概率

<!-- bilingual-en:start -->
*A random variable maps outcomes; its distribution transports probabilities*
<!-- bilingual-en:end -->

[[随机变量]] $X:\Omega\to\mathbb R$ 是一个可测函数。函数规则可以完全确定，随机的是这一次会发生哪个 $\omega$。两枚骰子里令 $X(d_1,d_2)=d_1+d_2$，则 $(1,6)$ 和 $(3,4)$ 都被映射为 7。于是 $X$ 保留了点数之和，却没有保留两枚骰子各自的点数。大写 $X$ 表示函数或尚未实现的随机数，小写 $x$ 表示一个具体取值或阈值。

<!-- bilingual-en:start -->
A [[随机变量|random variable]] is a measurable function. Its rule can be deterministic; uncertainty concerns the realised underlying outcome. Summing two dice maps several distinct outcomes to seven, retaining the sum while discarding the individual results. Uppercase X denotes the random variable; lowercase x denotes a value or threshold.
<!-- bilingual-en:end -->

slide 8 的 $P(X=7)$ 是简写。完整事件是 $\{\omega:X(\omega)=7\}$，其中有 $(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)$ 六个结果，所以概率为 $6/36=1/6$。对实数轴上的可测集合 $C$，同样有 $\mu_X(C)=P(\{\omega:X(\omega)\in C\})$。$\mu_X$ 就是 $X$ 的分布，称作原概率经 $X$ 得到的 pushforward measure；这里的 $C$ 是数值集合，不是原始试验结果的集合。

<!-- bilingual-en:start -->
On slide 8, P(X=7) abbreviates the event consisting of all outcomes whose sum is seven. Its six equiprobable outcomes give probability one sixth. More generally, the distribution assigns a measurable numerical set C the probability of all original outcomes mapped into C. This is the pushforward of the underlying probability measure.
<!-- bilingual-en:end -->

### 2.2 CDF 的四条性质，都可以从事件读出来

<!-- bilingual-en:start -->
*Reading the CDF properties through events*
<!-- bilingual-en:end -->

[[累积分布函数]]（cumulative distribution function，CDF）定义为 $F_X(x)=P(X\le x)$。“累积”指把阈值左边、连同阈值本身的概率都算进去。CDF 对离散、具有密度、混合以及其他分布都适用。先记定义，再看四条性质：

<!-- bilingual-en:start -->
The [[累积分布函数|CDF]] includes all probability to the left of a threshold, including the threshold itself. It applies to discrete, absolutely continuous, mixed, and singular distributions. Its properties follow from the events being accumulated:
<!-- bilingual-en:end -->

**单调不减。** 若 $a<b$，则 $\{X\le a\}\subseteq\{X\le b\}$，因此 $F_X(a)\le F_X(b)$。

**右连续。** 令阈值 $x+1/n$ 从右边降到 $x$，事件 $\{X\le x+1/n\}$ 也逐步缩小到 $\{X\le x\}$。概率对递减事件列连续，所以 $F_X(x+1/n)\to F_X(x)$。这不是要求 CDF 没有跳跃；跳跃发生时，函数值取跳上去以后的高度。

<!-- bilingual-en:start -->
**Nondecreasing:** increasing the threshold enlarges the event. **Right-continuous:** as thresholds decrease to x from above, their events decrease to the event X≤x; continuity of probability for decreasing events gives the limit. Right continuity allows jumps and assigns the value at a jump to its upper level.
<!-- bilingual-en:end -->

**两端极限。** 阈值向负无穷移动，累计概率趋于 0；向正无穷移动，实值 $X$ 的全部概率最终被计入，累计概率趋于 1。

**区间相减。** 因为 $\{X\le b\}$ 是 $\{X\le a\}$ 与 $\{a<X\le b\}$ 的互斥并，所以 $P(a<X\le b)=F_X(b)-F_X(a)$。左端严格、右端包含，是由所减掉的集合决定的。

<!-- bilingual-en:start -->
**Tail limits:** the CDF tends to zero at negative infinity and to one at positive infinity. **Interval subtraction:** the event X≤b splits into X≤a and a<X≤b. Subtracting the first probability explains the precise open-left, closed-right endpoints.
<!-- bilingual-en:end -->

$$\lim_{x\to-\infty}F_X(x)=0,\qquad\lim_{x\to\infty}F_X(x)=1,\qquad P(X=x)=F_X(x)-F_X(x-).$$

$F_X(x-)$ 读作“从左边逼近 $x$ 时 CDF 的极限”，等于 $P(X<x)$；最后一式说明跳跃高度正是该点的概率。两骰之和的 $F_X(3)=3/36$、$F_X(4)=6/36$、$F_X(5)=10/36$、$F_X(6)=15/36$。因此 $P(3<X\le5)=(10-3)/36=7/36$，对应点数和为 4 或 5 的 $3+4$ 个结果。

<!-- bilingual-en:start -->
The left limit excludes the point itself, so the jump height equals its point mass. For the sum of two dice, subtracting the cumulative probability at three from that at five gives seven thirty-sixths, corresponding to the outcomes whose sums are four or five.
<!-- bilingual-en:end -->

![[PSI-L2-distributions.png|900]]

图中左侧的 CDF 通过跳跃积累点质量；中间通过密度曲线下面积积累概率；右侧同时有一个跳跃和一段连续增长。下面各节会逐项算出后两种例子。

<!-- bilingual-en:start -->
The left CDF accumulates point masses through jumps, the middle accumulates density area, and the right combines an atom with continuous growth. The next sections calculate the latter two examples.
<!-- bilingual-en:end -->

### 2.3 PDF 给的是密度，面积才是概率

<!-- bilingual-en:start -->
*A density becomes a probability after integration*
<!-- bilingual-en:end -->

当分布绝对连续时，存在 [[概率密度函数]]（probability density function，PDF）$f_X$，使 $F_X(x)=\int_{-\infty}^{x}f_X(t)\,dt$。于是 $P(a<X\le b)=\int_a^b f_X(t)\,dt$。这里 $t$ 只是积分的临时变量；改成 $u$ 不会改变结果。在密度连续处，微积分基本定理给出 $F_X'(x)=f_X(x)$。

<!-- bilingual-en:start -->
An absolutely continuous distribution has a [[概率密度函数|density]] whose integral gives the CDF. Interval probabilities are areas under that density. The integration variable is a dummy symbol; at continuity points of the density, the fundamental theorem of calculus identifies it with the derivative of the CDF.
<!-- bilingual-en:end -->

要保留这个方向的条件：**CDF 连续，不足以保证有普通 PDF**，具体边界见 [[连续CDF不保证密度]]。本课遇到的均匀、正态等分布都有密度，因而可以放心用积分求概率。另一个区别见 [[PMF不等于密度]]：离散 PMF 的值 $p_X(x)$ 就是 $P(X=x)$；PDF 的高度 $f_X(x)$ 则不是点概率，甚至可以大于 1。

<!-- bilingual-en:start -->
Continuity of a CDF alone [[连续CDF不保证密度|does not guarantee a density]]. The uniform and normal distributions used here do have densities. A [[PMF不等于密度|PMF value is a point probability]], whereas a PDF value is a height and may exceed one.
<!-- bilingual-en:end -->

## 3. 用少量数字描述分布（slides 12–24）

<!-- bilingual-en:start -->
*Summarising a distribution with moments and quantiles*
<!-- bilingual-en:end -->

### 3.1 期望：每一个值都要乘上它的概率权重

<!-- bilingual-en:start -->
*Expectation weights each value by its probability*
<!-- bilingual-en:end -->

[[期望]] $E[X]$ 是按分布加权的平均。离散时逐项相加，有密度时把许多微小区间的贡献积分起来。有限期望通常要求 $E|X|<\infty$，不能默认每个分布都有均值。

<!-- bilingual-en:start -->
[[期望|Expectation]] is a distribution-weighted average: a sum for discrete outcomes and an integral for a density. A finite expectation requires absolute integrability; not every distribution has a mean.
<!-- bilingual-en:end -->

$$E[X]=\sum_x x\,p_X(x),\qquad E[X]=\int_{-\infty}^{\infty}x f_X(x)\,dx.$$

**离散完整例：公平骰子。** $E[X]=1(1/6)+2(1/6)+\cdots+6(1/6)=21/6=3.5$。均值并不必然是某次能出现的结果。要算平方的均值，平方的是骰子取值，权重仍是原来的概率：$E[X^2]=(1+4+9+16+25+36)/6=91/6$。而 $(E[X])^2=3.5^2=49/4$；两者不是同一项运算。

<!-- bilingual-en:start -->
**Discrete example: a fair die.** The mean is 3.5, which cannot occur on one roll. For the mean of the square, square each outcome while retaining its probability weight, giving 91/6. This differs from squaring the mean, which gives 49/4.
<!-- bilingual-en:end -->

这正是 [[LOTUS]]（law of the unconscious statistician）的用途：要算 $g(X)$ 的期望，可以直接使用 $X$ 的分布，不必先求 $g(X)$ 的完整分布。只需把公式里原先“取值 $x$”的位置换成“要平均的数 $g(x)$”。

<!-- bilingual-en:start -->
[[LOTUS]] computes the expectation of a function using the distribution of the original variable. There is no need to derive the full distribution of the transformed variable first: replace the outcome in the weighted average by its transformed value.
<!-- bilingual-en:end -->

$$E[g(X)]=\sum_x g(x)p_X(x)\quad\text{or}\quad E[g(X)]=\int g(x)f_X(x)\,dx.$$

**课堂辅助例：$f_X(x)=3x^2$，$0<x<1$，其他位置为 0。** 先检验总面积：$\int_0^1 3x^2\,dx=[x^3]_0^1=1$。这一步只检验它是一张有效密度。再计算三个不同对象：

<!-- bilingual-en:start -->
**Classroom illustration: density 3x² on (0,1).** Its integral is one, confirming normalisation. Normalisation is different from each of the following expectations:
<!-- bilingual-en:end -->

$$\begin{aligned}E[X]&=\int_0^1 x(3x^2)\,dx=3\int_0^1x^3\,dx=3\left[\frac{x^4}{4}\right]_0^1=\frac34,\\E[X^2]&=\int_0^1 x^2(3x^2)\,dx=3\int_0^1x^4\,dx=\frac35,\\E[2X^2+1]&=\int_0^1(2x^2+1)3x^2\,dx=\int_0^1(6x^4+3x^2)\,dx=\frac65+1=\frac{11}{5}.\end{aligned}$$

这里用了幂函数积分 $\int x^k dx=x^{k+1}/(k+1)+C$（$k\ne-1$）；$[H(x)]_a^b$ 表示 $H(b)-H(a)$。若只积 $f_X$ 得到 1，算的是总概率，不是期望。不过“期望算出 1”本身不必然错误，例如常数变量 $X=1$ 的期望确实是 1；应检查被积函数，而不是仅凭结果判断。

<!-- bilingual-en:start -->
The calculation uses the power rule and evaluates an antiderivative at its upper limit minus its lower limit. Integrating the density alone computes total probability. A mean of one is not automatically wrong; inspect whether the outcome factor is present rather than judging by the numerical answer alone.
<!-- bilingual-en:end -->

另一个课堂例 $f_X(x)=2(1-x)$（$0<x<1$）给出 $E[X]=\int_0^1(2x-2x^2)dx=[x^2-2x^3/3]_0^1=1/3$。这条密度把更大权重放在靠近 0 的位置，均值小于 $1/2$，与图形一致。[[原点矩与中心矩]]把这些计算整理成记号：$E[X^k]$ 是 $k$ 阶原点矩，$E[(X-\mu)^k]$ 是 $k$ 阶中心矩；更一般的 $E[g(X)]$ 是函数的期望，不必都称为幂矩。

<!-- bilingual-en:start -->
For the classroom density 2(1−x), weighting by x gives a mean of one third, consistent with greater density near zero. [[原点矩与中心矩|Raw and central moments]] distinguish powers of X from powers of its deviation from the mean. A general expectation of g(X) need not be a power moment.
<!-- bilingual-en:end -->

### 3.2 线性性为什么不需要独立

<!-- bilingual-en:start -->
*Why linearity does not require independence*
<!-- bilingual-en:end -->

[[期望线性性]]说，对可积的 $X,Y$ 和常数 $a,b$，$E[aX+bY]=aE[X]+bE[Y]$。原因是对同一个概率测度积分时，积分可以拆开、常数可以提出；联合分布是否分解并不参与这一步。特别地 $E[aX+b]=aE[X]+b$，因为常数 $b$ 的期望还是 $b$。

<!-- bilingual-en:start -->
[[期望线性性|Linearity of expectation]] follows from linearity of integration under one probability measure. It does not require factorising a joint distribution. The expectation of a constant remains that constant.
<!-- bilingual-en:end -->

[[期望与非线性变换不可交换]]：一般不能把 $E[g(X)]$ 改成 $g(E[X])$。以上例来说，$E[X^2]=3/5$，$(E[X])^2=9/16$。仿射函数 $g(x)=ax+b$ 保证相等；这不是说非线性函数在任何特殊分布下都绝不可能碰巧相等。

<!-- bilingual-en:start -->
In general, expectation and a nonlinear function cannot be interchanged. The density example gives 3/5 for the mean square but 9/16 for the squared mean. Affine functions guarantee equality; a nonlinear function may still give equality for a particular distribution.
<!-- bilingual-en:end -->

### 3.3 指示变量把概率问题变成期望问题

<!-- bilingual-en:start -->
*Indicators turn event probabilities into expectations*
<!-- bilingual-en:end -->

[[事件指示变量]] $I_A=\mathbf1_A$ 在 $A$ 发生时等于 1，否则等于 0。按照 [[指示变量期望]]，$E[I_A]=0P(A^c)+1P(A)=P(A)$。按照 [[指示变量的交事件乘积]]，$I_AI_B=I_{A\cap B}$，因为乘积只有在两个因子都为 1 时才为 1，所以 $E[I_AI_B]=P(A\cap B)$。这两式不要求 $A,B$ 独立。

<!-- bilingual-en:start -->
An [[事件指示变量|indicator]] equals one when its event occurs and zero otherwise. Its [[指示变量期望|expectation equals the event probability]]. The [[指示变量的交事件乘积|product of two indicators represents the intersection]], so its expectation is the joint probability. Neither identity requires independence.
<!-- bilingual-en:end -->

**课堂辅助例：10 次记录中出现 6 的次数。** 令 $I_i=\mathbf1\{\text{第 }i\text{ 次记录为 }6\}$，总数 $N=\sum_{i=1}^{10}I_i$。只要每次记录边际上都是公平骰子，就有 $E[N]=\sum_iE[I_i]=10/6=5/3$。即使先掷一次，再把同一个结果复制 10 遍，期望仍然是 $10(1/6)$。但分布和方差会变化：独立投掷时 $\operatorname{Var}(N)=10(1/6)(5/6)=25/18$；复制时 $N=10I_1$，方差为 $100(1/6)(5/6)=125/9$。后面解释为什么方差多出了协方差项。

<!-- bilingual-en:start -->
**Classroom illustration: count sixes in ten records.** Summing indicators gives expected count 5/3 whenever each marginal record is a fair die, even if one result is copied ten times. The variances differ: independent rolls give 25/18, whereas ten copies give 125/9. The covariance terms introduced later explain the difference.
<!-- bilingual-en:end -->

### 3.4 方差、标准差和标准误在描述谁

<!-- bilingual-en:start -->
*Variance, standard deviation, and standard error*
<!-- bilingual-en:end -->

[[方差]]用平均平方偏离描述分散程度：$\operatorname{Var}(X)=E[(X-\mu)^2]$，其中 $\mu=E[X]$。直接平均偏离会得到 $E[X-\mu]=0$，正负会抵消；平方让偏离都变成非负。设二阶矩有限，逐项展开：

<!-- bilingual-en:start -->
[[方差|Variance]] measures the expected squared deviation from the mean. Signed deviations average to zero, so squaring prevents cancellation. With a finite second moment, expanding gives:
<!-- bilingual-en:end -->

$$\begin{aligned}\operatorname{Var}(X)&=E[X^2-2\mu X+\mu^2]\\&=E[X^2]-2\mu E[X]+\mu^2\\&=E[X^2]-2\mu^2+\mu^2=E[X^2]-(E[X])^2.\end{aligned}$$

对 $f_X=3x^2$ 的例子，$\operatorname{Var}(X)=3/5-9/16=(48-45)/80=3/80$。标准差 $\operatorname{sd}(X)=\sqrt{\operatorname{Var}(X)}=\sqrt{3/80}\approx0.19365$。若 $X$ 的单位是英镑，方差是英镑平方，标准差仍是英镑。

<!-- bilingual-en:start -->
For the density 3x², the variance is 3/80 and the standard deviation is about 0.19365. Variance has squared units; standard deviation has the same units as the variable.
<!-- bilingual-en:end -->

[[方差的仿射变换]]可从中心化一步看清楚。令 $Y=aX+b$，那么 $E[Y]=a\mu+b$，所以 $Y-E[Y]=a(X-\mu)$；平方后再取期望便得到 $\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$。开平方得到 $\operatorname{sd}(aX+b)=|a|\operatorname{sd}(X)$。绝对值来自 $\sqrt{a^2}=|a|$，标准差不能为负。

<!-- bilingual-en:start -->
For an [[方差的仿射变换|affine transformation]], subtracting the new mean leaves a times the original centred variable. Squaring multiplies variance by a²; taking a square root multiplies standard deviation by |a|. Translation changes neither measure of spread.
<!-- bilingual-en:end -->

**slide 20 原例：英里换公里。** $Y=1.609X$，则 $\operatorname{Var}(Y)=1.609^2\operatorname{Var}(X)$、$\operatorname{sd}(Y)=1.609\operatorname{sd}(X)$。**课堂单位例：** 年收入均值 £30,000，标准差 £8,000；用“千英镑”表示，$Y=X/1000$ 的均值是 30，标准差是 8，方差是 64（千英镑）$^2$。每人加 £2,000 只把均值变成 £32,000；每人乘 1.15 则把均值变成 £34,500，标准差变成 £9,200。

<!-- bilingual-en:start -->
**Slide 20:** converting miles to kilometres multiplies standard deviation by 1.609 and variance by its square. **Classroom units example:** measuring income in thousands of pounds turns a £30,000 mean and £8,000 standard deviation into 30 and 8, with variance 64 in squared thousands. Adding £2,000 shifts only the mean; multiplying incomes by 1.15 rescales both mean and standard deviation.
<!-- bilingual-en:end -->

[[标准误含义|标准误（standard error）]]是统计量抽样分布的标准差；实际报告中通常使用它的估计值。若 $X_1,\ldots,X_n$ 独立同分布、方差为 $\sigma^2$，$\bar X=(X_1+\cdots+X_n)/n$ 的标准误为 $\sigma/\sqrt n$，估计标准误为 $s/\sqrt n$。$\operatorname{sd}(X)$ 描述单个观测有多分散，$SE(\bar X)$ 描述换一批样本后均值会怎样变动。与 [[03_PSI Lecture 1 - 识别、概率与估计|上一讲抽样分布]]相连：$n$ 是每批样本大小；重复模拟次数 $R$ 只是用来观察抽样分布，不是这里分母中的 $n$。

<!-- bilingual-en:start -->
A [[标准误含义|standard error]] is the standard deviation of a statistic’s sampling distribution, often estimated in practice. For an iid sample with variance sigma², the sample mean has standard error sigma divided by the square root of n. This describes variation between sample means, not variation between individual observations. In the [[03_PSI Lecture 1 - 识别、概率与估计|Lecture 1 sampling experiment]], n is the size of each sample; the number R of simulated repetitions is a different quantity.
<!-- bilingual-en:end -->

### 3.5 分位数：达到某个累计概率的最左门槛

<!-- bilingual-en:start -->
*Quantiles locate probability thresholds*
<!-- bilingual-en:end -->

[[总体分位数]]对 $0<p<1$ 定义为 $q_p=\inf\{x:F_X(x)\ge p\}$。先找出所有使累计概率至少达到 $p$ 的阈值，再取这些阈值的下确界（infimum）。对 CDF，这给出最左边达到该水平的位置。$q_{0.5}$ 是按此约定选出的中位数；它满足 $P(X\le q_{0.5})\ge1/2$ 和 $P(X\ge q_{0.5})\ge1/2$。

<!-- bilingual-en:start -->
A [[总体分位数|population quantile]] is the infimum of thresholds where the CDF reaches or exceeds p. This selects the leftmost threshold at that level. The median selected by this convention has at least half the probability at or below it and at least half at or above it.
<!-- bilingual-en:end -->

若 CDF 连续且在目标附近严格递增，可以直接解 $F_X(q_p)=p$。例如 $F_X(x)=x^3$ 给出 $q_p=p^{1/3}$，中位数为 $m=2^{-1/3}\approx0.79370$，不是 $\sqrt{1/2}$。离散时未必能解等号：若 $P(X=0)=0.3$、$P(X=2)=0.7$，$q_{0.5}=2$，但 $F_X(2)=1$。若两点各有 0.5 概率，广义分位数选 0，而区间 $[0,2]$ 中每个数都满足两侧概率至少一半的中位数条件。

<!-- bilingual-en:start -->
A continuous, locally strictly increasing CDF allows solving F(q)=p. For F(x)=x³, the median is the cube root of one half. A discrete CDF may jump past the desired level: a variable placing probabilities 0.3 and 0.7 at zero and two has median quantile two with CDF value one. With equal masses, the quantile convention selects zero although every point between zero and two satisfies the median inequalities.
<!-- bilingual-en:end -->

slide 21 的工资例中，$q_{0.5}=£18$、$q_{0.9}=£35$ 表示累计概率的门槛。在连续、没有点质量的情形可以说一半在 £18 以下、九成在 £35 以下；若大量工资恰好等于门槛，应说“至少一半不超过 £18”“至少九成不超过 £35”。总体分位数也不同于由有限数据计算的 [[样本分位数]]；后者存在不同插值约定。

<!-- bilingual-en:start -->
The wage example on slide 21 gives cumulative thresholds. Exact proportions below the thresholds require appropriate continuity; with ties, use at least half at or below £18 and at least ninety percent at or below £35. [[样本分位数|Sample quantiles]] are finite-data estimates and may use different interpolation conventions.
<!-- bilingual-en:end -->

### 3.6 偏度和峰度：形状需要相应的矩存在

<!-- bilingual-en:start -->
*Skewness and kurtosis require moments to exist*
<!-- bilingual-en:end -->

[[偏度]]是标准化三阶中心矩。减去均值把中心移到 0，除以标准差消除单位，立方保留左右偏离的符号。设 $\sigma>0$ 且三阶绝对矩有限，

<!-- bilingual-en:start -->
[[偏度|Skewness]] is the standardised third central moment. Centring removes location, scaling removes units, and cubing retains the direction of deviations. It requires positive variance and a finite third absolute moment:
<!-- bilingual-en:end -->

$$\operatorname{Skew}(X)=E\!\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]=\frac{E[(X-\mu)^3]}{\sigma^3}.$$

正偏通常画成长右尾，负偏通常画成长左尾。slides 22–23 说的是右偏时**通常**均值高于中位数；这不是偏度的定义，也不是适用于任意分布的充要判据。对称且相应矩存在会给出零偏度，反向不成立。对于 $f_X=3x^2$，多数概率靠近 1，尾巴向左；它的均值 $3/4$ 小于中位数 $2^{-1/3}$。若要严格核验方向，可以补算：

<!-- bilingual-en:start -->
Positive skew is often pictured with a long right tail and negative skew with a long left tail. The usual ordering of mean and median is a heuristic, not a definition or universal equivalence. Symmetry with the required moments implies zero skewness, but the converse fails. For 3x², most mass lies near one and the tail extends left. A direct check is:
<!-- bilingual-en:end -->

$$\begin{aligned}E[X^3]&=\int_0^1x^3(3x^2)\,dx=\frac12,\\E[(X-\mu)^3]&=E[X^3]-3\mu E[X^2]+3\mu^2E[X]-\mu^3\\&=\frac12-3\cdot\frac34\cdot\frac35+2\left(\frac34\right)^3=-\frac1{160}<0.\end{aligned}$$

这段三阶矩运算是整理时补入的核验，不作为课堂已经独立完成的证据。[[峰度]]则是标准化四阶中心矩 $\operatorname{Kurt}(X)=E[(X-\mu)^4]/\sigma^4$，要求有限四阶矩及正方差。四次方显著加重远离均值的取值，故不应只看峰顶高低。正态分布峰度为 3，excess kurtosis（超额峰度）是峰度减 3，因而为 0。slide 24 用 Cauchy 图形展示厚尾，但 Cauchy 没有有限均值与方差，不能给它套出一个普通有限峰度。参见 [[矩存在性的使用边界]]。

<!-- bilingual-en:start -->
The third-moment calculation is an added verification, not evidence of completed classroom work. [[峰度|Kurtosis]] is the standardised fourth central moment and requires a finite fourth moment and positive variance. It strongly weights large deviations rather than measuring peak height alone. Normal kurtosis is three, so excess kurtosis is zero. The Cauchy illustration has heavy tails but lacks the moments needed to define ordinary kurtosis; see [[矩存在性的使用边界|moment-existence conditions]].
<!-- bilingual-en:end -->

## 4. 两个分布与密度的准确含义（slides 26–35）

<!-- bilingual-en:start -->
*Two distributions and the meaning of density*
<!-- bilingual-en:end -->

### 4.1 均匀分布：相同长度的区间有相同概率

<!-- bilingual-en:start -->
*Uniform distributions assign equal probabilities to equal lengths*
<!-- bilingual-en:end -->

**slide 26 原例。** [[连续均匀分布]] $X\sim U[a,b]$，其中 $a<b$，表示在 $[a,b]$ 内相同长度的区间有相同概率。符号 $\sim$ 读作“服从……分布”。密度必须是常数；设高度为 $c$，总面积为 $c(b-a)=1$，所以 $c=1/(b-a)$。区间之外密度为 0。

<!-- bilingual-en:start -->
**Slide 26.** A [[连续均匀分布|continuous uniform variable]] assigns equal probabilities to equal-length intervals within its range. A constant density height times the range length must equal one, fixing that height at the reciprocal of the length.
<!-- bilingual-en:end -->

$$f_X(x)=\begin{cases}\dfrac1{b-a},&a<x<b,\\0,&\text{otherwise},\end{cases}\qquad F_X(x)=\begin{cases}0,&x<a,\\\dfrac{x-a}{b-a},&a\le x\le b,\\1,&x>b.\end{cases}$$

CDF 中间那一段来自 $\int_a^x (b-a)^{-1}dt=(x-a)/(b-a)$。[[均匀分布的矩|均值和方差]]也从积分得到；用 $b^2-a^2=(b-a)(a+b)$ 和 $b^3-a^3=(b-a)(b^2+ab+a^2)$ 消去分母：

<!-- bilingual-en:start -->
The middle part of the CDF is the density integrated from a to x. The mean and second moment follow by integrating and factoring the differences of squares and cubes:
<!-- bilingual-en:end -->

$$\begin{aligned}E[X]&=\frac1{b-a}\left[\frac{x^2}{2}\right]_a^b=\frac{b^2-a^2}{2(b-a)}=\frac{a+b}{2},\\E[X^2]&=\frac{b^3-a^3}{3(b-a)}=\frac{a^2+ab+b^2}{3},\\\operatorname{Var}(X)&=\frac{a^2+ab+b^2}{3}-\frac{(a+b)^2}{4}\\&=\frac{4a^2+4ab+4b^2-3a^2-6ab-3b^2}{12}=\frac{(b-a)^2}{12}.\end{aligned}$$

具体令 $X\sim U[2,6]$，则密度为 $1/4$，$P(3<X\le5)=(5-3)/4=1/2$，均值为 4，方差为 $16/12=4/3$。若区间缩成 $[0,1/2]$，密度高度变为 2，但总面积仍是 $2(1/2)=1$，再次说明密度高度不受“概率不超过 1”约束。

<!-- bilingual-en:start -->
For U[2,6], density is one quarter, the interval from three to five has probability one half, the mean is four, and variance is 4/3. A uniform distribution on [0,1/2] has density height two while its total area remains one.
<!-- bilingual-en:end -->

### 4.2 正态分布：均值控制位置，标准差控制宽度

<!-- bilingual-en:start -->
*The normal distribution separates location from scale*
<!-- bilingual-en:end -->

**slide 27 原例。** [[正态分布]]写作 $X\sim N(\mu,\sigma^2)$，要求 $\sigma>0$。括号里第二项是方差，不是标准差。$\exp(t)=e^t$；$\pi$ 是圆周率。密度为

<!-- bilingual-en:start -->
**Slide 27.** A [[正态分布|normal distribution]] is parameterised by a mean and a variance. Its standard deviation must be positive; the second parameter is the variance, not the standard deviation. The exponential notation means a power of e:
<!-- bilingual-en:end -->

$$f_X(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(x-\mu)^2}{2\sigma^2}\right],\qquad x\in\mathbb R.$$

$(x-\mu)^2$ 使左右等距离处具有相同密度；指数前的负号使远处密度衰减。增大 $\sigma$ 会把分布拉宽，同时前面的 $1/\sigma$ 降低高度，使总面积保持 1。标准正态 $Z\sim N(0,1)$ 的密度记作 $\phi(z)$，CDF 记作 $\Phi(z)=\int_{-\infty}^z\phi(t)dt$；小写 $\phi$ 是高度，大写 $\Phi$ 是累计概率。一般正态的 CDF 是 $F_X(x)=\Phi((x-\mu)/\sigma)$，第 5 节会推导这个标准化。

<!-- bilingual-en:start -->
The squared deviation gives symmetry, and the negative exponent reduces density far from the mean. A larger standard deviation spreads the distribution while lowering its height to preserve area. Lowercase phi denotes the standard normal density; uppercase Phi denotes its cumulative probability. Standardising the input gives the CDF of a general normal variable.
<!-- bilingual-en:end -->

正态密度的归一化使用高斯积分 $\int_{-\infty}^{\infty}e^{-z^2/2}dz=\sqrt{2\pi}$。在这个已知积分基础上，对称性给 $E[Z]=0$。又因 $\phi'(z)=-z\phi(z)$，分部积分可得 $E[Z^2]=[-z\phi(z)]_{-\infty}^{\infty}+\int\phi(z)dz=1$；边界项为 0，因为指数衰减快于一次幂。相同办法给 $E[Z^4]=[-z^3\phi(z)]_{-\infty}^{\infty}+3\int z^2\phi(z)dz=3$。因此 $X=\mu+\sigma Z$ 的均值、方差分别是 $\mu,\sigma^2$，峰度为 3。

<!-- bilingual-en:start -->
Normalisation uses the Gaussian integral. Symmetry yields zero standard-normal mean. Integration by parts, using the density derivative −zφ(z), yields second moment one and fourth moment three; exponential decay removes the boundary terms. Affine scaling then gives the general normal mean, variance, and kurtosis.
<!-- bilingual-en:end -->

### 4.3 为什么抽到精确的 π 的概率为零

<!-- bilingual-en:start -->
*Why an exact real value can have probability zero*
<!-- bilingual-en:end -->

**slides 28–32 原例。** 从 $[0,10]$ 均匀取数，密度始终为 $1/10$。要前一位与 $\pi$ 一致，区间是 $[3,4)$，概率 $1/10$；前两位一致，区间是 $[3.1,3.2)$，概率 $0.1/10=1/100$；前三位一致，区间是 $[3.14,3.15)$，概率 $0.01/10=1/1000$。继续缩小区间，概率趋于 0，而密度始终是 $1/10$。因此 $P(X=\pi)=0$ 与 $f_X(\pi)=1/10$ 并不矛盾。

<!-- bilingual-en:start -->
**Slides 28–32.** A uniform draw from [0,10] matches progressively more digits of pi in intervals of lengths one, one tenth, one hundredth, and so on. Their probabilities tend to zero while density remains one tenth. Zero point probability and positive density therefore agree.
<!-- bilingual-en:end -->

[[零概率不等于不可能|概率为零不等于事件集合为空]]。例如 $\{X=\pi\}$ 是一个有意义的可能取值事件，只是没有正的点质量。连续模型最终仍会实现某一个具体数；不能把不可数多个零概率单点直接用“可数可加性”相加。公理允许的是可数和。

<!-- bilingual-en:start -->
A zero-probability event need not be empty. An exact real value is meaningful but has no positive point mass in this model. A realised draw still has a particular value. Countable additivity does not permit an uncountable sum of singleton probabilities.
<!-- bilingual-en:end -->

### 4.4 密度的单位与局部概率

<!-- bilingual-en:start -->
*Density is probability per unit length*
<!-- bilingual-en:end -->

[[密度的局部概率解释]]给出密度高度的准确含义。在 $f_X$ 连续的 $x$ 附近，宽度很小的区间具有概率 $f_X(x)h+o(h)$；$o(h)$ 表示除以 $h$ 后会趋于 0 的误差。可以从 CDF 两端作差理解：

<!-- bilingual-en:start -->
[[密度的局部概率解释|Density has a local probability interpretation]] at continuity points: probability in a short interval equals height times width, plus an error negligible relative to width. This follows from the change in the CDF:
<!-- bilingual-en:end -->

$$P\!\left(x-\frac h2<X\le x+\frac h2\right)=F_X\!\left(x+\frac h2\right)-F_X\!\left(x-\frac h2\right)=f_X(x)h+o(h).$$

若 $X$ 以“米”为单位，$f_X$ 的单位是“每米”。高度本身有意义，但必须带着尺度来读；并非“只有密度比有意义”。对于两个等宽且足够窄的区间，若分母处密度为正，则概率比近似为 $f_X(x_1)/f_X(x_2)$。密度可以在单个点上改值而不改变任何区间积分，所以局部解释需要适当的连续版本，不能对任意改写后的单点高度作物理解释。这也把本节接回 [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性|SOFP 的一阶近似]]：导数乘上微小长度给出一阶变化量。

<!-- bilingual-en:start -->
If X is measured in metres, density is measured per metre. Its absolute height has a scale-dependent meaning, not merely a relative one. Equal narrow interval probabilities have approximately the ratio of their densities when the denominator density is positive. Since changing a density at one point leaves probabilities unchanged, this local interpretation uses an appropriate continuous version. The argument reuses the first-order approximation developed in [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性|SOFP Lecture 1]].
<!-- bilingual-en:end -->

### 4.5 混合分布：一部分概率在点上，一部分铺在区间里

<!-- bilingual-en:start -->
*Mixed distributions combine atoms and a continuous component*
<!-- bilingual-en:end -->

**课堂辅助例：保险赔付。** 以万元计，$X$ 有 0.8 的概率等于 0；剩余 0.2 的概率按 $U[0,10]$ 分布。这是 [[混合分布]]。CDF 在 0 处跳到 0.8，随后连续上升：

<!-- bilingual-en:start -->
**Classroom insurance illustration.** In units of ten thousand yuan, payment is zero with probability 0.8; with probability 0.2 it follows a uniform distribution on [0,10]. This [[混合分布|mixed distribution]] has both a CDF jump and continuous growth:
<!-- bilingual-en:end -->

$$F_X(x)=\begin{cases}0,&x<0,\\0.8+0.02x,&0\le x\le10,\\1,&x>10.\end{cases}$$

连续部分的密度贡献是 $0.2(1/10)=0.02$，积分只有 0.2；漏掉 0 处的 0.8，就漏掉了绝大部分概率。因此这张混合分布不能用一个普通的实数轴密度完整表示。对可积的 $g$，应计算 $E[g(X)]=0.8g(0)+0.02\int_0^{10}g(x)dx$。例如 $E[X]=0+0.02[x^2/2]_0^{10}=1$ 万元。

<!-- bilingual-en:start -->
The continuous component contributes density 0.02 and integrates to only 0.2. The remaining mass is concentrated at zero, so an ordinary density alone cannot describe the whole distribution. Expectations add the atom’s contribution to the continuous integral; the mean payment is one unit of ten thousand yuan.
<!-- bilingual-en:end -->

slide 35 和附录用 $\int g(x)\,dF_X(x)$ 统一写这些情况。$dF_X$ 表示“按 $X$ 的分布分配权重”：CDF 有跳跃，就贡献相应的点质量；有密度的部分，就贡献 $f_X(x)dx$。它不是把每个 $dF$ 都机械替换成导数。一般严格写法是对分布测度作 Lebesgue–Stieltjes 积分；Lebesgue 积分并不只用于连续变量。对当前课程，能把这套记号还原成正确的和或积分已经足够。

<!-- bilingual-en:start -->
The notation ∫g dF weights values by the distribution: jumps contribute atoms, while an absolutely continuous component contributes density times length. It does not mean replacing every dF mechanically by a derivative. The general measure-theoretic form is a Lebesgue–Stieltjes integral; Lebesgue integration is not limited to continuous variables. Here the practical requirement is to recover the correct sum or integral.
<!-- bilingual-en:end -->

## 5. 变换随机变量：先改写事件，再求分布（slides 38–42）

<!-- bilingual-en:start -->
*Transformations: rewrite the event before finding the distribution*
<!-- bilingual-en:end -->

### 5.1 一个稳定的做法：支持集加四步

<!-- bilingual-en:start -->
*A reliable workflow: range first, then four steps*
<!-- bilingual-en:end -->

[[随机变量的分布变换]]处理 $Y=g(X)$。单位换算、标准化、取对数和指示函数都属于变换。先写 $X$ 的可能范围，再求 $Y$ 的可能范围；随后依次写 $F_Y(y)=P(Y\le y)$、代入 $g(X)$、在 $X$ 的范围内解不等式、用 $F_X$ 求概率。有密度且可求导时，再由 $F_Y$ 求 $f_Y$。这套课堂辅助流程把 slide 39 的事件法展开，适用于比单调换元公式更广的情况。

<!-- bilingual-en:start -->
A [[随机变量的分布变换|distribution transformation]] includes rescaling, standardisation, logarithms, and indicators. Determine the source and transformed ranges first. Then write the target CDF, substitute the transformation, solve the inequality on the source range, and evaluate its probability using the original distribution. Differentiate only when a density calculation is justified.
<!-- bilingual-en:end -->

**课堂辅助例：$X\sim U[0,1]$，$Y=2X+1$。** 两端对应 $1,3$，故 $Y$ 的范围为 $[1,3]$。对 $1\le y\le3$，

<!-- bilingual-en:start -->
**Classroom illustration: Y=2X+1 for X uniform on [0,1].** The endpoints become one and three. Within that range:
<!-- bilingual-en:end -->

$$F_Y(y)=P(2X+1\le y)=P\!\left(X\le\frac{y-1}{2}\right)=F_X\!\left(\frac{y-1}{2}\right)=\frac{y-1}{2}.$$

必须把范围外也补齐：$y<1$ 时 CDF 为 0；$y>3$ 时为 1。区间内部导数为 $1/2$，因此 $Y\sim U[1,3]$。检查面积：$(3-1)(1/2)=1$；检查均值：$E[Y]=2(1/2)+1=2$。这两个检查分别检验分布归一化和变换计算。

<!-- bilingual-en:start -->
Complete the CDF with zero below one and one above three. Its interior derivative is one half, so Y is uniform on [1,3]. The area check gives one and linearity gives mean two, providing two independent checks.
<!-- bilingual-en:end -->

### 5.2 递减变换：不等号翻转，端点也要照顾

<!-- bilingual-en:start -->
*Decreasing transformations reverse inequalities*
<!-- bilingual-en:end -->

[[递减变换的CDF]]从同一个事件起步。若 $g$ 严格递减，$g(X)\le y$ 等价于 $X\ge g^{-1}(y)$。注意 $P(X\ge a)=1-P(X<a)=1-F_X(a-)$，不是在任何分布下都能写成 $1-F_X(a)$。slide 40 的 $1-F_X(g^{-1}(y))$ 用在其连续情形时成立；有点质量时需保留左极限。

<!-- bilingual-en:start -->
For a [[递减变换的CDF|strictly decreasing transformation]], the source inequality reverses. The event X≥a has probability one minus the left limit of the CDF at a. Slide 40’s expression without a left limit is valid in the continuous setting; atoms require the endpoint correction.
<!-- bilingual-en:end -->

$$F_Y(y)=1-F_X\bigl(g^{-1}(y)-\bigr).$$

**课堂辅助例：$X\sim U[0,1]$，$Y=3-2X$。** 两端变为 3 和 1，范围仍是 $[1,3]$。逐步移项：$3-2X\le y\Rightarrow-2X\le y-3\Rightarrow X\ge(3-y)/2$。最后一步除以负数，必须翻转不等号。连续性允许写

<!-- bilingual-en:start -->
**Classroom illustration: Y=3−2X.** Its range is again [1,3]. Subtract three, then divide by minus two and reverse the inequality. Continuity permits the following CDF calculation:
<!-- bilingual-en:end -->

$$F_Y(y)=1-F_X\!\left(\frac{3-y}{2}\right)=1-\frac{3-y}{2}=\frac{y-1}{2},\qquad1\le y\le3.$$

导数仍为 $1/2$，所以结果同样是 $U[1,3]$。$Y=1-X$ 则同理得到 $U[0,1]$。变换方向不同，不妨碍最终分布相同。若改用离散 $P(X=0)=P(X=1)=1/2$，令 $Y=-X$，则 $F_Y(0)=1$；错误地用 $1-F_X(0)$ 只得到 $1/2$，这说明左极限不是可随意丢掉的装饰。

<!-- bilingual-en:start -->
The derivative is still one half, so the increasing and decreasing affine transformations have the same target distribution. Reflecting a uniform variable about one half also preserves uniformity. With a discrete equal-probability variable taking zero and one, however, Y=−X has CDF value one at zero; omitting the left limit incorrectly gives one half.
<!-- bilingual-en:end -->

### 5.3 Jacobian 是怎样保护概率总量的

<!-- bilingual-en:start -->
*The Jacobian preserves probability mass*
<!-- bilingual-en:end -->

对具有密度的 $X$，若 $g$ 在有关区间上严格单调、连续可微且导数非零，链式法则给出

<!-- bilingual-en:start -->
For a density and a continuously differentiable monotone transformation with nonzero derivative, the chain rule gives:
<!-- bilingual-en:end -->

$$f_Y(y)=f_X\bigl(g^{-1}(y)\bigr)\left|\frac{d}{dy}g^{-1}(y)\right|.$$

[[密度换元的尺度因子]]中，$g^{-1}(y)$ 是“哪个原值变成了 $y$”；$|dx/dy|$ 是“一小段新长度对应多少旧长度”。旧区间和它的像代表同一组随机结果，故概率必须相同：$f_X(x)|dx|\approx f_Y(y)|dy|$。把 $|dy|$ 除到另一边，就是上式。拉长一个区间，密度要降低；压短一个区间，密度要提高。绝对值负责长度为正，不能因为变换递减就产生负密度。

<!-- bilingual-en:start -->
In the [[密度换元的尺度因子|density scale correction]], the inverse identifies the source value, while the absolute inverse derivative converts new length into old length. A source interval and its image represent the same outcomes, so their probabilities must agree. Stretching lowers density and compression raises it. The absolute value prevents orientation from producing a negative density.
<!-- bilingual-en:end -->

例如 $Y=2X+1$，$x=(y-1)/2$，$dx/dy=1/2$；$X$ 的密度 1 被乘成 $1/2$。递减的 $Y=3-2X$ 给 $dx/dy=-1/2$，取绝对值以后也为 $1/2$。多维时，“长度”换成面积或体积，因子换成逆变换 Jacobian 行列式的绝对值，见 [[多元换元公式]]。这与 SOFP 使用的 [[多元链式法则]]共享同一个局部线性近似思想。

<!-- bilingual-en:start -->
The inverse slopes of the two affine examples are one half and minus one half; both give density one half after taking the absolute value. In several dimensions, the [[多元换元公式|Jacobian determinant]] replaces the length factor with an area or volume factor. This uses the same local linear approximation underlying the [[多元链式法则|multivariable chain rule]].
<!-- bilingual-en:end -->

![[PSI-L2-transformations.png|900]]

### 5.4 平方变换：范围决定需要几个根

<!-- bilingual-en:start -->
*Squaring: the source range determines which roots count*
<!-- bilingual-en:end -->

**slide 41 原例：$X\sim U[0,1]$，$Y=X^2$。** 因为 $X$ 非负，$X^2\le y$ 在 $0\le y\le1$ 时等价于 $X\le\sqrt y$，所以 $F_Y(y)=\sqrt y$，而 $f_Y(y)=1/(2\sqrt y)$（$0<y<1$）。CDF 在 $y<0$ 为 0，在 $y>1$ 为 1。密度在 0 附近变得很高，却仍满足 $\int_0^1(2\sqrt y)^{-1}dy=[\sqrt y]_0^1=1$；$P(Y=0)=0$。

<!-- bilingual-en:start -->
**Slide 41.** Squaring a uniform variable on [0,1] uses only the nonnegative root. The resulting CDF is √y and density is 1/(2√y) in the unit interval. Although density diverges near zero, it integrates to one and there is no atom at zero.
<!-- bilingual-en:end -->

直觉上，$X\in[0,0.1]$ 的概率为 0.1，平方后落在宽度仅 0.01 的 $[0,0.01]$，所以平均密度变成 $0.1/0.01=10$。靠近零时平方把区间压得更厉害，密度因此集中。这说的是区间平均密度，不是声称该区间每个点的密度都等于 10。

<!-- bilingual-en:start -->
The first tenth of the source interval has probability 0.1 but is compressed into an interval of width 0.01. Its average target density is therefore ten. Compression is strongest near zero. Ten is an interval average, not the density at every point in that interval.
<!-- bilingual-en:end -->

**课堂扩展：若 $X$ 可正可负，不能漏掉负根。** 对 $Z\sim N(0,1)$、$Y=Z^2$，$y\ge0$ 时事件是 $-\sqrt y\le Z\le\sqrt y$，所以 $F_Y(y)=\Phi(\sqrt y)-\Phi(-\sqrt y)=2\Phi(\sqrt y)-1$；对 $y>0$ 求导得 $f_Y(y)=\phi(\sqrt y)/\sqrt y$，这就是一自由度卡方分布的密度。离散版本也要汇总所有原像：若 $X$ 在 $\{-1,0,1\}$ 上均匀，$P(X^2=1)=P(X=-1)+P(X=1)=2/3$，$P(X^2=0)=1/3$。

<!-- bilingual-en:start -->
**Classroom extension.** If the source can be negative, include both square roots. Squaring a standard normal gives CDF 2Φ(√y)−1 and, for positive y, density φ(√y)/√y: the chi-squared law with one degree of freedom. For a discrete uniform source on −1, zero, and one, both nonzero source values map to one and their probabilities must be added.
<!-- bilingual-en:end -->

单调逆函数公式遇到 $g'(x)=0$、平坦区间或多个分支时不能机械套用。平坦区间甚至可能把正概率压成一个点质量。先回到 CDF 的事件等式，再判断是否存在密度；参见 [[随机变量的分布变换]] 与 [[非单调换元需分支]]。

<!-- bilingual-en:start -->
Do not apply the inverse formula blindly at critical points, flat intervals, or multiple branches. A flat interval can create an atom. Return to the CDF event and then assess whether a density exists; see [[随机变量的分布变换|distribution transformations]] and [[非单调换元需分支|branchwise substitution]].
<!-- bilingual-en:end -->

### 5.5 正态标准化：除的是标准差

<!-- bilingual-en:start -->
*Normal standardisation divides by the standard deviation*
<!-- bilingual-en:end -->

[[正态标准化]]令 $Z=(X-\mu)/\sigma$，其中 $X\sim N(\mu,\sigma^2)$、$\sigma>0$。先减去均值，中心移动到 0；再除以标准差，单位变成“离均值多少个标准差”。下面不仅核验均值和方差，还核验整个密度。逆变换是 $x=\mu+\sigma z$，所以 $|dx/dz|=\sigma$：

<!-- bilingual-en:start -->
[[正态标准化|Normal standardisation]] first centres and then measures deviations in standard-deviation units. To establish the whole distribution rather than only its first two moments, use the inverse transformation x=μ+σz and its scale factor σ:
<!-- bilingual-en:end -->

$$\begin{aligned}f_Z(z)&=f_X(\mu+\sigma z)\,\sigma\\&=\frac{\sigma}{\sigma\sqrt{2\pi}}\exp\!\left[-\frac{(\mu+\sigma z-\mu)^2}{2\sigma^2}\right]\\&=\frac1{\sqrt{2\pi}}e^{-z^2/2}=\phi(z).\end{aligned}$$

**课堂完整例：$X\sim N(10,4)$。** 因为方差为 4，标准差是 2，所以 $Z=(X-10)/2$。求 $P(X\le13)$，同时对不等式两边减 10、除以 2，得到 $P(Z\le1.5)=\Phi(1.5)\approx0.93319$。再求区间概率：

<!-- bilingual-en:start -->
**Classroom example: X∼N(10,4).** Variance four means standard deviation two. Transforming the threshold thirteen gives 1.5 standard deviations and probability about 0.93319. For an interval:
<!-- bilingual-en:end -->

$$\begin{aligned}P(8<X\le13)&=P\!\left(\frac{8-10}{2}<Z\le\frac{13-10}{2}\right)\\&=P(-1<Z\le1.5)=\Phi(1.5)-\Phi(-1)\\&\approx0.93319-0.15866=0.77454.\end{aligned}$$

对任何有有限正方差的变量，中心化再除以标准差都能得到均值 0、方差 1；**只有原分布正态时**，才能因此得到标准正态。标准化不是把任意形状变成钟形的魔法。

<!-- bilingual-en:start -->
Any variable with finite positive variance can be standardised to mean zero and variance one. This gives a standard normal distribution only when the original variable is normal; standardisation does not make an arbitrary distribution bell-shaped.
<!-- bilingual-en:end -->

### 5.6 用自己的 CDF 变换自己

<!-- bilingual-en:start -->
*Transforming a variable through its own CDF*
<!-- bilingual-en:end -->

课堂中 $f_X=3x^2$、$F_X(x)=x^3$ 的例子，令 $Y=X^3$。在 $0\le y\le1$，$F_Y(y)=P(X\le y^{1/3})=(y^{1/3})^3=y$，所以 $Y\sim U[0,1]$。这里 $Y$ 恰好就是 $F_X(X)$；这是一条更一般的 [[概率积分变换]]：只要 $F_X$ 连续，就有 $F_X(X)\sim U[0,1]$。严格递增时，可直接用普通逆函数证明；一般连续 CDF 要用广义分位数处理平坦段，不要求密度处处为正。

<!-- bilingual-en:start -->
For the classroom density 3x², transforming X into X³ gives CDF y on the unit interval. This is a case of the [[概率积分变换|probability integral transform]]: a variable evaluated through its own continuous CDF is uniform. Ordinary inversion proves the strictly increasing case; general continuous CDFs use quantiles to handle flat parts.
<!-- bilingual-en:end -->

离散时一般不成立。例如 $X$ 在 0、1 上各取一半概率，$F_X(X)$ 只会取 $1/2$ 或 1，显然不是连续均匀分布。后续某些连续检验统计量在正确的零假设下产生均匀 $p$ 值，正是相同结构；有离散性、估计参数或复合零假设时需另行核验，不能直接推广。

<!-- bilingual-en:start -->
The result generally fails for a discrete variable: equal masses at zero and one transform into only one half and one. The same structure underlies exact uniform p-values for suitable continuous test statistics under the correct null. Discreteness, estimated parameters, and composite nulls require additional checks.
<!-- bilingual-en:end -->

## 6. 一张格子图串起联合、边际、条件与 Bayes（slides 44–59）

<!-- bilingual-en:start -->
*Joint, marginal, conditional, and Bayesian probabilities in one grid*
<!-- bilingual-en:end -->

### 6.1 一格、一行、一列分别在说什么

<!-- bilingual-en:start -->
*Reading a cell, a row, and a column*
<!-- bilingual-en:end -->

[[联合分布]]描述 $X,Y$ 同时怎样取值。slides 44–50 用 $X,Y\in\{0,1,\ldots,6\}$ 的格子图：横轴是 $X$，纵轴是 $Y$，一格 $p_{x,y}=P(X=x,Y=y)$ 对应“两件事同时成立”。逗号表示“且”。所有格子的概率非负且总和为 1。每个格子只是一个可能的数值对；在一般原始样本空间里，也可能有多个 $\omega$ 映到同一个数值对。

<!-- bilingual-en:start -->
A [[联合分布|joint distribution]] describes simultaneous values. In slides 44–50, X runs horizontally and Y vertically. Each cell gives the probability that both specified values occur. The entries are nonnegative and sum to one. In a more general underlying experiment, several original outcomes may map to the same numerical pair.
<!-- bilingual-en:end -->

![[PSI-L2-slide-50.png|900]]

上图保留 slide 50 的原始布局：橙色格子是 $p_{2,4}$，绿色列固定 $X=2$，蓝色行固定 $Y=4$。**固定一列不等于已经得到条件分布**，还要除以该列的总概率；**把一行相加**则是忽略 $X$ 后得到 $Y=4$ 的边际概率。

<!-- bilingual-en:start -->
This is the original layout of slide 50: the orange cell gives the joint event, the green column fixes X, and the blue row fixes Y. A column becomes a conditional distribution only after division by its total. Summing a row instead gives a marginal probability.
<!-- bilingual-en:end -->

**官方补充讲义第 3 页的数值例。** 为了把每一步算完，下面使用它的六格表，方向保持不变。内部六项是联合概率，边上是总和；不要把边际项又算进内部总和。

<!-- bilingual-en:start -->
**Official supplement, page 3.** The six-cell example below keeps the same orientation and makes every calculation explicit. Interior entries are joint probabilities; the margins are their totals and must not be counted again as additional cells.
<!-- bilingual-en:end -->

| $Y\backslash X$ | $0$ | $1$ | $2$ | $p_Y(y)$ |
|---|---:|---:|---:|---:|
| $1$ | $0.10$ | $0.20$ | $0.20$ | $0.50$ |
| $0$ | $0.20$ | $0.20$ | $0.10$ | $0.50$ |
| $p_X(x)$ | $0.30$ | $0.40$ | $0.30$ | $1.00$ |

### 6.2 边际化：把另一个变量加掉

<!-- bilingual-en:start -->
*Marginalisation sums out the other variable*
<!-- bilingual-en:end -->

[[边际分布]]只描述某一个变量。要 $Y$，就保留 $y$、遍历所有 $x$；要 $X$，就保留 $x$、遍历所有 $y$：$p_Y(y)=\sum_xp_{x,y}$，$p_X(x)=\sum_yp_{x,y}$。同一行的格子互斥，因为一次不可能同时有 $X=0$ 和 $X=1$；所以可以直接相加。

<!-- bilingual-en:start -->
A [[边际分布|marginal distribution]] describes one variable on its own. Hold its value fixed and sum over every possible value of the other variable. These cells are disjoint events, which justifies adding their probabilities.
<!-- bilingual-en:end -->

例如 $P(Y=1)=0.10+0.20+0.20=0.50$，$P(X=2)=0.20+0.10=0.30$。但是只知道行总和与列总和，通常不能恢复内部各格子；[[边际分布不定联合分布]]说明单变量的分布没有告诉我们它们怎样配对，也就没有完整的依赖信息。

<!-- bilingual-en:start -->
For example, the Y=1 row totals 0.50 and the X=2 column totals 0.30. [[边际分布不定联合分布|Marginals generally do not determine the joint distribution]]: the row and column totals do not reveal how probability is paired inside the table.
<!-- bilingual-en:end -->

### 6.3 条件化：先限制，再重新归一化

<!-- bilingual-en:start -->
*Conditioning restricts attention and renormalises*
<!-- bilingual-en:end -->

[[条件分布]]回答“已经知道 $X=x$ 后，$Y$ 的不确定性还是什么样”。离散情况下需要 $p_X(x)>0$，公式由 [[条件概率]]直接给出。

<!-- bilingual-en:start -->
A [[条件分布|conditional distribution]] describes the uncertainty about Y after a value of X is known. In the discrete case, the conditioning value must have positive probability, and the formula follows from [[条件概率|conditional probability]]:
<!-- bilingual-en:end -->

$$p_{Y\mid X}(y\mid x)=\frac{p_{x,y}}{p_X(x)}=\frac{p_{x,y}}{\sum_rp_{x,r}}.$$

分子是一格；分母是**整列**。$r$ 是临时求和下标，表示把这一列所有可能的 $Y$ 都加起来，不是额外引入第三个随机变量。固定 $X=2$ 后，

<!-- bilingual-en:start -->
The numerator is one cell; the denominator is the whole column. The symbol r is only a summation index, not a third random variable. For X=2:
<!-- bilingual-en:end -->

$$P(Y=0\mid X=2)=\frac{0.10}{0.30}=\frac13,\qquad P(Y=1\mid X=2)=\frac{0.20}{0.30}=\frac23.$$

两项加起来是 1，才构成新的分布。原来 $Y=1$ 的概率是 $1/2$，观察到 $X=2$ 后变成 $2/3$。反过来给定 $Y=1$，则固定一行并除以 0.50，得到 $P(X=0\mid Y=1)=0.2$、$P(X=1\mid Y=1)=0.4$、$P(X=2\mid Y=1)=0.4$。哪一个变量在竖线右边，就先固定哪一个。

<!-- bilingual-en:start -->
The two conditional probabilities sum to one. Observing X=2 changes the probability of Y=1 from one half to two thirds. Reversing the conditioning fixes the Y=1 row and gives probabilities 0.2, 0.4, and 0.4 for X. The variable to the right of the conditioning bar is the one held fixed.
<!-- bilingual-en:end -->

### 6.4 连续情形：先画支持集，再写积分上下限

<!-- bilingual-en:start -->
*Continuous distributions require the correct region of integration*
<!-- bilingual-en:end -->

若存在联合密度 $f_{X,Y}(x,y)$，一小块面积的概率约为密度乘面积，区域概率要做二重积分。[[边际分布|边际化]]把求和换成积分；固定的变量留下，另一个变量被积掉。[[条件分布|条件密度]]则把一条切片除以其面积：

<!-- bilingual-en:start -->
With a joint density, probabilities are integrals over regions. [[边际分布|Marginalisation]] integrates out the other variable; [[条件分布|conditioning]] instead normalises a slice by its area:
<!-- bilingual-en:end -->

$$f_X(x)=\int_{\mathbb R}f_{X,Y}(x,y)\,dy,\qquad f_{Y\mid X}(y\mid x)=\frac{f_{X,Y}(x,y)}{f_X(x)}\quad\text{when }f_X(x)>0.$$

连续 $X$ 通常满足 $P(X=x)=0$，这里不能写成两个事件概率的 $0/0$。密度比给出条件分布的一个版本，对 $X$ 的分布几乎处处定义即可；在特殊零概率位置任意改动，并不改变条件期望或整体概率结论。

<!-- bilingual-en:start -->
A continuous X generally assigns zero probability to each individual point, so this is not a ratio of event probabilities equal to zero over zero. The density formula supplies a version of the conditional distribution, determined almost everywhere under the marginal law of X.
<!-- bilingual-en:end -->

![[PSI-L2-triangle.png|900]]

**官方补充讲义第 8 页原例。** 令 $f_{X,Y}(x,y)=2$ 在三角形 $0<y<x<1$ 内成立，其他地方为 0。三角形面积为 $1/2$，乘密度 2 得总概率 1。固定 $x$ 时，$y$ 从 0 走到 $x$；固定 $y$ 时，$x$ 从 $y$ 走到 1：

<!-- bilingual-en:start -->
**Official supplement, page 8.** The joint density is two in the triangle 0<y<x<1 and zero elsewhere. Height two times area one half gives total probability one. Vertical and horizontal slices determine the integration limits:
<!-- bilingual-en:end -->

$$f_X(x)=\int_0^x2\,dy=[2y]_0^x=2x\quad(0<x<1),\qquad f_Y(y)=\int_y^1 2\,dx=[2x]_y^1=2(1-y)\quad(0<y<1).$$

所以 $f_{Y\mid X}(y\mid x)=2/(2x)=1/x$，范围是 $0<y<x$；即 $Y\mid X=x\sim U[0,x]$。当 $x=1/2$，条件密度为 2，范围为 $(0,1/2)$，于是 $P(Y<1/4\mid X=1/2)=\int_0^{1/4}2dy=1/2$。不加条件时要用另一张密度：$P(Y<1/4)=\int_0^{1/4}2(1-y)dy=[2y-y^2]_0^{1/4}=7/16$。条件化不仅更换密度高度，也可能改变允许的范围。

<!-- bilingual-en:start -->
Thus Y conditional on X=x is uniform on [0,x]. At x=1/2, the probability of falling below 1/4 is one half. Without conditioning, the marginal density instead gives 7/16. Conditioning can change both density height and the allowable range.
<!-- bilingual-en:end -->

### 6.5 Bayes：用同一个格子反转条件方向

<!-- bilingual-en:start -->
*Bayes reverses the conditioning direction through the same joint event*
<!-- bilingual-en:end -->

同一个联合概率可以沿列求，也可以沿行求：$p_{x,y}=p_{Y\mid X}(y\mid x)p_X(x)=p_{X\mid Y}(x\mid y)p_Y(y)$。把右边的 $p_Y(y)$ 除过去，就得到 [[Bayes法则]]。若用另一方向的条件概率展开分母，得到

<!-- bilingual-en:start -->
The same joint cell can be reached from a column or a row. Equating these two factorizations yields [[Bayes法则|Bayes’ rule]]. Expanding the marginal denominator gives:
<!-- bilingual-en:end -->

$$p_{X\mid Y}(x\mid y)=\frac{p_{Y\mid X}(y\mid x)p_X(x)}{\sum_{x'}p_{Y\mid X}(y\mid x')p_X(x')}.$$

$x'$ 只是遍历所有候选值的另一个下标。六格表中，$P(X=2\mid Y=1)=[(2/3)(0.30)]/0.50=0.40$，与直接用该行归一化一致。**先验** $p_X(x)$ 描述看到 $y$ 之前的分布；**似然** $p_{Y\mid X}(y\mid x)$ 把已看到的 $y$ 固定，比较不同候选 $x$ 产生该证据的可能性；**后验** $p_{X\mid Y}(x\mid y)$ 是看过证据后对 $x$ 归一化的分布。

<!-- bilingual-en:start -->
The primed x runs over all candidate values. The table example yields posterior probability 0.40 for X=2 after observing Y=1, agreeing with direct row normalisation. The prior describes X before the observation; the likelihood fixes observed y and varies the candidate x; the posterior is the normalised distribution over x after seeing y.
<!-- bilingual-en:end -->

这与上一讲的 [[似然函数]]、[[先验分布]]、[[后验分布]]是同一条更新链。似然不必对候选参数加起来为 1；后验则必须。连续参数时，把候选参数上的求和改成积分，仍然是“似然乘先验，再按全部候选归一化”。

<!-- bilingual-en:start -->
This is the same chain connecting [[似然函数|likelihood]], [[先验分布|prior]], and [[后验分布|posterior]] in Lecture 1. Likelihood need not sum to one over candidates, while posterior must. Continuous parameters use an integral for that normalisation.
<!-- bilingual-en:end -->

### 6.6 违约信号原例：80% 不是看到信号后的违约率

<!-- bilingual-en:start -->
*The default-signal example*
<!-- bilingual-en:end -->

**slide 58 与补充讲义第 5 页原例。** $H$ 表示借款人将违约，$S$ 表示出现风险信号。已知 $P(H)=0.10$、$P(S\mid H)=0.80$、$P(S\mid H^c)=0.20$。先算两条产生信号的路径：违约且报警的概率是 $0.10(0.80)=0.08$；不违约但报警的概率是 $0.90(0.20)=0.18$。二者互斥，按 [[全概率公式]]相加得到 $P(S)=0.26$。

<!-- bilingual-en:start -->
**Slide 58 and supplement page 5.** Default has prior probability 0.10. Signals occur with probability 0.80 among defaulters and 0.20 among non-defaulters. The joint probabilities of the two signal-producing paths are 0.08 and 0.18. The [[全概率公式|law of total probability]] gives total signal probability 0.26.
<!-- bilingual-en:end -->

$$P(H\mid S)=\frac{P(S\mid H)P(H)}{P(S)}=\frac{0.80\cdot0.10}{0.26}=\frac4{13}\approx0.30769.$$

按这些比例想象 1,000 人：100 人违约，其中 80 人报警；900 人不违约，其中 180 人报警。共 260 个报警，真正违约的是 80 个，所以比例为 $80/260$。似然中的 0.80 是“违约者里谁报警”，后验中的 0.30769 是“报警者里谁违约”；这两个分母对应不同人群。

<!-- bilingual-en:start -->
In a population with these exact proportions, 1,000 borrowers include 100 defaulters producing 80 signals and 900 non-defaulters producing 180 signals. Of 260 signals, 80 come from defaulters. The likelihood conditions on the default group; the posterior conditions on the signal group.
<!-- bilingual-en:end -->

### 6.7 赔率形式：证据把原赔率乘多少倍

<!-- bilingual-en:start -->
*Odds form measures how evidence multiplies prior odds*
<!-- bilingual-en:end -->

[[Bayes赔率形式]]把概率 $p$ 改写成赔率 $p/(1-p)$。赔率不是概率：例如 $p=1/4$ 对应赔率 $1/3$，表示每 1 份支持事件，配 3 份支持补事件。分别写出 $P(H\mid S)$ 和 $P(H^c\mid S)$，相除后共同分母 $P(S)$ 抵消：

<!-- bilingual-en:start -->
[[Bayes赔率形式|Bayesian odds]] compare the probability of an event to its complement. Odds are not probabilities: probability one quarter corresponds to odds one to three. Taking the ratio of the two posterior probabilities cancels the shared evidence normaliser:
<!-- bilingual-en:end -->

$$\frac{P(H\mid S)}{P(H^c\mid S)}=\frac{P(H)}{P(H^c)}\frac{P(S\mid H)}{P(S\mid H^c)}.$$

本例先验赔率为 $0.1/0.9=1/9$，likelihood ratio（似然比）为 $0.8/0.2=4$，所以后验赔率为 $4/9$。把赔率 $o$ 还原为概率，用 $p=o/(1+o)$，于是 $(4/9)/(1+4/9)=4/13$。似然比大于 1，证据提高该事件的赔率；等于 1 则不改变；小于 1 则降低。

<!-- bilingual-en:start -->
Here prior odds are 1/9 and the likelihood ratio is four, giving posterior odds 4/9. Converting odds o back to probability via o/(1+o) gives 4/13. A likelihood ratio above, equal to, or below one raises, preserves, or lowers the odds.
<!-- bilingual-en:end -->

## 7. 条件矩：在每一组里重新计算（slides 61–64）

<!-- bilingual-en:start -->
*Conditional moments recalculate averages within groups*
<!-- bilingual-en:end -->

### 7.1 固定 x 是一个数，放回 X 就是随机变量

<!-- bilingual-en:start -->
*Fixing x gives a number; substituting X gives a random variable*
<!-- bilingual-en:end -->

[[条件期望]]先在给定信息后对剩余随机性作平均。在六格表中，已知 $X=2$ 时，$Y=0,1$ 的概率分别为 $1/3,2/3$，所以 $E[Y\mid X=2]=0(1/3)+1(2/3)=2/3$。一般把 $m(x)=E[Y\mid X=x]$ 看成一个函数：给它某个 $x$，它返回这一组的均值。

<!-- bilingual-en:start -->
[[条件期望|Conditional expectation]] averages the uncertainty remaining after information is supplied. In the six-cell table, conditioning on X=2 gives mean Y equal to two thirds. The function m(x) maps a particular group value to its conditional mean.
<!-- bilingual-en:end -->

$$m(0)=\frac{0.10}{0.30}=\frac13,\qquad m(1)=\frac{0.20}{0.40}=\frac12,\qquad m(2)=\frac{0.20}{0.30}=\frac23.$$

$m(2)=2/3$ 是数；$m(X)=E[Y\mid X]$ 仍是随机变量。观察 $X$ 以前，不知道会用哪一组均值；它以 0.30、0.40、0.30 的概率分别取 $1/3,1/2,2/3$。所以对它再取期望或方差都有意义。条件期望既不是“总是常数”，也不是“知道 $X$ 就完全知道 $Y$”；知道的是这组的平均值。

<!-- bilingual-en:start -->
The value m(2) is a number, while m(X) remains random before X is observed. Its three possible values have probabilities 0.30, 0.40, and 0.30. It therefore has its own expectation and variance. Knowing a conditional mean need not reveal the actual outcome.
<!-- bilingual-en:end -->

### 7.2 h(X,Y) 的条件期望：固定什么，平均什么

<!-- bilingual-en:start -->
*Conditioning a function of two variables*
<!-- bilingual-en:end -->

slide 61 写 $E[h(X,Y)\mid X=x]=\sum_yh(x,y)p_{Y\mid X}(y\mid x)$。$h$ 只是某个已知函数；固定 $x$ 后，取值 $h(x,y)$ 随 $y$ 变化，权重则是新的条件概率。以 $H=X+2Y$ 为例，在 $X=2$ 的列里，$H$ 只能取 2 和 4，因而

<!-- bilingual-en:start -->
Slide 61 averages h(x,y) over y using conditional probabilities while holding x fixed. For H=X+2Y, the X=2 column gives possible values two and four:
<!-- bilingual-en:end -->

$$E[H\mid X=2]=2\cdot\frac13+4\cdot\frac23=\frac{10}{3}.$$

也可以写 $E[X+2Y\mid X=x]=x+2E[Y\mid X=x]$。按照 [[条件期望取出已知量]]，给定 $X$ 后，任何已知的 $X$ 的函数都可以提出，只要所需乘积可积；例如 $E[XY\mid X]=X E[Y\mid X]$。这不是无条件情形下的 $E[XY]=E[X]E[Y]$，后者一般需要额外条件。

<!-- bilingual-en:start -->
Alternatively, conditional linearity gives x+2m(x). A function already determined by X can be taken outside a conditional expectation when the required quantities are integrable. This does not justify factorising an unconditional product expectation.
<!-- bilingual-en:end -->

### 7.3 条件方差使用各组自己的中心

<!-- bilingual-en:start -->
*Conditional variance centres within each group*
<!-- bilingual-en:end -->

[[条件方差]]为 $\operatorname{Var}(Y\mid X=x)=E[(Y-m(x))^2\mid X=x]=E[Y^2\mid X=x]-m(x)^2$。这里减的是这一列的均值，不能先减总体均值 $E[Y]$。六格表中的 $Y$ 只有 0 和 1，所以 $Y^2=Y$，即 [[二元结果条件方差]]中的 Bernoulli 计算：

<!-- bilingual-en:start -->
[[条件方差|Conditional variance]] measures squared deviations from each group’s own mean, not from the overall mean. In the table Y is binary, so its square equals itself and [[二元结果条件方差|Bernoulli conditional variance]] applies:
<!-- bilingual-en:end -->

$$\begin{aligned}v(0)&=\frac13-\left(\frac13\right)^2=\frac29,\\v(1)&=\frac12-\left(\frac12\right)^2=\frac14,\\v(2)&=\frac23-\left(\frac23\right)^2=\frac29.\end{aligned}$$

例如在 $X=2$ 时，按定义直接算也是 $(0-2/3)^2(1/3)+(1-2/3)^2(2/3)=4/27+2/27=2/9$。而 $H=X+2Y$ 在这列里只是给 $2Y$ 加上常数 2，所以 $\operatorname{Var}(H\mid X=2)=4(2/9)=8/9$。

<!-- bilingual-en:start -->
Directly weighting the squared deviations in the X=2 column also gives 2/9. Since H in this group is twice Y plus a constant, its conditional variance is four times as large, or 8/9.
<!-- bilingual-en:end -->

## 8. 分组以后怎样回到总体（slides 66–72）

<!-- bilingual-en:start -->
*Returning from group summaries to the population*
<!-- bilingual-en:end -->

### 8.1 全期望：组内平均，再按组的概率加权

<!-- bilingual-en:start -->
*Total expectation weights the group means*
<!-- bilingual-en:end -->

slides 66–69 先用格子总量说明求和顺序：可以逐格加，也可以先算行总和、再把各行相加。若每格等权，平均值也能这样拆；一旦各组概率不同，第二层平均必须使用组概率。[[全期望公式]]（law of iterated expectations，LIE）写成 $E[H]=E[E[H\mid X]]$，前提是 $E|H|<\infty$。

<!-- bilingual-en:start -->
Slides 66–69 explain summation order through a grid. Summing cells directly equals summing within rows and then across rows. For averages, unequal groups require probability weights. The [[全期望公式|law of iterated expectations]] averages the conditional expectation to recover the original expectation, assuming integrability.
<!-- bilingual-en:end -->

对 $H=h(X,Y)$，把条件期望的定义代入，就能看到分母怎样被外层权重抵消：

<!-- bilingual-en:start -->
Substituting the conditional distribution shows exactly how the outer group weights cancel the inner normalising denominators:
<!-- bilingual-en:end -->

$$\begin{aligned}E[E[H\mid X]]&=\sum_x\left[\sum_yh(x,y)\frac{p_{x,y}}{p_X(x)}\right]p_X(x)\\&=\sum_x\sum_yh(x,y)p_{x,y}=E[H].\end{aligned}$$

只对 $p_X(x)>0$ 的列计算；零概率列无需定义初等比值。可数无限求和时，可积性保证必要的求和交换。这个公式是 [[塔式法则]]在外层不再保留任何非平凡信息时的特例。

<!-- bilingual-en:start -->
Only columns of positive probability need enter the elementary ratios. Integrability justifies the required rearrangement for countable sums. This is a special case of the [[塔式法则|tower property]] in which no nontrivial information remains at the outer level.
<!-- bilingual-en:end -->

**沿用六格表。** 直接看 $Y$ 的边际分布，$E[Y]=0(0.5)+1(0.5)=0.5$。改为先算列均值：

<!-- bilingual-en:start -->
**Using the same table.** The marginal mean of Y is one half. Averaging its column means gives the same result:
<!-- bilingual-en:end -->

$$E[m(X)]=0.30\cdot\frac13+0.40\cdot\frac12+0.30\cdot\frac23=0.10+0.20+0.20=0.50.$$

这里三个均值的简单平均碰巧也是 $1/2$，只是表的对称造成的，不能据此省略权重。换成两个概率为 0.9 和 0.1、均值为 2 和 8 的组，总体均值为 $0.9(2)+0.1(8)=2.6$，不是 $(2+8)/2=5$。对 $H=X+2Y$，各列均值为 $2/3,2,10/3$，再加权得到 $0.3(2/3)+0.4(2)+0.3(10/3)=2$，也等于 $E[X]+2E[Y]=1+1$。

<!-- bilingual-en:start -->
The table’s unweighted mean happens to agree because of symmetry, not because weights are optional. Groups with probabilities 0.9 and 0.1 and means two and eight have overall mean 2.6 rather than five. The same calculation for H=X+2Y gives mean two, agreeing with unconditional linearity.
<!-- bilingual-en:end -->

**连续三角形例继续。** 已知 $Y\mid X=x\sim U[0,x]$，故 $E[Y\mid X=x]=x/2$，而 $f_X(x)=2x$。因此 $E[Y]=\int_0^1(x/2)(2x)dx=\int_0^1x^2dx=1/3$。直接用 $f_Y(y)=2(1-y)$ 也得到 $\int_0^1 y\,2(1-y)dy=1/3$。内层用条件分布，外层用条件变量自身的边际分布。

<!-- bilingual-en:start -->
**Continuing the triangular example.** The conditional mean is x/2 and the marginal density of X is 2x. Integrating their product gives one third, matching direct integration under the marginal density of Y. The inner average uses the conditional distribution; the outer average uses the marginal distribution of the conditioning variable.
<!-- bilingual-en:end -->

这也解释了回归里常用的一步：若 $E[\varepsilon\mid X]=0$，则 $E[\varepsilon]=E[E[\varepsilon\mid X]]=0$；若乘积可积，还得到 $E[X\varepsilon]=E[X E[\varepsilon\mid X]]=0$。第二个等号先用了条件期望中提出已知量，再用了全期望。

<!-- bilingual-en:start -->
This justifies a common regression argument: zero conditional mean implies zero unconditional mean and, when the product is integrable, zero expected product with X. The latter result combines pulling out the known factor with total expectation.
<!-- bilingual-en:end -->

### 8.2 全方差：组内分散加组间均值分散

<!-- bilingual-en:start -->
*Total variance separates within-group and between-group variation*
<!-- bilingual-en:end -->

[[全方差公式]]要求有限二阶矩。记 $m(X)=E[Y\mid X]$，$v(X)=\operatorname{Var}(Y\mid X)$，则

<!-- bilingual-en:start -->
The [[全方差公式|law of total variance]] requires a finite second moment. Writing the conditional mean as m(X) and conditional variance as v(X):
<!-- bilingual-en:end -->

$$\boxed{\operatorname{Var}(Y)=\underbrace{E[v(X)]}_{\text{within groups}}+\underbrace{\operatorname{Var}(m(X))}_{\text{between group means}}.}$$

第一项先量每组围绕自己均值的波动，再按组概率平均；第二项量这些组均值围绕总体均值的波动。条件方差 $v(X)$、条件方差的期望 $E[v(X)]$、条件均值的方差 $\operatorname{Var}(m(X))$ 是三个不同对象，不能交换括号的位置。

<!-- bilingual-en:start -->
The first term averages each group’s spread around its own mean. The second measures the spread of group means around the population mean. Conditional variance, its expectation, and the variance of conditional expectation are different objects; moving the brackets changes the meaning.
<!-- bilingual-en:end -->

**六格表的完整检验。** 因为 $Y\sim\operatorname{Bernoulli}(1/2)$，总方差为 $1/4$。组内部分是

<!-- bilingual-en:start -->
**Full numerical check using the table.** Y is Bernoulli with parameter one half, so total variance is one quarter. The within-group contribution is:
<!-- bilingual-en:end -->

$$E[v(X)]=\frac3{10}\frac29+\frac4{10}\frac14+\frac3{10}\frac29=\frac1{15}+\frac1{10}+\frac1{15}=\frac7{30}.$$

条件均值的总体平均是 $1/2$，所以组间部分为

<!-- bilingual-en:start -->
The average conditional mean is one half, so the between-group contribution is:
<!-- bilingual-en:end -->

$$\begin{aligned}\operatorname{Var}(m(X))&=\frac3{10}\left(\frac13-\frac12\right)^2+\frac4{10}\left(\frac12-\frac12\right)^2+\frac3{10}\left(\frac23-\frac12\right)^2\\&=\frac3{10}\frac1{36}+0+\frac3{10}\frac1{36}=\frac1{60}.\end{aligned}$$

$$\frac7{30}+\frac1{60}=\frac{14}{60}+\frac1{60}=\frac{15}{60}=\frac14.\quad\checkmark$$

**连续三角形例再检验一次。** $E[X]=2/3$、$E[X^2]=1/2$，所以 $\operatorname{Var}(X)=1/2-4/9=1/18$。条件分布均匀给出 $m(X)=X/2$、$v(X)=X^2/12$，于是 $E[v(X)]=(1/2)/12=1/24$、$\operatorname{Var}(m(X))=(1/4)(1/18)=1/72$，相加为 $1/18$。直接用 $E[Y]=1/3$、$E[Y^2]=1/6$ 得到 $1/6-1/9=1/18$，再次一致。

<!-- bilingual-en:start -->
**Continuous check.** In the triangular model, X has mean 2/3, second moment 1/2, and variance 1/18. The conditional uniform law gives mean X/2 and variance X²/12. Expected conditional variance is 1/24 and variance of the conditional mean is 1/72, summing to 1/18, the direct marginal variance of Y.
<!-- bilingual-en:end -->

> [!note]- 展开全方差证明（PDF 第 90 页，non-examinable）
> **附录补充：为什么交叉项为零。** 令 $\mu=E[Y]$、$m=m(X)$。加减同一个 $m$，得到 $Y-\mu=(Y-m)+(m-\mu)$。平方并取期望：
> $$\operatorname{Var}(Y)=E[(Y-m)^2]+E[(m-\mu)^2]+2E[(Y-m)(m-\mu)].$$
> 第一项由全期望等于 $E[\operatorname{Var}(Y\mid X)]$；第二项因 $E[m]=\mu$ 等于 $\operatorname{Var}(m)$。处理第三项时，$m-\mu$ 已由 $X$ 确定，可以提出：
> $$\begin{aligned}E[(Y-m)(m-\mu)]&=E\!\left[E[(Y-m)(m-\mu)\mid X]\right]\\&=E\!\left[(m-\mu)E[Y-m\mid X]\right]\\&=E[(m-\mu)(m-m)]=0.\end{aligned}$$
> 这里消失的原因是零条件均值，不是声称残差与 $X$ 独立。有限二阶矩保证这些乘积和期望可用。
>
> <!-- bilingual-en:start -->
> **Optional appendix proof.** Decompose each deviation into its residual from the group mean and the group mean’s deviation from the overall mean. Squaring yields within-group, between-group, and cross terms. Total expectation identifies the first term; the average group mean equals the overall mean. In the cross term, the group-mean deviation is known given X, while the conditional mean of the residual is zero. Thus the cross term vanishes without requiring independence.
> <!-- bilingual-en:end -->

与 [[条件期望投影]]的联系是：平方损失下，用 $E[Y\mid X]$ 预测 $Y$，剩余均方误差就是组内项。若 $\operatorname{Var}(Y)>0$，$\operatorname{Var}(E[Y\mid X])/\operatorname{Var}(Y)$ 描述条件均值所解释的总体方差比例；它不自动等于任意线性回归的样本 $R^2$，因为条件均值可能非线性，且这里讨论的是总体量。

<!-- bilingual-en:start -->
The connection to [[条件期望投影|conditional expectation as a projection]] is that its prediction error under squared loss equals the within-group contribution. The ratio of between-group variance to total variance measures the population variance explained by the conditional mean. It is not automatically the sample R² of an arbitrary linear regression, since the conditional mean may be nonlinear and the quantities here are population quantities.
<!-- bilingual-en:end -->

## 9. 三种不同强度的关系（slides 74–85）

<!-- bilingual-en:start -->
*Three levels of dependence restrictions*
<!-- bilingual-en:end -->

### 9.1 独立性控制整张条件分布

<!-- bilingual-en:start -->
*Independence fixes the whole conditional distribution*
<!-- bilingual-en:end -->

[[事件独立]]的定义是 $P(A\cap B)=P(A)P(B)$。当 $P(B)>0$，它等价于 $P(A\mid B)=P(A)$；乘积定义在零概率事件时仍能使用。对 [[随机变量独立]]，要求一方取值落在任意可测集合 $C$、另一方落在任意可测集合 $D$ 的事件都满足这个分解：

<!-- bilingual-en:start -->
[[事件独立|Event independence]] means factorisation of the intersection probability. With a positive conditioning probability, this is equivalent to unchanged conditional probability. The factorisation definition also handles null events. [[随机变量独立|Random-variable independence]] requires factorisation for every pair of measurable value sets:
<!-- bilingual-en:end -->

$$P(X\in C,Y\in D)=P(X\in C)P(Y\in D).$$

这不是只检查某一个点或某一个均值。[[随机变量独立的分布判据]]允许用联合 CDF 分解检查；离散情形也可逐格检查 $p_{x,y}=p_X(x)p_Y(y)$。等价地，[[条件分布不变推出独立]]：给定 $X=x$ 后，$Y$ 的整张条件分布对几乎所有 $x$ 都与边际分布一致。

<!-- bilingual-en:start -->
Independence is not a check of one point or one moment. [[随机变量独立的分布判据|Distribution criteria]] permit checking the joint CDF, or every PMF cell in a discrete model. Equivalently, [[条件分布不变推出独立|the entire conditional law is unchanged]] for almost every conditioning value.
<!-- bilingual-en:end -->

[[独立同分布]]（iid）包含两个要求：每个观测服从同一分布；观测之间相互独立。相同直方图或相同边际分布不保证独立，前面“复制同一个骰子 10 次”就是反例。多个变量的相互独立还强于两两独立，见 [[两两独立不推出相互独立]]。

<!-- bilingual-en:start -->
[[独立同分布|Independent and identically distributed]] combines a common marginal distribution with mutual independence. Identical marginals alone do not suffice, as copying one die result shows. With more than two variables, [[两两独立不推出相互独立|pairwise independence is weaker than mutual independence]].
<!-- bilingual-en:end -->

### 9.2 条件独立：分组后独立，合起来未必

<!-- bilingual-en:start -->
*Conditional independence may disappear after pooling*
<!-- bilingual-en:end -->

[[条件独立]] $A\perp B\mid W$ 表示在相同的 $W$ 信息下，$P(A\cap B\mid W)=P(A\mid W)P(B\mid W)$。slide 75 的雨伞例说：已知天气后，两个人各自决定是否带伞，决策相互独立；但两人都在雨天更可能带伞，忽略天气后就会一起变化。

<!-- bilingual-en:start -->
[[条件独立|Conditional independence]] factorises probabilities after the same information W is given. The umbrella example on slide 75 makes decisions independent within a weather state, although pooling weather states makes the decisions move together.
<!-- bilingual-en:end -->

**给原例补上数值。** 设雨天概率为 $1/2$；雨天每人带伞概率 0.8，晴天为 0.2，且在各天气组内独立。每人的边际带伞概率是 $(0.8+0.2)/2=0.5$。两人同时带伞的概率为 $(0.8^2+0.2^2)/2=0.34$，不是 $0.5^2=0.25$。所以条件独立并不推出边际独立；反过来也不成立，完整双向边界见 [[条件独立不等于边际独立]]。

<!-- bilingual-en:start -->
**A numerical elaboration of that example.** Suppose rainy and dry days are equally likely, with individual umbrella probabilities 0.8 and 0.2 respectively. Conditional independence gives joint umbrella probability 0.34 after pooling, whereas the product of the marginals is 0.25. [[条件独立不等于边际独立|Conditional and marginal independence do not imply each other]].
<!-- bilingual-en:end -->

### 9.3 均值独立只固定一个条件矩

<!-- bilingual-en:start -->
*Mean independence fixes one conditional moment*
<!-- bilingual-en:end -->

[[均值独立]]的方向很重要：“$Y$ 对 $X$ 均值独立”表示 $E[Y\mid X]=E[Y]$ 几乎处处，要求 $Y$ 可积。无论看到哪个有意义的 $X$ 值，预测 $Y$ 的平均值都不变。但条件方差、尾部或其他形状仍可变化。它一般不保证 $E[X\mid Y]=E[X]$，后面第一个反例会直接展示这种不对称。

<!-- bilingual-en:start -->
[[均值独立|Mean independence]] is directional. Y is mean independent of X when its conditional mean equals its unconditional mean almost surely, assuming integrability. Conditional variance or shape may still change. Reversing X and Y gives a different condition.
<!-- bilingual-en:end -->

### 9.4 条件标准化何时真的得到独立

<!-- bilingual-en:start -->
*When conditional standardisation yields independence*
<!-- bilingual-en:end -->

**slide 78 原例。** 假设各组确实满足 $Y\mid X=x\sim N(\mu(x),\sigma^2(x))$，且 $\sigma(x)>0$。令 $Z=(Y-\mu(X))/\sigma(X)$。给定 $X=x$ 后，$\mu(x),\sigma(x)$ 都是常数，因此每组的 $Z$ 都服从 $N(0,1)$。整张条件分布相同，故由 [[条件分布不变推出独立]]得到 $Z\perp X$。这是 [[条件正态标准化]]的结论。

<!-- bilingual-en:start -->
**Slide 78.** If each conditional group is normal with positive scale, subtracting its conditional mean and dividing by its conditional standard deviation gives a standard normal law in every group. An [[条件分布不变推出独立|unchanged full conditional law]] implies independence. This is the conclusion of [[条件正态标准化|conditional normal standardisation]].
<!-- bilingual-en:end -->

如果只知道各组有均值和方差，标准化后得到的只是每组均值 0、方差 1；不同组仍可能有不同偏度、峰度或离散形状，因而不一定独立。这里起作用的是完整的正态分布假设，不是“减均值、除标准差”这两个操作单独保证独立。

<!-- bilingual-en:start -->
Without the conditional normal model, group standardisation fixes only the first two moments. Different groups may retain different skewness, kurtosis, or discrete shapes. Independence comes from the common full conditional distribution, not from the algebraic rescaling alone.
<!-- bilingual-en:end -->

### 9.5 协方差与相关系数压缩了多少信息

<!-- bilingual-en:start -->
*Covariance and correlation compress dependence into one number*
<!-- bilingual-en:end -->

有限二阶矩下，[[协方差]]是两个中心化变量乘积的期望：

<!-- bilingual-en:start -->
With finite second moments, [[协方差|covariance]] is the mean product of centred variables:
<!-- bilingual-en:end -->

$$\begin{aligned}\operatorname{Cov}(X,Y)&=E[(X-\mu_X)(Y-\mu_Y)]\\&=E[XY]-\mu_XE[Y]-\mu_YE[X]+\mu_X\mu_Y\\&=E[XY]-E[X]E[Y].\end{aligned}$$

它为正时，同向偏离均值的乘积在加权平均中占优势；为负时，反向偏离占优势。[[相关系数]]用两个标准差标准化：$\rho=\operatorname{Cov}(X,Y)/(\sigma_X\sigma_Y)$。当两个方差有限且严格为正，$\rho\in[-1,1]$，且没有单位；若某个变量是常数，协方差仍可为 0，但 Pearson 相关系数因分母为 0 而未定义。

<!-- bilingual-en:start -->
Positive covariance means same-direction deviations dominate the weighted product; negative covariance means opposite-direction deviations dominate. [[相关系数|Pearson correlation]] divides by both standard deviations to remove units. It lies between minus one and one when both variances are finite and positive; a constant variable makes the correlation undefined.
<!-- bilingual-en:end -->

不相关指协方差为 0。这只说一个中心化乘积的平均为 0，不能说一个变量对另一个“没有任何关系”。在线性预测的意义下，协方差为 0 会给出零总体斜率；非线性关系仍可能很强，见 [[零协方差不推独立]]。

<!-- bilingual-en:start -->
Uncorrelatedness sets one expected centred product to zero. It does not remove every relationship. It yields zero slope in a population linear projection while allowing strong nonlinear dependence; see [[零协方差不推独立|zero covariance does not imply independence]].
<!-- bilingual-en:end -->

### 9.6 三层蕴含及其证明

<!-- bilingual-en:start -->
*The hierarchy and why the forward implications hold*
<!-- bilingual-en:end -->

在相关矩存在时，[[独立性强弱关系]]是：独立 $\Rightarrow$ 均值独立 $\Rightarrow$ 不相关。第一步因为条件分布都不变，按它计算的均值当然不变。第二步可以逐行写成

<!-- bilingual-en:start -->
Under the required moment conditions, [[独立性强弱关系|the hierarchy]] runs from independence to mean independence to zero covariance. An unchanged full distribution has an unchanged mean. The second implication follows explicitly from total expectation:
<!-- bilingual-en:end -->

$$\begin{aligned}E[XY]&=E[E[XY\mid X]]\\&=E[X E[Y\mid X]]\\&=E[X E[Y]]=E[X]E[Y],\end{aligned}\qquad\therefore\operatorname{Cov}(X,Y)=0.$$

均值独立只需相应一阶矩；这里为保证乘积和协方差都可用，可以统一假设 $X,Y$ 有有限二阶矩。以下两个反例分别否定两条反向箭头。

<!-- bilingual-en:start -->
Mean independence itself requires only the relevant first moment. Finite second moments for both variables are a convenient sufficient condition for the product and covariance calculations here. The following examples disprove the two reverse implications separately.
<!-- bilingual-en:end -->

### 9.7 反例一：均值独立，但并不独立（slide 83）

<!-- bilingual-en:start -->
*Counterexample one: mean independence without independence*
<!-- bilingual-en:end -->

设 $X\sim\operatorname{Bernoulli}(1/2)$，独立的 $U$ 以各 $1/2$ 概率取 $-1,+1$，并定义 $Y=XU$。$X=0$ 时一定有 $Y=0$；$X=1$ 时 $Y=U$，于是非零联合概率只有三项：

<!-- bilingual-en:start -->
Let X be Bernoulli with parameter one half and let independent U take minus and plus one with equal probabilities. Set Y=XU. Only three joint outcomes have positive probability:
<!-- bilingual-en:end -->

| $(X,Y)$ | $(0,0)$ | $(1,-1)$ | $(1,1)$ |
|---|---:|---:|---:|
| 联合概率 / Joint probability | $1/2$ | $1/4$ | $1/4$ |

两组的均值为 $E[Y\mid X=0]=0$、$E[Y\mid X=1]=(-1)(1/2)+1(1/2)=0$，总体均值也是 0。因此 $Y$ 对 $X$ 均值独立。可是一旦知道 $X=0$，$Y=0$ 的概率变成 1；不加条件时这个概率只有 $1/2$，所以并不独立。两组条件方差分别是 0 和 1：均值没变，分布宽度变了。这就是 [[均值独立不推独立]]。

<!-- bilingual-en:start -->
Both conditional means are zero, as is the unconditional mean, so Y is mean independent of X. Yet observing X=0 changes the probability of Y=0 from one half to one. The conditional variances are zero and one. Thus [[均值独立不推独立|mean independence does not imply independence]].
<!-- bilingual-en:end -->

方向也不能反过来：$E[X]=1/2$，但 $E[X\mid Y=0]=0$，$E[X\mid Y=1]=E[X\mid Y=-1]=1$。所以 $X$ 不对 $Y$ 均值独立。同一个反例同时说明“均值独立”为什么必须说清谁对谁。

<!-- bilingual-en:start -->
The reverse direction fails: the mean of X is one half, but its conditional mean is zero when Y=0 and one when Y is nonzero. X is therefore not mean independent of Y.
<!-- bilingual-en:end -->

### 9.8 反例二：不相关，但条件均值明显变化（slide 84）

<!-- bilingual-en:start -->
*Counterexample two: zero covariance with a changing conditional mean*
<!-- bilingual-en:end -->

设 $X$ 在 $-1,0,1$ 上等概率取值，$Y=X^2$。三个数值对为 $(-1,1),(0,0),(1,1)$，各有概率 $1/3$。完整计算为

<!-- bilingual-en:start -->
Let X be uniform on −1, zero, and one, and let Y=X². The three joint outcomes have equal probability:
<!-- bilingual-en:end -->

$$E[X]=\frac{-1+0+1}{3}=0,\qquad E[Y]=\frac{1+0+1}{3}=\frac23,\qquad E[XY]=E[X^3]=\frac{-1+0+1}{3}=0.$$

因此 $\operatorname{Cov}(X,Y)=0-0(2/3)=0$。但 $E[Y\mid X=x]=x^2$，特别是 $E[Y\mid X=0]=0\ne2/3$，故 $Y$ 不对 $X$ 均值独立。见 [[零协方差不推均值独立]]。左右两个非零点的乘积贡献正好抵消，不代表 $X$ 对预测 $Y$ 没用：知道 $X$ 后甚至完全知道 $Y$。

<!-- bilingual-en:start -->
The covariance is zero, but the conditional mean of Y equals x² and changes with x. Hence [[零协方差不推均值独立|zero covariance does not imply mean independence]]. Opposite-signed product contributions cancel even though X completely determines Y.
<!-- bilingual-en:end -->

这个例子还说明方向性：给定 $Y=1$，$X=-1$ 与 $X=1$ 仍各占一半，均值为 0；给定 $Y=0$，$X=0$，均值也为 0。因此反向的 $E[X\mid Y]=E[X]=0$ 反而成立。

<!-- bilingual-en:start -->
Direction matters here too: conditional on Y=1, the two possible X values average to zero; conditional on Y=0, X is zero. Thus X is mean independent of Y even though Y is not mean independent of X.
<!-- bilingual-en:end -->

### 9.9 和的方差为何多出一项（slide 85）

<!-- bilingual-en:start -->
*Why the variance of a sum includes a cross term*
<!-- bilingual-en:end -->

[[和的方差协方差项]]来自平方展开。令 $A_c=A-E[A]$、$B_c=B-E[B]$；先用期望线性性中心化，再用 $(a+b)^2=a^2+2ab+b^2$：

<!-- bilingual-en:start -->
The [[和的方差协方差项|covariance term in a sum’s variance]] comes directly from squaring the centred sum:
<!-- bilingual-en:end -->

$$\begin{aligned}\operatorname{Var}(A+B)&=E[(A_c+B_c)^2]\\&=E[A_c^2]+2E[A_cB_c]+E[B_c^2]\\&=\operatorname{Var}(A)+\operatorname{Var}(B)+2\operatorname{Cov}(A,B).\end{aligned}$$

两变量时，方差相加的准确条件是协方差为 0；独立只是一个更强的充分条件。刚才 $Y=X^2$ 的例子里 $E[X^2]=2/3$，所以 $\operatorname{Var}(X)=2/3$；$Y^2=Y$，所以 $\operatorname{Var}(Y)=2/3-(2/3)^2=2/9$。两者虽依赖，但协方差为 0，故 $\operatorname{Var}(X+Y)=2/3+2/9=8/9$。直接看 $X+Y$：它以 $2/3$ 概率取 0、以 $1/3$ 概率取 2，方差为 $4/3-(2/3)^2=8/9$。

<!-- bilingual-en:start -->
For two variables, zero covariance is exactly the condition for variance additivity; independence is stronger than necessary. In the squared-variable example, variances 2/3 and 2/9 add to 8/9 despite dependence. Directly, their sum is zero with probability two thirds and two with probability one third, confirming the same result.
<!-- bilingual-en:end -->

对于三个或更多变量，公式是 $\operatorname{Var}(\sum_iX_i)=\sum_i\operatorname{Var}(X_i)+2\sum_{i<j}\operatorname{Cov}(X_i,X_j)$。逐项协方差为 0 是充分条件；一般说“方差相加”的必要充分条件是这些协方差的总和为 0，因为正负项也可能抵消。

<!-- bilingual-en:start -->
For a larger finite sum, include every pairwise covariance. Having each covariance equal zero is sufficient. Equality with the sum of the variances requires only that their total be zero, because positive and negative terms may cancel.
<!-- bilingual-en:end -->

## 10. 为什么这些区别会影响 OLS（slide 86）

<!-- bilingual-en:start -->
*How the distinctions affect OLS*
<!-- bilingual-en:end -->

### 10.1 零斜率只说明线性拟合的特定性质

<!-- bilingual-en:start -->
*A zero slope describes a particular linear fit*
<!-- bilingual-en:end -->

含截距的一元 OLS 写为 $y_i=\beta_0+\beta_1x_i+\varepsilon_i$。$i$ 标记第几个观测；$\beta_0,\beta_1$ 是总体模型参数；带帽子的 $\hat\beta_0,\hat\beta_1$ 是用样本算出的估计量。[[一元 OLS 斜率]]在 $x$ 有样本变动时为

<!-- bilingual-en:start -->
A simple OLS model includes an intercept and a slope. The index labels observations; population parameters differ from their sample estimates, denoted by hats. The [[一元 OLS 斜率|sample slope]] is defined when the regressor varies in the sample:
<!-- bilingual-en:end -->

$$\hat\beta_1=\frac{\sum_{i=1}^n(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^n(x_i-\bar x)^2}=\frac{\widehat{\operatorname{Cov}}(x,y)}{\widehat{\operatorname{Var}}(x)},\qquad\hat\beta_0=\bar y-\hat\beta_1\bar x.$$

协方差和方差必须采用相同归一化因子（都除 $n$，或都除 $n-1$），这个共同因子才会抵消。若 $\hat\beta_1=0$，只推出这个样本的协方差为 0。在上面三个点各出现同样次数的 $y=x^2$ 样本中，拟合直线是 $\hat y=2/3$，斜率为 0，但曲线关系显然存在。若随机抽样得到的三点次数不平衡，样本斜率未必恰好为 0；不要把总体协方差等于 0 当作每个样本都精确为 0。

<!-- bilingual-en:start -->
The same normalising factor must be used in sample covariance and sample variance so that it cancels. A zero sample slope means zero sample covariance. A balanced sample from the three-point square relation has fitted line 2/3, despite its deterministic nonlinear relation. An unbalanced random sample need not have exactly zero slope even when population covariance is zero.
<!-- bilingual-en:end -->

更不能由零斜率断言“没有因果影响”，或由非零斜率断言“存在因果影响”。[[概率依赖与因果]]与 [[识别与估计]]仍然约束着这个解释；本讲新增的是看清概率关系的强弱，并没有取消上一讲的识别问题。

<!-- bilingual-en:start -->
Neither a zero nor a nonzero slope by itself establishes a causal conclusion. [[概率依赖与因果|Dependence versus causality]] and [[识别与估计|identification versus estimation]] continue to constrain interpretation.
<!-- bilingual-en:end -->

### 10.2 无偏：给定完整样本的 x，误差平均为零

<!-- bilingual-en:start -->
*Unbiasedness uses zero conditional mean given the full design*
<!-- bilingual-en:end -->

把模型代进斜率公式。由于 $\sum_i(x_i-\bar x)=0$，常数项和误差均值项都抵消，得到

<!-- bilingual-en:start -->
Substituting the model into the slope formula, the centred regressor sums to zero, eliminating the intercept and error-mean terms:
<!-- bilingual-en:end -->

$$\begin{aligned}\hat\beta_1&=\frac{\sum_i(x_i-\bar x)[\beta_1(x_i-\bar x)+(\varepsilon_i-\bar\varepsilon)]}{\sum_i(x_i-\bar x)^2}\\&=\beta_1+\frac{\sum_i(x_i-\bar x)\varepsilon_i}{\sum_i(x_i-\bar x)^2}.\end{aligned}$$

[[零条件均值无偏性]]使用 $E[\varepsilon_i\mid x_1,\ldots,x_n]=0$（每个 $i$ 都成立）。给定整个 $x$ 样本，分母与各个系数都成为固定数；在分母非零且相关期望存在的条件下，

<!-- bilingual-en:start -->
[[零条件均值无偏性|Zero conditional mean]] conditions on the full regressor sample for every observation. Once the full sample is fixed, the denominator and all weights are fixed numbers. With a nonzero denominator and the required expectations:
<!-- bilingual-en:end -->

$$E[\hat\beta_1\mid x_1,\ldots,x_n]=\beta_1+\frac{\sum_i(x_i-\bar x)E[\varepsilon_i\mid x_1,\ldots,x_n]}{\sum_i(x_i-\bar x)^2}=\beta_1.$$

若估计量可积，再用全期望得到 $E[\hat\beta_1]=\beta_1$。同样由 $\hat\beta_0=\bar y-\hat\beta_1\bar x$ 得到截距无偏。这个结论不要求误差正态，也不要求同方差；但把 $E[\varepsilon_i\mid x_i]=0$ 升级成对完整样本条件化，需要独立随机抽样等相应依据。

<!-- bilingual-en:start -->
If the estimator is integrable, total expectation gives unconditional unbiasedness. The intercept follows from its formula. Normality and homoskedasticity are unnecessary for this conclusion. Moving from conditioning on one observation’s regressor to the full sample requires justification, such as independent sampling.
<!-- bilingual-en:end -->

### 10.3 一致：总体正交，还要有稳定的大样本平均

<!-- bilingual-en:start -->
*Consistency requires orthogonality and stable sample moments*
<!-- bilingual-en:end -->

[[OLS一致性条件]]讨论的是 $n\to\infty$ 时的概率极限，区别于每个有限 $n$ 下的期望。作为本讲可操作的一组充分条件，设 $(X_i,\varepsilon_i)$ 独立同分布、二阶矩有限、$\operatorname{Var}(X)>0$。大数定律让样本均值与二阶样本矩收敛，前面的误差项于是满足

<!-- bilingual-en:start -->
[[OLS一致性条件|OLS consistency]] concerns a probability limit as sample size grows, rather than the expectation at each finite size. One convenient sufficient setup is iid observation-error pairs, finite second moments, and positive regressor variance. Laws of large numbers stabilise the sample moments:
<!-- bilingual-en:end -->

$$\hat\beta_1-\beta_1=\frac{n^{-1}\sum_iX_i\varepsilon_i-\bar X\bar\varepsilon}{n^{-1}\sum_iX_i^2-\bar X^2}\xrightarrow{p}\frac{E[X\varepsilon]-E[X]E[\varepsilon]}{E[X^2]-E[X]^2}=\frac{\operatorname{Cov}(X,\varepsilon)}{\operatorname{Var}(X)}.$$

所以当 $\operatorname{Cov}(X,\varepsilon)=0$ 时，斜率一致；若还要求截距一致，需要 $E[\varepsilon]=0$。此时总体正交也可以写成 $E[X\varepsilon]=0$。符号 $\xrightarrow{p}$ 读作依概率收敛，详细定义沿用 [[依概率收敛]]与 [[估计量一致性]]。

<!-- bilingual-en:start -->
Zero covariance makes the slope consistent; consistency of the intercept additionally requires zero mean error. With zero mean error, orthogonality can be written as zero expected product. The arrow denotes [[依概率收敛|convergence in probability]], the mode of convergence used in [[估计量一致性|estimator consistency]].
<!-- bilingual-en:end -->

slide 86 列的是 headline conditions，必须连同这些背景条件使用。只有零协方差，却没有样本矩的大数规律，或 $X$ 没有总体变动，都不能完成这个证明。反过来，有限样本中比值的期望通常不等于总体矩之比，因此这里的一致性推导不自动给出无偏性。对时间序列、聚类或重尾资料，要用匹配的数据条件替换这组 iid 假设。

<!-- bilingual-en:start -->
Slide 86 gives headline conditions that need these background assumptions. Orthogonality alone does not establish convergence or identify a slope. Also, a finite-sample ratio’s expectation is generally not a ratio of population moments, so this consistency argument does not establish unbiasedness. Dependent or heavy-tailed data require appropriate alternative conditions.
<!-- bilingual-en:end -->

## 11. 回看时沿着哪些问题检查理解

<!-- bilingual-en:start -->
*Questions for reading the argument back*
<!-- bilingual-en:end -->

这讲的主线可以用六个问题串回去。先尝试说明理由，再回到对应小节查证。它们是阅读检查，不代表任何人已经完成了练习。

<!-- bilingual-en:start -->
Six questions reconstruct the lecture’s argument. Explain the reason before returning to the relevant section. These are reading prompts, not evidence of completed practice.
<!-- bilingual-en:end -->

1. $\omega$、事件 $A$、事件族 $\mathcal F$、随机变量 $X$ 的类型分别是什么？为什么有限空间也可以采用比所有子集更小的 σ-代数？
2. 已知 $F_X$，怎样求 $P(a<X\le b)$ 和 $P(X=a)$？何时可以用普通密度积分？
3. 算 $E[g(X)]$ 时，哪一部分是取值，哪一部分是权重？为什么期望线性性不要求独立，而方差相加要看协方差？
4. 对 $Y=3-2X$，哪一步改变不等号方向？对 $Y=X^2$，什么时候要保留两个根？
5. 在六格表里，边际化、条件化、Bayes 反转各自改变了什么？为什么 $E[Y\mid X]$ 仍能有方差？
6. 两个反例各否定哪一条反向箭头？OLS 无偏与一致的推导中，分别在哪一步使用了相应条件？

<!-- bilingual-en:start -->
&nbsp;
**1.** Identify the types of an outcome, event, event collection, and random variable; explain a coarse sigma-algebra on a finite space.<br>
**2.** Recover interval and point probabilities from a CDF, and state when density integration is valid.<br>
**3.** Separate transformed values from probability weights in LOTUS; distinguish linearity from variance additivity.<br>
**4.** Locate the inequality reversal in a decreasing transformation and decide when squaring requires both roots.<br>
**5.** Distinguish marginalisation, conditioning, and Bayesian reversal in the table; explain the randomness of a conditional mean.<br>
**6.** Identify the reverse implication refuted by each counterexample and locate the assumptions used in the OLS proofs.
<!-- bilingual-en:end -->

**共享主题地图：** [[概率空间、条件概率与 Bayes 法则.canvas]] · [[随机变量、分布与矩.canvas]] · [[条件期望.canvas]] · [[OLS 线性回归.canvas]]

## 来源与核验

<!-- bilingual-en:start -->
*Sources and checks*
<!-- bilingual-en:end -->

课程主干来自 Tom Glinnan 的 [[Lecture 2 - Statistics I.pdf]]：slides 4–10 支持概率、随机变量与 CDF/PDF；12–24 支持矩、指示变量和分位数；26–35 支持均匀、正态与密度解释；38–42 支持变换；44–59 支持联合、条件与 Bayes；61–72 支持条件矩与两条分组定律；74–86 支持独立性、反例与 OLS 衔接。PDF 第 89–90 页用于核对积分记号和可选全方差证明。

<!-- bilingual-en:start -->
Tom Glinnan’s [[Lecture 2 - Statistics I.pdf|lecture slides]] provide the course sequence and notation: probability and distributions (4–10), moments and quantiles (12–24), density interpretation (26–35), transformations (38–42), joint and conditional distributions (44–59), conditional moments and decomposition laws (61–72), and independence and OLS (74–86). PDF pages 89–90 support integral notation and the optional variance proof.
<!-- bilingual-en:end -->

[[Joint, Marginal and Conditional Distributions - Summary.pdf|官方补充讲义]]第 2–3 页支持格子解释与六格数值原例；第 4–5 页支持全概率及违约信号例；第 6–8 页支持连续边际化、条件密度与三角形原例。六格表的条件矩、全方差运算与三角形的矩计算是基于原例展开的推导。两骰引入、保险混合分布、$3x^2$ 与 $2(1-x)$ 密度、仿射变换及正态标准化计算来自课堂辅助讲解，并已逐项复算。图中的概率模型按正文公式绘制；格子原图直接保留 slide 50。

<!-- bilingual-en:start -->
The [[Joint, Marginal and Conditional Distributions - Summary.pdf|official supplement]] supports the grid and six-cell example (2–3), total probability and the default signal (4–5), and the continuous triangular example (6–8). The moment calculations extend those exact examples. Classroom illustrations are recalculated from their stated models. Generated figures use the displayed formulas, and the original grid is retained from slide 50.
<!-- bilingual-en:end -->

[[概率积分变换]]的连续 CDF 条件另以 [Duke STA 611 Lecture 5，印刷第 9 页](https://www2.stat.duke.edu/courses/Fall19/sta611.01/Lecture/Lecture05.pdf#page=16)核对。正文将课堂简写补充为明确条件：有限空间也需要选择事件族；一般 PDF 要求绝对连续性；递减变换保留必要的左极限；均值与中位数的次序不是偏度定义；Cauchy 的相关矩不存在；OLS 的 headline conditions 需要可积性、识别与收敛条件配合。

<!-- bilingual-en:start -->
[Duke STA 611 Lecture 5, printed page 9](https://www2.stat.duke.edu/courses/Fall19/sta611.01/Lecture/Lecture05.pdf#page=16) verifies the continuous-CDF condition for the [[概率积分变换|probability integral transform]]. The text makes implicit assumptions explicit for measurable events, densities, decreasing transformations, skewness, Cauchy moments, and the finite- and large-sample OLS arguments.
<!-- bilingual-en:end -->

