---
aliases:
- A uniform Lp bound implies uniform integrability
student_os: knowledge-atom
atom_id: PROB-MG-023
atom_set: martingales-stopping
atom_type: theorem
status: source-checked
requires:
- '[[一致可积]]'
related:
- '[[L1有界不推一致可积]]'
- '[[Markov不等式]]'
part_of:
- '[[鞅与停时.canvas]]'
---

<!-- bilingual-en:start -->
# Uniform integrability from a higher moment
<!-- bilingual-en:end -->

高阶矩有界推出一致可积

<!-- bilingual-en:start -->
A higher absolute moment charges large values more heavily than the first moment does. If a whole family can afford only a fixed amount of this heavier charge, its large values cannot keep contributing a fixed amount to the first moment. That is how a higher-moment bound gives [[一致可积|uniform integrability]].
<!-- bilingual-en:end -->

高阶绝对矩对大值的惩罚比一阶矩更重。若整个族的这种高阶代价都有同一个上界，越来越大的数值便不能持续保留一份固定的一阶矩贡献。这就是高阶矩界推出[[一致可积]]的原因。

> [!important] One common exponent and one common bound · 同一个阶数和同一个界
> <!-- bilingual-en:start -->
> Let $p>1$ be fixed. If a family of real random variables satisfies $\sup_i E|X_i|^p\le C<\infty$, then it is uniformly integrable. The expectation is of the powered absolute value, not the power of an expectation. No independence is required.
> <!-- bilingual-en:end -->
>
> 固定一个$p>1$。若实随机变量族满足$\sup_i E|X_i|^p\le C<\infty$，则一致可积。这里先取绝对值的$p$次幂，再求期望；不能改成先求期望再乘方，也不要求变量相互独立。

<!-- bilingual-en:start -->
In the language of $L^p$ spaces, $\|X_i\|_p=(E|X_i|^p)^{1/p}$, so bounding these moments is equivalent to a uniform $L^p$ norm bound. Integrability of each member follows already from $|x|\le1+|x|^p$, giving $E|X_i|\le1+C$.
<!-- bilingual-en:end -->

用$L^p$空间的记号，$\|X_i\|_p=(E|X_i|^p)^{1/p}$，所以矩的一致有界等价于$L^p$范数的一致有界。每个成员也自动可积，因为$|x|\le1+|x|^p$给出$E|X_i|\le1+C$。

<!-- bilingual-en:start -->
## Converting the extra power into a tail bound
<!-- bilingual-en:end -->

## 为什么高一阶的矩能压住尾部

<!-- bilingual-en:start -->
Fix $K>0$. On the event $|X_i|>K$, the factor $|X_i|^{p-1}$ is at least $K^{p-1}$. Divide the $p$th power by this lower bound:
<!-- bilingual-en:end -->

固定$K>0$。在$|X_i|>K$的事件上，额外因子$|X_i|^{p-1}$至少为$K^{p-1}$，于是

$$|X_i|\mathbf1_{\{|X_i|>K\}}\le\frac{|X_i|^p}{K^{p-1}},\qquad
\sup_i E[|X_i|\mathbf1_{\{|X_i|>K\}}]\le\frac C{K^{p-1}}\longrightarrow0.
$$

<!-- bilingual-en:start -->
The right-hand side has no family index. Given $\varepsilon>0$ and $C>0$, any $K>(C/\varepsilon)^{1/(p-1)}$ works for every member. If $C=0$, all members vanish almost surely and the conclusion is immediate. Here “higher” means any real $p>1$, not necessarily an integer or a whole order above one.
<!-- bilingual-en:end -->

右边不含成员指标。若$C>0$，给定$\varepsilon>0$，取$K>(C/\varepsilon)^{1/(p-1)}$便能同时控制全部成员；$C=0$时所有成员几乎必然为零，结论直接成立。“高阶”可以是任意实数$p>1$，不要求整数，也不必恰好比1高一个整阶。

> [!example] A second-moment check · 二阶矩例子
> <!-- bilingual-en:start -->
> If $\sup_nE[X_n^2]\le9$, take $p=2$. The uniform tail expectation is at most $9/K$. A cutoff $K=900$ bounds it by 0.01; taking $K>900$ makes the bound strictly below 0.01.
> <!-- bilingual-en:end -->
>
> 若$\sup_nE[X_n^2]\le9$，取$p=2$，统一尾部期望不超过$9/K$。$K=900$给出不超过0.01的界；$K>900$使这个界严格小于0.01。

<!-- bilingual-en:start -->
## What this condition does and does not require
<!-- bilingual-en:end -->

条件的边界

<!-- bilingual-en:start -->
At $p=1$, the estimate becomes $C$, which does not decrease with $K$. The [[L1有界不推一致可积|rare-spike counterexample]] shows that this is a real obstruction, not just a weak proof. Conversely, the higher-moment condition is sufficient, not necessary: a family consisting of a single integrable heavy-tailed variable is uniformly integrable even when all its moments of order greater than one are infinite.
<!-- bilingual-en:end -->

$p=1$时，这个界变成不随$K$下降的$C$；[[L1有界不推一致可积|稀有尖峰反例]]说明这确实是结论的障碍，而不仅是证明不够巧。反过来，高阶矩条件只是充分条件：只含一个可积重尾变量的族仍一致可积，即使它所有大于1阶的矩都无穷。

<!-- bilingual-en:start -->
For an explicit example of the last point, let $Y$ take $e^k$ with probability $c e^{-k}/k^2$, $k=1,2,\ldots$, where $c$ normalizes the probabilities. Then $EY=c\sum k^{-2}<\infty$, while $EY^p=c\sum e^{(p-1)k}/k^2=\infty$ for every $p>1$. The first series converges by the [[p级数判别|p-series test]] with exponent two. For any $p>1$, the terms of the second series eventually grow rather than tend to zero. The one-member family $\{Y\}$ is UI because integrability makes its own tail expectation vanish.
<!-- bilingual-en:end -->

具体地，令$Y$以概率$c e^{-k}/k^2$取$e^k$，$k=1,2,\ldots$，其中$c$使概率和为1。于是$EY=c\sum k^{-2}<\infty$，而任意$p>1$都有$EY^p=c\sum e^{(p-1)k}/k^2=\infty$。第一个级数由指数为2的[[p级数判别]]收敛；任意$p>1$时，第二个级数的项最终会增长而不趋零。单成员族$\{Y\}$仍一致可积，因为可积性保证它自己的尾部期望趋零。

## 来源与核验

- [MIT 18.445, Lecture 18, slide 3](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/e131f4c516e7f67ae27487d6a80f41d0_MIT18_445S15_lecture18.pdf#page=3)：固定$p>1$的一致$L^p$界是UI充分条件。
- [Durrett, Theorem 4.6.2, printed p.245](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=253)：超线性矩控制的一般形式；此处取幂函数并展开尾部估计。单成员重尾例子由正文给出的概率与级数直接核算。
