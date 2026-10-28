---
aliases:
  - Bernoulli distribution
  - 伯努利分布
student_os: knowledge-atom
atom_id: PROB-RV-016
atom_set: random-variables-distributions-moments
atom_type: definition
status: source-checked
requires:
  - "[[概率质量函数]]"
related:
  - "[[事件指示变量]]"
leads_to:
  - "[[二项分布]]"
  - "[[Bernoulli和的二项分布]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

<!-- bilingual-en:start -->
# Bernoulli distribution
<!-- bilingual-en:end -->
*伯努利分布*

<!-- bilingual-en:start -->
A **Bernoulli distribution** describes one yes-or-no outcome recorded as a number: $1$ for yes and $0$ for no. Its parameter $p$ tells us the probability of recording $1$. Once we know $p$, we also know the probability of $0$, because these are the only two possibilities.
<!-- bilingual-en:end -->

**Bernoulli 分布（伯努利分布）**描述一个“是或否”的结果：把“是”记成 $1$，“否”记成 $0$。参数 $p$ 告诉我们取值为 $1$ 的概率。因为只有这两种可能，知道了 $p$，也就知道了取值为 $0$ 的概率。

> [!example] Will it rain? · 会不会下雨？
> <!-- bilingual-en:start -->
> In a toy model, choose a day and record $X=1$ if it rains that day and $X=0$ otherwise. Assume the probability of rain is $0.4$. Then $X$ has a Bernoulli distribution with $p=0.4$: it is $1$ with probability $0.4$ and $0$ with probability $0.6$. The [[随机变量|random variable]] $X$ is the numerical recording rule; the distribution tells us how likely its possible values are.
> <!-- bilingual-en:end -->
>
> 在一个示意模型里，选取一天：当天下雨就记 $X=1$，否则记 $X=0$。假设下雨概率是 $0.4$，那么 $X$ 就服从参数 $p=0.4$ 的 Bernoulli 分布：以 $0.4$ 的概率取 $1$，以 $0.6$ 的概率取 $0$。[[随机变量]] $X$ 规定如何记录结果，分布则说明各个可能数值有多大概率出现。

<!-- bilingual-en:start -->
## Writing the two probabilities
<!-- bilingual-en:end -->
*把两个概率写下来*

<!-- bilingual-en:start -->
The notation $P(X=1)$ means “the probability that $X$ equals $1$”. The symbol $\sim$ means “has the distribution”, so $X\sim\operatorname{Bernoulli}(p)$ is a short way of specifying the following [[概率质量函数|probability mass function]]—the probability attached to each possible value.
<!-- bilingual-en:end -->

记号 $P(X=1)$ 表示“$X$ 等于 $1$ 的概率”，$\sim$ 表示“服从这个分布”。因此，$X\sim\operatorname{Bernoulli}(p)$ 是下面这份[[概率质量函数]]的简写，也就是为每个可能取值规定概率。

> [!abstract] A Bernoulli model · Bernoulli 模型
> <!-- bilingual-en:start -->
> For $0\le p\le1$,
> <!-- bilingual-en:end -->
>
> 对于 $0\le p\le1$，
>
> $$P(X=1)=p,\qquad P(X=0)=1-p.$$
>
> <!-- bilingual-en:start -->
> All other values have probability zero. The two displayed probabilities add to one. At $p=0$ the outcome is always $0$; at $p=1$ it is always $1$. These endpoints are included in the definition.
> <!-- bilingual-en:end -->
>
> 其他数值的概率都是零，列出的两个概率相加为一。当 $p=0$ 时结果总为 $0$，当 $p=1$ 时结果总为 $1$；定义也包括这两个端点。

<!-- bilingual-en:start -->
The outcome labelled $1$ is often called “success”, but the label only tells us which event we are recording. It could represent rain, a machine failure, or a correct answer. For any specified event $A$, its [[事件指示变量|indicator]] $I_A$ is $1$ when $A$ occurs and $0$ otherwise, so $I_A\sim\operatorname{Bernoulli}(P(A))$.
<!-- bilingual-en:end -->

取值 $1$ 常被叫作“成功”，但这个名称只标明我们记录的是哪个事件，并不表示结果好坏；它可以代表下雨、机器故障或答对题目。对一个已经明确的事件 $A$，它的[[事件指示变量]] $I_A$ 在事件发生时为 $1$，否则为 $0$，因此 $I_A\sim\operatorname{Bernoulli}(P(A))$。

<!-- bilingual-en:start -->
## Why the mean equals $p$
<!-- bilingual-en:end -->
*为什么均值就是 $p$*

<!-- bilingual-en:start -->
The [[期望|expectation]] $E[X]$ is the probability-weighted average of the possible values. For a Bernoulli variable, it is $0(1-p)+1p=p$. In the rain example, the mean is $0.4$ even though one day's recorded value can only be $0$ or $1$.
<!-- bilingual-en:end -->

[[期望]] $E[X]$ 是把各个可能取值按概率加权后的平均数。对 Bernoulli 变量，这个计算就是 $0(1-p)+1p=p$。所以雨天例子的均值为 $0.4$，尽管某一天实际记录的数值只能是 $0$ 或 $1$。

<!-- bilingual-en:start -->
The [[方差|variance]] measures the average squared distance from that mean. Since squaring either $0$ or $1$ leaves it unchanged, $X^2=X$ and $E[X^2]=p$. Using $\operatorname{Var}(X)=E[X^2]-E[X]^2$ gives $p-p^2$:
<!-- bilingual-en:end -->

[[方差]]衡量取值偏离均值的平方平均有多大。因为 $0$ 和 $1$ 平方后都不变，所以 $X^2=X$，并且 $E[X^2]=p$。代入 $\operatorname{Var}(X)=E[X^2]-E[X]^2$，得到 $p-p^2$：

$$E[X]=p,\qquad \operatorname{Var}(X)=p(1-p).$$

> [!warning] Several observations need more information · 多个观测还需要说明它们的关系
> <!-- bilingual-en:start -->
> Saying that each observation is Bernoulli does not tell us whether observations are independent or whether they share the same $p$. For example, copying one rainy-day indicator creates two identical Bernoulli variables, with the second completely determined by the first. The additional requirements are explained in [[独立同分布|independent and identically distributed variables]]. Counting the number of ones across independent trials with a common $p$ leads to a [[Bernoulli和的二项分布|binomial distribution]].
> <!-- bilingual-en:end -->
>
> 说每个观测都服从 Bernoulli 分布，还没有说明它们是否独立，也没有说明它们的 $p$ 是否相同。例如，把同一天的下雨记录抄一遍，会得到两个相同的 Bernoulli 变量，第二个完全由第一个决定。这些额外要求见[[独立同分布]]；如果多个试验独立且有共同的 $p$，其中 $1$ 的总个数服从[[Bernoulli和的二项分布|二项分布]]。

> [!question]- Read the parameter · 读懂参数
> <!-- bilingual-en:start -->
> If $X\sim\operatorname{Bernoulli}(0.2)$, what is the probability that $X=0$?
>
> **Answer:** $1-0.2=0.8$. The parameter gives the probability of $1$.
> <!-- bilingual-en:end -->
>
> 若 $X\sim\operatorname{Bernoulli}(0.2)$，$X=0$ 的概率是多少？
>
> **答案：** $1-0.2=0.8$。参数给出的是取值为 $1$ 的概率。

## 来源与核验

<!-- bilingual-en:start -->
- [MIT 18.05 Probability Reading, Class 4 §§3.1–3.2 and Class 5 §§3.1, 3.3](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf): supports the Bernoulli probabilities, success/failure coding, mean and variance formulas, and the binomial count of independent trials with a common $p$. The rain scenario is an illustrative application of this definition.
<!-- bilingual-en:end -->

- [MIT 18.05 Probability Reading，Class 4 §§3.1–3.2 与 Class 5 §§3.1、3.3](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf)：支持 Bernoulli 的两个概率、成功与失败的编码、均值和方差公式，以及具有共同 $p$ 的独立试验所产生的二项计数。雨天情境是依据定义构造的示意例子。
