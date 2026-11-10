---
student_os: knowledge-atom
atom_id: PROB-CONC-013
aliases:
  - Conditional Jensen inequality
  - Conditional Jensen's inequality
  - 条件 Jensen 不等式
atom_type: theorem
status: source-checked
---

<!-- bilingual-en:start -->
# Conditional Jensen's inequality
<!-- bilingual-en:end -->

*条件 Jensen 不等式*

<!-- bilingual-en:start -->
Suppose an average is allowed to use information, such as which bus route a passenger takes. **Conditional Jensen's inequality** compares two predictions based on that same information: transform the predicted value, or predict the transformed value. For a convex transform, the first prediction is no larger than the second, almost surely.
<!-- bilingual-en:end -->

假设计算平均值时可以利用信息，例如乘客搭乘哪条公交线路。**条件 Jensen 不等式**比较两个使用相同信息的预测：先预测数值再变换，或者直接预测变换后的数值。对于凸变换，第一个预测几乎必然不大于第二个。

<!-- bilingual-en:start -->
The [[条件期望|conditional expectation]] $E[Y\mid\mathcal G]$ uses only the information recorded by a [[σ-代数|sigma-algebra]] $\mathcal G$. With finitely many information groups, it is the mean within the group that occurred. Those group means can differ, so the conditional expectation is itself a random variable. [[期望的Jensen不等式|Ordinary Jensen's inequality]] compares two numbers; the conditional version compares two such information-based random variables.
<!-- bilingual-en:end -->

[[条件期望]] $E[Y\mid\mathcal G]$ 只使用[[σ-代数]] $\mathcal G$ 所记录的信息。如果信息把结果分成有限个组，它就是实际所属组内的平均。各组的均值可以不同，因此条件期望本身仍是随机变量。[[期望的Jensen不等式|普通 Jensen 不等式]]比较两个数；条件版本比较两个由信息决定的随机变量。

> [!example] A mean wait and a mean squared wait · 平均等待与平均平方等待
> <!-- bilingual-en:start -->
> Use the bus model from the conditional-expectation card. Let $R$ be the route and $Y$ the wait in minutes. The possible pairs $(R,Y)$ are $(0,0),(0,4),(1,4),(1,8)$, with probabilities $1/8,1/8,3/8,3/8$. Knowing $R$ leaves two equally likely waits within either route. For the convex function $\varphi(y)=y^2$, the comparisons are
> <!-- bilingual-en:end -->
>
> 沿用条件期望卡的公交模型。令 $R$ 表示线路，$Y$ 表示等待分钟数。组合 $(R,Y)$ 为 $(0,0),(0,4),(1,4),(1,8)$，对应概率 $1/8,1/8,3/8,3/8$。知道 $R$ 后，每条线路内都还剩两个等可能的等待时间。对凸函数 $\varphi(y)=y^2$，比较如下：
>
> | Route / 线路 | Possible waits / 可能等待 | $E[Y\mid R]$ | $(E[Y\mid R])^2$ | $E[Y^2\mid R]$ |
> |---|---|---:|---:|---:|
> | $R=0$ | $0,4$ | $2$ | $4$ | $8$ |
> | $R=1$ | $4,8$ | $6$ | $36$ | $40$ |
>
> <!-- bilingual-en:start -->
> In either information state, the square of the mean is at most the mean of the square: $4\le8$ and $36\le40$. These two comparisons together say $(E[Y\mid R])^2\le E[Y^2\mid R]$. We are not claiming that the actual squared wait always exceeds its route's squared mean: on route $0$, a realized wait of zero is possible.
> <!-- bilingual-en:end -->
>
> 无论处于哪个信息状态，均值的平方都不大于平方的均值：$4\le8$、$36\le40$。把两组比较合起来，就是 $(E[Y\mid R])^2\le E[Y^2\mid R]$。这并不声称每次实际等待的平方都大于该线路均值的平方：线路 $0$ 完全可能实际等零分钟。

<!-- bilingual-en:start -->
## The general statement and “almost surely”
<!-- bilingual-en:end -->

*一般表述与“几乎必然”的含义*

> [!important] Keep the information fixed · 比较时使用同一份信息
> <!-- bilingual-en:start -->
> On a [[概率空间|probability space]] $(\Omega,\mathcal F,P)$, let $Y$ be real and integrable, and let $\mathcal G\subseteq\mathcal F$ be a sigma-algebra of available information. Suppose $\varphi:\mathbb R\to\mathbb R$ is finite-valued and [[凹凸性|convex]], with $E|\varphi(Y)|<\infty$. Then
> <!-- bilingual-en:end -->
>
> $$
> \boxed{\varphi\!\left(E[Y\mid\mathcal G]\right)
> \le E[\varphi(Y)\mid\mathcal G]\quad\text{almost surely}.}
> $$
>
> 在[[概率空间]] $(\Omega,\mathcal F,P)$ 上，令实随机变量 $Y$ 可积，$\mathcal G\subseteq\mathcal F$ 是表示可用信息的 σ-代数。若 $\varphi:\mathbb R\to\mathbb R$ 是有限值[[凹凸性|凸函数]]，并且 $E|\varphi(Y)|<\infty$，就有上面的几乎必然不等式。

<!-- bilingual-en:start -->
“Almost surely” means that the set of outcomes violating the comparison has probability zero. Conditional expectations are only determined up to changes on a null set, so the theorem does not assign a prescribed value at every zero-probability information point. It needs no independence between $Y$ and the information. The left-hand side is also integrable under these hypotheses, as the proof below explains.
<!-- bilingual-en:end -->

“几乎必然”表示违反比较的结果集合概率为零。条件期望允许在零概率集合上更换版本，因此定理不会指定每个零概率信息点必须取什么值。这里不要求 $Y$ 与已知信息独立。在这些条件下，左边也可积，理由见下面的证明。

<!-- bilingual-en:start -->
With no information, the conditional means are ordinary expectations, recovering ordinary Jensen. If $\mathcal G$ already determines $Y$, both sides equal $\varphi(Y)$ almost surely. For a concave function, apply the result to its negative to reverse the inequality, retaining the same integrability and domain checks.
<!-- bilingual-en:end -->

如果没有信息，条件均值就是普通期望，因而恢复普通 Jensen。如果 $\mathcal G$ 已经确定 $Y$，两边几乎必然都等于 $\varphi(Y)$。对凹函数，可将结论用于它的相反数，让不等号反向，但仍须检查可积性与定义域。

> [!note]- Why the rule survives conditioning · 为什么条件化后仍然成立
> <!-- bilingual-en:start -->
> A finite convex function on $\mathbb R$ can be built as the supremum of a countable collection of affine lower bounds $\ell_k(x)=a_kx+b_k$. Each line stays below the curve, and together they approach it arbitrarily closely at every input. If the function is affine, one line suffices. Here $a_k,b_k$ are fixed real numbers, and $k=1,2,\ldots$ indexes the lines.
> <!-- bilingual-en:end -->
>
> 实数轴上的有限值凸函数可以写成可数条仿射下界 $\ell_k(x)=a_kx+b_k$ 的上确界。每条直线都位于曲线下方，合在一起能在每个输入处任意逼近曲线；函数本身若是仿射函数，一条直线就够。$a_k,b_k$ 是固定实数，$k=1,2,\ldots$ 用来编号。
>
> <!-- bilingual-en:start -->
> Put $m=E[Y\mid\mathcal G]$ and $H=E[\varphi(Y)\mid\mathcal G]$. Conditional expectation preserves order and is linear in fixed coefficients, so for each $k$,
> <!-- bilingual-en:end -->
>
> 记 $m=E[Y\mid\mathcal G]$、$H=E[\varphi(Y)\mid\mathcal G]$。条件期望保序，而且对固定系数满足线性性，所以对每个 $k$，
>
> $$H\ge E[a_kY+b_k\mid\mathcal G]=a_km+b_k\quad\text{almost surely}.$$
>
> <!-- bilingual-en:start -->
> There are only countably many lines, so the union of their probability-zero exceptional sets still has probability zero. Outside that union, all these inequalities hold together. Taking their supremum gives $H\ge\sup_k\ell_k(m)=\varphi(m)$. This countability step avoids incorrectly requiring every member of an uncountable family to share one null set.
> <!-- bilingual-en:end -->
>
> 因为只有可数条直线，它们各自的零概率异常集取并后仍是零概率集。在这个并集之外，所有不等式同时成立；取上确界得到 $H\ge\sup_k\ell_k(m)=\varphi(m)$。可数性保证了这一步，不能对不可数族未经论证就假定存在共同的零概率异常集。
>
> <!-- bilingual-en:start -->
> Finally, one lower line gives $\ell_1(m)\le\varphi(m)\le H$. Both outer quantities are integrable, hence $|\varphi(m)|\le|\ell_1(m)|+|H|$ is integrable too. No extra assumption about the transformed conditional mean was silently added.
> <!-- bilingual-en:end -->
>
> 最后，任取其中一条下界就有 $\ell_1(m)\le\varphi(m)\le H$。外侧两项都可积，因此 $|\varphi(m)|\le|\ell_1(m)|+|H|$ 也可积，并没有额外偷加“变换后的条件均值可积”这一假设。

<!-- bilingual-en:start -->
For example, $\varphi(y)=|y|$ gives $|E[Y\mid\mathcal G]|\le E[|Y|\mid\mathcal G]$. The square version requires $E[Y^2]<\infty$ and gives the second-moment bound used by [[条件期望投影|least-squares prediction]]. Applying the same comparison to future values of a martingale is the mechanism behind [[凸变换产生次鞅|convex transforms producing submartingales]].
<!-- bilingual-en:end -->

例如，取 $\varphi(y)=|y|$ 得到 $|E[Y\mid\mathcal G]|\le E[|Y|\mid\mathcal G]$。平方版本要求 $E[Y^2]<\infty$，得到[[条件期望投影|最小二乘预测]]使用的二阶矩界。把同样的比较用于鞅的未来值，就是[[凸变换产生次鞅]]的机制。

## 来源与核验

<!-- bilingual-en:start -->
- [Durrett, Theorems 4.1.9–4.1.11, printed pp. 210–212 / PDF pp. 218–220](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf#page=219): conditional order and linearity, conditional Jensen with both integrability assumptions, the countable affine-minorant proof and moment contraction. The sandwich argument explicitly supplies integrability of the transformed conditional mean.
- [MIT 18.445, Lecture 15, PDF pp. 6–8](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/fb97374ded39d0e4a0047564af28dc93_MIT18_445S15_lecture15.pdf#page=7): the information-based definition, almost-sure uniqueness, Jensen on page 7 and the conditional-expectation properties. The bus comparisons are calculated from the existing definition card's probability table.
<!-- bilingual-en:end -->

- Durrett 定理 4.1.9–4.1.11（印刷第 210–212 页／PDF 第 218–220 页）：核对条件期望的保序与线性性、两项可积性假设、条件 Jensen 的可数仿射下界证明及矩收缩。正文用上下夹住的办法明确说明变换后的条件均值可积。
- MIT 18.445 第 15 讲，PDF 第 6–8 页：核对信息定义、几乎必然唯一性、第 7 页的 Jensen 及条件期望性质。公交比较直接按已有定义卡的概率表计算。
