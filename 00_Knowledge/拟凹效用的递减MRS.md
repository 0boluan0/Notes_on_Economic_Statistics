---
student_os: knowledge-atom
atom_id: OPT-TOOLS-034
aliases:
- Diminishing marginal rate of substitution
- Diminishing MRS
status: source-checked
requires:
- '[[效用函数]]'
- '[[无差异曲线]]'
- '[[拟凹函数]]'
- '[[上轮廓集]]'
- '[[边际替代率]]'
- '[[导数判凹凸]]'
---

<!-- bilingual-en:start -->
# Diminishing MRS under quasiconcave utility
<!-- bilingual-en:end -->

*拟凹效用的递减 MRS*

<!-- bilingual-en:start -->
As a consumer moves to more of good 1 and less of good 2 **along one indifference curve**, convex preference regions can make the willingness to give up good 2 fall. The precise conclusion is that the [[边际替代率|marginal rate of substitution]] is **nonincreasing**: it may stay constant. Quasiconcavity supplies the shape of the preferred region; regularity and positive marginal utilities let us express that shape through a derivative ratio.
<!-- bilingual-en:end -->

当消费者**沿同一条无差异曲线**增加商品1、减少商品2时，凸的偏好区域可以使其放弃商品2的意愿下降。准确结论是[[边际替代率]]**非增**，允许保持常数。拟凹性提供较好区域的形状，正则性和正边际效用才使我们能够用导数比率表达这种形状。

<!-- bilingual-en:start -->
## Specify the curve and the region above it
<!-- bilingual-en:end -->

*明确曲线以及它上方的区域*

<!-- bilingual-en:start -->
Take nonempty open intervals $I,J$ of quantities, and the rectangular consumption region $R=I\times J$. Let $u:R\to\mathbb R$ have continuous first [[偏导数|partial derivatives]], with $u_1,u_2>0$ throughout $R$. Fix a utility level $k$ and suppose its [[无差异曲线|indifference curve]] is a $C^1$ graph $x_2=\phi(x_1)$ for every $x_1\in I$, with $\phi(I)\subseteq J$. An indifference curve is the [[水平集|set of bundles having the same utility]]. The [[隐函数定理|implicit function theorem]] justifies such a graph locally when $u_2\ne0$; one point's theorem does not automatically provide a global graph over an arbitrary interval.
<!-- bilingual-en:end -->

取两个非空开区间 $I,J$ 作为数量范围，令矩形消费区域为 $R=I\times J$。设 $u:R\to\mathbb R$ 有连续一阶[[偏导数]]，且 $R$ 上处处 $u_1,u_2>0$。固定效用水平 $k$，假设相应[[无差异曲线]]在每个 $x_1\in I$ 都可写成 $C^1$ 图像 $x_2=\phi(x_1)$，其中 $\phi(I)\subseteq J$。无差异曲线是[[水平集|具有相同效用的组合集合]]。$u_2\ne0$ 时，[[隐函数定理]]能在局部保证这种表示；某一点的定理结论并不自动给出任意区间上的全局图像。

<!-- bilingual-en:start -->
For fixed $x_1$, positivity of $u_2$ makes utility strictly increase with $x_2$, by the [[拉格朗日中值定理|mean value theorem]]. Hence the [[上轮廓集|upper contour region]] at this level is precisely the part above the graph, including the graph itself:
<!-- bilingual-en:end -->

固定 $x_1$ 后，$u_2>0$ 通过[[拉格朗日中值定理]]使效用随 $x_2$ 严格增加。因此，这一水平的[[上轮廓集]]恰好是图像上方的部分，包括图像本身：

$$
\begin{aligned}
C_k&=\{(x_1,x_2)\in R:u(x_1,x_2)\ge k\}\\
&=\{(x_1,x_2)\in R:x_2\ge\phi(x_1)\}.\end{aligned}
$$

<!-- bilingual-en:start -->
Assume $C_k$ is [[凸集|convex]]. [[拟凹函数|Quasiconcavity]] of $u$ on the convex region $R$ guarantees this for every level; for the conclusion on this particular curve, convexity of this one upper region suffices.
<!-- bilingual-en:end -->

假设 $C_k$ 是[[凸集]]。$u$ 在凸域 $R$ 上[[拟凹函数|拟凹]]，就保证每个水平都满足这一条件；若只研究这条曲线，当前这个上方区域为凸便已足够。

<!-- bilingual-en:start -->
## From convex regions to ordered slopes
<!-- bilingual-en:end -->

*从凸区域到斜率有序*

<!-- bilingual-en:start -->
Choose $a,b\in I$ and $0\le t\le1$. The two graph points $(a,\phi(a))$ and $(b,\phi(b))$ lie in $C_k$. Their convex combination remains in $C_k$ and in $R$. The “above the graph” description therefore gives
<!-- bilingual-en:end -->

任取 $a,b\in I$ 和 $0\le t\le1$。两点 $(a,\phi(a))$、$(b,\phi(b))$ 都在 $C_k$ 中，它们的凸组合仍在 $C_k$ 和 $R$ 中。利用“位于图像上方”的描述，便得到

$$
\phi(ta+(1-t)b)\le t\phi(a)+(1-t)\phi(b).
$$

<!-- bilingual-en:start -->
This is convexity of the one-variable function $\phi$. For $a<z<b$, rearrange this chord inequality at $z$ to obtain the secant bounds below, then let $z$ approach $a$ from the right in the first inequality and $b$ from the left in the second.
<!-- bilingual-en:end -->

这正是一元函数 $\phi$ 的凸性。对 $a<z<b$，把 $z$ 处的弦不等式整理成下面的割线界；随后在第一个不等式令 $z$ 从右侧趋于 $a$，在第二个令 $z$ 从左侧趋于 $b$。

$$
\begin{gathered}
\frac{\phi(z)-\phi(a)}{z-a}
\le\frac{\phi(b)-\phi(a)}{b-a}
\le\frac{\phi(b)-\phi(z)}{b-z},\\[4pt]
\phi'(a)\le\frac{\phi(b)-\phi(a)}{b-a}\le\phi'(b).
\end{gathered}
$$

<!-- bilingual-en:start -->
Thus $\phi'$ is nondecreasing; this is the relevant [[导数判凹凸|derivative criterion for convexity]], with no second derivative required. Meanwhile the implicit derivative formula gives $\phi'=-u_1/u_2<0$. Negating the ordered derivatives reverses their inequality:
<!-- bilingual-en:end -->

因此 $\phi'$ 非减；这就是所需的[[导数判凹凸|凸性导数判据]]，不需要二阶导数。同时隐式求导给出 $\phi'=-u_1/u_2<0$。给有序导数加负号会反转不等号：

$$
MRS_{12}(a,\phi(a))=-\phi'(a)
\ge-\phi'(b)=MRS_{12}(b,\phi(b)),\qquad a<b.
$$

<!-- bilingual-en:start -->
On a regular decreasing graph segment, convexity of the region above the indifference curve implies convexity of its boundary function. Its derivative is nondecreasing, so the [[边际替代率|MRS]], the negative of that derivative, is nonincreasing. This needs the stated smoothness, sign, and local graph assumptions; [[拟凹函数|quasiconcavity]] alone permits corners and does not ensure strict decrease.
<!-- bilingual-en:end -->

设二商品效用可微、$u_1,u_2>0$，在所考察区间的正规无差异曲线写成 $x_2=\phi(x_1)$，且相应上轮廓集在该区域是其上方。若上轮廓集凸，则边界函数 $\phi$ 凸，故其导数非减。由[[边际替代率]]，$MRS_{12}=-\phi'$，因此随 $x_1$ 增加而非增。

这个推理要求同一条无差异曲线、正边际效用和光滑图像表示；[[拟凹函数|拟凹性]]本身允许折点，不能无条件保证处处存在该比率。它也不把弱递减升级为严格递减。

<!-- bilingual-en:start -->
## A falling rate and a constant rate
<!-- bilingual-en:end -->

*下降速率与常数速率*

<!-- bilingual-en:start -->
For $u=x_1x_2$ on the positive quadrant, a fixed level $k>0$ has $\phi(x_1)=k/x_1$ and $MRS_{12}=k/x_1^2$. The derivative of this rate is $-2k/x_1^3<0$, so it strictly falls. At utility $12$, the bundles $(2,6)$ and $(3,4)$ have MRS values $3$ and $4/3$. For $u=x_1+x_2$, however, $\phi(x_1)=k-x_1$ on $0<x_1<k$ and MRS is constantly $1$. The theorem includes this weakly diminishing case.
<!-- bilingual-en:end -->

对正象限上的 $u=x_1x_2$，固定水平 $k>0$ 给出 $\phi(x_1)=k/x_1$、$MRS_{12}=k/x_1^2$。该速率的导数是 $-2k/x_1^3<0$，所以严格下降。在效用 $12$ 上，$(2,6)$ 与 $(3,4)$ 的 MRS 分别是 $3$、$4/3$。但对 $u=x_1+x_2$，在 $0<x_1<k$ 上有 $\phi(x_1)=k-x_1$，MRS 恒为 $1$；定理包含这种弱递减情况。

> [!warning] Hold the level and the hypotheses fixed · 固定水平并核对条件
> <!-- bilingual-en:start -->
> For the same quasiconcave utility $u=x_1x_2$, moving from $(1,1)$ to $(2,8)$ increases good 1 but raises MRS from $1$ to $4$: the utility levels differ. Positive marginal utilities alone also do not suffice. For $u=x_1^2+x_2^2$ on the positive quadrant, the level $4$ has slope $-x_1/x_2$ and MRS rises along its quarter-circle. Its upper region is not convex: $(1,\sqrt3)$ and $(\sqrt3,1)$ have utility $4$, but their midpoint has utility $2+\sqrt3<4$.
> <!-- bilingual-en:end -->
>
> 即使同样是拟凹效用 $u=x_1x_2$，从 $(1,1)$ 移到 $(2,8)$，商品1增加，MRS 却从 $1$ 升到 $4$，因为效用水平不同。仅有正边际效用也不够：对正象限上的 $u=x_1^2+x_2^2$，水平 $4$ 的斜率为 $-x_1/x_2$，MRS 沿其四分之一圆弧上升。该上方区域不凸：$(1,\sqrt3)$、$(\sqrt3,1)$ 的效用都是 $4$，中点效用却为 $2+\sqrt3<4$。

<!-- bilingual-en:start -->
Finally, $u=\min\{x_1,x_2\}$ is quasiconcave but has a kink on $x_1=x_2>0$. The set-based mixture conclusion remains meaningful there, while the smooth ratio proof does not apply at the kink. The conclusion concerns the selected regular curve segment; boundary points of a consumption domain need their own one-sided interpretation.
<!-- bilingual-en:end -->

最后，$u=\min\{x_1,x_2\}$ 虽然拟凹，却在 $x_1=x_2>0$ 有折点。基于集合的混合结论在那里仍有意义，但光滑比率证明不适用于折点。结论针对所选的正规曲线段；消费域边界点需要另行作单侧解释。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [CORE Econ, *The Economy 1.0*, Leibniz 3.2.1, “Convex preferences” and Cobb–Douglas example](https://books.core-econ.org/the-economy-v1/book/text/leibniz-03-02-01.html): movements on a fixed indifference curve and the positive-marginal-utility ratio. Weak convexity only gives nonincrease; strict claims require additional conditions.
- [Boyd and Vandenberghe, *Convex Optimization*, §§3.1.3, 3.1.7, 3.4.1, PDF pp.83–84, 89, 109–111](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=83): convex-function derivative bounds, convex epigraphs and convex upper contour sets. The local-rectangle version and full secant argument are proved above.
- [MIT 18.022, *The implicit function theorem*, PDF pp.1–2](https://math.mit.edu/~larsh/teaching/F2007/handouts/implicit.pdf): the local regular graph and derivative calculation; global continuation is not asserted.
<!-- bilingual-en:end -->

核验范围：CORE Econ Leibniz 3.2.1 的 Convex preferences 与 Cobb–Douglas 例支持固定无差异曲线上的移动及正边际效用比率；弱凸性只给非增，严格说法另需条件。Boyd 指定页支持凸函数导数界、凸上方图形及凸上轮廓集；局部矩形版本与完整割线论证在正文给出。MIT 隐函数讲义PDF1–2页支持局部正规图像和导数计算，不声称全局延拓。

<!-- bilingual-en:start -->
- The slide gives the geometric interpretation; the text supplies positive marginal utilities, a regular graph, and a convex upper region before deriving nonincreasing MRS.
- The linked course exposition retains the detailed reasoning, assumptions, examples, and context.
<!-- bilingual-en:end -->

- [[EC400 Slides Lecture 1.pdf#page=59|EC400 SOFP 原材料，PDF p.59]]：课件提供递减 MRS 的几何解释；正文补明正边际效用、正规曲线和凸上方区域条件，再由凸边界的非减导数推导。
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#8.4 经济含义：喜欢混合究竟有多强|SOFP Lecture 1 对应讲解]]：保留完整推导、条件、例子及课程语境。
