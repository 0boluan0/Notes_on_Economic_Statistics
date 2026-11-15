---
aliases:
  - KKT necessary conditions
  - KKT necessity
  - KKT必要条件
student_os: knowledge-atom
atom_id: OPT-MV-018
atom_set: multivariable-optimization
atom_type: necessary-condition
status: source-checked
requires:
  - "[[KKT条件]]"
  - "[[LICQ条件]]"
  - "[[局部极值与绝对极值]]"
related:
  - "[[Slater强对偶]]"
  - "[[LICQ不是KKT必要条件]]"
  - "[[KKT充分性]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

<!-- bilingual-en:start -->
# KKT necessity at a local optimum
<!-- bilingual-en:end -->

*KKT 的局部最优必要条件*

<!-- bilingual-en:start -->
At a constrained maximum, the objective can still point uphill: the constraints may block the improving moves. The [[KKT条件|KKT conditions]] express this balance using the objective gradient, constraint gradients, and coefficients called multipliers. **KKT necessity** says that a local optimum must admit such multipliers when the constraint description is sufficiently regular. It gives a condition that every qualifying optimum must pass, rather than a test that makes every passing point optimal.
<!-- bilingual-en:end -->

在有约束的最大点，目标函数仍可能朝某个方向上升，只是约束挡住了改善的移动。[[KKT条件]]用目标梯度、约束梯度，以及称为乘子的系数表达这种平衡。**KKT 必要性**说的是：约束表示足够规则时，局部最优点一定配得上这样的乘子。它给出合乎前提的最优点必须通过的条件；通过这些条件的点，仍需另外判断是否最优。

<!-- bilingual-en:start -->
## What the theorem assumes
<!-- bilingual-en:end -->

*定理从哪些前提出发*

<!-- bilingual-en:start -->
Let $x$ have $n\ge1$ real coordinates and lie in an [[开集|open set]] $U\subseteq\mathbb R^n$. We maximize a real-valued objective $f(x)$ subject to finitely many inequalities $g_i(x)\le0$, $i=1,\ldots,m$, and equalities $h_j(x)=0$, $j=1,\ldots,p$. Either count may be zero. Write their common feasible set as $F$. All restrictions must be included in $F$; the ambient domain $U$ is open so that ordinary derivatives can be taken around the candidate.
<!-- bilingual-en:end -->

令 $x$ 有 $n\ge1$ 个实坐标，位于[[开集]] $U\subseteq\mathbb R^n$ 中。我们最大化实值目标 $f(x)$，满足有限条不等式 $g_i(x)\le0$（$i=1,\ldots,m$）与等式 $h_j(x)=0$（$j=1,\ldots,p$）；任一类约束都可以没有。共同可行集记为 $F$，所有限制都要计入其中。外层定义域 $U$ 为开集，是为了能在候选点周围使用通常的导数。

<!-- bilingual-en:start -->
A [[局部极值与绝对极值|local maximizer relative to $F$]] is a feasible $x^*$ for which some $\varepsilon>0$ satisfies $f(x)\le f(x^*)$ whenever $x\in F$ and $\|x-x^*\|<\varepsilon$. Only nearby **feasible** inputs are compared; $x^*$ may lie on the boundary of $F$. Assume $f$, every $g_i$, and every $h_j$ are $C^1$ on an open neighborhood of $x^*$ contained in $U$: their first partial derivatives exist and are continuous there. Checking derivatives only at $x^*$ does not establish this hypothesis.
<!-- bilingual-en:end -->

相对于 $F$ 的[[局部极值与绝对极值|局部最大点]]是一个可行点 $x^*$：存在 $\varepsilon>0$，使所有满足 $\|x-x^*\|<\varepsilon$ 的 $x\in F$ 都有 $f(x)\le f(x^*)$。比较的只是附近的**可行**输入，因此 $x^*$ 可以在 $F$ 的边界上。假设 $f$、每个 $g_i$ 和每个 $h_j$ 在 $U$ 内某个包含 $x^*$ 的开邻域上为 $C^1$，即各一阶偏导在那里存在且连续。只检查 $x^*$ 这一点的导数，不能验证该前提。

<!-- bilingual-en:start -->
The active index set is $A(x^*)=\{i:g_i(x^*)=0\}$. A concrete sufficient qualification is [[LICQ条件|LICQ]]: the indexed family consisting of **all equality gradients and all active inequality gradients** is linearly independent at $x^*$. A [[全微分#实值函数、梯度与全微分|gradient]] is the column of first partial derivatives. Include every listed constraint in this test; a repeated or zero gradient cannot be discarded, and activity is determined by $g_i(x^*)=0$, before any multiplier has been found.
<!-- bilingual-en:end -->

活动约束指标集为 $A(x^*)=\{i:g_i(x^*)=0\}$。一种具体而充分的约束资格是 [[LICQ条件|LICQ]]：**全部等式梯度与全部活动不等式梯度**构成的带指标向量族，在 $x^*$ 处线性无关。[[全微分#实值函数、梯度与全微分|梯度]]是各一阶偏导排成的列。检验时必须保留每条列出的约束，不能丢弃重复或零梯度；活动性由 $g_i(x^*)=0$ 决定，无须先求乘子。

> [!theorem] A local necessity theorem · 局部最优的必要性定理
> <!-- bilingual-en:start -->
> For a differentiable problem, a local optimum satisfying an appropriate constraint qualification such as LICQ admits inequality and equality multipliers satisfying primal feasibility, dual feasibility, stationarity, and complementary slackness.
> <!-- bilingual-en:end -->
>
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad h(x)=0,\qquad g(x)\le0.
> $$
> 若 $f,g,h$ 在局部最优点 $x^*$ 附近为 $C^1$，且 $x^*$ 满足适当约束资格，例如 [[LICQ条件|LICQ]]，则存在 $\lambda^*\ge0$ 与不受符号限制的 $\nu^*$，使 [[KKT条件]] 的原问题可行性、对偶可行性、驻点和互补松弛同时成立。

<!-- bilingual-en:start -->
In the maximization convention used here, the [[KKT条件|Lagrangian]] is $\mathcal L(x,\lambda,\nu)=f(x)-\sum_i\lambda_i g_i(x)-\sum_j\nu_j h_j(x)$. The conclusion supplies $\lambda^*\in\mathbb R^m$, with every component nonnegative, and unrestricted $\nu^*\in\mathbb R^p$. Together with the already assumed feasibility, they satisfy the following equations. Stars identify this point and its coefficients; they do not mean that arbitrary solutions of these equations have already been proved optimal.
<!-- bilingual-en:end -->

按这里的最大化约定，[[KKT条件|Lagrangian]] 为 $\mathcal L(x,\lambda,\nu)=f(x)-\sum_i\lambda_i g_i(x)-\sum_j\nu_j h_j(x)$。结论提供各分量非负的 $\lambda^*\in\mathbb R^m$ 与符号不限的 $\nu^*\in\mathbb R^p$；连同已经假设的可行性，它们满足以下方程。星号标记这个点及其系数，并不表示方程的任意解都已经被证明最优。

$$
\begin{aligned}
g_i(x^*)&\le0\quad(i=1,\ldots,m),&h_j(x^*)&=0\quad(j=1,\ldots,p),\\
\lambda_i^*&\ge0\quad(i=1,\ldots,m),&\lambda_i^*g_i(x^*)&=0\quad(i=1,\ldots,m),\\
\nabla f(x^*)-\sum_{i=1}^{m}\lambda_i^*\nabla g_i(x^*)
-\sum_{j=1}^{p}\nu_j^*\nabla h_j(x^*)&=0.
\end{aligned}
$$

<!-- bilingual-en:start -->
This theorem needs no concavity, convexity, or global optimality. For minimization, apply it to the objective $-f$ and convert the stationarity signs consistently. LICQ and local optimality are separate hypotheses: a regular feasible point need not be optimal, and a local optimum need not have a regular constraint description.
<!-- bilingual-en:end -->

这个定理不要求凹性、凸性或全局最优。处理最小化时，可以对目标 $-f$ 应用它，再一致地转换驻点方程的符号。LICQ 与局部最优是不同的前提：规则的可行点不一定最优，局部最优点的约束表示也不一定规则。

> [!example] A boundary maximum with a nonzero derivative · 导数不为零的边界最大点
> <!-- bilingual-en:start -->
> Maximize $f(x)=x+x^2$ subject to $g_1(x)=x\le0$ and $g_2(x)=-x-2\le0$, so $F=[-2,0]$. At $x^*=0$, every feasible $x$ with $-1<x<0$ gives $f(x)=x(1+x)<0=f(0)$; hence $0$ is a strict local maximizer relative to $F$. Only $g_1$ is active, and $g_1'(0)=1$, so LICQ holds. The coefficients $\lambda_1^*=1$, $\lambda_2^*=0$ give stationarity $1-1\cdot1-0\cdot(-1)=0$. Both are nonnegative, and both complementary products are zero. The nonzero objective derivative is balanced by the upper-bound constraint. Yet $f(-2)=2>f(0)=0$: this local optimum is not global.
> <!-- bilingual-en:end -->
>
> 最大化 $f(x)=x+x^2$，约束为 $g_1(x)=x\le0$ 与 $g_2(x)=-x-2\le0$，所以 $F=[-2,0]$。在 $x^*=0$ 处，所有满足 $-1<x<0$ 的可行点都有 $f(x)=x(1+x)<0=f(0)$，故 $0$ 是相对于 $F$ 的严格局部最大点。只有 $g_1$ 活动，而 $g_1'(0)=1$，所以 LICQ 成立。取 $\lambda_1^*=1$、$\lambda_2^*=0$，驻点式为 $1-1\cdot1-0\cdot(-1)=0$；两乘子都非负，两个互补乘积都为零。非零的目标导数由上界约束配平。不过 $f(-2)=2>f(0)=0$，这个局部最优并非全局最优。

<!-- bilingual-en:start -->
## Why the theorem needs a constraint qualification
<!-- bilingual-en:end -->

## 约束资格为什么出现在定理里

<!-- bilingual-en:start -->
A constraint qualification ensures that active constraint normals adequately describe the local feasible geometry. With the degenerate constraint $x^2\le0$, zero is the only feasible point but the active gradient vanishes, so ordinary KKT stationarity cannot represent the nonzero objective derivative.
<!-- bilingual-en:end -->

约束资格保证活动约束的一阶法向量足以描述可行域附近的局部几何。若约束表示在最优点退化，局部最优仍可能存在，却没有普通 KKT 乘子。例如最大化 $f(x)=x$、约束 $x^2\le0$ 时，唯一可行点是零，但活动约束梯度也为零，驻点方程无法用它表示 $f'(0)=1$。

<!-- bilingual-en:start -->
Here the original constraint permits no move away from $0$, whereas its first-order approximation is $0\cdot d\le0$ for every direction $d$. That approximation incorrectly permits even the improving direction $d=1$. Algebraically, stationarity would require $1-\lambda\cdot0=0$, which no finite $\lambda\ge0$ satisfies. All functions are $C^1$ and $0$ is both a local and global maximizer, but LICQ fails because the active gradient is zero. The example shows why a theorem cannot simply omit all qualification assumptions; it does not say that LICQ is the only possible qualification.
<!-- bilingual-en:end -->

这里原约束不允许离开 $0$，但其一阶近似却是对所有方向 $d$ 都成立的 $0\cdot d\le0$，连改善目标的方向 $d=1$ 也被放行。代数上，驻点式要求 $1-\lambda\cdot0=0$，没有任何有限的 $\lambda\ge0$ 能满足。所有函数都是 $C^1$，而且 $0$ 既是局部又是全局最大点，但活动梯度为零使 LICQ 失败。这个例子说明定理不能直接删掉所有资格前提，并不是说 LICQ 是唯一可用的资格。

<!-- bilingual-en:start -->
## How LICQ produces the multipliers
<!-- bilingual-en:end -->

*LICQ 怎样给出乘子：把约束当作附近的新坐标*

<!-- bilingual-en:start -->
The proof below explains both the existence and the signs of the coefficients. Its extra tools are the [[逆函数定理|inverse function theorem]], which allows a locally reversible change of coordinates when its derivative matrix is invertible, and the [[多元链式法则|multivariable chain rule]]. The key move is to use the active constraint values themselves as some of the new coordinates. Then curved boundaries become simple coordinate bounds, locally.
<!-- bilingual-en:end -->

下面的证明同时解释系数为什么存在、为什么具有相应符号。额外用到的是[[逆函数定理]]：导数矩阵可逆时，可以在附近做可逆的坐标变换；以及[[多元链式法则]]。关键一步是把活动约束函数的取值本身作为部分新坐标，这样弯曲边界在局部就变成简单的坐标限制。

<!-- bilingual-en:start -->
Write the active indices as $i_1,\ldots,i_q$ and let $r=q+p$. LICQ says that the $r$ rows $\nabla g_{i_k}(x^*)^T$ and $\nabla h_j(x^*)^T$ are independent, so $r\le n$. The superscript $T$ denotes [[矩阵转置|transpose]], turning a gradient column into a row. Complete them to a [[基|basis]] of the row vectors in $\mathbb R^n$ by adding fixed rows $a_{r+1}^T,\ldots,a_n^T$. Such a completion is possible by repeatedly adding a vector outside the current span until it spans all of $\mathbb R^n$. Define the $n$-coordinate map $T$ below. Every displayed constraint is zero at $x^*$, and the added linear coordinates are also zero there, so $T(x^*)=0$.
<!-- bilingual-en:end -->

把活动指标写成 $i_1,\ldots,i_q$，令 $r=q+p$。LICQ 表示 $\nabla g_{i_k}(x^*)^T$ 与 $\nabla h_j(x^*)^T$ 这 $r$ 个行向量线性无关，所以 $r\le n$。上标 $T$ 表示[[矩阵转置|转置]]，将梯度列变成行。添加固定行向量 $a_{r+1}^T,\ldots,a_n^T$，将它们补成 $\mathbb R^n$ 行向量空间的一组[[基]]：只需在尚未张成整个空间时，不断加入当前张成空间之外的向量。由此定义下面的 $n$ 维坐标变换 $T$。所列约束在 $x^*$ 都为零，补充的线性坐标也为零，故 $T(x^*)=0$。

$$
T(x)=\bigl(g_{i_1}(x),\ldots,g_{i_q}(x),
h_1(x),\ldots,h_p(x),
a_{r+1}^T(x-x^*),\ldots,a_n^T(x-x^*)\bigr).
$$

<!-- bilingual-en:start -->
The derivative matrix $DT(x^*)$ has exactly those basis rows, so it is invertible. Since $T$ is $C^1$ nearby, the inverse function theorem supplies a $C^1$ inverse $x=T^{-1}(z)$ on an open neighborhood of $z=0$. Shrink that neighborhood if necessary: finitely many inactive inequalities remain strictly negative by continuity, and its inverse image lies within the neighborhood where $x^*$ is optimal. Thus, in these coordinates, the feasible points are exactly the nearby points obeying
<!-- bilingual-en:end -->

导数矩阵 $DT(x^*)$ 的行恰好是这组基，因此它可逆。由于 $T$ 在附近为 $C^1$，逆函数定理在 $z=0$ 的某个开邻域上提供 $C^1$ 逆变换 $x=T^{-1}(z)$。必要时缩小该邻域：由连续性，有限条非活动不等式仍严格小于零；其逆像也落在 $x^*$ 局部最优的邻域内。因此，新坐标中的附近可行点恰好满足

$$
z_k\le0\quad(1\le k\le q),\qquad
z_{q+j}=0\quad(1\le j\le p),
$$

<!-- bilingual-en:start -->
while the remaining coordinates $z_{r+1},\ldots,z_n$ are unrestricted locally. Put $\Phi(z)=f(T^{-1}(z))$. It is $C^1$, and $0$ locally maximizes it under these coordinate restrictions. Let $e_k$ denote the vector with coordinate $k$ equal to one and all other coordinates zero. For a free coordinate $k>r$, both $te_k$ and $-te_k$ are feasible for small positive $t$; the [[无约束一阶条件|one-variable derivative test]] gives $\partial_k\Phi(0)=0$. For an inequality coordinate $k\le q$, the move $-te_k$ is feasible, so
<!-- bilingual-en:end -->

而剩余坐标 $z_{r+1},\ldots,z_n$ 在局部没有限制。令 $\Phi(z)=f(T^{-1}(z))$，则它为 $C^1$，且 $0$ 在这些坐标限制下局部最大。记 $e_k$ 为第 $k$ 个坐标等于一、其余坐标为零的向量。对于自由坐标 $k>r$，当 $t>0$ 足够小时，$te_k$ 与 $-te_k$ 都可行，因此[[无约束一阶条件|一元导数判别]]给出 $\partial_k\Phi(0)=0$。对于不等式坐标 $k\le q$，$-te_k$ 可行，于是

$$
\Phi(-te_k)-\Phi(0)=-t\,\partial_k\Phi(0)+o(t)\le0
\quad\Longrightarrow\quad \partial_k\Phi(0)\ge0.
$$

<!-- bilingual-en:start -->
Here $o(t)/t\to0$ as $t\downarrow0$; dividing the inequality by positive $t$ gives the sign. Set $\lambda_{i_k}^*=\partial_k\Phi(0)\ge0$ for active inequalities and $\lambda_i^*=0$ for inactive ones. Equality coordinates cannot be varied feasibly on their own, so local optimality imposes no sign on their partial derivatives; set $\nu_j^*=\partial_{q+j}\Phi(0)$. Finally, the chain rule applied to $f=\Phi\circ T$ gives
<!-- bilingual-en:end -->

这里 $o(t)/t\to0$（$t\downarrow0$），将不等式除以正数 $t$ 即得该符号。对活动不等式取 $\lambda_{i_k}^*=\partial_k\Phi(0)\ge0$，对非活动不等式取 $\lambda_i^*=0$。等式坐标不能单独沿可行方向变化，因此局部最优不限制相应偏导的符号；取 $\nu_j^*=\partial_{q+j}\Phi(0)$。最后，对 $f=\Phi\circ T$ 使用链式法则，得到

$$
\nabla f(x^*)=DT(x^*)^T\nabla\Phi(0)
=\sum_{k=1}^{q}\lambda_{i_k}^*\nabla g_{i_k}(x^*)
+\sum_{j=1}^{p}\nu_j^*\nabla h_j(x^*).
$$

<!-- bilingual-en:start -->
The added free-coordinate terms vanish because their coefficients are zero. This is stationarity. Each active inequality has $g_i(x^*)=0$, and each inactive inequality has $\lambda_i^*=0$, so every [[互补松弛|complementary product]] is zero. Feasibility was assumed and nonnegativity was just proved: all four KKT classes now hold. Empty constraint families or an absence of free coordinates simply remove the corresponding terms. This argument also shows why continuity of derivatives on a neighborhood, rather than derivatives at just one point, is used in this theorem.
<!-- bilingual-en:end -->

补充自由坐标的项因系数为零而消失，留下的就是驻点方程。每条活动不等式都有 $g_i(x^*)=0$，每条非活动不等式都有 $\lambda_i^*=0$，因此所有[[互补松弛|互补乘积]]都为零。可行性来自假设，非负性刚刚证明，四类 KKT 条件至此全部成立。某类约束为空或没有自由坐标时，相应项直接省去即可。这个论证也说明了本定理为什么使用邻域中的导数连续性，而不只检查某一点的导数。

<!-- bilingual-en:start -->
## Necessary does not mean sufficient
<!-- bilingual-en:end -->

## 必要不等于充分

<!-- bilingual-en:start -->
The theorem only places a local optimum inside the KKT candidate set. In a nonconvex problem, a KKT point need not be the desired maximum. The linked KKT sufficiency theorem supplies a separate global guarantee under convex structure. LICQ is a convenient sufficient qualification, not a necessary condition for KKT multipliers to exist.
<!-- bilingual-en:end -->

这条定理只说局部最优必须进入 KKT 候选集。一般非凸问题中的 KKT 点仍可能不是所求最大点。若再满足凹目标、凸不等式与仿射等式等结构，可由 [[KKT充分性]] 证明全局最优。LICQ 是一条方便的充分资格，却不是 KKT 乘子存在的必要条件，反例见 [[LICQ不是KKT必要条件]]。

<!-- bilingual-en:start -->
The phrase “places a local optimum inside the KKT candidate set” keeps the theorem's qualification hypothesis. Failure of LICQ alone leaves multiplier existence undecided; the examples in [[LICQ不是KKT必要条件|LICQ is sufficient but not necessary for KKT existence]] show that distinction. [[Slater强对偶|Slater's theorem]] gives a different route under its own convex assumptions. None of these qualifications changes the direction of this local necessity theorem.
<!-- bilingual-en:end -->

“局部最优进入 KKT 候选集”这句话仍带着定理的资格前提。单独看到 LICQ 失败，还不能判断乘子是否存在，[[LICQ不是KKT必要条件]]中的例子专门解释了这一区别。[[Slater强对偶]]在其自身的凸性等前提下提供另一条途径；这些资格都不会改变本局部必要性定理的推导方向。

> [!warning] A KKT point can be a local minimum · KKT 点也可能是局部最小点
> <!-- bilingual-en:start -->
> Return to $f(x)=x+x^2$ on $[-2,0]$. At $\widehat x=-1/2$, $g_1(\widehat x)=-1/2$ and $g_2(\widehat x)=-3/2$, so both inequalities are slack. Taking both multipliers as zero gives stationarity $f'(\widehat x)=1+2(-1/2)=0$, nonnegativity, feasibility, and complementary slackness. LICQ also holds because the active family is empty. Yet $f(x)=(x+1/2)^2-1/4$, so $\widehat x$ is a strict local minimum, and nearby distinct feasible inputs have larger values. Even KKT together with $C^1$ and LICQ does not prove a local maximum; the local-maximum hypothesis cannot be inferred from the conclusion.
> <!-- bilingual-en:end -->
>
> 回到 $[-2,0]$ 上的 $f(x)=x+x^2$。在 $\widehat x=-1/2$ 处，$g_1(\widehat x)=-1/2$、$g_2(\widehat x)=-3/2$，两条不等式都松弛。把两乘子取为零，就满足驻点式 $f'(\widehat x)=1+2(-1/2)=0$，以及非负性、可行性和互补松弛。活动向量族为空，所以 LICQ 也成立。然而 $f(x)=(x+1/2)^2-1/4$，故 $\widehat x$ 是严格局部最小点，附近其他可行输入的值都更大。即使同时满足 KKT、$C^1$ 与 LICQ，也没有证明局部最大；不能从结论倒推出局部最大这一前提。

<!-- bilingual-en:start -->
For a global certificate, open [[KKT充分性|KKT sufficiency]] and check its additional shape assumptions. That is a separate theorem. In particular, the necessity result here allows a nonconcave objective and nonlinear equalities, while the standard concave-maximization sufficiency theorem uses a concave objective, convex inequalities, and affine equalities.
<!-- bilingual-en:end -->

若要证明全局最优，应打开[[KKT充分性]]并核对其额外的形状前提，这是另一个定理。本页的必要性允许目标不凹、等式非线性；标准凹最大化充分性定理则使用凹目标、凸不等式与仿射等式。

> [!question]- Check the direction · 自检
> <!-- bilingual-en:start -->
> If a candidate in a differentiable nonconvex problem satisfies all KKT conditions, does the necessity theorem above prove that it is a local maximizer?
> <!-- bilingual-en:end -->
>
> 若一个可微非凸问题的候选点满足全部 KKT 条件，上述必要性定理是否已证明它是局部最大？
>
> <!-- bilingual-en:start -->
> **Answer:** No. The theorem runs from a local optimum under a suitable qualification to KKT. The reverse direction needs separate sufficient conditions.
> <!-- bilingual-en:end -->
>
> **答案：** 没有。定理的方向是“合适资格下局部最优推出 KKT”；反向需要另行的充分条件。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for KKT necessity under a constraint qualification.
- The EC400 SOFP notes were checked for the maximisation sign convention and the four KKT classes.
- [MIT 6.7220, Lecture 7](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/mit6_7220_s25_lec07.pdf), PDF pp. 3–5, Definition L7.1 and Theorem L7.3: the KKT system, independence of all equality and active inequality gradients, and the need to distinguish continuous differentiability nearby from differentiability only at the point.
- [Stanford CME307/MS&E311, Lecture 7](https://web.stanford.edu/class/msande311/lecture07.pdf), PDF pp. 2–3 and 5–7: $C^1$ functions, local optimality, regularity, feasible curves, and the necessary-condition theorem. Its minimization convention is converted here to maximization with $g_i\le0$.
- [Stanford Math 174A, “The Inverse Function Theorem”](https://math.stanford.edu/~andras/174A-2.pdf), Theorem 1, PDF p. 1: the local $C^1$ inverse used in the coordinate proof. That proof is a direct derivation here; both polynomial examples are checked directly as well.
- Boyd and Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), §5.5.3, printed pp. 243–244 / PDF pp. 257–258: separates KKT arising from attained primal/dual optima with zero gap from sufficiency under convex structure, and states the additional Slater route. It is not used as the source of the general local LICQ theorem.
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：直接核对可微约束问题中约束资格与 KKT 必要性的关系。
- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程的最大化号约定和四类条件。
- MIT 讲义 PDF 第 3–5 页的 Definition L7.1 与 Theorem L7.3 支持四类条件、全部等式及活动不等式梯度的线性无关前提，以及“邻域内连续可微”与“只在该点可微”的区别。
- Stanford CME307/MS&E311 第 7 讲 PDF 第 2–3、5–7 页支持 $C^1$、局部最优、规则性、可行曲线与必要性定理；本文将其最小化号约定一致转换为 $g_i\le0$ 的最大化形式。
- Stanford Math 174A 逆函数定理讲义 PDF 第 1 页 Theorem 1 支持证明中的局部 $C^1$ 逆变换；坐标证明在本文直接推导，两个多项式例子也均直接计算核对。
- Boyd 与 Vandenberghe，*Convex Optimization* §5.5.3，印刷第 243–244 页／PDF 第 257–258 页，用于区分零对偶间隙且原、对偶最优均达到时的 KKT、凸结构下的充分性，以及 Slater 的另一条途径，不将其冒充一般局部 LICQ 定理的来源。

<!-- bilingual-en:start -->
The local EC400 notes, PDF pp. 29–30 and 34, supply the maximization sign convention. Their page-34 shortcuts about active constraints, zero equality multipliers, and interchangeable qualifications are not assumptions of the theorem above: activity is $g_i(x^*)=0$, every equality gradient is retained, and LICQ and Slater have distinct hypotheses and scopes.
<!-- bilingual-en:end -->

本地 EC400 讲义 PDF 第 29–30、34 页提供最大化号约定。第 34 页对活动约束、零等式乘子及资格可互换的简写，不作为本页定理的前提：活动性是 $g_i(x^*)=0$，全部等式梯度均保留，LICQ 与 Slater 各有不同的假设和适用范围。
