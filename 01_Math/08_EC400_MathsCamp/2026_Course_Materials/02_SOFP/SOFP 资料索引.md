# SOFP 资料索引

SOFP 是 **Static Optimisation and Fixed Points（静态优化与不动点）**。

## 课程主线

### Lecture 1 — Tools for Optimisation

完整讲解与例题：[[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性]]。原始手写与课堂记录见该笔记的材料入口。

- Quadratic forms and determinants
- Taylor expansion
- Concavity、convexity、quasi-concavity、quasi-convexity

### Lecture 2 — Optimisation

- Unconstrained optimisation
- Equality constraints and the Lagrange theorem
- Inequality constraints、Kuhn–Tucker theorem、Arrow–Enthoven theorem

### Lecture 3 — Comparative Statics and Fixed Points

- Envelope theorem
- Implicit function theorem
- Correspondences
- Fixed-point theorems and the theorem of the maximum

## Lecture 2–3 多元优化学习主线
<!-- bilingual-en:start -->
*Multivariable Optimisation Path for Lectures 2–3*
<!-- bilingual-en:end -->

![[多元优化.canvas]]

这张地图先把“最优点是否存在”“一阶条件找到什么”“怎样证明局部或全局最优”分开，再进入约束资格、KKT、比较静态与价值函数。箭头表示数学依赖或推论关系，不表示学习掌握度。
<!-- bilingual-en:start -->
The map first separates attainment, first-order candidates, and local or global certification. It then moves through constraint qualifications, KKT, comparative statics, and value functions. Arrows represent mathematical dependence or implication, not mastery.
<!-- bilingual-en:end -->

### 从存在与候选到局部、全局结论
<!-- bilingual-en:start -->
*From Existence and Candidates to Local and Global Conclusions*
<!-- bilingual-en:end -->

![[极值定理]]

极值定理只回答“有没有最优解”，不告诉我们解在哪里。对开放域内的可微问题，一阶条件再把内部极值缩小为零梯度候选点。
<!-- bilingual-en:start -->
The extreme-value theorem answers whether an optimum is attained, not where it is. For a differentiable problem on an open domain, the first-order condition then narrows interior extrema to zero-gradient candidates.
<!-- bilingual-en:end -->

![[无约束一阶条件]]

零梯度仍只是候选信号。严格定号的 Hessian 给出局部充分判据；若只是半定，就必须继续检查高阶项或其他结构。
<!-- bilingual-en:start -->
A zero gradient is still only a candidate signal. A strictly signed Hessian gives a local sufficient test; a merely semidefinite Hessian requires higher-order terms or other structure.
<!-- bilingual-en:end -->

![[Hessian 局部极小判据]]

![[Hessian 局部极大判据]]

![[半定Hessian无结论]]

点上的二阶检验只负责局部分类。要把局部最大升级为全局最大，必须同时在整个凸可行域上控制目标的凹性。
<!-- bilingual-en:start -->
A pointwise second-order test provides only local classification. Promoting a local maximum to a global one requires concavity across the entire convex feasible set.
<!-- bilingual-en:end -->

![[凸优化全局性]]

严格凹性可以在解已经取得时进一步给出唯一性，但取得性仍需要独立证明。
<!-- bilingual-en:start -->
Once attainment has been established, strict concavity can add uniqueness. It does not establish attainment by itself.
<!-- bilingual-en:end -->

![[严格凹不保证解存在]]

### 加入等式与不等式约束
<!-- bilingual-en:start -->
*Adding Equality and Inequality Constraints*
<!-- bilingual-en:end -->

等式约束先把可行移动限制到切空间。Lagrange 条件找出正则局部最优必须满足的驻点方程，随后的约束二阶条件才负责局部验证。
<!-- bilingual-en:start -->
Equality constraints first restrict feasible movement to the tangent space. The Lagrange condition locates stationary candidates that a regular local optimum must satisfy; constrained second-order conditions then provide local certification.
<!-- bilingual-en:end -->

![[Lagrange必要条件]]

![[乘子不是精确罚金]]

![[等式约束二阶条件]]

![[严格最优不推系统非奇异]]

乘子在这里先是驻点系统中的法向量系数，不自动等于罚函数参数。而且，即使二阶条件证明了严格局部最优，也不能反推比较静态所需的系统 Jacobian 可逆。
<!-- bilingual-en:start -->
Here a multiplier is first a coefficient of constraint normals in the stationarity system, not automatically a penalty parameter. Even a strict local optimum certified by second-order reasoning does not imply the nonsingular system Jacobian required for comparative statics.
<!-- bilingual-en:end -->

加入不等式约束后，KKT 把原问题可行性、对偶可行性、驻点和互补松弛组成一个候选系统。必要性与充分性是这个系统的两个不同逻辑方向。
<!-- bilingual-en:start -->
With inequality constraints, KKT combines primal feasibility, dual feasibility, stationarity, and complementary slackness into one candidate system. Necessity and sufficiency are two different logical directions for that system.
<!-- bilingual-en:end -->

![[KKT条件]]

![[互补松弛]]

![[绑定不推正乘子]]

互补松弛可以从“严格松弛”推出“零乘子”，却不能从“绑定”反推“正乘子”。接下来要问的是：什么条件保证局部最优真的会进入 KKT 系统？
<!-- bilingual-en:start -->
Complementary slackness allows “strictly slack” to imply a zero multiplier, but it does not allow “binding” to imply a positive multiplier. The next question is what guarantees that a local optimum actually enters the KKT system.
<!-- bilingual-en:end -->

![[LICQ条件]]

![[KKT必要性]]

![[LICQ保证乘子唯一]]

![[LICQ不是KKT必要条件]]

在一般可微问题中，LICQ 通过活动约束的局部几何提供一条必要性路线，还能保证已存在的乘子唯一。它失败时，KKT 乘子仍可能存在。
<!-- bilingual-en:start -->
In a general differentiable problem, LICQ supplies one local-geometric route to KKT necessity and also makes existing multipliers unique. Its failure does not by itself rule out KKT multipliers.
<!-- bilingual-en:end -->

![[Slater条件]]

![[Slater强对偶]]

![[Slater不等于LICQ]]

在凸问题中，Slater 改从严格可行点出发，通过强对偶与乘子存在性补上“最优 ⇒ KKT”。它与 LICQ 不是同一条件，也不是“KKT ⇒ 全局最优”所需的假设。
<!-- bilingual-en:start -->
In a convex problem, Slater starts from a strictly feasible point and uses strong duality and multiplier existence to supply “optimality $\Rightarrow$ KKT.” It is neither equivalent to LICQ nor required for the reverse certification from KKT to global optimality.
<!-- bilingual-en:end -->

![[KKT充分性]]

![[严格凹不保证乘子唯一]]

凹目标、凸不等式和仿射等式使 KKT 候选成为全局最优证书。严格凹性可以识别唯一的最优选择，但重复或相关约束仍可以让乘子不唯一。
<!-- bilingual-en:start -->
A concave objective with convex inequalities and affine equalities turns a KKT candidate into a certificate of global optimality. Strict concavity may identify a unique optimal choice, but duplicated or dependent constraints can still leave multipliers nonunique.
<!-- bilingual-en:end -->

### 从最优选择转向参数变化与最优价值
<!-- bilingual-en:start -->
*From Optimal Choices to Parameter Changes and Optimal Values*
<!-- bilingual-en:end -->

比较静态必须先分清两个对象：值函数记录“最好能有多好”，最优解对应记录“哪些选择做到了”。
<!-- bilingual-en:start -->
Comparative statics must first separate two objects: the value function records how good the best achievable value is, while the solution correspondence records which choices attain it.
<!-- bilingual-en:end -->

![[值函数]]

![[最优解对应]]

若要跟踪某条局部最优选择分支，就把最优性条件写成隐式方程组，并单独检查其 Jacobian 非奇异。
<!-- bilingual-en:start -->
To track a local optimizer branch, write the optimality conditions as an implicit system and independently check that its Jacobian is nonsingular.
<!-- bilingual-en:end -->

![[隐函数比较静态]]

若只问最优价值如何变化，包络定理在正则可微分支上可以绕过选择导数。
<!-- bilingual-en:start -->
If only the optimal value is of interest, the envelope theorem can bypass the optimizer derivative along a regular differentiable branch.
<!-- bilingual-en:end -->

![[包络定理]]

当当前最优解不唯一时，Danskin 定理在它的明确假设下用活动最优解直接计算价值的方向导数。
<!-- bilingual-en:start -->
When current optimizers are not unique, Danskin's theorem uses the active optimizer set to compute directional derivatives of the value under its stated assumptions.
<!-- bilingual-en:end -->

![[Danskin定理]]

选择分支发生跳跃时，价值可能仍然可微，也可能出现折点；必须直接检查价值函数。
<!-- bilingual-en:start -->
When an optimizer branch jumps, the value may remain differentiable or may develop a kink. The value function itself must be checked directly.
<!-- bilingual-en:end -->

![[优化器跳跃不推价值不可微]]

影子价格是约束参数的包络导数应用：它表示边际放宽约束时最优价值的一阶变化，不是一个无条件的市场价或精确罚金。
<!-- bilingual-en:start -->
A shadow price is the envelope derivative with respect to a constraint parameter: it measures the first-order value effect of a marginal relaxation, not an unconditional market price or exact penalty.
<!-- bilingual-en:end -->

![[影子价格]]

## 2026 讲义与 slides

- [[EC400 SOFP Syllabus.pdf|2026 Syllabus]]
- [[EC400 SOFP Coursepack.pdf|2026 Coursepack]]
- [[EC400 Lecture Notes SOFP.pdf|完整 Lecture Notes（50 页）]]
- [[EC400 Slides Lecture 1.pdf|Lecture 1 slides]]
- [[EC400 Slides Lecture 2.pdf|Lecture 2 slides]]
- [[EC400 Slides Lecture 3.pdf|Lecture 3 slides]]
- [[EC400 Visualizer Notes 2026.pdf|Visualizer Notes 2026]]
- [[EC400 Visualizer Notes 2019.pdf|Visualizer Notes 2019]]
- [[SOFP_notes_02-08-2026.pdf|Marie's Notes（Class Group 3/4）]]
- [[EC400 Problem Sets.pdf|Problem Sets]]
- [[EC400 Problem Set Solutions.pdf|Problem Set Solutions]]

![[EC400 Lecture Notes SOFP.pdf#height=620]]

## 2026 年考试资料（2026-09-29 新增）

- [[EC400 SOFP Exam 2026.pdf|2026 Exam（2 页）]] · [[EC400 SOFP Exam 2026 Solutions.pdf|2026 Solutions（5 页）]]
- [Moodle — EC400 Exam Papers and Solutions 2026](https://moodle.lse.ac.uk/mod/folder/view.php?id=2218698)

## 历年试题

2026-09-25 归档：`Past_Exams/` 已收齐当时 Moodle 开放的 2011、2012、2013、2016–2025 试题与 solutions，共 26 份 PDF。文件名保留 Moodle 原名，方便与课程页面核对。
