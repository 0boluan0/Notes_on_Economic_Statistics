---
aliases:
  - "Edgeworth Box"
  - "Edgeworth box economy"
  - "契约曲线"
  - "福利经济学基本定理"
  - "Fundamental Welfare Theorems"
status: source-checked
---

# 交换经济、Edgeworth box 与福利定理
<!-- bilingual-en:start -->
*Exchange Economies, the Edgeworth Box, and the Welfare Theorems*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在总资源已经给定时，判断两个人通过交换能否让彼此变得更好、什么时候所有交换收益已经用尽，以及竞争市场为何可能走到这种有效配置。
> **具体锚点：** A 有很多食物却缺住所，B 恰好相反；只要两人愿意用不同的比率交换食物和住所，就存在双方都获益的交易。
> **核心难点：** Edgeworth box 不是一张普通供求图。箱内每一点同时表示两个人的完整配置；契约曲线只表示 Pareto 有效，不表示公平；福利定理还依赖价格接受、完整市场、无外部性和凸性等条件。
> **为什么重要：** 它把 MRS、竞争均衡、效率与公平放进同一张图，也解释了为什么“市场结果有效”与“初始禀赋合理”是两个问题。
> **继续：** 先按“禀赋点 → 互利透镜 → 契约曲线 → 价格线”读图；要评价哪一个有效配置更值得选择，再回到 [[福利经济学与政府干预#Pareto、补偿与社会福利|Pareto、补偿与社会福利]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Given a fixed stock of resources, determine whether two people can trade to make both better off, when all gains from trade have been exhausted, and why a competitive market may reach such an efficient allocation.
> **Concrete anchor:** A has plenty of food but little shelter, while B has the opposite endowment. If they are willing to exchange food for shelter at different rates, a mutually beneficial trade exists.
> **Central difficulty:** An Edgeworth box is not an ordinary supply-and-demand diagram. Every point represents both people's complete allocations at once; the contract curve identifies Pareto efficiency, not fairness; and the welfare theorems require conditions such as price taking, complete markets, no externalities, and convexity.
> **Why it matters:** The diagram places the MRS, competitive equilibrium, efficiency, and equity in one model, showing why “the market outcome is efficient” and “the initial endowment is fair” are separate claims.
> **Continue:** Read the diagram in the order “endowment → mutually beneficial lens → contract curve → price line.” To judge which efficient allocation is socially preferable, return to [[福利经济学与政府干预#Pareto、补偿与社会福利|Pareto, Compensation, and Social Welfare]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - MIT OCW 14.03, [Lecture 10: General Equilibrium in a Pure Exchange Economy](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/e45ec68f98dcb7bcd7529866c0c44dc6_MIT14_03F16_lec10.pdf)：核验 Edgeworth box 的构造、互利透镜、契约曲线、竞争均衡与两条福利定理。
> - MIT OCW 14.42, [Lecture 3: Efficiency and Markets](https://ocw.mit.edu/courses/14-42-environmental-policy-and-economics-spring-2011/fab704c104e59101921d89a3f8307edb_MIT14_42S11_note03.pdf)：核验交换与生产效率条件、MRS/MRT 和福利定理失效时要检查的市场条件。
> <!-- bilingual-en:start -->
> - MIT OCW 14.03, [Lecture 10: General Equilibrium in a Pure Exchange Economy](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/e45ec68f98dcb7bcd7529866c0c44dc6_MIT14_03F16_lec10.pdf): verifies the construction of the Edgeworth box, the mutually beneficial lens, the contract curve, competitive equilibrium, and the two welfare theorems.
> - MIT OCW 14.42, [Lecture 3: Efficiency and Markets](https://ocw.mit.edu/courses/14-42-environmental-policy-and-economics-spring-2011/fab704c104e59101921d89a3f8307edb_MIT14_42S11_note03.pdf): verifies the exchange and production efficiency conditions, the roles of the MRS and MRT, and the market conditions to inspect when the welfare theorems fail.
> <!-- bilingual-en:end -->

## 箱子怎样同时表示两个人
<!-- bilingual-en:start -->
*How One Box Represents Two People at Once*
<!-- bilingual-en:end -->

设经济中只有两个人 A、B 和两种商品 $x$、$y$。总禀赋固定为 $(\bar x,\bar y)$。A 从左下角读自己的消费，B 从右上角反向读自己的消费，因此箱内任一点都自动给出四个数量：
<!-- bilingual-en:start -->
Suppose the economy contains two people, A and B, and two goods, $x$ and $y$. The aggregate endowment is fixed at $(\bar x,\bar y)$. A reads consumption from the lower-left origin, while B reads it in the opposite direction from the upper-right origin. Every point in the box therefore specifies four quantities at once:
<!-- bilingual-en:end -->

$$
(x_A,y_A),\qquad (x_B,y_B)=(\bar x-x_A,\bar y-y_A).
$$

这两个资源约束意味着，箱内的点是可行配置；箱外的点需要超过现有总资源。箱子的宽和高分别是两种商品的总量，不是价格或效用。
<!-- bilingual-en:start -->
These two resource constraints make every point inside the box a feasible allocation; a point outside the box would require more than the available resources. The width and height of the box are the aggregate quantities of the two goods, not prices or utility levels.
<!-- bilingual-en:end -->

初始禀赋点 $E$ 说明交易前谁拥有什么。通过 A、B 在 $E$ 处的无差异曲线，可以找到一片“透镜”：透镜中的配置使至少一人严格更好，同时另一人不变差。只要透镜非空，初始配置就还没有用尽交换收益。
<!-- bilingual-en:start -->
The initial-endowment point $E$ records who owns what before trade. A's and B's indifference curves through $E$ enclose a lens-shaped region: allocations inside the lens make at least one person strictly better off without making the other worse off. If this lens is non-empty, the initial allocation has not exhausted the gains from trade.
<!-- bilingual-en:end -->

## 从 MRS 不同到契约曲线
<!-- bilingual-en:start -->
*From Unequal MRSs to the Contract Curve*
<!-- bilingual-en:end -->

边际替代率 $MRS_{xy}$ 表示，在效用不变时，一个人愿意为多得到一单位 $x$ 放弃多少 $y$。若同一配置上两人的 MRS 不同，他们对交换比率的评价不同，通常可以在两者之间找到一个价格，使双方都愿意交易。
<!-- bilingual-en:start -->
The marginal rate of substitution $MRS_{xy}$ measures how much $y$ a person is willing to give up for one additional unit of $x$ while holding utility constant. If the two people have different MRSs at the same allocation, they value the exchange rate differently, so one can usually find a trading rate between the two at which both are willing to trade.
<!-- bilingual-en:end -->

在内点、偏好光滑且局部非饱和时，交换的 Pareto 效率条件是：
<!-- bilingual-en:start -->
At an interior allocation with smooth, locally non-satiated preferences, Pareto efficiency in exchange requires:
<!-- bilingual-en:end -->

$$
MRS_{xy}^{A}=MRS_{xy}^{B}.
$$

直觉不是“两个数字碰巧相等”，而是双方已经没有彼此都能接受的边际交换方向。若两条无差异曲线还会相交，就还能画出新的互利透镜，原配置便不是 Pareto 有效。
<!-- bilingual-en:start -->
The intuition is not merely that two numbers happen to match. Equality means there is no remaining marginal direction of trade that both people would accept. If their indifference curves still cross, another mutually beneficial lens can be drawn, so the allocation is not Pareto efficient.
<!-- bilingual-en:end -->

所有 Pareto 有效配置的集合称为契约曲线。名字容易误导：它不是已经签署的合同，也不表示某一个唯一均衡；它汇集了许多不同分配结果，其中有些极端偏向 A，有些极端偏向 B。
<!-- bilingual-en:start -->
The set of all Pareto-efficient allocations is called the contract curve. The name can mislead: it is not a signed contract and does not identify one unique equilibrium. It contains many distributive outcomes, some heavily favouring A and others heavily favouring B.
<!-- bilingual-en:end -->

切点条件只适用于内点。若某个人只想要一种商品，Pareto 有效配置可能位于箱子的边或角，无法用“两条曲线相切”识别；此时要直接检查是否仍存在可行的 Pareto 改进。
<!-- bilingual-en:start -->
The tangency condition applies only to interior allocations. If one person values only one good, an efficient allocation may lie on an edge or at a corner and cannot be identified by tangency. In that case, test directly whether any feasible Pareto improvement remains.
<!-- bilingual-en:end -->

## 价格怎样选出一个竞争均衡
<!-- bilingual-en:start -->
*How Prices Select a Competitive Equilibrium*
<!-- bilingual-en:end -->

给定价格比 $p_x/p_y$，穿过初始禀赋点的预算线说明每个人可以按什么比率出售禀赋并购买另一种商品。价格接受者各自在预算线上选择最偏好的消费束；竞争均衡还要求两人的选择能够同时实现，即每种商品总需求等于总禀赋。
<!-- bilingual-en:start -->
Given a price ratio $p_x/p_y$, the budget line through the initial endowment shows the rate at which each person can sell endowment goods and buy the other good. Each price-taking consumer chooses the most preferred affordable bundle. A competitive equilibrium additionally requires the two choices to be jointly feasible: aggregate demand for every good must equal the aggregate endowment.
<!-- bilingual-en:end -->

在内点最优选择上，两人都面对同一个价格比，因此：
<!-- bilingual-en:start -->
At an interior optimum, both people face the same price ratio, so:
<!-- bilingual-en:end -->

$$
MRS_{xy}^{A}=\frac{p_x}{p_y}=MRS_{xy}^{B}.
$$

价格的作用是把两张偏好图分开并让市场出清。若 $x$ 超额需求，调整相对价格会抑制对 $x$ 的需求并鼓励供给；直到各人的最优选择彼此兼容。价格不是凭空制造资源，而是在可行配置中协调谁持有什么。
<!-- bilingual-en:start -->
Prices separate the two preference maps and coordinate market clearing. If there is excess demand for $x$, an adjustment in its relative price discourages demand and encourages supply until individual choices become mutually compatible. Prices do not create resources; they coordinate ownership within the feasible set.
<!-- bilingual-en:end -->

## Worked example：从互利交易到均衡
<!-- bilingual-en:start -->
*Worked Example: From Mutually Beneficial Trade to Equilibrium*
<!-- bilingual-en:end -->

设总资源为 $(10,10)$，A 的初始禀赋是 $(8,2)$，B 的初始禀赋是 $(2,8)$，两人的效用均为 $u(x,y)=\sqrt{xy}$。对于这种偏好：
<!-- bilingual-en:start -->
Let the aggregate endowment be $(10,10)$. A initially owns $(8,2)$ and B owns $(2,8)$, and both have utility $u(x,y)=\sqrt{xy}$. Under these preferences:
<!-- bilingual-en:end -->

$$
MRS_{xy}=\frac{MU_x}{MU_y}=\frac{y}{x}.
$$

在初始点，$MRS_A=2/8=0.25$，而 $MRS_B=8/2=4$。A 愿意用较少的 $y$ 换 $x$，B 对 $x$ 的相对评价更高；两人的边际评价不同，所以存在互利交换空间。
<!-- bilingual-en:start -->
At the initial allocation, $MRS_A=2/8=0.25$, whereas $MRS_B=8/2=4$. A is willing to give up relatively little $y$ for $x$, while B values $x$ more highly at the margin. Their marginal valuations differ, so mutually beneficial trade is possible.
<!-- bilingual-en:end -->

若 $p_x=p_y=1$，两人的禀赋财富都为 10。Cobb–Douglas 消费者把一半财富用于每种商品，因此 A、B 都选择 $(5,5)$。总需求恰为 $(10,10)$，市场出清；两人的 MRS 都为 1，也等于价格比。
<!-- bilingual-en:start -->
If $p_x=p_y=1$, each person's endowment wealth equals 10. A Cobb–Douglas consumer spends half of wealth on each good, so both A and B choose $(5,5)$. Aggregate demand is exactly $(10,10)$, the markets clear, and both MRSs equal 1, which also equals the price ratio.
<!-- bilingual-en:end -->

两人的效用都从 $\sqrt{16}=4$ 上升到 $\sqrt{25}=5$。因此 $(5,5)$ 不仅满足切点条件，而且相对于给定初始禀赋是双方都严格受益的竞争均衡。换一组初始禀赋，即使总资源与偏好不变，也可能由同一套市场机制选出契约曲线上的另一个点。
<!-- bilingual-en:start -->
Each person's utility rises from $\sqrt{16}=4$ to $\sqrt{25}=5$. Thus $(5,5)$ not only satisfies the tangency condition but is also a competitive equilibrium that makes both people strictly better off relative to the stated endowment. A different initial distribution, even with unchanged aggregate resources and preferences, may lead the same market mechanism to select another point on the contract curve.
<!-- bilingual-en:end -->

## 两条福利定理分别说什么
<!-- bilingual-en:start -->
*What the Two Welfare Theorems Actually Say*
<!-- bilingual-en:end -->

> [!source] 本节依据
> MIT 14.03 Lecture 10 用 Edgeworth box 展示竞争均衡、第一福利定理以及通过重新分配禀赋支持不同有效配置的第二福利定理；MIT 14.42 Lecture 3 同时列出完整产权、价格接受、信息和交易成本等现实边界。
> <!-- bilingual-en:start -->
> MIT 14.03 Lecture 10 uses the Edgeworth box to present competitive equilibrium, the First Welfare Theorem, and the Second Welfare Theorem's support of different efficient allocations after endowment redistribution. MIT 14.42 Lecture 3 also identifies practical boundaries involving complete property rights, price taking, information, and transaction costs.
> <!-- bilingual-en:end -->

**第一福利定理：** 在适当条件下，每一个竞争均衡都是 Pareto 有效的。方向是“均衡 → 有效”。它没有说明均衡公平，也没有说市场一定存在、唯一或稳定；它更没有说现实中的外部性、市场力量和缺失市场会自动消失。
<!-- bilingual-en:start -->
**First Welfare Theorem:** Under suitable conditions, every competitive equilibrium is Pareto efficient. Its direction is “equilibrium → efficiency.” It does not say that the equilibrium is equitable, that equilibrium necessarily exists, is unique, or is stable, or that externalities, market power, and missing markets disappear in reality.
<!-- bilingual-en:end -->

**第二福利定理：** 在偏好和技术具有适当凸性等条件下，每一个 Pareto 有效配置都可以通过恰当的初始禀赋再分配，再交给竞争市场来支持。方向是“有效配置 + 合适禀赋 → 均衡”。理论上这把分配目标与交易效率分开：先用一次性转移选择起点，再让价格协调边际选择。
<!-- bilingual-en:start -->
**Second Welfare Theorem:** With suitable convexity and related conditions on preferences and technology, every Pareto-efficient allocation can be supported as a competitive equilibrium after an appropriate redistribution of initial endowments. Its direction is “efficient allocation + suitable endowment → equilibrium.” In theory, this separates distribution from exchange efficiency: choose the starting point through lump-sum transfers, then let prices coordinate marginal choices.
<!-- bilingual-en:end -->

现实税收通常不是无行为效应的一次性禀赋转移。政府不知道每个人的能力和偏好，税基会改变，执行也有成本，所以第二福利定理不是“先随意再分配、市场都会无损修复”的政策处方。
<!-- bilingual-en:start -->
Real-world taxes are generally not behaviour-free lump-sum transfers of endowments. Governments do not know everyone's ability and preferences, tax bases respond, and enforcement is costly. The Second Welfare Theorem is therefore not a policy recipe claiming that any redistribution can be followed by a costless market correction.
<!-- bilingual-en:end -->

## 从交换扩展到生产：MRS 与 MRT
<!-- bilingual-en:start -->
*Extending from Exchange to Production: MRS and MRT*
<!-- bilingual-en:end -->

纯交换模型固定总资源；加入生产后，经济还必须决定生产多少 $x$ 和 $y$。边际转换率 $MRT_{xy}$ 是生产可能性边界的斜率绝对值，表示多生产一单位 $x$ 必须放弃多少 $y$。
<!-- bilingual-en:start -->
The pure-exchange model fixes aggregate resources. Once production is added, the economy must also choose how much $x$ and $y$ to produce. The marginal rate of transformation $MRT_{xy}$ is the absolute slope of the production-possibility frontier and measures how much $y$ must be forgone to produce one additional unit of $x$.
<!-- bilingual-en:end -->

在内点的总体 Pareto 效率配置中，消费者之间的 MRS 要相等，生产者之间的 MRT 要相等，而且共同的消费边际评价还要等于生产的机会成本：
<!-- bilingual-en:start -->
At an interior allocation that is Pareto efficient for the economy as a whole, consumers' MRSs must be equal, producers' MRTs must be equal, and the common marginal valuation in consumption must equal the opportunity cost in production:
<!-- bilingual-en:end -->

$$
MRS_{xy}^{A}=MRS_{xy}^{B}=\frac{p_x}{p_y}=MRT_{xy}.
$$

若 $MRS>MRT$，消费者愿意为额外的 $x$ 放弃的 $y$ 多于生产上实际需要牺牲的 $y$，经济可以通过多生产 $x$、少生产 $y$ 创造改进；反向不等式则给出相反调整。
<!-- bilingual-en:start -->
If $MRS>MRT$, consumers are willing to give up more $y$ for an extra unit of $x$ than production actually requires, so the economy can improve the allocation by producing more $x$ and less $y$. The reverse inequality calls for the opposite adjustment.
<!-- bilingual-en:end -->

## 诊断顺序与失败边界
<!-- bilingual-en:start -->
*Diagnostic Sequence and Failure Boundaries*
<!-- bilingual-en:end -->

读一道 Edgeworth box 题时，依次检查：

1. 箱子的宽高和两个原点分别表示什么；
2. 给定点是否满足资源可行性；
3. 初始无差异曲线之间是否存在互利透镜；
4. 是内点切线问题，还是边界/角点问题；
5. 若问竞争均衡，是否同时满足个人最优和市场出清；
6. 若问社会评价，是否把 Pareto 效率误当作公平。
<!-- bilingual-en:start -->
When reading an Edgeworth-box problem, check in this order:

1. what the box dimensions and the two origins represent;
2. whether the stated point satisfies resource feasibility;
3. whether the initial indifference curves enclose a mutually beneficial lens;
4. whether the problem has an interior tangency or an edge or corner solution;
5. if a competitive equilibrium is requested, whether both individual optimality and market clearing hold;
6. if a social judgement is requested, whether Pareto efficiency is being mistaken for equity.
<!-- bilingual-en:end -->

最常见的错误是看到 $MRS_A=MRS_B$ 就宣布“公平且均衡”。这个等式只提供内点交换效率；要成为竞争均衡，还要有一条由价格和禀赋决定的共同预算线并使市场出清；要判断公平，则必须另给分配标准或社会福利权重。
<!-- bilingual-en:start -->
The most common mistake is to see $MRS_A=MRS_B$ and declare the allocation “fair and in equilibrium.” That equality establishes only interior exchange efficiency. Competitive equilibrium also requires a common budget line determined by prices and endowments, together with market clearing. Equity requires a separate distributive criterion or social-welfare weights.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么 MRS 不相等意味着通常还有交换收益？
<!-- bilingual-en:start -->
*Explain in your own words: Why do unequal MRSs usually imply remaining gains from trade?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 两人的边际交换意愿不同，通常可以在两者的 MRS 之间找到一个交换比率：一方愿意按该比率买，另一方愿意卖，因此双方都能走向更高的无差异曲线。
> <!-- bilingual-en:start -->
> Their marginal willingness to trade differs. A trading rate can usually be found between the two MRSs at which one person wants to buy and the other wants to sell, allowing both to reach higher indifference curves.
> <!-- bilingual-en:end -->

### 契约曲线上的点是否都同样公平？
<!-- bilingual-en:start -->
*Are all points on the contract curve equally equitable?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不是。契约曲线只表示没有剩余的 Pareto 改进；它可以包含几乎把全部资源给 A 或给 B 的极端配置。公平判断需要分配标准，不能由效率条件单独推出。
> <!-- bilingual-en:start -->
> No. The contract curve only means that no Pareto improvement remains; it can include extreme allocations giving almost everything to A or to B. Equity requires a distributive criterion and cannot be inferred from efficiency alone.
> <!-- bilingual-en:end -->

### 一个点满足 $MRS_A=MRS_B$，为什么仍未必是竞争均衡？
<!-- bilingual-en:start -->
*A point satisfies $MRS_A=MRS_B$. Why might it still fail to be a competitive equilibrium?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它还必须可行，并且存在一组价格，使两个人在各自由禀赋决定的预算约束下都主动选择该点，同时所有市场出清。MRS 相等本身没有验证预算可达性和出清。
> <!-- bilingual-en:start -->
> The point must also be feasible, and prices must exist at which both people voluntarily choose it under the budgets generated by their endowments while every market clears. Equal MRSs alone do not establish affordability or market clearing.
> <!-- bilingual-en:end -->

### 第一与第二福利定理的方向分别是什么？
<!-- bilingual-en:start -->
*What is the direction of each welfare theorem?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 第一福利定理从竞争均衡推出 Pareto 有效；第二福利定理在凸性等条件下，从目标 Pareto 有效配置与合适的初始禀赋再分配推出可支持它的竞争均衡。
> <!-- bilingual-en:start -->
> The First Welfare Theorem maps competitive equilibrium to Pareto efficiency. Under conditions including convexity, the Second maps a target Pareto-efficient allocation plus a suitable redistribution of initial endowments to a competitive equilibrium that supports it.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- MIT OCW 14.03, [Lecture 10: General Equilibrium in a Pure Exchange Economy](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/e45ec68f98dcb7bcd7529866c0c44dc6_MIT14_03F16_lec10.pdf)：支持两人两商品的资源约束、Edgeworth box、互利透镜、契约曲线、Walrasian 价格调整、竞争均衡和福利定理；二次核对了角点例外与第二定理的禀赋再分配含义。
- MIT OCW 14.42, [Lecture 3: Efficiency and Markets](https://ocw.mit.edu/courses/14-42-environmental-policy-and-economics-spring-2011/fab704c104e59101921d89a3f8307edb_MIT14_42S11_note03.pdf)：支持 $MRS$、价格比和 $MRT$ 的连接，以及完整产权、价格接受、信息与交易成本等边界。
- [[02_Economy/02_public finance财政学/1_导论/03_规范分析——Negative analysis|财政学规范分析课堂记录]]：核对本课程实际出现的 Edgeworth box、契约曲线、MRS/MRT、两条福利定理和效率—公平顺序；课堂原文仅作为课程语境，不独立承担理论核验。
<!-- bilingual-en:start -->
- MIT OCW 14.03, [Lecture 10: General Equilibrium in a Pure Exchange Economy](https://ocw.mit.edu/courses/14-03-microeconomic-theory-and-public-policy-fall-2016/e45ec68f98dcb7bcd7529866c0c44dc6_MIT14_03F16_lec10.pdf): supports the two-person, two-good resource constraints, the Edgeworth box, the mutually beneficial lens, the contract curve, Walrasian price adjustment, competitive equilibrium, and the welfare theorems; the corner-solution exception and the endowment-redistribution meaning of the Second Welfare Theorem were checked again.
- MIT OCW 14.42, [Lecture 3: Efficiency and Markets](https://ocw.mit.edu/courses/14-42-environmental-policy-and-economics-spring-2011/fab704c104e59101921d89a3f8307edb_MIT14_42S11_note03.pdf): supports the links among the $MRS$, the price ratio, and the $MRT$, together with boundaries involving complete property rights, price taking, information, and transaction costs.
- [[02_Economy/02_public finance财政学/1_导论/03_规范分析——Negative analysis|Public-finance classroom record on normative analysis]]: confirms that this course actually covered the Edgeworth box, the contract curve, the MRS/MRT conditions, the two welfare theorems, and the efficiency–equity sequence. The classroom text supplies course context rather than independent theoretical verification.
<!-- bilingual-en:end -->
