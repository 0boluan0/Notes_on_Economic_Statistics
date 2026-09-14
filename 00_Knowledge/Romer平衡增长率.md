---
aliases:
  - 研发配置与消费Euler条件共同决定Romer模型的平衡增长率
  - Balanced growth rate in the Romer model
student_os: knowledge-atom
atom_id: MACRO-ENDO-018
atom_type: proposition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 研发配置与消费Euler条件共同决定Romer模型的平衡增长率
<!-- bilingual-en:start -->
*Research allocation and the consumption Euler condition jointly determine the balanced growth rate in the Romer model.*
<!-- bilingual-en:end -->

[[Romer品种扩张模型]]的对称平衡增长路径中，设计 $A$、物质资本 $K$、产出 $Y$ 和消费 $C$ 按同一常数 $g$ 增长；每个品种的用量 $x$、利率 $r$ 和部门人力配置不变。增长率来自研究回报与消费储蓄选择的共同约束，并非直接指定一个正研发比例后就结束模型。
<!-- bilingual-en:start -->
On the symmetric balanced growth path of the [[Romer品种扩张模型|Romer expanding-variety model]], designs $A$, physical capital $K$, output $Y$ and consumption $C$ grow at the same constant rate $g$. Quantity per variety, the interest rate and sectoral human-capital allocations remain constant. Growth is jointly constrained by research returns and household saving decisions.
<!-- bilingual-en:end -->

原文令总人力资本 $H$ 和劳动力 $L$ 固定，$\delta>0$ 为研发效率。记 $\gamma=1-\alpha-\beta\in(0,1)$、$\Lambda=\alpha/[\gamma(1-\gamma)]$，其中 $\alpha,\beta>0$。[[Romer研发人力配置]] 已给出技术与市场一侧的关系：
<!-- bilingual-en:start -->
The original model fixes total human capital $H$ and labor $L$, with research productivity $\delta>0$. Define $\gamma=1-\alpha-\beta\in(0,1)$ and $\Lambda=\alpha/[\gamma(1-\gamma)]$ for positive $\alpha,\beta$. [[Romer研发人力配置|Research allocation]] supplies the technology-and-market relation:
<!-- bilingual-en:end -->

$$g=\delta H-\Lambda r.$$

家庭以贴现率 $\rho>0$ 评价未来，瞬时效用为 $u(C)=(C^{1-\sigma}-1)/(1-\sigma)$，$\sigma>0$；$\sigma=1$ 时取 $\log C$。其消费 Euler 条件将消费增长与利率联系起来：
<!-- bilingual-en:start -->
Households discount at $\rho>0$ and have instantaneous utility $u(C)=(C^{1-\sigma}-1)/(1-\sigma)$ with $\sigma>0$, using $\log C$ when $\sigma=1$. Their consumption Euler condition relates consumption growth to the interest rate:
<!-- bilingual-en:end -->

$$g=\frac{r-\rho}{\sigma},\qquad r=\rho+\sigma g.$$

把后一式代入第一式并解出 $g$，得到
<!-- bilingual-en:start -->
Substitution into the research relation and solution for $g$ gives:
<!-- bilingual-en:end -->

$$
\boxed{g=\frac{\delta H-\Lambda\rho}{1+\Lambda\sigma}},\qquad
r=\rho+\sigma g,\qquad H_A=\frac g\delta,\quad H_Y=H-H_A.
$$

正增长内点需要 $\delta H>\Lambda\rho$；不满足时须检查零研发的长期角点。还要求效用积分与终端条件适当，即 $\rho>(1-\sigma)g$，等价于 $r>g$；$\sigma\ge1$ 时这条条件自动成立。由中间品最优式还可复算 $rK=\gamma^2Y$，所以资源账给出 $C/Y=1-g\gamma^2/r>0$。
<!-- bilingual-en:start -->
Positive interior growth requires $\delta H>\Lambda\rho$; otherwise the long-run zero-research corner must be considered. Finiteness and the terminal condition require $\rho>(1-\sigma)g$, equivalently $r>g$, which holds automatically when $\sigma\ge1$. The intermediate producer's optimum also gives $rK=\gamma^2Y$, so resource feasibility implies $C/Y=1-g\gamma^2/r>0$.
<!-- bilingual-en:end -->

用一组可连贯复算的教学参数，时间以年计：$\alpha=\beta=1/4$、$\delta=0.05$、$\eta=5$、$H=1.8$、$L=1$、$\rho=0.01$、$\sigma=1$。其中 $\eta$ 是单位耐用品的资源成本，$\delta$ 仍是研发效率。由 $\gamma=1/2$、$\Lambda=1$ 得
<!-- bilingual-en:start -->
For a constructed example with time measured in years, take $\alpha=\beta=1/4$, $\delta=0.05$, $\eta=5$, $H=1.8$, $L=1$, $\rho=0.01$ and $\sigma=1$. Here $\eta$ is the resource cost per durable unit and $\delta$ remains research productivity. With $\gamma=1/2$ and $\Lambda=1$, the solution is:
<!-- bilingual-en:end -->

$$
g=0.04,\quad r=0.05,\quad H_A=0.8,\quad H_Y=1;
$$
$$
x=1,\quad p=0.5,\quad \pi=0.25,\quad P_A=5;
\qquad Y=A,\quad K=5A,\quad C=0.8A.
$$

这组数把 [[Romer中间品加价|租价与收益]]、[[Romer专利价值|专利现值]] 和人力配置接起来：研究工资 $\delta AP_A=0.25A$，生产工资 $\alpha Y/H_Y=0.25A$；资本增长 $\dot K=5\dot A=0.2A$，恰等于 $Y-C$。若初始 $A=10$ 且已在该路径上，后来设计翻倍到20时，产出从10到20、资本从50到100；新增实体机器由持续储蓄支付，翻倍时间为 $\log2/0.04$。
<!-- bilingual-en:start -->
These values connect [[Romer中间品加价|rental returns]], [[Romer专利价值|patent value]] and research allocation. Both research and production wages equal $0.25A$, and capital accumulation is $5\dot A=0.2A=Y-C$. Starting on this path at $A=10$, doubling designs to 20 also doubles output from 10 to 20 and capital from 50 to 100. Saving finances the additional machines; doubling takes $\log2/0.04$ time units.
<!-- bilingual-en:end -->

公式描述的是**平衡增长配置**。若初始 $K/A$ 不等于路径要求的比率，不能断言经济立刻按此配置增长；Romer 原文明确没有分析从任意初始状态收敛的转轨。本例是模型内部的一致性演示，不是某国参数校准或预测。
<!-- bilingual-en:start -->
The formula describes a balanced allocation. If the initial capital-to-design ratio differs from the required ratio, it does not establish immediate growth on this path. Romer explicitly leaves transitional convergence from arbitrary initial conditions unanalyzed. The example demonstrates internal consistency, not a country calibration or forecast.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=19)，印刷S88、S90–S93（PDF第19、21–24页）：偏好和消费Euler关系、平衡增长定义、式(11)–(13)、非负研究投入及有限效用条件、未分析转轨的声明。
- 已从工资配置与Euler式重新联立，并用式(4)–(5)复算数例的 $x,p,\pi,P_A$、工资相等、$\dot K=Y-C$ 和正消费。$\eta$ 与 $\delta$ 保留原文含义。
<!-- bilingual-en:start -->
The original equations supply the equilibrium growth relation and its restrictions. The example was independently checked against machine demand, operating returns, patent valuation, wage equality, goods-market feasibility and positive consumption.
<!-- bilingual-en:end -->
