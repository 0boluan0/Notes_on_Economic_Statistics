---
aliases:
  - 研发与最终品部门的工资比较决定Romer模型中的研究人力配置
  - Allocation of research human capital in the Romer model
student_os: knowledge-atom
atom_id: MACRO-ENDO-017
atom_type: proposition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 研发与最终品部门的工资比较决定Romer模型中的研究人力配置
<!-- bilingual-en:start -->
*The comparison of research and production wages determines the allocation of human capital to research in the Romer model.*
<!-- bilingual-en:end -->

在 [[Romer品种扩张模型]] 中，总人力资本 $H$ 固定，可以分为最终品生产的 $H_Y$ 和研发的 $H_A$。研究人力有机会成本：同一份能力进入实验室，就不能同时在当期最终品部门生产。两个部门都活跃且人力可流动时，均衡要求两边每单位人力资本的工资相等。
<!-- bilingual-en:start -->
In the [[Romer品种扩张模型|Romer expanding-variety model]], fixed total human capital $H$ is divided between final production, $H_Y$, and research, $H_A$. Research has an opportunity cost because that capacity cannot simultaneously produce final output. When both sectors are active and human capital is mobile, equilibrium equalizes their remuneration per unit.
<!-- bilingual-en:end -->

由 [[Romer研发知识外溢]]，一单位研究人力产生 $\delta A$ 的设计流量，每个设计可按 [[Romer专利价值|专利价格]] $P_A$ 出售；最终品竞争厂商按人力资本的边际产出付薪。因此
<!-- bilingual-en:start -->
With [[Romer研发知识外溢|research knowledge spillovers]], one unit of research human capital generates a design flow of $\delta A$, sold at [[Romer专利价值|patent price]] $P_A$. Competitive final producers pay the marginal product of human capital. Consequently:
<!-- bilingual-en:end -->

$$
w_R=\delta AP_A,\qquad w_Y=\frac{\alpha Y}{H_Y},\qquad
\delta AP_A=\frac{\alpha Y}{H_Y}.
$$

这里 $\delta>0$ 是研发效率，$A>0$ 是设计存量。为了在原文的对称平衡增长配置上解出分配，记 $\gamma=1-\alpha-\beta\in(0,1)$，$B=H_Y^\alpha L^\beta$。由生产式和中间品经营收益有 $Y=ABx^\gamma$、$P_A=\gamma(1-\gamma)Bx^\gamma/r$。代入工资等式后，共同的 $ABx^\gamma$ 约去：
<!-- bilingual-en:start -->
Here $\delta$ measures research productivity and $A$ is the design stock. On the original symmetric balanced allocation, write $\gamma=1-\alpha-\beta\in(0,1)$ and $B=H_Y^\alpha L^\beta$. Production and operating returns imply $Y=ABx^\gamma$ and $P_A=\gamma(1-\gamma)Bx^\gamma/r$. Substitution cancels the common term $ABx^\gamma$.
<!-- bilingual-en:end -->

$$
\frac{\delta\gamma(1-\gamma)}r=\frac\alpha{H_Y},\qquad
H_Y=\frac{\Lambda r}\delta,
$$
$$
\Lambda=\frac\alpha{\gamma(1-\gamma)},\qquad
H_A=H-\frac{\Lambda r}\delta,\qquad
g_A=\delta H_A=\delta H-\Lambda r.
$$

这就把研发比例从外部设定变成了两种职业回报的比较。比如 $H=1.8$、$\delta=0.05$、$\Lambda=1$、$r=0.05$，便有 $H_Y=1$、$H_A=0.8$。研究使用的是0.8单位既有人力资本，不能把它读成总人力资本每期增加0.8。
<!-- bilingual-en:start -->
Research allocation is now determined by the competing returns rather than imposed as an external research share. For $H=1.8$, $\delta=0.05$, $\Lambda=1$ and $r=0.05$, the allocation is $H_Y=1$ and $H_A=0.8$. Research uses 0.8 units of existing human capital; it does not add 0.8 to total human capital each period.
<!-- bilingual-en:end -->

内点公式还必须满足 $H_A>0$。若算出非正的研究投入，应检查 $H_A=0$ 的角点：研究的潜在工资不能超过最终品工资，不能继续强用工资等式得出负研究人力。这张卡把给定利率下的配置接到增长；利率与消费者储蓄选择怎样共同闭合，见 [[Romer平衡增长率]]。
<!-- bilingual-en:start -->
The interior formula also requires positive research input. A nonpositive candidate calls for the $H_A=0$ corner, where potential research remuneration cannot exceed the production wage. Negative research input is not a feasible allocation. This relation connects the interest rate to research; [[Romer平衡增长率|balanced growth]] closes the rate jointly with household saving decisions.
<!-- bilingual-en:end -->

## 来源与核验

- Romer（1990），[Endogenous Technological Change](https://web.stanford.edu/~klenow/Romer_1990.pdf#page=22)，印刷S91–S92（PDF第22–23页），式(8)–(12)：专利价值、跨部门工资、$H_Y$ 和增长—利率关系；印刷S85（PDF第16页）：固定总人力资本约束与研究工资。
- 已逐步代入并约去共同项；$\gamma$ 缩写原文 $1-\alpha-\beta$，$\Lambda$ 保持原文定义。数例仅计算部门配置，完整资源与偏好校验见增长率卡。
<!-- bilingual-en:start -->
Equations (8)–(12) establish the wage comparison and allocation. The substitutions were reworked with the original definition of $\Lambda$. The numerical allocation is embedded in the fully checked growth example.
<!-- bilingual-en:end -->
