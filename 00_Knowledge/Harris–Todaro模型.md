---
aliases:
  - Harris–Todaro模型是以城市预期收入与农村收入的比较来决定迁移及城市失业均衡的两部门模型
  - Harris–Todaro migration model
student_os: knowledge-atom
atom_id: DEV-MIG-020
atom_type: definition
status: source-checked
---

# Harris–Todaro模型是以城市预期收入与农村收入的比较来决定迁移及城市失业均衡的两部门模型

Harris–Todaro 模型把农村就业、城市现代部门岗位和城市求职者放在同一个劳动力分配问题中：城市工资高，但获得该工资的岗位有限；农村劳动者比较的是城市**预期收入**与放弃的农村收入。迁移扩大城市劳动力池、改变获聘概率，因而可以在仍有城市失业时停止。本卡采用无迁移成本、未获城市岗位收入为零的基准；多期、成本与风险的决策条件见[[迁移收益现值]]。
<!-- bilingual-en:start -->
The Harris–Todaro model allocates labour between rural employment and an urban labour pool competing for limited modern-sector jobs. Migrants compare expected urban earnings with rural earnings forgone. Migration changes the hiring probability and may stop while urban unemployment remains. This benchmark has no migration cost and zero income without an urban job; see [[迁移收益现值|the present value of migration]] for horizons, costs and risk.
<!-- bilingual-en:end -->

## 把岗位与求职人数一起写进均衡

总劳动力为 $L$，农业就业为 $L_A$，城市劳动力为 $S$，城市现代部门就业为 $M$。农业充分就业；城市劳动者同质，按[[城市就业概率换算|周期随机分配规则]]获得岗位，每人最多一个岗位。所有数量均为同一时期的劳动者人数，不是城市全部居民。
<!-- bilingual-en:start -->
Let total labour be $L$, agricultural employment $L_A$, the urban labour pool $S$, and modern-sector urban employment $M$. Agriculture has full employment; homogeneous urban workers receive at most one job under a [[城市就业概率换算|periodic random-allocation rule]]. All quantities count workers on the same time basis, not all urban residents.
<!-- bilingual-en:end -->

$$
L=L_A+S,\qquad 0<M\le S,\qquad
p=\frac{M}{S},\qquad E[w_U]=p\bar w_M.
$$

城市制度工资 $\bar w_M$ 高于灵活工资市场出清水平；给定生产条件，企业劳动需求决定 $M$。Harris 与 Todaro（1970）令农业收入为劳动边际产出价值 $w_A=Pq'(L_A)$，农业边际产出为正且递减；$P$ 是用制造品计价的农产品价格。该原文还让 $P$ 随两部门相对产量变化，并没有普遍固定粮价。
<!-- bilingual-en:start -->
The institutional urban wage exceeds the flexible-wage clearing level, and firms' labour demand determines $M$ under given production conditions. Harris and Todaro (1970) set rural earnings equal to agricultural marginal-product value, $Pq'(L_A)$, with positive diminishing marginal product. Agricultural prices are expressed in manufactured goods and respond to relative outputs in their full model; food prices are not universally fixed.
<!-- bilingual-en:end -->

若城市预期收入较高，劳动力向城市移动；城市岗位暂不变时，$S$ 上升使 $M/S$ 下降，同时离农可提高留在农业中的边际产出。允许双向调整、位于内点且不存在额外迁移门槛时，停止迁移的条件是：
<!-- bilingual-en:start -->
Higher expected urban earnings induce migration. With jobs temporarily fixed, a larger urban pool lowers $M/S$; leaving agriculture can also raise rural marginal product. At an interior equilibrium with two-way adjustment and no additional migration threshold:
<!-- bilingual-en:end -->

$$
w_A=\bar w_M\frac{M}{S},\qquad
U=S-M,\qquad u=\frac{U}{S}=1-\frac{w_A}{\bar w_M}.
$$

相等的是**预期收入**，不是每位城市劳动者实际拿到的工资。只要 $0<w_A<\bar w_M$，基准均衡的 $u>0$；这并不等于每个人领取 $w_A$。均衡也不是个体满意、社会福利最优或迁移不存在摩擦的证明。
<!-- bilingual-en:start -->
Expected earnings are equalised, not every worker's realised wage. If $0<w_A<\bar w_M$, benchmark equilibrium unemployment is positive; workers do not each receive $w_A$. Equilibrium is not proof of universal satisfaction, social optimality or frictionless migration.
<!-- bilingual-en:end -->

## 一个把概率解出来的最小例子

以下是固定产品价格的教学特例，不是原文数据。令 $L=100$、$\bar w_M=120$，企业在该工资下提供 $M=20$ 个岗位，农村收入函数为 $w_A=120-L_A$。由 $L_A=100-S$，有 $w_A=20+S$；均衡要求 $2400/S=20+S$，可行解为 $S=40$。因此 $L_A=60$、$p=1/2$、$U=20$、$u=50\%$，城乡比较的收入都是60。这里的 $p$ 来自均衡，而非另行假定一个“成功率”。
<!-- bilingual-en:start -->
In this constructed fixed-price example, $L=100$, the wage is 120, employers offer 20 jobs, and rural earnings are $120-L_A$. Hence rural earnings equal $20+S$ and equilibrium solves $2400/S=20+S$, giving $S=40$. Rural employment is 60, the job probability is one-half, unemployment is 20 and its rate is 50%; both expected urban and rural earnings are 60. The probability is solved jointly with migration.
<!-- bilingual-en:end -->

若未获现代岗位者从事[[非正规就业]]，他们仍是就业者，不能把 $S-M$ 全部计入[[失业率|公开失业]]。收入不为零时也要重写预期收益。这个扩展和异质性会改变模型的计算，不能只更换状态名称。课堂双原点图的读法见[[迁移双原点图]]；工资冲击与岗位冲击分别见[[城市工资与迁移反应]]、[[岗位创造与诱致迁移]]。
<!-- bilingual-en:start -->
Workers in [[非正规就业|informal employment]] remain employed; $S-M$ cannot all be counted as [[失业率|open unemployment]]. Positive non-modern earnings require a revised expected-income calculation. Such extensions and heterogeneity change the model, not merely its labels. See [[迁移双原点图|the dual-origin diagram]], [[城市工资与迁移反应|wage changes]] and [[岗位创造与诱致迁移|job creation]] for the corresponding applications.
<!-- bilingual-en:end -->

## 来源与核验

- Harris & Todaro（1970），[Migration, Unemployment and Development](https://www.aeaweb.org/aer/top20/60.1.126-142.pdf)，印刷第127–129页，PDF第2–4页，式(1)–(9)与脚注7–9：支持随机岗位分配、价格及边际产出条件、劳动约束和预期收入均衡。本文符号与固定价格数例为教学改写；原文的农村部门福利归属假设不是本卡求迁移停止条件的必要前提。
- Todaro & Smith（2012），[Economic Development, 11th ed.](https://business.tiu.edu.iq/wp-content/uploads/2019/02/Economic-Development-11th-Edition-Michael-P.-Todaro-Stephen-C.-Smith-1.pdf)，印刷第340–341页，PDF第366–367页：支持课堂两部门呈现、式(7.1)及正非正规收入需要修改预期收入的边界。
- [[02_Economy/10_发展经济学/发展经济学拍屏ppt.pdf#page=38|课堂PDF第38页]]：定位课堂符号与模型范围，不把历史教材的统计用语替代当代就业分类。
<!-- bilingual-en:start -->
The original paper supports allocation, production and equilibrium assumptions. The textbook supports the classroom formulation and its informal-income extension. Notation and the fixed-price numerical example are constructed for teaching; the classroom PDF supplies course context.
<!-- bilingual-en:end -->
