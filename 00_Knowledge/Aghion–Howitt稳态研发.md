---
aliases:
  - 研发自由进入与制造劳动需求共同确定Aghion–Howitt模型的稳态研发劳动
  - Stationary research in the Aghion–Howitt model
student_os: knowledge-atom
atom_id: MACRO-ENDO-023
atom_type: proposition
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 研发自由进入与制造劳动需求共同确定Aghion–Howitt模型的稳态研发劳动
<!-- bilingual-en:start -->
*Free entry into research and manufacturing labour demand jointly determine stationary research employment in the Aghion–Howitt model.*
<!-- bilingual-en:end -->

这个最小闭合模型问：同一批熟练劳动中，有多少用于现在生产，多少用于寻找下一代技术？采用 Aghion–Howitt（1992）的线性研发与 Cobb–Douglas 特例：风险中性使利率等于固定时间偏好率 $r>0$；熟练劳动总量为 $N$，研发劳动为 $n_j$，制造劳动及中间品数量为 $x_j=N-n_j$。**此处 $n_j$ 是劳动量，与 AK 公式中的人口增长率 $n$ 不同。** 下标 $j$ 数创新代际，日历时间另记为 $\tau$。
<!-- bilingual-en:start -->
The model allocates skilled labour between current production and the search for new technology. Use the linear-research, Cobb–Douglas case of Aghion–Howitt: risk neutrality makes the interest rate equal the constant time-preference rate $r>0$; $N$ is total skilled labour, $n_j$ research labour, and $x_j=N-n_j$ manufacturing labour and intermediate output. Here $n_j$ is labour, unlike population growth $n$ in an AK equation. The index $j$ counts innovation generations; calendar time is $\tau$.
<!-- bilingual-en:end -->

固定非熟练劳动吸收入最终品技术，$0<\alpha<1$。每次成功使生产率乘以 $\gamma>1$；创新足够大，使旧专利不约束新垄断者的定价（drastic innovation）。本例中其条件为 $\gamma\ge\alpha^{-\alpha}$。制造一单位中间品用一单位熟练劳动；研发按 [[研发的泊松到达|强度]] $\lambda n_j$ 产生创新，$\lambda>0$。
<!-- bilingual-en:start -->
Fixed unskilled labour is absorbed into final production, with $0<\alpha<1$. Each discovery multiplies productivity by $\gamma>1$. Innovations are drastic, so the previous patent does not constrain monopoly pricing; here this requires $\gamma\ge\alpha^{-\alpha}$. One unit of intermediate output uses one skilled worker, and research generates discoveries at [[研发的泊松到达|intensity]] $\lambda n_j$, with $\lambda>0$.
<!-- bilingual-en:end -->

$$
y_j=A_jx_j^\alpha,\qquad A_j=A_0\gamma^j,\qquad x_j=N-n_j.
$$

竞争性最终品企业给出逆需求 $p_j=A_j\alpha x_j^{\alpha-1}$。中间品垄断者付工资 $w_j$，选择 $x_j$ 最大化 $(p_j-w_j)x_j$；由边际收入等于工资得到
<!-- bilingual-en:start -->
Competitive final producers generate inverse demand $p_j=A_j\alpha x_j^{\alpha-1}$. The intermediate monopolist chooses $x_j$ to maximise $(p_j-w_j)x_j$. Equating marginal revenue to the skilled wage gives
<!-- bilingual-en:end -->

$$
w_j=A_j\alpha^2x_j^{\alpha-1},\qquad
p_j=\frac{w_j}{\alpha},\qquad
\pi_j=\frac{1-\alpha}{\alpha}w_jx_j.
$$

外部研发企业每增加一单位劳动，付工资 $w_j$，获得期望奖励 $\lambda V_{j+1}$。利用 [[创新租金的替代风险|成功后利润的价值]]，自由进入满足
<!-- bilingual-en:start -->
An outside researcher pays $w_j$ for another unit of labour and receives expected reward $\lambda V_{j+1}$. Using the [[创新租金的替代风险|value of post-discovery profits]], free entry requires
<!-- bilingual-en:end -->

$$
w_j\ge\lambda V_{j+1},\qquad n_j\ge0,\qquad
n_j(w_j-\lambda V_{j+1})=0,
\qquad V_{j+1}=\frac{\pi_{j+1}}{r+\lambda n_{j+1}}.
$$

在位者做出下一次创新会替代自己的旧租金，获得的增量价值小于外部进入者；在线性、充分大创新的设定下，研发由外部企业进行。自由进入令研发预期净收益为零，仍允许成功者取得用来补偿投入与失败风险的利润。
<!-- bilingual-en:start -->
An incumbent's next discovery would replace its own rents, giving it a smaller incremental reward than an outsider. Under linear research and drastic innovation, outsiders conduct research. Free entry eliminates expected net research returns while allowing successful innovators to earn the profits that reward inputs and failure risk.
<!-- bilingual-en:end -->

稳态中 $n_j=\bar n$，制造数量不随代际改变，而工资和利润随 $A_j$ 同比例增长，故 $\pi_{j+1}=\gamma\pi_j$。正研发时消去工资，得到
<!-- bilingual-en:start -->
In a stationary allocation $n_j=\bar n$, manufacturing quantity stays fixed across generations, while wages and profits scale with productivity, so $\pi_{j+1}=\gamma\pi_j$. For positive research, eliminating the wage yields
<!-- bilingual-en:end -->

$$
1=\frac{\lambda\gamma(1-\alpha)(N-\bar n)}{\alpha(r+\lambda\bar n)},
\qquad
\bar n=\max\left\{0,
\frac{\lambda\gamma(1-\alpha)N-\alpha r}
{\lambda[\alpha+\gamma(1-\alpha)]}\right\}.
$$

最小例：$\alpha=0.5,\gamma=1.5,N=1,\lambda=0.1,r=0.08$，满足 $1.5>\sqrt2$。代入得 $\bar n=0.28,x=0.72$。取当代 $A_j=1$，则 $w_j\approx0.294628,\pi_j\approx0.212132$；下一代的 $V_{j+1}=1.5\pi_j/(0.08+0.028)\approx2.946278$，所以 $\lambda V_{j+1}=w_j$，劳动与研发条件对上。
<!-- bilingual-en:start -->
With $\alpha=0.5,\gamma=1.5,N=1,\lambda=0.1,r=0.08$, the drastic condition holds. Research is $\bar n=0.28$ and manufacturing $x=0.72$. At current productivity one, $w_j\approx0.294628$ and $\pi_j\approx0.212132$. The next patent is worth about $2.946278$, verifying $\lambda V_{j+1}=w_j$.
<!-- bilingual-en:end -->

这是**劳动分配稳态**，产出仍随机跳跃，见 [[随机创新的增长口径]]。它给市场均衡，不保证福利最优，见 [[私人与社会研发]]。固定质量步长、无资本积累及即时劳动调整是本例的范围；原文还允许非稳态均衡，不能将唯一稳态读成所有动态路径唯一。
<!-- bilingual-en:start -->
Stationarity concerns the labour allocation; output still jumps randomly, as described in [[随机创新的增长口径|growth measures under random innovation]]. This is a market equilibrium, not necessarily the [[私人与社会研发|social optimum]]. The example fixes innovation size and omits capital accumulation and labour adjustment frictions. A unique stationary allocation does not establish a unique dynamic path.
<!-- bilingual-en:end -->

## 来源与核验

- Aghion 与 Howitt（1992），[原文](https://web2.econ.ku.dk/dalgaard/growth/growth07/AghionHowitt.pdf#page=6)，印刷页327–333、340（PDF第6–12、19页），式(2.1)–(2.3)、(2.8)、(2.10)、(2.12)、(3.4)、(5.3)：核对资源、需求、利润、自由进入、代际时序和drastic条件。闭式与数值独立推导，$j$改为创新代际索引以免混淆日历时间。
<!-- bilingual-en:start -->
The cited equations establish technology, resources, monopoly choices, research entry, timing and the drastic-innovation condition. The closed form and numerical solution are independently derived; $j$ explicitly labels innovation generations.
<!-- bilingual-en:end -->
