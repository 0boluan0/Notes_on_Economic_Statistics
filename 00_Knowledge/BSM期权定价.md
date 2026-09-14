---
aliases:
  - BSM期权定价用对数正态风险中性分布计算欧式看涨与看跌的价值
  - Black–Scholes–Merton option pricing
student_os: knowledge-atom
atom_id: FI-MKT-006
atom_type: method
status: source-checked
requires:
  - "[[风险中性定价]]"
related:
  - "[[Delta]]"
  - "[[隐含波动率]]"
  - "[[Delta与实值概率]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# BSM期权定价用对数正态风险中性分布计算欧式看涨与看跌的价值
<!-- bilingual-en:start -->
*BSM option pricing values European calls and puts under a risk-neutral lognormal distribution*
<!-- bilingual-en:end -->

BSM 公式把**欧式到期支付**放进 [[风险中性定价]]。这里取现货 $S>0$、执行价 $K>0$、剩余年数 $\tau=T-t>0$，常数连续复利利率 $r$、连续股息率 $q$、波动率 $\sigma>0$；无摩擦，可持续交易并按模型利率融资。模型下股价连续，风险中性对数价格增量为正态：
<!-- bilingual-en:start -->
The BSM formula applies [[风险中性定价|risk-neutral valuation]] to European terminal payoffs. Use positive $S,K,\tau,\sigma$, constant continuously compounded $r$ and dividend yield $q$, frictionless continuous trading and model-consistent financing. Prices have continuous paths and risk-neutral log increments are normal:
<!-- bilingual-en:end -->

$$\log(S_T/S)=(r-q-\tfrac12\sigma^2)\tau+\sigma\sqrt\tau Z,\qquad Z\sim N(0,1).$$

记 $N$ 为标准正态累计分布函数，先算
<!-- bilingual-en:start -->
Let $N$ denote the standard normal cumulative distribution function. First calculate:
<!-- bilingual-en:end -->

$$d_1=\frac{\log(S/K)+(r-q+\tfrac12\sigma^2)\tau}{\sigma\sqrt\tau},\qquad d_2=d_1-\sigma\sqrt\tau.$$

然后分别代入：
<!-- bilingual-en:start -->
Then substitute into the call and put formulas respectively:
<!-- bilingual-en:end -->

$$C=Se^{-q\tau}N(d_1)-Ke^{-r\tau}N(d_2),$$
$$P=Ke^{-r\tau}N(-d_2)-Se^{-q\tau}N(-d_1).$$

为什么看涨有两项？把支付写成 $S_T1_{S_T>K}-K1_{S_T>K}$：第二项是满足行权条件时支付执行价的贴现期望；第一项是那部分状态下收到股票的贴现期望。对数正态积分分别给出 $N(d_2)$ 与 $N(d_1)$，不能把两者随意交换。
<!-- bilingual-en:start -->
Split the call payoff into $S_T1_{S_T>K}-K1_{S_T>K}$. The discounted strike payment uses the exercise-event probability; the stock receipt uses a price-weighted expectation. Evaluating the lognormal integrals gives $N(d_2)$ and $N(d_1)$ respectively; they are not interchangeable.
<!-- bilingual-en:end -->

取 $S=K=50,r=0.05,q=0,\sigma=0.20,\tau=0.5$，有 $d_1\approx0.247487,d_2\approx0.106066$，从同一组未提前舍入的数据得到 $C\approx3.444364$、$P\approx2.209860$。差额核对为 $C-P=50-50e^{-0.025}\approx1.234504$。这是公式的内部核算，不是期权真实市价预测。
<!-- bilingual-en:start -->
With $S=K=50,r=0.05,q=0,\sigma=0.20,\tau=0.5$, unrounded inputs give $C\approx3.444364$ and $P\approx2.209860$. Their difference is $50-50e^{-0.025}\approx1.234504$. This checks formula consistency, not actual market-price accuracy.
<!-- bilingual-en:end -->

加入 $q$ 时，价格前项和 $d_1$ 中的漂移都要改变。美式合约须先判断 [[美式行权决策]]；只有满足 [[无股息看涨不提前行权]] 等适用条件时，才可据此复用欧式价格。离散股息、跳跃或随机波动率也须选相应估值方法，不能原样代入此式当成精确答案。由市场价格倒求 $\sigma$ 是 [[隐含波动率]]；对输入求偏导才得到 [[Greeks]]。
<!-- bilingual-en:start -->
Dividend yield changes both the prefactor and the drift in $d_1$. An American contract requires an [[美式行权决策|exercise-versus-continuation decision]]; the European price applies only under qualifying conditions such as [[无股息看涨不提前行权|the non-dividend American-call result]]. Discrete dividends, jumps or stochastic volatility also require appropriate valuation methods. Inverting a market price gives [[隐含波动率|implied volatility]]; differentiating with respect to inputs gives [[Greeks|Greeks]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Kohn–Allen, [*Derivative Securities*, §4](https://math.nyu.edu/~kohn/derivative.securities/2007/section4.pdf), pp. 6–9：对数正态风险中性分布及期权期望；[§5](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf), pp. 1–3：看涨公式和偏导的起点。
- Haugh, [*A Brief Review of Derivatives Pricing & Hedging*](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf), pp. 9–10、13–17：BSM、股息调整及模型边界。
- 课程 [[02_Economy/07_金融机构与风险管理/08_操作员如何管理风险暴露]] §1.1.4 提供参数与学习顺序；半年的看涨/看跌由本页公式重新计算，原稿代入差错由相邻校注说明。
<!-- bilingual-en:start -->
*Sources and verification*

- Kohn–Allen, [*Derivative Securities*, Section 4](https://math.nyu.edu/~kohn/derivative.securities/2007/section4.pdf), pp. 6–9, supports the risk-neutral lognormal distribution and option expectations. [Section 5](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf), pp. 1–3, gives the call formula and the starting point for its derivatives.
- Haugh, [*A Brief Review of Derivatives Pricing & Hedging*](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf), pp. 9–10 and 13–17, supports BSM valuation, dividend adjustments and model boundaries.
- Section 1.1.4 of the [[02_Economy/07_金融机构与风险管理/08_操作员如何管理风险暴露|original course chapter]] supplies the parameters and learning sequence. The half-year call and put were recalculated from this page's formulas; adjacent notes explain the original substitution errors.
<!-- bilingual-en:end -->
