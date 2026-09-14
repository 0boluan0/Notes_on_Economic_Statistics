---
aliases:
  - "在线性头寸和联合正态风险因子近似下，方差—协方差法用损失均值与协方差矩阵解析计算组合 VaR"
  - "Variance-covariance VaR is a delta-normal portfolio-loss approximation"
  - "Delta-normal 参数 VaR"
  - "Delta-normal VaR"
student_os: knowledge-atom
atom_id: RM-VAR-014
atom_set: var-es-backtesting
atom_type: estimation-method
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[VaR定义]]"
  - "[[风险度量口径]]"
  - "[[随机向量]]"
  - "[[协方差矩阵]]"
related:
  - "[[Delta-Gamma价格近似]]"
  - "[[相关系数]]"
  - "[[Copula]]"
  - "[[风险窗口权衡]]"
  - "[[VaR时间缩放]]"
  - "[[VaR采样误差]]"
  - "[[风险模型验证]]"
contrasts_with:
  - "[[历史模拟法]]"
  - "[[风险蒙特卡洛]]"
---

# 在线性头寸和联合正态风险因子近似下，方差—协方差法用损失均值与协方差矩阵解析计算组合 VaR
<!-- bilingual-en:start -->
*Under linear positions and jointly normal risk-factor changes, variance-covariance VaR gives an analytic portfolio-loss quantile*
<!-- bilingual-en:end -->

> [!summary] 先把头寸线性化，再把联合分布压成均值和协方差
> 方差—协方差 VaR，也常称 Delta-normal VaR，用风险因子的均值、协方差和头寸的一阶敏感度直接得到损失分位点。它计算快，但准确性完全依赖线性化、分布和协方差稳定性。

令未来风险因子变动为
$$
\Delta s\sim N(\mu,\Sigma),
$$
组合价值的一阶变化近似为
$$
\Delta V\approx\delta'\Delta s,
$$
其中 $\delta$ 是当前头寸对各风险因子的局部敏感度。采用损失为正的约定，
$$
L=-\Delta V,
$$
于是
$$
\mu_L=-\delta'\mu,
\qquad
\sigma_L^2=\delta'\Sigma\delta.
$$
若 $z_\alpha$ 是标准正态分布的 $\alpha$ 分位点，
$$
\operatorname{VaR}_\alpha(L)
=
\mu_L+z_\alpha\sigma_L.
$$

若把均值近似为零，公式简化为
$$
\operatorname{VaR}_\alpha(L)
\approx
z_\alpha\sqrt{\delta'\Sigma\delta}.
$$
这一步是建模近似，不是 VaR 定义。若模型从收益而不是损失出发，均值符号也必须相应转换。

## “参数法”不能无限外推

方差—协方差法是参数法的一种经典形式，但参数模型并不都等于正态或只使用协方差。采用 $t$ 分布、偏态分布或其他完整参数分布时，分位常数和尾部性质都会改变。因此，Delta-normal 路线不能代表所有参数 VaR，也不能把它们都简化成 $z_\alpha\sigma$。

该近似容易在以下地方失效：

- 期权或含嵌入选择权的头寸有显著 Gamma、Vega 和路径依赖，一阶 Delta 可能接近零但真实损失并不小；
- 收益偏态、厚尾或跳跃时，正态分位低估或错画尾部；
- 危机中相关性和波动率改变，历史协方差矩阵不再代表预测期；
- 风险因子映射遗漏基差、波动率曲面或流动性冲击；
- 参数估计误差和持有期缩放误差仍会传进最终 VaR。

若要保留非线性，可使用 Delta–Gamma 近似或在历史/Monte Carlo 情景下全重估；方法选择取决于头寸结构，而不是只看计算速度。

> [!question]- 自检
> 一个期权组合当前总 Delta 接近零，方差—协方差 VaR 是否可以直接判为接近零？
>
> **答案：** 不可以。Delta-normal 只读取一阶敏感度；Gamma、Vega、跳跃和波动率曲面变化仍可能造成显著损失，应使用经验证的高阶近似或全重估。
>
> <!-- bilingual-en:start -->
> **Self-check:** If an option portfolio's total Delta is close to zero, can its variance–covariance VaR be declared close to zero?
>
> **Answer:** No. Delta-normal VaR reads only first-order sensitivity. Gamma, Vega, jumps, and volatility-surface changes can still cause material losses, requiring a validated higher-order approximation or full revaluation.
> <!-- bilingual-en:end -->

## 来源与核验

- [J.P. Morgan/Reuters, *RiskMetrics—Technical Document*, Fourth Edition](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)：核对风险因子映射、协方差聚合和解析 VaR 方法。
- Hull, *Risk Management and Financial Institutions*：交叉核对 Delta-normal 公式、线性头寸和非正态/非线性边界。
- [[02_Economy/07_金融机构与风险管理/14_VaR参数法和模拟法]]：提供课程例题和三类方法比较的教学语境。
