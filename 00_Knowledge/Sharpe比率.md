---
aliases:
  - Sharpe比率是相容期间内超额收益均值与其标准差之比
  - Sharpe ratio
  - 夏普比率
  - 夏普指数
student_os: knowledge-atom
atom_id: FI-MV-023
atom_type: definition
status: source-checked
part_of:
  - "[[均值—方差投资组合理论.canvas]]"
requires:
  - "[[无风险利率口径]]"
related:
  - "[[Alpha基准依赖性]]"
leads_to:
  - "[[负Sharpe排序边界]]"
  - "[[切点组合]]"
---

# Sharpe比率是相容期间内超额收益均值与其标准差之比
<!-- bilingual-en:start -->
*The Sharpe ratio divides mean excess return by its standard deviation over a consistent period*
<!-- bilingual-en:end -->

Sharpe 比率衡量相对于指定基准的平均差额收益，每单位差额收益波动对应多少回报。本主题采用无风险基准，令 $D=R_P-R_f$；在均值有限且 $0<\operatorname{sd}(D)<\infty$ 时，事前比率为 $S=E(D)/\operatorname{sd}(D)$。若单期 $R_f=r_f$ 确定，则分母可直接换成组合标准差 $\sigma_P$。
<!-- bilingual-en:start -->
The Sharpe ratio measures mean return relative to a specified benchmark per unit of variability in that differential return. Here the benchmark is risk-free, so $D=R_P-R_f$. With finite mean and $0<\operatorname{sd}(D)<\infty$, the ex ante ratio is $S=E(D)/\operatorname{sd}(D)$. For a certain one-period $R_f=r_f$, the denominator can be replaced by portfolio standard deviation $\sigma_P$.
<!-- bilingual-en:end -->

$$
S=\frac{E(R_P)-r_f}{\sigma_P}\qquad(R_f=r_f\text{ 确定},\ \sigma_P>0).
$$

事后样本比率使用逐期差额 $D_t=R_{P,t}-R_{f,t}$ 的样本均值和标准差，而不是把某一次实现收益代入期望收益的位置。若 $R_{f,t}$ 随期变化，应先逐期相减再求标准差。样本标准差可采用下面的 $T-1$ 约定，比较时须统一频率、期间、币种和费用口径。
<!-- bilingual-en:start -->
The historical ratio uses the sample mean and standard deviation of period-by-period differences $D_t=R_{P,t}-R_{f,t}$, not one realized return in place of expected return. If $R_{f,t}$ varies across observations, subtract it period by period before computing the standard deviation. The sample standard deviation may use the following $T-1$ convention; comparisons require matching frequencies, horizons, currencies, and fee conventions.
<!-- bilingual-en:end -->

$$
\bar D=\frac1T\sum_{t=1}^TD_t,\qquad
s_D=\sqrt{\frac1{T-1}\sum_{t=1}^T(D_t-\bar D)^2},\qquad
\widehat S=\frac{\bar D}{s_D}\quad(T\ge2,\ s_D>0).
$$

正、零、负比率分别表示平均超额收益为正、零、负，但负值的排序有 [[负Sharpe排序边界|独立边界]]。若标准差为零，通常的比值未定义：全持无风险资产给出 $0/0$，不应把它记成 Sharpe 为零。非零确定超额收益也不能靠“分母为零”自动生成一个普通有限比率。
<!-- bilingual-en:start -->
Positive, zero, and negative ratios indicate the corresponding sign of mean excess return, but negative values have a [[负Sharpe排序边界|separate ranking limitation]]. With zero standard deviation, the ordinary ratio is undefined: an entirely risk-free investment gives $0/0$, not a Sharpe ratio of zero. A nonzero certain excess return likewise does not produce an ordinary finite ratio through division by zero.
<!-- bilingual-en:end -->

例如期望收益 9%、同期间无风险收益 3%、波动率 20%，则 $S=0.30$。历史高 Sharpe 不保证未来高 Sharpe，也不直接证明 [[Alpha基准依赖性|投资能力]]；该指标本身没有纳入与其他持仓的相关性。跨期换算还依赖收益聚合和序列相关条件，不能无条件乘以年化平方根。
<!-- bilingual-en:start -->
For expected return 9%, a matching risk-free return 3%, and volatility 20%, $S=0.30$. A high historical Sharpe ratio guarantees neither a high future ratio nor [[Alpha基准依赖性|investment skill]], and does not incorporate correlations with other holdings. Time aggregation also depends on compounding and serial dependence, so square-root annualization is not unconditional.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Sharpe（1994），The Sharpe Ratio，作者全文](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm)，The Ratio、Time Dependence、Correlations 及脚注 1：支持事前／事后差额收益定义、样本标准差口径及时间和相关性边界；零分母与 0.30 算例按定义核验。
<!-- bilingual-en:start -->
- [Sharpe (1994), The Sharpe Ratio, author's full text](https://web.stanford.edu/~wfsharpe/art/sr/SR.htm), The Ratio, Time Dependence, Correlations, and endnote 1, supports ex ante/ex post differential-return definitions, the sample convention, and time and correlation limitations. The zero-denominator boundary and the 0.30 example are checked directly from the definition.
<!-- bilingual-en:end -->
