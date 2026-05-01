---
aliases:
- Greeks Calculation
- Greeks计算
- 期权Greeks计算
tags:
- procedure
- derivatives
---
# Greeks Calculation

## 这张卡什么时候用

给出 Black-Scholes 参数或期权定价模型，要求计算 Delta、Gamma、Vega、Theta、Rho 时使用。

## 输入

- 标的价格 $S$。
- 行权价 $K$。
- 无风险利率 $r$。
- 剩余期限 $T$。
- 波动率 $\sigma$。
- 看涨或看跌。

## 输出

- [[Delta]]、[[Gamma]]、[[Vega]]、[[Theta]]、[[Rho]]。

## Step 1：计算 $d_1$ 和 $d_2$

$$
d_1=\frac{\ln(S/K)+(r+\sigma^2/2)T}{\sigma\sqrt{T}},
\qquad
d_2=d_1-\sigma\sqrt{T}
$$

所有时间、利率、波动率必须使用同一年度化口径。

## Step 2：计算 Delta

看涨：

$$
\Delta_c=N(d_1)
$$

看跌：

$$
\Delta_p=N(d_1)-1
$$

## Step 3：计算 Gamma

$$
\Gamma=\frac{N'(d_1)}{S\sigma\sqrt T}
$$

同一参数下，看涨和看跌 Gamma 相同。

## Step 4：计算 Vega

$$
Vega=S\sqrt T N'(d_1)
$$

如果题目用“波动率上升 1 个百分点”的口径，记得把公式结果按 0.01 缩放。

## Step 5：计算 Theta

看涨：

$$
\Theta_c=-\frac{SN'(d_1)\sigma}{2\sqrt T}-rKe^{-rT}N(d_2)
$$

看跌：

$$
\Theta_p=-\frac{SN'(d_1)\sigma}{2\sqrt T}+rKe^{-rT}N(-d_2)
$$

确认题目是按年、按月还是按日报告 Theta。

## Step 6：计算 Rho

看涨：

$$
\rho_c=KTe^{-rT}N(d_2)
$$

看跌：

$$
\rho_p=-KTe^{-rT}N(-d_2)
$$

## 检查点

- 看涨 Delta 在 0 到 1 之间；看跌 Delta 在 -1 到 0 之间。
- Gamma 和 Vega 对普通欧式看涨/看跌通常为正。
- Theta 符号最容易因定义方向混乱。
- Greeks 是局部敏感度，不替代完整重新定价。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Greeks Hedging Map]]
- [[Delta Approximation]]
- [[Delta-Gamma Approximation]]
