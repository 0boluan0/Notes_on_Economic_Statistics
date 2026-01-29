---
aliases:
  - Greeks Calculation
  - 期权Greeks计算
tags:
  - procedure
  - 07_金融机构与风险管理
---

# Greeks计算

## 适用场景

计算期权等衍生品对标的资产价格、波动率、时间等变量的敏感度，用于风险管理、对冲策略设计和组合风险评估。

## 所需数据/条件

- 期权定价模型（Black-Scholes公式或数值定价器）
- 标的资产当前价格 $S$
- 执行价格 $K$
- 无风险利率 $r$
- 到期时间 $T$
- 波动率 $\sigma$
- 期权类型（看涨/看跌）

## 计算步骤

### 步骤 1：计算基础参数

$$ d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} $$
$$ d_2 = d_1 - \sigma\sqrt{T} $$

**注意点**：确保所有参数单位一致（时间单位与利率匹配）。

### 步骤 2：计算Delta

**看涨期权**：
$$ \Delta_c = \frac{\partial c}{\partial S} = N(d_1) $$

**看跌期权**：
$$ \Delta_p = \frac{\partial p}{\partial S} = N(d_1) - 1 $$

其中 $N(\cdot)$ 为标准正态分布累积函数。

**注意点**：看涨期权Delta为正，看跌期权Delta为负；均在[0,1]或[-1,0]范围内。

### 步骤 3：计算Gamma

$$ \Gamma = \frac{\partial^2 V}{\partial S^2} = \frac{N'(d_1)}{S\sigma\sqrt{T}} $$

其中 $N'(\cdot)$ 为标准正态密度函数。

**注意点**：看涨和看跌期权的Gamma相同；对于看涨期权Gamma总是正值。

### 步骤 4：计算Theta

**看涨期权**：
$$ \Theta_c = -\frac{S N'(d_1)\sigma}{2\sqrt{T}} - r K e^{-rT} N(d_2) $$

**看跌期权**：
$$ \Theta_p = -\frac{S N'(d_1)\sigma}{2\sqrt{T}} + r K e^{-rT} N(-d_2) $$

**注意点**：Theta通常为负值（时间衰减），深度实值看跌期权Theta可能为正。

### 步骤 5：计算Vega

$$ \text{Vega} = \frac{\partial V}{\partial \sigma} = S\sqrt{T} N'(d_1) $$

**注意点**：看涨和看跌期权的Vega相同；波动率越高，期权价值越大。

### 步骤 6：计算Rho

**看涨期权**：
$$ \rho_c = \frac{\partial c}{\partial r} = K T e^{-rT} N(d_2) $$

**看跌期权**：
$$ \rho_p = \frac{\partial p}{\partial r} = -K T e^{-rT} N(-d_2) $$

**注意点**：利率上升增加看涨期权价值，降低看跌期权价值。

## 关键公式

**标准正态密度函数**：
$$ N'(x) = \frac{1}{\sqrt{2\pi}} e^{-x^2/2} $$

**Delta-Gamma近似**：
$$ \Delta V \approx \Delta \times \Delta S + \frac{1}{2}\Gamma \times (\Delta S)^2 $$

**Theta-Vega-Rho近似**：
$$ \Delta V \approx \Theta \times \Delta t + \text{Vega} \times \Delta \sigma + \rho \times \Delta r $$

## 常见问题

1. **符号混淆**：Theta通常定义为-∂V/∂t（时间流逝损失），正值表示损失而非收益。
2. **单位不统一**：确保时间、利率、波动率的单位匹配。
3. **深度实值看跌**：此时Theta可能为正，时间流逝反而增加期权价值。
4. **近似误差**：一阶近似在小变动时准确，大幅波动需考虑高阶项。

## 相关概念
[[Delta]]
[[Gamma]]
[[Vega]]
[[Theta]]
[[Rho]]
