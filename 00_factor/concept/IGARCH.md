---
aliases:
- 积整GARCH
- Integrated GARCH
- IGARCH
tags:
- 时间序列
- 波动建模
- concept
---
$IGARCH（Integrated GARCH）指满足 \sum \alpha_i + \sum \beta_j = 1 的 GARCH 模型，具有高度持久的波动记忆，长期无条件方差不存在。$

## 定义（以 GARCH(1,1) 为例）

$$
\begin{cases}
\varepsilon_t = \nu_t \sqrt{h_t} \\
h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2 + \beta_1 h_{t-1},\quad \alpha_1+\beta_1=1
\end{cases}
$$

性质：
- 冲击的影响不衰减（单位根型方差过程）；
- 长期方差发散（无固定无条件二阶矩）；
- 适合“极强持久性”的波动序列。

## Connections

- 相关：[[GARCH]]、[[ARCH]]、[[TARCH]]、[[EGARCH]]
- 估计与检验：[[Maximum Likelihood Estimation|极大似然估计]]、[[ARCH Effects Test|ARCH效应检验]]

## $source_notes$

- [[04_波动建模 Modeling Volatility#3.1 IGARCH]]

