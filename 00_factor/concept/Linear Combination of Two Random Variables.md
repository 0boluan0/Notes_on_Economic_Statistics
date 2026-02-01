---
aliases:
- 两个随机变量线性组合
tags:
- 数学
- concept
---

假设有两个随机变量 $X$ 和 $Y$，它们的方差分别为 $\mathrm{Var}(X)$ 和 $\mathrm{Var}(Y)$，协方差为 $\mathrm{Cov}(X, Y)$。考虑它们的线性组合：

$$
Z = aX + bY
$$

其中 $a$ 和 $b$ 是常数。那么 $Z$ 的方差公式为：
$$
\mathrm{Var}(Z) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y) + 2ab\mathrm{Cov}(X, Y)
$$

如果 $X$ 和 $Y$ **独立**，那么 $\mathrm{Cov}(X, Y) = 0$，公式简化为：

$$
\mathrm{Var}(Z) = a^2 \mathrm{Var}(X) + b^2 \mathrm{Var}(Y)
$$
