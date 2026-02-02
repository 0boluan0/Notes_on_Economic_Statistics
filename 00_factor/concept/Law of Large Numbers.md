---
aliases:
- 大数定律
- LLN
tags:
- proof
- 数学基础
- concept
---
# 大数定律（LLN）

## 定理内容

**弱大数定律（WLLN）**：

设 $\{X_n\}_{n=1}^{\infty}$ 为独立同分布（i.i.d.）随机变量序列，满足：
$$E[|X_n|] < \infty \quad \text{（期望存在有限）}$$

记 $\mu = E[X_n]$ 为共同期望。定义样本均值：
$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$$

则样本均值**依概率收敛**到真实期望：
$$\bar{X}_n \xrightarrow{p} \mu$$

即对于任意 $\varepsilon > 0$：
$$\lim_{n \to \infty} P\{|\bar{X}_n - \mu| < \varepsilon\} = 1$$

**强大数定律（SLLN）**（Kolmogorov）：

若进一步满足 $E[X_n^2] < \infty$（方差有限），则样本均值**几乎处处收敛**到真实期望：
$$\bar{X}_n \xrightarrow{a.s.} \mu$$

即：
$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

## 证明思路

弱大数定律可用**切比雪夫不等式**证明，大大简化了条件。强大数定律的证明较复杂，通常使用 Borel-Cantelli 引理。

## 证明过程（弱大数定律）

### 步骤 1：写出切比雪夫不等式

对于随机变量 $Z$ 和任意 $\varepsilon > 0$：
$$P\{|Z - E[Z]| \geq \varepsilon\} \leq \frac{\text{Var}(Z)}{\varepsilon^2}$$

### 步骤 2：计算样本均值的期望和方差

设 $X_i$ 独立同分布，$E[X_i] = \mu$，$\text{Var}(X_i) = \sigma^2$。

$$
\begin{aligned}
E[\bar{X}_n] &= E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n \mu = \mu \\
\text{Var}(\bar{X}_n) &= \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
\end{aligned}
$$

### 步骤 3：应用切比雪夫不等式

对于任意 $\varepsilon > 0$：
$$
\begin{aligned}
P\{|\bar{X}_n - \mu| \geq \varepsilon\}
&\leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} \\
&= \frac{\sigma^2/n}{\varepsilon^2} \\
&= \frac{\sigma^2}{n \varepsilon^2}
\end{aligned}
$$

### 步骤 4：取极限

$$
\begin{aligned}
\lim_{n \to \infty} P\{|\bar{X}_n - \mu| \geq \varepsilon\}
&\leq \lim_{n \to \infty} \frac{\sigma^2}{n \varepsilon^2} \\
&= 0
\end{aligned}
$$

因此：
$$\lim_{n \to \infty} P\{|\bar{X}_n - \mu| < \varepsilon\} = 1$$

这证明了弱大数定律：
$$\bar{X}_n \xrightarrow{p} \mu$$

### 步骤 5：依概率收敛的定义

$\bar{X}_n \xrightarrow{p} \mu$ 意味着：
$$\forall \varepsilon > 0, \quad \lim_{n \to \infty} P\{|\bar{X}_n - \mu| > \varepsilon\} = 0$$

即随着样本量增大，样本均值与真实期望的距离超过任意小的正数的概率趋近于 0。

## 推广：矩阵形式的大数定律

设 $\{X_i\}_{i=1}^n$ 为独立同分布 $k \times 1$ 随机向量序列，满足：
$$E[X_i] = \mu, \quad \text{Var}(X_i) = \Sigma \text{（有限）}$$

则：
$$\frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{p} \mu$$

更一般地，对于标量函数 $g(X_i)$：
$$\frac{1}{n}\sum_{i=1}^n g(X_i) \xrightarrow{p} E[g(X_i)]$$

**遍历性假设**：

对于平稳过程 $\{X_t\}$，大数定律要求**遍历性**（ergodicity）：
$$\frac{1}{T}\sum_{t=1}^T g(X_t) \xrightarrow{p} E[g(X_t)]$$

遍历性保证了时间平均收敛到空间平均。

## 结论

大数定律揭示了**样本均值的稳定性**：

1. **平均值的收敛**：随着观测数量增加，样本平均趋近于总体平均
2. **统计推断基础**：使从样本推断总体成为可能
3. **经济意义**："平均定律"——长期来看，结果会趋于期望值

**弱收敛与强收敛的区别**：

| 收敛类型 | 符号 | 含义 | 强度 |
|----------|------|------|------|
| **依概率收敛** | $\xrightarrow{p}$ | $P\{|\bar{X}_n - \mu| > \varepsilon\} \to 0$ | 较弱 |
| **几乎处处收敛** | $\xrightarrow{a.s.}$ | $P\{\lim \bar{X}_n = \mu\} = 1$ | 较强 |

几乎处处收敛蕴含依概率收敛，但反之不成立。

**应用**：

1. **OLS 的一致性证明**：利用 LLN 证明样本矩收敛到总体矩
2. **矩估计方法（MM）**：$\hat{\theta}_{MM}$ 满足样本矩方程 $\to$ 总体矩方程
3. **蒙特卡洛积分**：$\frac{1}{n}\sum f(X_i) \approx E[f(X)]$

**违反大数定律的情形**：

1. **期望不存在**：如柯西分布（$X \sim \text{Cauchy}$）
2. **非独立**：强相关序列的平均值可能不收敛
3. **非平稳**：参数随时间变化

## 相关概念
[[00_factor/concept/Central Limit Theorem|中心极限定理]]
[[Convergence in Probability|依概率收敛]]
