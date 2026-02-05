---
aliases:
- 协整定理
- Granger representation theorem
- Engle-Granger representation theorem
- Cointegration theorem
- Cointegration
tags:
- proof
- 06_时间序列分析
- concept
---
# 协整定理（Granger 表示定理）

## 定理内容

**协整定义**（Engle-Granger, 1987）：

设 n 维向量时间序列 $x_t = (x_{1t}, x_{2t}, \ldots, x_{nt})'$，其中每个分量都是 $I(1)$ 过程（一阶单整）。如果存在 $n \times 1$ 的常数向量 $\beta$（$\beta \neq 0$），使得线性组合：
$$
e_t = \beta' x_t
$$
是 $I(0)$ 平稳过程，则称 $x_t$ 的各分量是**协整的**，记为 $x_t \sim CI(1,1)$。

**Granger 表示定理**：

如果一组 $I(1)$ 变量存在 $r$ 个独立的协整关系（$0 < r \leq n-1$），则可以将其 $\text{VAR}(p)$ 模型改写为误差修正模型（VECM）形式，二者等价。

## 证明思路

从 VAR(p) 模型出发，通过差分变换引入 $\pi$ 矩阵，利用秩分解证明 $\pi$ 矩阵的秩等于协整关系的数量。

## 证明过程

### 步骤 1：写出 VAR(p) 模型

考虑 n 个 $I(1)$ 变量的 p 阶向量自回归模型：
$$
x_t = A_1 x_{t-1} + A_2 x_{t-2} + \cdots + A_p x_{t-p} + \varepsilon_t
$$
其中 $A_i$ 为 $n \times n$ 系数矩阵，$\varepsilon_t$ 为 $n \times 1$ 白噪声向量。

### 步骤 2：引入差分和滞后算子

使用滞后算子 $L$（$L^i x_t = x_{t-i}$）：
$$
x_t = (A_1 L + A_2 L^2 + \cdots + A_p L^p) x_t + \varepsilon_t
$$
移项得：
$$
(I - A_1 L - A_2 L^2 - \cdots - A_p L^p) x_t = \varepsilon_t
$$

### 步骤 3：展开为差分形式

利用恒等式 $(1 - L) = \Delta$，将算子多项式重新组合：
$$
I - A_1 L - A_2 L^2 - \cdots - A_p L^p = \pi_0(I - L) + \pi (I - L) + \sum_{i=1}^{p-1} \pi_i (L^i - L^{i+1})
$$
其中：
- $\pi_0 = I - \sum_{i=1}^p A_i$
- $\pi = -\pi_0 = \sum_{i=1}^p A_i - I$

整理后得到误差修正形式：
$$
\Delta x_t = \pi_0 + \pi x_{t-1} + \sum_{i=1}^{p-1} \pi_i \Delta x_{t-i} + \varepsilon_t
$$

### 步骤 4：分析 $\pi$ 矩阵的秩

关键在于 $\pi x_{t-1}$ 这一项。令 $\text{rank}(\pi) = r$。

**情形 1：$r = 0$**

$\pi$ 为零矩阵，VECM 退化为纯差分形式：
$$
\Delta x_t = \pi_0 + \sum_{i=1}^{p-1} \pi_i \Delta x_{t-i} + \varepsilon_t
$$

这意味着没有协整关系，所有变量都是独立的随机游走。

**情形 2：$r = n$**

$\pi$ 满秩，$x_{t-1}$ 可以用 $\Delta x_{t-1}, \ldots$ 表示，意味着 $x_t$ 本身平稳，与 $I(1)$ 假设矛盾。

**情形 3：$0 < r < n$**

$\pi$ 的秩为 $r$，可以进行秩分解：
$\pi = \alpha \beta'$

其中：
- $\alpha$ 为 $n \times r$ 矩阵（调整系数矩阵）
- $\beta$ 为 $n \times r$ 矩阵（协整向量矩阵）

此时 VECM 为：
$$
\Delta x_t = \pi_0 + \alpha (\beta' x_{t-1}) + \sum_{i=1}^{p-1} \pi_i \Delta x_{t-i} + \varepsilon_t
$$

### 步骤 5：证明 $\beta$ 为协整向量

项 $\beta' x_{t-1}$ 是 $r \times 1$ 向量。由于 $\Delta x_t$ 是 $I(0)$ 平稳的（差分阶数减 1），而 $\sum_{i=1}^{p-1} \pi_i \Delta x_{t-i}$ 也是 $I(0)$，因此 $\alpha (\beta' x_{t-1})$ 必须是平稳的。

因为 $\alpha$ 满秩（$n \times r$，秩为 $r$），这意味着：
$\beta' x_t$ 是 $I(0)$ 平稳向量。

这正是协整的定义！$\beta$ 的每一列都是一个协整向量。

### 步骤 6：确定协整关系数量

$\pi$ 矩阵的秩 $r$ 等于独立的协整关系数量。

- 如果 $r = 1$，存在 1 个协整关系（如 2 个变量共享 1 个长期均衡）
- 如果 $r > 1$，存在 $r$ 个独立的协整关系
- $n - r$ 是系统中独立的共同随机趋势数量

### 步骤 7：误差修正机制

VECM 解释了协整的经济学含义：

$$
\Delta x_t = \pi_0 + \alpha e_{t-1} + \sum_{i=1}^{p-1} \pi_i \Delta x_{t-i} + \varepsilon_t
$$

其中 $e_t = \beta' x_t$ 是**长期均衡误差**。

- 如果 $e_{t-1} > 0$（变量组合偏离均衡上方），调整系数 $\alpha$ 将拉动 $\Delta x_t$ 向负方向调整
- 如果 $e_{t-1} < 0$，调整系数 $\alpha$ 将拉动 $\Delta x_t$ 向正方向调整

这体现了误差修正机制：当系统偏离长期均衡时，会逐步回到均衡。

## 结论

**Granger 表示定理的核心结论**：

1. **协整等价于误差修正形式**：存在协整关系当且仅当可以写成 VECM 形式
2. **秩与协整关系一一对应**：$\pi$ 矩阵的秩 $r$ 等于协整关系数量
3. **经济含义明确**：VECM 同时刻画了短期动态（差分项）和长期（误差修正项）

**检验协整的两种方法**：

1. **Engle-Granger 两步法**：估计长期关系，检验残差平稳性
2. **Johansen 方法**：基于 VAR 的最大似然估计，通过 $\pi$ 矩阵的特征值检验秩 $r$

## 相关概念
[[Engle-Granger Two-Step Test|EG两步检验法]]
[[Augmented Dickey-Fuller Test|ADF检验]]
[[ARMA|ARMA模型]]
