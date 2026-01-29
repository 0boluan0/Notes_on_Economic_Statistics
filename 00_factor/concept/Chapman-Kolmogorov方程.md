---
aliases:
  - Chapman-Kolmogorov equation
  - C-K 方程
  - CK 方程
tags:
  - proof
  - 05_随机过程
---

# Chapman-Kolmogorov 方程

## 定理内容

对于时间齐次的马尔可夫链 $\{X_n, n \geq 0\}$，状态空间为 $S = \{0, 1, 2, \ldots\}$，一步转移概率为：
$$p_{ij} = P\{X_{n+1} = j \mid X_n = i\}$$

$m$ 步转移概率为：
$$p_{ij}^{(m)} = P\{X_{n+m} = j \mid X_n = i\}$$

**Chapman-Kolmogorov 方程**（C-K 方程）：
$$p_{ij}^{(m+n)} = \sum_{k \in S} p_{ik}^{(m)} p_{kj}^{(n)}$$

对于任意状态 $i, j \in S$ 和任意正整数 $m, n$。

**矩阵形式**：
$$P^{(m+n)} = P^{(m)} \cdot P^{(n)}$$

其中 $P^{(m)} = (p_{ij}^{(m)})$ 为 $m$ 步转移概率矩阵。

## 证明思路

从条件概率的定义出发，利用马尔可夫性质（无后效性）和全概率公式，将 $m+n$ 步转移分解为中间状态 $k$ 的转移。

## 证明过程

### 步骤 1：定义多步转移概率

根据定义：
$$p_{ij}^{(m+n)} = P\{X_{n+m+n} = j \mid X_0 = i\}$$

### 步骤 2：引入中间状态

考虑在第 $m$ 步时的状态 $X_m = k$。利用全概率公式：
$$
\begin{aligned}
P\{X_{m+n} = j \mid X_0 = i\}
&= \sum_{k \in S} P\{X_{m+n} = j, X_m = k \mid X_0 = i\} \\
&= \sum_{k \in S} P\{X_{m+n} = j \mid X_m = k, X_0 = i\} \cdot P\{X_m = k \mid X_0 = i\}
\end{aligned}
$$

### 步骤 3：应用马尔可夫性质

**马尔可夫性（无后效性）**：未来状态只依赖于当前状态，与过去历史无关。

因此：
$$P\{X_{m+n} = j \mid X_m = k, X_0 = i\} = P\{X_{m+n} = j \mid X_m = k\}$$

由于过程是时间齐次的（转移概率不随时间变化）：
$$P\{X_{m+n} = j \mid X_m = k\} = P\{X_n = j \mid X_0 = k\} = p_{kj}^{(n)}$$

### 步骤 4：代入并化简

将马尔可夫性结果代入：
$$
\begin{aligned}
p_{ij}^{(m+n)}
&= \sum_{k \in S} P\{X_{n} = j \mid X_0 = k\} \cdot P\{X_m = k \mid X_0 = i\} \\
&= \sum_{k \in S} p_{kj}^{(n)} \cdot p_{ik}^{(m)} \\
&= \sum_{k \in S} p_{ik}^{(m)} p_{kj}^{(n)}
\end{aligned}
$$

### 步骤 5：矩阵形式证明

设 $P^{(m)} = (p_{ij}^{(m)})$ 为 $m$ 步转移概率矩阵。

根据矩阵乘法定义：
$$(P^{(m)} \cdot P^{(n)})_{ij} = \sum_{k \in S} p_{ik}^{(m)} p_{kj}^{(n)}$$

由 C-K 方程：
$$(P^{(m+n)})_{ij} = \sum_{k \in S} p_{ik}^{(m)} p_{kj}^{(n)}$$

因此：
$$P^{(m+n)} = P^{(m)} \cdot P^{(n)}$$

### 步骤 6：推论

**推论 1**：$n$ 步转移概率矩阵
$$P^{(n)} = P^n$$

即 $n$ 步转移概率矩阵等于一步转移概率矩阵的 $n$ 次幂。

**推论 2**：Chapman-Kolmogorov(方程的物理意义**

从状态 $i$ 经过 $m+n$ 步到达状态 $j$，等价于：
1. 先从 $i$ 经过 $m$ 步到达某个中间状态 $k$
2. 再从 $k$ 经过 $n$ 步到达 $j$
3. 对所有可能的中间状态 $k$ 求和

## 结论

Chapman-Kolmogorov 方程是马尔可夫链理论的基石，其重要性体现在：

1. **计算多步转移概率**：通过矩阵幂运算计算任意步数的转移概率
2. **稳态分布求解**：通过 $P^{(n)} = P^n$ 的极限分析稳态分布
3. **状态分类**：帮助分析状态的可达性、常返性、周期性等

**连续时间马尔可夫链的 C-K 方程**：

对于连续时间马尔可夫链，设 $p_{ij}(t)$ 为 $t$ 时刻从状态 $i$ 转移到 $j$ 的概率：

$$p_{ij}(s+t) = \sum_{k \in S} p_{ik}(s) p_{kj}(t)$$

由此可导出 Kolmogorov 前向/后向微分方程：
- 前向方程：$\frac{d}{dt} p_{ij}(t) = \sum_{k} p_{ik}(t) q_{kj}$
- 后向方程：$\frac{d}{dt} p_{ij}(t) = \sum_{k} q_{ik} p_{kj}(t)$

其中 $Q = (q_{ij})$ 为生成矩阵（或强度矩阵）。

## 相关概念
[[马尔可夫过程]]
[[平稳随机过程]]
