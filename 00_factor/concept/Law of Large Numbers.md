---
aliases: [Law of Large Numbers, LLN, 大数定律]
tags: [concept, probability, statistics]
---
# Law of Large Numbers

## 它是什么

大数定律说明：在适当条件下，大量重复观测的样本均值接近共同期望。对 IID 随机变量 $X_1,X_2,\ldots$，令
$$
\bar X_n=\frac1n\sum_{i=1}^nX_i,
\qquad \mu=\mathbb E[X_1].
$$

弱大数定律（WLLN）的结论是
$$
\bar X_n\xrightarrow{p}\mu,
$$
即对每个 $\varepsilon>0$，
$$
\Pr(|\bar X_n-\mu|\ge\varepsilon)\to0.
$$

## 假设必须与证明匹配

- IID 且 $\mathbb E|X_1|<\infty$ 足以得到经典弱大数定律，也足以得到 IID 情形的强大数定律。
- 6.042J 使用 [[Chebyshev Inequality]] 的初等证明时，需要更强的有限方差条件 $\operatorname{Var}(X_1)=\sigma^2<\infty$；此时偏差概率至多为 $\sigma^2/(n\varepsilon^2)$。
- “非独立就一定失败”并不正确；许多弱依赖或遍历序列也有相应大数定律，但需要另行陈述条件。

## 最小直觉

独立样本的和的方差随 $n$ 增长，而平均值除以 $n$，所以样本均值方差缩小为 $\sigma^2/n$。完整初等推导见 [[Weak Law of Large Numbers Proof]]。

## 边界

柯西分布的期望不存在，其 IID 样本均值不会按上述结论稳定到有限均值。大数定律也不等于“有限样本一定接近期望”。

## 关联卡片

- [[IID]]
- [[Convergence in Probability]]
- [[Chebyshev Inequality]]
- [[Confidence Interval]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
