---
aliases:
- 自回归条件异方差
- ARCH
tags:
- concept
---
>[!note] **ARCH(1)模型定义：**
> $$
> \epsilon_t = \nu_t \sqrt{\alpha_0 + \alpha_1 \epsilon_{t-1}^2}
> $$
> 或者写作：
> $$
> \begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t} \\
> h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2
> \end{cases}
> $$
> 其中 $\nu_t$ 是一列独立同分布（i.i.d.）的随机变量，满足 $\mathbb{E}(\nu_t)=0$，$\operatorname{Var}(\nu_t)=1$。
> $\alpha_0$ 和 $\alpha_1$ 为常数参数，且 $\alpha_0 > 0$ 确保为正，$0 \leq \alpha_1 < 1$ 保证平稳性。
> $h_t$ 表示 $\varepsilon_t$ 在 $t$ 期的条件方差（即 $h_t = \mathrm{Var}(\varepsilon_t \mid \mathcal{F}_{t-1})$）。
> 这里 $\epsilon_t$ 可以看作我们感兴趣序列（例如资产回报率）的均值已滤除后的**随机扰动**（即残差）。模型表示此残差的方差并不恒定，而是由上一期残差的平方 $\epsilon_{t-1}^2$ 决定。

所以它是一个鞅差分。因为 $t-1$ 期的所有项在算期望的时候都能提出来。

**关键**：在 ARCH(1) 中，$\epsilon_t$ 的 $t-1$ 期条件方差为 $\alpha_0 + \alpha_1 \epsilon_{t-1}^2$。（无条件方差就是对条件方差再取一次期望，得到 $\bar{h} = \frac{\alpha_0}{1-\alpha_1}$。）

**其中**，常数项 $\alpha_0$ 不能被删除。因为如果给 $\epsilon_t=\alpha_1 \epsilon_{t-1}^2$ 两侧同时取期望，最后算出来 $\alpha_1$ 的值一定为 1。

## 相关链接

- 扩展模型：[[GARCH]], [[TARCH]], [[EGARCH]]
- 相关概念：[[Volatility Clustering|波动聚集]], [[Conditional Heteroskedasticity|条件异方差]], [[Historical Volatility|历史波动率]], [[Implied Volatility|隐含波动率]], [[Realized Volatility|已实现波动率]]

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
