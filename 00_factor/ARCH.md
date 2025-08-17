---
aliases:
  - 自回归条件异方差
---



>[!note] **ARCH(1)模型定义：**
>$$\epsilon_t = \nu_t \sqrt{\alpha_0 + \alpha_1 \epsilon_{t-1}^2}$$或者写作:$$\begin{cases}
\varepsilon_t = \nu_t \sqrt{h_t}  \\
h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2
\end{cases}$$
>其中${\nu_t}$是一列独立同分布（i.i.d.）的随机变量，满足
>$\mathbb{E}(\nu_t)=0, \operatorname{Var}(\nu_t)=1$。
>$\alpha_0$和$\alpha_1$为常数参数，且$\alpha_0 > 0$确保是正的，$0 \leq \alpha_1 < 1$保证平稳性。
>$h_t$表示$\varepsilon_t$在$t$期的条件方差（即$h_t = \mathrm{Var}(\varepsilon_t \mid \mathcal{F}_{t-1})$)
>这里$\epsilon_t$可以看作我们感兴趣序列（例如资产回报率）的均值已滤除后的**随机扰动**（即残差）。模型表示此残差的方差并不恒定，而是由上一期残差的平方$\epsilon_{t-1}^2$决定。

所以==它是一个鞅差分==.因为t-1期的所有项在算期望的时候都能提出来.

<span style="color: yellow;">关键</span>: 在ARCH(1)中，$\epsilon_t$的t-1期条件方差为$\alpha_0 + \alpha_1 \epsilon_{t-1}^2$。(无条件方差就是对条件方差再取一次期望,得到$\bar{h} = \frac{\alpha_0}{1-\alpha_1}$ )

==其中,常数项$\alpha_{0}$不能被删除.因为,如果给$\epsilon_{1}=\alpha_1 \epsilon_{t-1}^2$两侧同时取期望,最后算出来$\alpha_{1}$的值一定为1