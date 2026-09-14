---
aliases:
  - "M-M-1 队列的系统内人数仅在 lambda 小于 mu 时有几何平稳分布"
  - M/M/1 stationary system size
  - M/M/1 stationary number in system
  - M/M/1 系统内人数分布
student_os: knowledge-atom
atom_id: PROB-CTMC-023
atom_set: continuous-time-markov-chains
atom_type: application
status: source-checked
mastery_state: unassessed
requires:
  - "[[生灭链平稳递推]]"
  - "[[可数CTMC稳态存在]]"
related:
  - "[[齐次泊松过程]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# M-M-1 队列的系统内人数仅在 lambda 小于 mu 时有几何平稳分布
<!-- bilingual-en:start -->
*The number of customers in an M/M/1 system has a geometric stationary distribution only when $\lambda<\mu$*
<!-- bilingual-en:end -->

> [!summary] 系统内人数包含正在服务的顾客
> M/M/1 有 Poisson 到达率 $\lambda>0$、一个指数服务率 $\mu>0$ 的服务器和无限等待空间。令 $X_t$ 为系统内总人数，包括正在服务和正在等待的顾客，则
> $$
> q_{n,n+1}=\lambda\quad(n\ge0),
> \qquad
> q_{n,n-1}=\mu\quad(n\ge1).
> $$
> 令 $\rho=\lambda/\mu$。平稳候选满足 $\pi_n=\pi_0\rho^n$，可归一化当且仅当 $\rho<1$；此时
> $$
> \pi_n=(1-\rho)\rho^n.
> $$
> <!-- bilingual-en:start -->
> The geometric weights form a stationary probability law exactly when the arrival rate is below the service rate.
> <!-- bilingual-en:end -->

该链的出口率至多 $\lambda+\mu$，所以不会爆炸。若 $\lambda=\mu$，embedded random walk 零常返，几何权重全为常数而不可归一化；若 $\lambda>\mu$，链向上暂态。由[[可数CTMC稳态存在]]，只有 $\lambda<\mu$ 的正常返情形存在平稳概率。

有限容量 M/M/1/K 的状态空间只有 $\{0,\ldots,K\}$，对任意正 $\lambda,\mu$ 都存在平稳分布，不能把无限等待空间的 $\lambda<\mu$ 条件原样套过去。

> [!example] 数值检查
> 若 $\lambda=2$、$\mu=3$，则 $\rho=2/3$，空系统概率 $\pi_0=1/3$，且
> $$
> \pi_n=\frac13\left(\frac23\right)^n.
> $$
> 若交换两率，形式权重随 $n$ 增长，无法归一化。

> [!question]- 自检
> 为什么只写递推 $\pi_{n+1}=\rho\pi_n$ 还不能证明无限容量 M/M/1 有平稳分布？
>
> **答案：** 还必须检查几何级数可归一化；这正要求 $\rho<1$，并对应链正常返。

## 来源与核验

- [Ward Whitt, CTMC notes, Example 4.3](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：核对 M/M/1 生灭生成率、$\rho<1$ 条件与几何平稳分布。
- [Columbia CTMC course notes](https://www.columbia.edu/~ks20/4106-18-Fall/Notes-CTMC.pdf)：核对 M/M/1 holding rates、embedded random walk 与竞争时钟构造。
