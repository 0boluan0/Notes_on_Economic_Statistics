---
aliases:
- M/M/1 stationary system size
- M/M/1 stationary number in system
- M/M/1 系统内人数分布
student_os: knowledge-atom
atom_id: PROB-CTMC-023
atom_set: continuous-time-markov-chains
atom_type: application
status: source-checked
requires:
- '[[生灭链平稳递推]]'
- '[[可数CTMC稳态存在]]'
related:
- '[[齐次泊松过程]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# The stationary number in an M/M/1 system
<!-- bilingual-en:end -->

*M/M/1 的平稳系统人数*

<!-- bilingual-en:start -->
A single-server queue can be busy without growing forever in distribution. The M/M/1 stationary law answers a concrete question: if we observe this system in equilibrium, what is the probability of finding $0,1,2,\ldots$ customers **in the system**? The count includes the customer being served, not just those waiting in line.
<!-- bilingual-en:end -->

单服务台队列可以持续繁忙，却仍有稳定的概率分布。M/M/1 的平稳人数律回答一个具体问题：在平衡中观察系统，发现**系统内**有 $0,1,2,\ldots$ 位顾客的概率各是多少？人数包括正在接受服务的顾客，不只是在队伍中等待的人。

<!-- bilingual-en:start -->
## What is being modelled
<!-- bilingual-en:end -->

*先明确到达、服务和计数*

<!-- bilingual-en:start -->
Assume arrivals form a [[齐次泊松过程|homogeneous Poisson process]] of rate $0<\lambda<\infty$. There is one server. Service durations are independent identically distributed [[指数分布|exponential random variables]] of rate $0<\mu<\infty$, independent of the arrival process. Their mean is $1/\mu$, not $\mu$. There is unlimited waiting space; customers are served first come, first served, the server works whenever someone is present, and no customer abandons the system.

The two “M”s describe the memoryless arrival and service mechanisms; the “1” counts servers. Let $X_t$ be the total number in the system at time $t$. If $X_t=3$, one customer is being served and two are waiting. An arrival raises $X_t$ by one; a service completion lowers it by one.
<!-- bilingual-en:end -->

假设顾客按速率 $0<\lambda<\infty$ 的[[齐次泊松过程]]到达。系统只有一个服务台；各顾客的服务时长独立同分布，服从速率 $0<\mu<\infty$ 的[[指数分布]]，并与到达过程独立。服务均值是 $1/\mu$，不是 $\mu$。等待空间无限，先到先服务，有顾客时服务台就工作，顾客不会中途离开。

两个“M”表示到达和服务中的无记忆机制，“1”表示一个服务台。记 $X_t$ 为时刻 $t$ 的系统总人数。$X_t=3$ 意味着一人正在服务、两人等待；一次到达使它增加一，一次服务完成使它减少一。

<!-- bilingual-en:start -->
This is a [[生灭链|birth–death chain]] on $\{0,1,2,\ldots\}$ with

$$\lambda_n=\lambda\quad(n\ge0),\qquad
\mu_n=\mu\quad(n\ge1),\qquad\mu_0=0.$$

When the system is nonempty, only one person is being served, so the total departure rate is $\mu$, **not $n\mu$**. The [[指数分布无记忆性|memoryless property]] makes the remaining service time exponential with the same rate. Competition with the next arrival gives total exit rate $\lambda+\mu$ when $n\ge1$; at $0$ the exit rate is only $\lambda$. These rates are bounded, so [[有界率非爆炸|infinitely many changes cannot occur in finite time]].
<!-- bilingual-en:end -->

这构成状态空间 $\{0,1,2,\ldots\}$ 上的[[生灭链]]，出生率始终为 $\lambda$，非空状态的死亡率为 $\mu$，而 $\mu_0=0$。非空时只有一人正在接受服务，所以总离开速率是 $\mu$，**不是 $n\mu$**。[[指数分布无记忆性]]使剩余服务时长仍有同样的指数率。它与下一次到达竞争，故非空时总出口率为 $\lambda+\mu$，空系统时只有 $\lambda$。这些率有统一上界，因此[[有界率非爆炸|有限时间内不会发生无限次改变]]。

<!-- bilingual-en:start -->
## Balance, then sum all the probabilities
<!-- bilingual-en:end -->

*先做平衡，再把全部概率相加*

<!-- bilingual-en:start -->
Let $\pi_n$ denote the [[CTMC平稳分布|stationary probability]] of $n$ customers. Across the edge $n\leftrightarrow n+1$, [[生灭链平稳递推|adjacent balance]] requires

$$\pi_n\lambda=\pi_{n+1}\mu.$$

Set $\rho=\lambda/\mu$, the dimensionless ratio of arrival rate to service capacity. Then $\pi_{n+1}=\rho\pi_n$, so $\pi_n=\pi_0\rho^n$. These relative weights are probabilities only if their total is finite. The [[几何级数|geometric series]] sums to $1/(1-\rho)$ exactly when $0<\rho<1$.
<!-- bilingual-en:end -->

记 $\pi_n$ 为系统有 $n$ 人的[[CTMC平稳分布|平稳概率]]。边 $n\leftrightarrow n+1$ 上的[[生灭链平稳递推|相邻平衡]]要求 $\pi_n\lambda=\pi_{n+1}\mu$。令 $\rho=\lambda/\mu$，这是没有单位的到达率与服务能力之比。递推得到 $\pi_{n+1}=\rho\pi_n$，进而 $\pi_n=\pi_0\rho^n$。这些相对权重只有总和有限才可能归一化；[[几何级数]]在 $0<\rho<1$ 时才有有限和 $1/(1-\rho)$。

> [!important] The equilibrium law · 平衡人数律
> <!-- bilingual-en:start -->
> For the infinite-capacity M/M/1 model with positive finite $\lambda$ and $\mu$, a stationary probability distribution exists exactly when $\rho=\lambda/\mu<1$. It is unique and
>
> $$\pi_n=(1-\rho)\rho^n,\qquad n=0,1,2,\ldots.$$
>
> This is the zero-based convention for a [[几何分布|geometric distribution]], with success parameter $1-\rho$. Its mean is $\rho/(1-\rho)$. In particular, $\pi_0=1-\rho$ is the idle probability and $1-\pi_0=\rho$ is the busy probability. These describe equilibrium time observations of the process, not the proportions just after genuine jumps.
> <!-- bilingual-en:end -->
>
> 对正且有限的 $\lambda,\mu$、无限容量的 M/M/1，平稳概率分布存在当且仅当 $\rho=\lambda/\mu<1$。此时它唯一，且 $\pi_n=(1-\rho)\rho^n$，$n=0,1,2,\ldots$。这属于[[几何分布]]从零开始计数的口径，成功参数是 $1-\rho$，均值为 $\rho/(1-\rho)$。特别地，$\pi_0=1-\rho$ 是空闲概率，$\rho$ 是忙碌概率。这是对平衡过程中时间观察的描述，不是真实跳跃后访问状态的次数比例。

> [!example] Two arrivals per hour, three services per hour · 到达率2、服务率3
> <!-- bilingual-en:start -->
> If $\lambda=2$ and $\mu=3$ per hour, then $\rho=2/3$ and
>
> $$\pi_0=1/3,\qquad\pi_1=2/9,\qquad
> \Pr(X\ge2)=4/9,\qquad \mathbb E[X]=2.$$
>
> Here $X$ is an equilibrium system count. Having at least one person **waiting** means $X\ge2$, not $X\ge1$. With no one waiting, the system may be empty or may have one customer in service, so that probability is $\pi_0+\pi_1=5/9$.
> <!-- bilingual-en:end -->
>
> 若每小时到达率 $\lambda=2$、服务率 $\mu=3$，则 $\rho=2/3$，空系统概率为 $1/3$，恰一人的概率为 $2/9$，至少两人的概率为 $4/9$，系统平均人数为 $2$。这里 $X$ 是平衡人数。至少一人**等待**对应 $X\ge2$，不是 $X\ge1$；无人等待时，可能系统全空，也可能只有一人正在服务，因此概率是 $\pi_0+\pi_1=5/9$。

<!-- bilingual-en:start -->
## What happens at and beyond the boundary
<!-- bilingual-en:end -->

*等号及超过边界时会怎样*

<!-- bilingual-en:start -->
At $\lambda=\mu$, all weights are equal and their infinite sum diverges. The chain is null recurrent: it returns, but its mean calendar return time is infinite. At $\lambda>\mu$, the weights grow and the chain is transient. Its [[嵌入跳链|jump chain]] moves upward with probability $\lambda/(\lambda+\mu)$ and downward with probability $\mu/(\lambda+\mu)$ away from zero, explaining the upward tendency in the latter case. These [[常返与停留时间|recurrence classifications]] agree with the absence of a stationary probability law; swapping the numerical rates above therefore destroys normalizability.

When $\rho<1$, “stable” does not mean there is a fixed upper bound on the number of customers. The stationary law assigns positive probability to every finite count, and its mean increases without bound as $\rho$ approaches $1$ from below. Starting from an arbitrary initial count is also different from already being in equilibrium; the displayed $\pi_n$ is the stationary law.
<!-- bilingual-en:end -->

当 $\lambda=\mu$，所有权重相等，无限和发散。链零常返：会回来，但平均日历回返时间无穷。当 $\lambda>\mu$，权重随人数增长，链暂态。其[[嵌入跳链]]在非零状态向上、向下的概率分别为 $\lambda/(\lambda+\mu)$ 与 $\mu/(\lambda+\mu)$，后一情形的向上趋势由此可见。这些[[常返与停留时间|常返分类]]与没有平稳概率律相一致；把数值例的到达率、服务率交换，就会失去可归一化性。

$\rho<1$ 时，“稳定”也不代表人数有一个固定上限：平稳律给每个有限人数正概率，而且当 $\rho$ 从下方趋近 $1$，其均值会无限增大。从任意初始人数启动也不同于已经处于平衡；上面的 $\pi_n$ 是平稳律。

<!-- bilingual-en:start -->
Finite capacity changes the conclusion. If an M/M/1/$K$ system admits at most $K$ customers **including the one in service**, arrivals at $K$ are blocked, so $\lambda_K=0$. For any positive $\lambda,\mu$, its weights $1,\rho,\ldots,\rho^K$ have a finite sum and can be normalized. At $\rho=1$ the law is uniform on $\{0,\ldots,K\}$. The condition $\rho<1$ belongs to the unlimited-capacity model, not to every single-server queue. If arrivals are absent ($\lambda=0$, $\mu>0$), the separate degenerate model has stationary law concentrated at zero.
<!-- bilingual-en:end -->

有限容量会改变结论。若 M/M/1/$K$ 至多容纳 $K$ 人，**包含服务中的一人**，满员时到达被拒绝，所以 $\lambda_K=0$。任意正的 $\lambda,\mu$ 都使有限权重列 $1,\rho,\ldots,\rho^K$ 可归一化；$\rho=1$ 时，在 $\{0,\ldots,K\}$ 上均匀。$\rho<1$ 是无限容量模型的条件，不是所有单服务台队列的条件。若根本没有到达（$\lambda=0$、$\mu>0$），另一个退化模型的平稳律集中在零。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, *Continuous-Time Markov Chains* (2012), Example 4.3, PDF p.25 / printed p.23](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf#page=25): infinite waiting room, geometric stationary law, its mean and the finite-capacity contrast.
- [Sigman (2018), FIFO M/M/1 example, PDF p.4; balance calculation, PDF pp.16–18](https://www.columbia.edu/~ks20/4106-18-Fall/Notes-CTMC.pdf#page=4): independent arrival/service mechanisms, total system count, holding rates, recursion and the recurrence classification.
- [Cambridge (2015), §3.2, Theorem 3.1 and Example 3.2, PDF pp.34–35](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=34): bounded-rate non-explosion, the $\rho<1$ threshold and mean system size. The $\lambda=2,\mu=3$ probabilities and finite-capacity endpoints are computed directly.
<!-- bilingual-en:end -->

- Whitt（2012）例4.3，PDF25／印刷23页：无限等待空间、几何平稳律、均值与有限容量对照。
- Sigman（2018）FIFO M/M/1 例，PDF4页；平衡计算，PDF16–18页：到达与服务机制独立、系统总人数、停留率、递推和常返分类。
- Cambridge（2015）§3.2、定理3.1与例3.2，PDF34–35页：有界率非爆炸、$\rho<1$ 门槛及平均人数。$\lambda=2,\mu=3$ 的概率和有限容量端点另行直接计算。
