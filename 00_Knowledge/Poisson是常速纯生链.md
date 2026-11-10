---
aliases:
- Poisson process as a pure-birth CTMC
- Constant-rate pure-birth representation of a Poisson process
- 泊松过程的纯生链表示
student_os: knowledge-atom
atom_id: PROB-CTMC-021
atom_set: continuous-time-markov-chains
atom_type: equivalence
status: source-checked
requires:
- '[[纯生链]]'
- '[[齐次泊松过程]]'
related:
- '[[泊松指数等价]]'
- '[[非齐次泊松过程]]'
- '[[生灭链]]'
leads_to:
- '[[CTMC共同率估计]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Poisson process as a constant-rate pure-birth chain
<!-- bilingual-en:end -->

*Poisson 过程的常速纯生表示*

<!-- bilingual-en:start -->
A [[纯生链|pure-birth chain]] fixes the route $0,1,2,\ldots$ and assigns a waiting clock to each level. A [[齐次泊松过程|homogeneous Poisson process]] is the special case in which **every level uses the same positive rate**. The number of arrivals so far does not speed up or slow down the next clock. This connects a description by waiting times with a description by counts in time intervals.
<!-- bilingual-en:end -->

[[纯生链]]固定路线 $0,1,2,\ldots$，并给每层配一个等待时钟。[[齐次泊松过程]]是其中**每层都使用同一个正速率**的特例：已经到达多少次，不会让下一个时钟变快或变慢。这把“按等待间隔描述”与“按时间区间中的计数描述”连在一起。

> [!example] Two arrivals per hour · 平均每小时两次到达
> <!-- bilingual-en:start -->
> Take a rate $\lambda=2$ per hour and start with count zero. The mean first wait is half an hour. After the count reaches $1$, the next mean wait is still half an hour, and the same holds at $2,3,\ldots$. The chance of no arrival in a half-hour interval is $e^{-1}$; the chance of exactly one is also $e^{-1}$. Equal means do not make the realised waits equal.
> <!-- bilingual-en:end -->
>
> 取速率 $\lambda=2$ 次/小时，从零计数开始。第一次平均等待为半小时；计数到达 $1$ 后，下一次平均仍等半小时，到 $2,3,\ldots$ 时也是如此。半小时内没有到达的概率为 $e^{-1}$，恰好一次的概率也为 $e^{-1}$。平均值相等，不表示每次实际等待相等。

<!-- bilingual-en:start -->
## The complete equivalence
<!-- bilingual-en:end -->

*完整的等价条件*

<!-- bilingual-en:start -->
Fix $0<\lambda<\infty$. Consider the standard right-continuous counting paths starting at $N(0)=0$, with unit upward jumps and no downward jumps. The following describe the same process law: a homogeneous Poisson count of rate $\lambda$; a time-homogeneous pure-birth CTMC with
<!-- bilingual-en:end -->

固定 $0<\lambda<\infty$。考虑从 $N(0)=0$ 出发、右连续、每次只向上跳一且不下降的标准计数路径。以下两种说法描述同一个过程律：速率为 $\lambda$ 的齐次 Poisson 计数；以及满足下列生成率的时间齐次纯生 CTMC：

$$q_{n,n+1}=\lambda,\qquad q_{nn}=-\lambda,\qquad
q_{nj}=0\quad(j\notin\{n,n+1\}),\qquad n=0,1,2,\ldots.$$

<!-- bilingual-en:start -->
The entries are [[生成矩阵约束|generator rates]], with rows as departure states. The full CTMC assumptions matter: unit jumps, the Markov rule, time homogeneity and these rates specify fresh independent exponential waits. Merely saying “the average arrival rate is $\lambda$” does not specify that process. Nonexplosion follows here, since $\sum_{n=0}^{\infty}1/\lambda=\infty$ by the [[纯生链爆炸判据|pure-birth criterion]].
<!-- bilingual-en:end -->

这些条目是以行为出发状态的[[生成矩阵约束|生成率]]。完整的 CTMC 假设很重要：单位跳跃、Markov 规律、时间齐次及上述速率，共同规定了逐次独立的指数等待。仅说“平均到达率为 $\lambda$”没有规定这个过程。这里还自动非爆炸，因为 $\sum_{n=0}^{\infty}1/\lambda=\infty$，满足[[纯生链爆炸判据]]。

<!-- bilingual-en:start -->
To see the link, call the wait at count $n$ $H_n$. The pure-birth construction makes $H_0,H_1,\ldots$ [[独立同分布|independent and identically distributed]] with law $\operatorname{Exp}(\lambda)$. At a fixed observation time, the unfinished wait has the same remaining distribution by [[指数分布无记忆性|exponential memorylessness]], and the unused later waits are independent of the past. Thus the future added count is a fresh copy independent of the earlier record. The [[泊松指数等价|Poisson–exponential equivalence]] then gives, for $0\le s<t$,
<!-- bilingual-en:end -->

具体地，把计数为 $n$ 时的等待记为 $H_n$。纯生构造使 $H_0,H_1,\ldots$ [[独立同分布]]，共同服从 $\operatorname{Exp}(\lambda)$。在一个固定观测时点，未结束的等待由[[指数分布无记忆性]]保留相同的剩余分布；尚未使用的后续等待又与过去独立。因此，未来新增计数是一个与既往记录独立的新副本。[[泊松指数等价]]于是给出：对 $0\le s<t$，

$$\Pr\{N(t)-N(s)=k\}
=e^{-\lambda(t-s)}\frac{[\lambda(t-s)]^k}{k!},\qquad k=0,1,2,\ldots.$$

<!-- bilingual-en:start -->
Counts in disjoint intervals are mutually independent. The displayed probability is the Poisson law, with parameter $\lambda(t-s)$, the mean count in that interval; $k!$ denotes the product $1\cdot2\cdots k$, with $0!=1$. Equal-length intervals have the same law. Conversely, a homogeneous Poisson process has independent exponential interarrival times and unit jumps, so its next-state rate at every count is exactly $\lambda$. This is equality of joint process laws, not just agreement of one-time distributions.
<!-- bilingual-en:end -->

不相交区间中的计数相互独立。上式是参数为 $\lambda(t-s)$ 的 Poisson 分布，这个参数等于该区间的平均到达数；$k!$ 表示乘积 $1\cdot2\cdots k$，约定 $0!=1$。等长区间具有相同分布。反过来，齐次 Poisson 过程具有独立指数到达间隔和单位跳跃，所以每个计数状态的下一步速率恰好都是 $\lambda$。这是联合过程律的等价，不仅是单时点分布相同。

<!-- bilingual-en:start -->
## Which changes preserve this description?
<!-- bilingual-en:end -->

*改变什么条件后，模型会不同？*

<!-- bilingual-en:start -->
If the initial level is $n_0$ instead of zero while all rates remain $\lambda$, then $X_t-n_0$ is the homogeneous Poisson count. The level $X_t$ itself has a shifted distribution. If $\lambda=0$, every level is absorbing and the count started at zero stays zero. One may include this as a **degenerate rate-zero Poisson convention**, but it uses an infinite next waiting time, not an ordinary finite [[指数分布|exponential distribution]]. The positive-rate definition and equivalence above do not need that convention.
<!-- bilingual-en:end -->

若初始层为 $n_0$ 而非零，各层速率仍为 $\lambda$，则 $X_t-n_0$ 才是从零开始的齐次 Poisson 计数，层数 $X_t$ 的分布整体平移。若 $\lambda=0$，每层都是吸收态，从零开始的计数永远为零。可以另行把它纳入**零率 Poisson 退化约定**，但此时下一次等待为无穷，而不是普通取有限值的[[指数分布]]。上述正率定义及等价不依赖这个扩展约定。

<!-- bilingual-en:start -->
If the rates instead depend on the current count, the model can remain time homogeneous without being Poisson. For example, $\lambda_n=(n+1)\lambda$ makes the mean next wait at counts $0,1,2$ equal to $1/\lambda$, $1/(2\lambda)$ and $1/(3\lambda)$. Its reciprocal-rate series is harmonic, so it is nonexplosive. But the chance of no next birth in duration $h$ after reaching count $n$ is $e^{-(n+1)\lambda h}$, which depends on that count. These conditional future laws cannot be those of the constant-rate Poisson model. The rates $\lambda_n=n+2$ have the same distinction.
<!-- bilingual-en:end -->

若速率依赖当前计数，模型仍可以时间齐次，却不再是 Poisson。例如 $\lambda_n=(n+1)\lambda$ 使计数为 $0,1,2$ 时的下一次平均等待分别为 $1/\lambda$、$1/(2\lambda)$、$1/(3\lambda)$。它的倒数率级数为调和型，所以不爆炸；但到达计数 $n$ 后，经过 $h$ 仍没有下一次出生的概率是 $e^{-(n+1)\lambda h}$，依赖当前计数。这些条件未来分布不是常速 Poisson 模型的分布。$\lambda_n=n+2$ 也具有同样的区别。

<!-- bilingual-en:start -->
A different modification is a deterministic rate depending on calendar time, $\lambda(t)$. With the independent-increment Poisson construction and a nonnegative locally integrable intensity, this gives a [[非齐次泊松过程|nonhomogeneous Poisson process]]: the count over $(s,t]$ has parameter $\int_s^t\lambda(u)\,du$. Increments remain independent, but equal-length intervals need not have the same distribution. This is calendar-time dependence, rather than dependence on the current count.
<!-- bilingual-en:end -->

另一种修改是使用依赖日历时间的确定性速率 $\lambda(t)$。在独立增量的 Poisson 构造下，若强度非负且局部可积，就得到[[非齐次泊松过程]]：区间 $(s,t]$ 的计数参数为 $\int_s^t\lambda(u)\,du$。增量仍独立，但等长区间未必同分布。这是日历时间依赖，而不是当前计数依赖。

> [!warning] Marginal counts do not determine a process · 边际计数不能决定整个过程
> <!-- bilingual-en:start -->
> Knowing only $N(t)\sim\operatorname{Poisson}(\lambda t)$ for each $t$ does not establish independent increments, the Markov property, or the constant-rate pure-birth representation. Those one-time statements leave the dependence between times unspecified. Check the path and joint-law conditions, as in [[边际分布不定时序依赖|the marginal-versus-dependence distinction]], before using this equivalence.
> <!-- bilingual-en:end -->
>
> 仅知道每个 $t$ 都有 $N(t)\sim\operatorname{Poisson}(\lambda t)$，不能证明独立增量、Markov 性或常速纯生表示。逐时点的陈述没有指定不同时点之间的依赖。应用等价前，应检查路径与联合律条件，参见[[边际分布不定时序依赖|边际分布与时序依赖的区别]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge Applied Probability (2015), §1.4, Theorems 1.4 and 1.6, PDF pp.6–8; §1.5 pp.10–12](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=6): independent exponential construction, Poisson increment characterization and state-dependent birth rates.
- [MIT 6.262 Lecture 5, PDF pp.1–3 / slides 2–5, and PDF pp.6–7 / slides 12–13](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/9f76a2356ac8abdd15a8d06c151f723b_MIT6_262S11_lec05.pdf#page=1): arrival epochs, independent exponential intervals, restarting at a fixed time, independent stationary increments and the Poisson count formula; slides 12–13 support the integrated-intensity formula for a deterministic time-varying rate. The zero-rate extension and shifted initial count are explicitly separated conventions; the numerical example follows from the formula.
<!-- bilingual-en:end -->

- Cambridge 第1.4节定理1.4与1.6（PDF第6–8页）及第1.5节（第10–12页）核对独立指数构造、Poisson 增量刻画和状态依赖出生率。
- MIT 第5讲 PDF第1–3页、幻灯片2–5核对到达时刻、独立指数间隔、固定时点后的重新开始、平稳独立增量与计数公式；PDF第6–7页、幻灯片12–13核对确定性时变强度的积分参数。零率与平移起点在正文中作为单独约定说明，数值例直接代入公式。
