---
aliases:
- Time-homogeneous CTMC transition function
- Homogeneous CTMC transition function
- CTMC 齐次转移函数
- 时间齐次 CTMC
student_os: knowledge-atom
atom_id: PROB-CTMC-037
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
- '[[CTMC定义]]'
related:
- '[[DTMC时间齐次转移核]]'
leads_to:
- '[[CTMC指数停留时间]]'
- '[[CTMC转移半群]]'
- '[[生成矩阵约束]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Time-homogeneous CTMC transition function
<!-- bilingual-en:end -->
*CTMC 的时间齐次转移函数*

<!-- bilingual-en:start -->
A **time-homogeneous transition function** gives the same forecast whenever the starting state and the elapsed time are the same. For a working device, the probability of failing within the next hour is then the same whether that hour begins at 8 a.m. or 8 p.m. A two-hour forecast can still differ from a one-hour forecast: homogeneity removes the starting date, not the interval length.
<!-- bilingual-en:end -->

**时间齐次转移函数**表示：起始状态相同、经过时长相同时，就使用同一转移概率。对正常设备而言，接下来一小时内故障的概率，不论这一小时从早上8点还是晚上8点开始，都相同。两小时的预测仍可以不同于一小时的预测；齐次性去掉的是起始日期，不是区间长度。

<!-- bilingual-en:start -->
Start with a [[CTMC定义|continuous-time Markov chain]] $X_t$ on a finite or countable state set $S$. The [[Markov性|Markov property]] already allows the earlier state history to be discarded once the current state is known. In general, however, a transition still needs **two clock times**. Write $p_{ij}(s,t)$ for the probability of state $j$ at time $t$ when starting in state $i$ at time $s$, with $s\le t$. The first index is the departure state, the second the destination.
<!-- bilingual-en:end -->

从有限或可数状态集 $S$ 上的[[CTMC定义|连续时间 Markov 链]] $X_t$ 出发。[[Markov性]]已经允许在知道当前状态后，丢弃更早的状态历史。但一般的转移仍需要**两个日历时点**。用 $p_{ij}(s,t)$ 表示在时点 $s$ 从状态 $i$ 出发，到时点 $t$ 处于状态 $j$ 的概率，其中 $s\le t$。第一个下标是出发状态，第二个是到达状态。

> [!abstract] One elapsed-time argument · 只留下经过时长
> <!-- bilingual-en:start -->
> Time homogeneity means there are functions $p_{ij}(h)$, for elapsed times $h\ge0$, such that
> <!-- bilingual-en:end -->
>
> 时间齐次是说：存在以经过时长 $h\ge0$ 为自变量的函数 $p_{ij}(h)$，使得
>
> $$p_{ij}(s,s+h)=p_{ij}(h)\quad\text{for all }s,h\ge0.$$
>
> <!-- bilingual-en:start -->
> Under a specified starting law, this gives $\Pr(X_{s+h}=j\mid X_s=i)=p_{ij}(h)$ whenever $\Pr(X_s=i)>0$. A model can also specify a transition row from a currently unreachable state. That row is part of the model, not a ratio obtained by dividing by zero.
> <!-- bilingual-en:end -->
>
> 在某个指定的起始概率律下，只要 $\Pr(X_s=i)>0$，就有 $\Pr(X_{s+h}=j\mid X_s=i)=p_{ij}(h)$。模型也可以规定从当前不可达状态出发的转移行；这是模型本身的规定，不是把两个零概率相除得到的结果。

<!-- bilingual-en:start -->
Collect the entries into $P(h)=(p_{ij}(h))$, using the [[Markov矩阵左右约定|row convention]]: each row fixes the starting state. In a model that retains all probability in $S$, every entry is nonnegative and every row sums to one. At $h=0$, the state has not changed, so $p_{ii}(0)=1$ and $p_{ij}(0)=0$ for $i\ne j$. Thus $P(0)=I$, the [[单位矩阵|identity matrix]] (or the identity kernel on a countable set). If an explosive process is killed and its cemetery state is omitted, rows on $S$ can instead sum to less than one; [[最小CTMC|the missing mass is its probability of having been killed]].
<!-- bilingual-en:end -->

采用[[Markov矩阵左右约定|行约定]]，把条目排成 $P(h)=(p_{ij}(h))$：每一行固定出发状态。若模型的全部概率都保留在 $S$ 中，元素非负、每行和为一。$h=0$ 时状态没有改变，所以 $p_{ii}(0)=1$，而 $i\ne j$ 时 $p_{ij}(0)=0$；即 $P(0)=I$，这里 $I$ 是[[单位矩阵]]，在可数集合上则是相应的恒等核。若爆炸过程被杀死，而矩阵没有列入墓地状态，$S$ 上的行和就可能小于一；[[最小CTMC|缺失部分是已经被杀死的概率]]。

<!-- bilingual-en:start -->
## A device that fails once
<!-- bilingual-en:end -->
*只发生一次故障的设备*

<!-- bilingual-en:start -->
Suppose a working device fails after an [[指数分布|exponential waiting time]] of rate $\lambda>0$, and then stays failed. The defining survival probability is $\Pr(H>h)=e^{-\lambda h}$, where $H$ is the waiting time. The rate has units of inverse time; $\lambda h$ is dimensionless. The [[指数分布无记忆性|memoryless property]] says that, conditional on still working, the remaining wait has that same law. With states ordered as working $W$, failed $F$,
<!-- bilingual-en:end -->

设正常设备等待一个速率为 $\lambda>0$ 的[[指数分布|指数等待时间]]后故障，以后一直保持故障。等待时间 $H$ 的生存概率是 $\Pr(H>h)=e^{-\lambda h}$。速率的单位是时间的倒数，$\lambda h$ 没有单位。[[指数分布无记忆性|无记忆性]]表示：已知设备仍然正常，剩余等待仍服从同一个规律。按正常 $W$、故障 $F$ 排列状态，有

$$P(h)=\begin{pmatrix}
e^{-\lambda h}&1-e^{-\lambda h}\\
0&1
\end{pmatrix}.$$

<!-- bilingual-en:start -->
The first row separates “still working” from “already failed”; these probabilities add to one. The second row records that a failed device never recovers. If one hour of survival has probability $1/2$, two hours have survival probability $(1/2)^2=1/4$, and failure probability $3/4$. Those numbers do not depend on which clock time the interval starts at.
<!-- bilingual-en:end -->

第一行区分“仍正常”和“已经故障”，两者概率和为一；第二行表示故障后不会恢复。若存活一小时的概率为 $1/2$，两小时存活概率就是 $(1/2)^2=1/4$，故障概率为 $3/4$。这些数字不依赖从哪个日历时刻开始观察。

> [!warning] A fixed rule can produce a changing distribution · 固定规则不等于分布固定
> <!-- bilingual-en:start -->
> Starting surely at $W$ gives the probability row $\mu(0)=(1,0)$. At time $h$, it is $\mu(h)=\mu(0)P(h)=(e^{-\lambda h},1-e^{-\lambda h})$. Thus the transition function is homogeneous while the state distribution changes. This is the same [[齐次不等于平稳|rule-versus-distribution distinction]] encountered in discrete time; continuous time does not remove it.
> <!-- bilingual-en:end -->
>
> 若确定从 $W$ 出发，初始概率行为 $\mu(0)=(1,0)$；时点 $h$ 的概率行则为 $\mu(h)=\mu(0)P(h)=(e^{-\lambda h},1-e^{-\lambda h})$。所以转移函数时间齐次，状态分布却随时间变化。这与离散时间中的[[齐次不等于平稳|规则和分布的区别]]相同，不会因时间连续而消失。

<!-- bilingual-en:start -->
## When the clock matters
<!-- bilingual-en:end -->
*日历时点仍然重要时*

<!-- bilingual-en:start -->
A device can instead face rate $\lambda$ during the day and rate $2\lambda$ at night, with no repair. For an hour lying entirely within one regime, the survival probabilities are $e^{-\lambda}$ and $e^{-2\lambda}$ when hours are the time unit. The model can still be Markov: the known clock and the current working/failed state determine the forecast. It is not time homogeneous, because equal-duration intervals use different rules. An interval crossing the day/night boundary must account for both parts.
<!-- bilingual-en:end -->

也可以规定：白天故障速率为 $\lambda$，夜间为 $2\lambda$，仍然不维修。以小时为单位时，完全落在白天或夜间的一小时，其存活概率分别为 $e^{-\lambda}$ 和 $e^{-2\lambda}$。模型仍可满足 Markov 性：已知时钟和当前正常/故障状态，便能确定预测；但它不时间齐次，因为等长区间使用不同规则。跨过昼夜分界的区间，则要分别考虑两部分。

<!-- bilingual-en:start -->
A homogeneous CTMC needs a whole family $\{P(h):h\ge0\}$, not just one fixed “one-step” matrix. The [[CTMC转移半群|semigroup identity]] explains how two intervals compose. Short-time rates and [[生成矩阵约束|generator matrices]] are another description under the relevant regularity assumptions; they are not part of the definition of homogeneity itself.
<!-- bilingual-en:end -->

齐次 CTMC 需要整族 $\{P(h):h\ge0\}$，而不只是一个固定的“一步”矩阵。两段时间怎样拼接，由[[CTMC转移半群|半群恒等式]]说明；短时速率与[[生成矩阵约束|生成矩阵]]则是在适当正则条件下的另一种描述，并不属于时间齐次的定义本身。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [MIT 6.436J, Lecture 24, Definition 2, PDF p.5](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=5): the separation of Markov dependence from calendar-time homogeneity.
- [Cambridge Applied Probability (2015), §1.1 PDF p.3; §1.3 p.6; §§1.6–1.7 pp.12–16](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=3): transition families, zero elapsed time, exponential survival and the minimal-process boundary. The positive-rate failure example and the day/night comparison are direct constructions and calculations.
- [Whitt, Continuous-Time Markov Chains, §2, PDF pp.3–4 / printed pp.2–3](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=3): transition probabilities, homogeneity and propagation of an initial probability row. “Stationary transition probabilities” there refers to the rule, not an automatically stationary initial law.
<!-- bilingual-en:end -->

- MIT 第5页定义2核对 Markov 依赖条件与日历时间齐次性的区别。
- Cambridge 第1.1节（PDF第3页）、第1.3节（第6页）、第1.6–1.7节（第12–16页）核对转移函数族、零时长、指数生存概率与最小过程边界。正速率故障例和昼夜比较直接构造、计算。
- Whitt 第2节（PDF第3–4页，印刷第2–3页）核对转移概率、齐次性及初始概率行的更新；其中“stationary transition probabilities”描述规则，不表示初始分布自动平稳。
