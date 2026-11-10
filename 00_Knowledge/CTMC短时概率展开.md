---
aliases:
- Infinitesimal transition probabilities
- Short-time CTMC expansion
- 短时转移概率
student_os: knowledge-atom
atom_id: PROB-CTMC-004
atom_set: continuous-time-markov-chains
atom_type: distinction
status: source-checked
requires:
- '[[生成矩阵约束]]'
related:
- '[[CTMC转移半群]]'
leads_to:
- '[[Kolmogorov前后向方程]]'
part_of:
- '[[连续时间马尔可夫链.canvas]]'
---

<!-- bilingual-en:start -->
# Short-time transition probabilities
<!-- bilingual-en:end -->

*CTMC 短时概率展开*

<!-- bilingual-en:start -->
A transition rate tells us how a probability begins to change when we start the clock. Multiplying that rate by a very short duration gives a first approximation. The approximation improves as the duration shrinks; a rate of $5$ per year does not mean a probability of $5$ over a year.
<!-- bilingual-en:end -->

转移速率描述的是：刚开始计时时，概率怎样变化。把速率乘上一段很短的时长，可以得到第一步近似；时长越小，这个近似越准确。每年 $5$ 的速率，不代表一年内有 $5$ 的概率。

<!-- bilingual-en:start -->
Consider a [[CTMC定义|pure-jump CTMC]] with a [[CTMC时间齐次转移函数|time-homogeneous transition rule]]. Use the [[生成矩阵约束|stable conservative generator]] $Q=(q_{ij})$, so each state's [[出口率|exit rate]] $q_i=-q_{ii}$ is finite. The [[Markov矩阵左右约定|row convention]] gives $p_{ij}(h)=\Pr_i(X_h=j)$. The duration $h$ is positive; we study its [[函数极限|limit]] as $h\downarrow0$.
<!-- bilingual-en:end -->

考虑具有[[CTMC时间齐次转移函数|时间齐次转移规律]]的[[CTMC定义|纯跳 CTMC]]。使用[[生成矩阵约束|稳定、保守的生成矩阵]] $Q=(q_{ij})$，各状态的[[出口率]] $q_i=-q_{ii}$ 都有限。采用[[Markov矩阵左右约定|行约定]]，$p_{ij}(h)=\Pr_i(X_h=j)$。时长 $h$ 为正，我们考察 $h\downarrow0$ 时的[[函数极限]]。

<!-- bilingual-en:start -->
## A derivative at zero gives the first-order term
<!-- bilingual-en:end -->

*零点导数给出一阶项*

<!-- bilingual-en:start -->
Starting from $i$, the zero-time probability is $0$ for $j\ne i$ and $1$ for $j=i$. Dividing the change in probability by elapsed time gives the [[导数|right derivative]]:
<!-- bilingual-en:end -->

从 $i$ 出发，零时长时到达 $j\ne i$ 的概率为 $0$，仍在 $i$ 的概率为 $1$。用概率的变化除以经过的时长，得到右[[导数]]：

$$q_{ij}=\lim_{h\downarrow0}\frac{p_{ij}(h)}h\quad(j\ne i),\qquad
q_{ii}=\lim_{h\downarrow0}\frac{p_{ii}(h)-1}{h}.$$

> [!abstract] First-order probabilities · 一阶概率
> <!-- bilingual-en:start -->
> Equivalently, for each fixed pair of states,
> <!-- bilingual-en:end -->
>
> 等价地，对每一对固定状态，
>
> $$p_{ij}(h)=q_{ij}h+o(h)\quad(j\ne i),\qquad
> p_{ii}(h)=1-q_i h+o(h).$$

<!-- bilingual-en:start -->
The symbol $o(h)$ denotes a remainder $r(h)$ such that $r(h)/h\to0$. There can be a different remainder for each pair $(i,j)$. It need not be positive, and it is not simply zero. In words: after dividing the error by $h$, the error tends to zero. For example, a constant times $h^2$ is $o(h)$ because $h^2/h=h\to0$.
<!-- bilingual-en:end -->

记号 $o(h)$ 表示一个余项 $r(h)$，满足 $r(h)/h\to0$。不同状态对 $(i,j)$ 可以有不同的余项。它不一定为正，也不等于直接省去。意思是：误差再除以 $h$ 后仍趋于零。例如，常数倍的 $h^2$ 就属于 $o(h)$，因为 $h^2/h=h\to0$。

<!-- bilingual-en:start -->
## An exact example with at most one jump
<!-- bilingual-en:end -->

*最多只跳一次的精确例子*

<!-- bilingual-en:start -->
A working device fails at rate $5$ per year and is never repaired. The [[指数分布|exponential waiting-time law]] gives
<!-- bilingual-en:end -->

一台正常设备以每年 $5$ 的速率故障，此后不再维修。由[[指数分布|指数等待时间分布]]，

$$p_{01}(h)=1-e^{-5h},\qquad p_{00}(h)=e^{-5h}.$$

<!-- bilingual-en:start -->
At $h=0.01$ years, the linear approximation is $5h=0.05$, while the exact failure probability is about $0.0487706$. Their difference is about $-0.0012294$. As $h$ shrinks, $(1-e^{-5h}-5h)/h\to0$. At one year the linear expression would be $5$, which cannot be a probability; the exact expression remains below $1$.
<!-- bilingual-en:end -->

当 $h=0.01$ 年时，线性近似为 $5h=0.05$，精确故障概率约为 $0.0487706$，两者之差约为 $-0.0012294$。随着 $h$ 缩小，$(1-e^{-5h}-5h)/h\to0$。若取一年，线性表达式就变成 $5$，根本不可能是概率；精确表达式则始终小于 $1$。

> [!warning] The remainder is not just multiple jumps · 余项不只来自多次跳跃
> <!-- bilingual-en:start -->
> This device can jump only once, but its exact probability still differs from $5h$. Even the probability of no jump, $e^{-5h}$, has nonlinear terms. In a model that allows repeated jumps, multi-jump paths also contribute; they are not the only source of the remainder. Likewise, being back in state $i$ at time $h$ may include leaving and returning, so $p_{ii}(h)$ is not generally the probability of no departure.
> <!-- bilingual-en:end -->
>
> 这台设备最多只跳一次，但精确概率仍不等于 $5h$。即使“没有跳跃”的概率 $e^{-5h}$，也有非线性项。在允许反复跳跃的模型里，多跳路径也会贡献误差，但它们不是余项的唯一来源。同样，时刻 $h$ 又处于 $i$，可能是离开后返回，所以 $p_{ii}(h)$ 一般不等于从未离开的概率。

<!-- bilingual-en:start -->
## Finite matrices and an infinite-state boundary
<!-- bilingual-en:end -->

*有限矩阵与无限状态的边界*

<!-- bilingual-en:start -->
For finitely many states, collect all entries into $P(h)$. With the [[单位矩阵|identity matrix]] $I$, which describes zero elapsed time, the expansion becomes
<!-- bilingual-en:end -->

有限状态时，把所有条目合成 $P(h)$。用描述零时长的[[单位矩阵]] $I$，可写成

$$P(h)=I+hQ+o(h).$$

<!-- bilingual-en:start -->
Measured by the [[Frobenius范数|Frobenius norm]]—the square root of the sum of squared entry magnitudes—the matrix remainder divided by $h$ tends to zero. In fact the finite-state [[CTMC矩阵指数|matrix exponential]] gives a remainder bounded by a constant times $h^2$ for sufficiently small $h$. Even if $I+hQ$ happens to have nonnegative entries and row sums one, it remains an approximation and need not equal $P(h)$.
<!-- bilingual-en:end -->

用[[Frobenius范数]]，即各元素模平方相加后开平方，衡量矩阵余项的大小；这个大小除以 $h$ 后趋于零。事实上，有限状态的[[CTMC矩阵指数|矩阵指数表达式]]还给出局部常数倍 $h^2$ 的误差上界。即使 $I+hQ$ 恰好元素非负、行和为一，它仍然只是近似，未必等于 $P(h)$。

<!-- bilingual-en:start -->
For a countable stable Q-matrix, the short-time statement is entrywise for fixed $i,j$ in its minimal pure-jump construction. It does not automatically give an error bound uniform over all starting states. For example, let state $i_n$ jump only to its own absorbing state $a_n$, at rate $n$ for each positive integer $n$. Every path makes at most one jump, yet for every fixed $h>0$ some diagonal approximation $1-nh$ is negative. The exact diagonal is $e^{-nh}$. Its remainder $e^{-nh}-1+nh$ is unbounded as $n\to\infty$ at fixed $h$. Thus these entrywise expansions are not uniform over the states. Unbounded rates require a specified norm, domain and additional control before such a stronger statement can be made.
<!-- bilingual-en:end -->

对可数状态的稳定 Q 矩阵，在其最小纯跳构造中，短时结论是对固定 $i,j$ 逐项成立的，并不自动给出对所有起点一致的误差界。例如，对每个正整数 $n$，令状态 $i_n$ 只以速率 $n$ 跳到自己的吸收态 $a_n$。每条路径最多跳一次，但对任意固定 $h>0$，总有某个对角近似 $1-nh$ 为负。精确对角概率为 $e^{-nh}$，余项 $e^{-nh}-1+nh$ 在固定 $h$、$n\to\infty$ 时无界。因此，这些逐项展开并不对所有状态一致成立。速率无界时，需要指定范数、定义域并提供额外控制，才能作更强的断言。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Whitt, Continuous-Time Markov Chains, §3.2](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=8), PDF pp. 8–10, equations (3.1)–(3.6): right derivatives, the meaning of $o(h)$ and finite-state matrix formulation.
- [Cambridge, Applied Probability](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf#page=15), PDF pp. 15–18, Theorems 1.23 and 1.27–1.29: countable minimal-chain derivatives and finite-state matrix-exponential expansion. The absorbing-device and unbounded-rate examples are computed directly here.
- [MIT 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=6), PDF pp. 6–8: holding-time interpretation and generator rates.
<!-- bilingual-en:end -->

- Whitt §3.2，PDF 第 8–10 页，式 (3.1)–(3.6)：右导数、$o(h)$ 的含义和有限状态矩阵形式。
- Cambridge，PDF 第 15–18 页，定理 1.23、1.27–1.29：可数最小链的导数，以及有限状态矩阵指数展开。单向故障和无界速率例子在本文直接计算。
- MIT 第 24 讲，PDF 第 6–8 页：停留时间解释与生成率。
