---
aliases:
  - CTMC reversibility
  - Reversible continuous-time Markov chain
  - 连续时间链可逆性
student_os: knowledge-atom
atom_id: PROB-CTMC-034
atom_set: continuous-time-markov-chains
atom_type: definition
status: source-checked
requires:
  - "[[CTMC定义]]"
  - "[[CTMC平稳分布]]"
related:
  - "[[DTMC可逆性]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# CTMC reversibility
<!-- bilingual-en:end -->

*CTMC 可逆性*

<!-- bilingual-en:start -->
Record a system after it has been started in equilibrium, then play the recording backwards. A reversible chain has the same probability law in both directions: even knowing the model, an observer cannot infer a preferred time direction from the distribution of the observed states. This is a statement about random paths, not about each individual path being a palindrome.
<!-- bilingual-en:end -->

把系统从平衡分布启动后录下来，再倒放。可逆链在正放和倒放时具有相同的概率规律：即使知道模型，也无法从观测状态的分布中辨认一个偏好的时间方向。这讨论随机路径的规律，不要求每一条实际路径都是回文。

<!-- bilingual-en:start -->
Fix a [[CTMC时间齐次转移函数|time-homogeneous CTMC]] with transition probabilities $p_{ij}(t)$. A [[CTMC平稳分布|stationary distribution]] $\pi$ satisfies $\pi P(t)=\pi$; starting with $X_0\sim\pi$ keeps every observation time distributed as $\pi$. Reversibility additionally compares their **joint** laws, called [[有限维分布|finite-dimensional distributions]].
<!-- bilingual-en:end -->

固定一个[[CTMC时间齐次转移函数|时间齐次 CTMC]]，转移概率记为 $p_{ij}(t)$。[[CTMC平稳分布|平稳分布]] $\pi$ 满足 $\pi P(t)=\pi$；从 $X_0\sim\pi$ 启动后，每个观察时点的分布都是 $\pi$。可逆性还要求比较多个时点的**联合**规律，即[[有限维分布]]。

> [!important] Reverse the time axis · 反转时间轴
> <!-- bilingual-en:start -->
> The chain is reversible relative to $\pi$ if, under the stationary initial law, for every $0=t_0<t_1<\cdots<t_m=T$, the two vectors below have the same joint distribution. The notation $\overset d=$ means equality in distribution, not equality on every path.
> <!-- bilingual-en:end -->
>
> 若在平稳初始分布下，对任意 $0=t_0<t_1<\cdots<t_m=T$，下面两个向量都有相同联合分布，就称链相对于 $\pi$ 可逆。记号 $\overset d=$ 表示同分布，不是逐条路径都相等。

$$
(X_{t_0},X_{t_1},\ldots,X_{t_m})
\overset d=
(X_{T-t_0},X_{T-t_1},\ldots,X_{T-t_m}).
$$

<!-- bilingual-en:start -->
For observations at times 0, 1 and 3, the comparison is $(X_0,X_1,X_3)$ versus $(X_3,X_2,X_0)$. Time 1 is reflected to time 2 about the midpoint $3/2$. With unequal gaps, simply reversing the original list of sampled values would not reflect the same observation schedule.
<!-- bilingual-en:end -->

例如观察时刻为0、1、3，比较的是 $(X_0,X_1,X_3)$ 与 $(X_3,X_2,X_0)$。关于中点 $3/2$ 反射后，时刻1变成时刻2。间隔不等时，仅把已采样数值的列表倒序，并没有按同一个观察时间表反转时间轴。

<!-- bilingual-en:start -->
## What can reveal a time direction?
<!-- bilingual-en:end -->

## 怎样看出时间方向？

<!-- bilingual-en:start -->
Already at two observation times, reversibility requires $\pi_ip_{ij}(h)=\pi_jp_{ji}(h)$. The left side is the probability of seeing $i$ and then $j$ after time $h$; the right side is the reversed observation. In the two-state working/repair model with failure rate 2 and repair rate 3, $\pi=(3/5,2/5)$ and
<!-- bilingual-en:end -->

只看两个时点，可逆性就要求 $\pi_ip_{ij}(h)=\pi_jp_{ji}(h)$。左边是先看到 $i$，隔 $h$ 后看到 $j$ 的概率；右边是反向观察的概率。对故障率2、修复率3的工作/维修模型，$\pi=(3/5,2/5)$，并且

$$
p_{01}(h)=\frac25(1-e^{-5h}),
\qquad
p_{10}(h)=\frac35(1-e^{-5h}).
$$

<!-- bilingual-en:start -->
Both weighted probabilities equal $\frac6{25}(1-e^{-5h})$. The [[CTMC详细平衡与可逆|finite-state equivalence theorem]] shows why its balanced instantaneous rates guarantee symmetry for every finite observation schedule, not just this pair.
<!-- bilingual-en:end -->

两个加权概率都等于 $\frac6{25}(1-e^{-5h})$。[[CTMC详细平衡与可逆|有限状态等价定理]]进一步说明，逐对瞬时速率流的平衡，为什么能保证任意有限观察时间表的对称，而不只是这一对时点。

> [!warning] Equilibrium does not erase circulation · 平衡不一定消除环流
> <!-- bilingual-en:start -->
> A three-state chain that only jumps $1\to2\to3\to1$ at equal rates can have constant, uniform marginal distributions. Its short-time transitions still reveal a clockwise direction. Stationarity and reversibility therefore answer different questions; the complete calculation is in the [[CTMC详细平衡与可逆|equivalence theorem]].
> <!-- bilingual-en:end -->
>
> 三状态链若只按相同速率沿 $1\to2\to3\to1$ 跳转，可以一直保持均匀的单时点分布；短时间转移却仍显露顺时针方向。因此平稳性和可逆性回答不同问题，完整计算见[[CTMC详细平衡与可逆|等价定理]]。

<!-- bilingual-en:start -->
The initial distribution matters. Even the reversible two-state model, when started deterministically in state 0, is not a stationary reversible path law: $X_0=0$ surely, whereas $X_T$ can be 1. Saying that its generator is reversible relative to $\pi$ refers to initialization with $\pi$. If some $\pi_i=0$, a stationary path never visits those states; equality of forward and reversed generators is required only on the positive-probability support. The reversed path is being compared through finite observations; reversing a right-continuous step path does not automatically preserve the same convention at jump instants.
<!-- bilingual-en:end -->

初始分布很重要。即使这个两状态模型相对于 $\pi$ 可逆，若固定从状态0开始，所得路径律也不是平稳可逆的：$X_0$ 必为0，$X_T$ 却可能为1。说生成矩阵相对于 $\pi$ 可逆，指的是按 $\pi$ 初始化。若某些 $\pi_i=0$，平稳路径不会访问这些状态；正反向生成矩阵相同的说法只约束正概率支撑。本定义通过有限时点观察比较倒放路径；直接倒放右连续阶梯路径，不会自动保留跳跃时刻的同一种取值约定。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Cambridge, Applied Probability, §2.6, PDF pp. 28–30](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf): stationary time reversal, rate detailed balance and their relationship.
- [Whitt, Continuous-Time Markov Chains (2012), §6, PDF pp. 33–36](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf): reversed transition rates and detailed balance. The finite-state proofs and examples above also check the conclusions directly.
<!-- bilingual-en:end -->

- [Cambridge, Applied Probability，§2.6，PDF 第28–30页](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：平稳时间反演、速率详细平衡及两者的关系。
- [Whitt, Continuous-Time Markov Chains（2012），§6，PDF 第33–36页](https://www.columbia.edu/~ww2040/Whitt_CTMCnotes121312.pdf)：反演生成率与详细平衡；正文另以有限状态证明和具体算例直接核对。
