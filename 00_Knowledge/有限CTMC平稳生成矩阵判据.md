---
aliases:
  - "有限状态 CTMC 的平稳分布恰是生成矩阵的归一化非负左零向量"
  - Finite-state CTMC stationary generator criterion
  - CTMC stationary equation
student_os: knowledge-atom
atom_id: PROB-CTMC-038
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC平稳分布]]"
  - "[[生成矩阵约束]]"
  - "[[CTMC矩阵指数]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[有限不可约CTMC稳态]]"
  - "[[可数CTMC稳态存在]]"
  - "[[跳链频率时间加权]]"
leads_to:
  - "[[CTMC详细平衡与可逆]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 有限状态 CTMC 的平稳分布恰是生成矩阵的归一化非负左零向量
<!-- bilingual-en:start -->
*The stationary distributions of a finite-state CTMC are exactly the normalised nonnegative left null vectors of its generator*
<!-- bilingual-en:end -->

> [!summary] 有限状态下的等价计算判据
> 对有限状态、时间齐次、保守 CTMC，在行概率约定下，概率向量 $\pi$ 是平稳分布，当且仅当
> $$
> \pi Q=0,
> \qquad \pi_i\ge0,
> \qquad \sum_i\pi_i=1.
> $$
> 也就是说，半群定义 $\pi P(t)=\pi$ 对所有 $t\ge0$ 成立，恰好等价于 $\pi$ 是生成矩阵的归一化非负左零向量。
>
> <!-- bilingual-en:start -->
> For a finite-state homogeneous conservative CTMC under the row-vector convention, a probability vector is stationary exactly when it is a nonnegative normalised left null vector of the generator.
> <!-- bilingual-en:end -->

## 为什么等价
<!-- bilingual-en:start -->
*Why the conditions are equivalent*
<!-- bilingual-en:end -->

若 $\pi P(t)=\pi$ 对所有 $t\ge0$ 成立，在 $t=0$ 处求右导数并使用 $P'(0)=Q$，便得到 $\pi Q=0$。反过来，有限状态下
$$
P(t)=e^{tQ}.
$$
若 $\pi Q=0$，则 $\pi Q^k=0$ 对每个 $k\ge1$ 成立，因此矩阵指数级数给出
$$
\pi P(t)=\pi e^{tQ}=\pi.
$$
非负与归一化保证这个左零向量确实是概率分布。

<!-- bilingual-en:start -->
Stationarity implies $\pi Q=0$ by differentiating $\pi P(t)=\pi$ at zero. Conversely, in finite state spaces $P(t)=e^{tQ}$; if $\pi Q=0$, every positive power term vanishes after left multiplication by $\pi$, so $\pi e^{tQ}=\pi$. Nonnegativity and normalisation make the left null vector a probability distribution.
<!-- bilingual-en:end -->

> [!example] 两状态链
> 对
> $$
> Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix},
> $$
> 取 $\pi=(1/3,2/3)$，则 $\pi Q=(0,0)$，且两个分量非负、总和为一。因此 $\pi$ 是平稳分布。
>
> <!-- bilingual-en:start -->
> For $Q=\begin{pmatrix}-2&2\\1&-1\end{pmatrix}$, the vector $\pi=(1/3,2/3)$ satisfies $\pi Q=0$, has nonnegative entries, and sums to one. The criterion therefore identifies it as stationary.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 在有限状态下，找到一个非零向量 $x$ 满足 $xQ=0$，是否已经得到平稳分布？
>
> **答案：** 未必。还要检查各分量非负，并把总质量正规化为一；零和向量或含负分量的左零向量不是概率分布。
>
> <!-- bilingual-en:start -->
> **Self-check:** In a finite state space, does any nonzero vector $x$ satisfying $xQ=0$ already give a stationary distribution?
>
> **Answer:** Not necessarily. Its entries must be nonnegative and its total mass must be normalised to one; a zero-sum or signed left null vector is not a probability distribution.
> <!-- bilingual-en:end -->

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

- 行概率约定要求解左方程 $\pi Q=0$；恒等式 $Q\mathbf1=0$ 只是生成矩阵的零行和，不是在求平稳分布。
- 该判据判断给定概率向量是否平稳，不单独保证平稳分布唯一，也不保证从任意初始分布收敛。
- 可数状态下，无穷求和、微分与半群生成元之间需要额外条件；形式上的归一化 $\pi Q=0$ 解不能替代对保守性、非爆炸和 regularity 的检查。

<!-- bilingual-en:start -->
- Under the row-vector convention one solves the left equation $\pi Q=0$; the identity $Q\mathbf1=0$ merely records zero generator row sums.
- The criterion tests stationarity, not uniqueness or convergence from every initial distribution.
- In countable state spaces, interchanging infinite sums, differentiation, and the semigroup generator requires additional assumptions; a formal normalised solution of $\pi Q=0$ does not replace checks of conservativity, non-explosion, and regularity.
<!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 6.436J, Lecture 24, Proposition 2](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7)：核对非负、归一化与生成矩阵左零向量对有限状态 CTMC 平稳分布的等价判据。
- [[CTMC矩阵指数]]：核对有限状态下 $P(t)=e^{tQ}$，并支持从 $\pi Q=0$ 推到 $\pi P(t)=\pi$ 的证明方向。
- [Cambridge Applied Probability notes, §2.4](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对 $Q$-不变测度的表述及其在可数状态空间中的额外过程边界。
