---
aliases:
  - "周期不阻止可数不可约正常返链的转移概率 Cesàro 平均收敛到平稳分布"
  - Cesaro convergence of periodic Markov chains
  - Cesàro convergence of transition probabilities
student_os: knowledge-atom
atom_id: PROB-DTMC-014
atom_set: discrete-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[正常返与零常返]]"
  - "[[不可约链]]"
  - "[[Markov稳态分布]]"
  - "[[状态周期]]"
related:
  - "[[有限链逐步收敛]]"
  - "[[稳态唯一不推收敛]]"
  - "[[Markov时间平均收敛]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

# 周期不阻止可数不可约正常返链的转移概率 Cesàro 平均收敛到平稳分布
<!-- bilingual-en:start -->
*Periodicity does not prevent the Cesàro averages of transition probabilities in a countable irreducible positive recurrent chain from converging to its stationary distribution*
<!-- bilingual-en:end -->

> [!summary] 转移概率的 Cesàro 极限
> 设 DTMC 的状态空间可数，且链不可约、正常返，唯一平稳分布为 $\pi$。那么对任意状态 $i,j$，
> $$
> \frac1N\sum_{n=0}^{N-1}p_{ij}^{(n)}\longrightarrow\pi_j,
> $$
> 这里不要求非周期。结论是对每个固定 $i,j$ 的逐坐标平均收敛，不是在说单个序列 $p_{ij}^{(n)}$ 本身收敛。
> <!-- bilingual-en:start -->
> In a countable irreducible positive recurrent chain, the Cesàro average of each transition-probability sequence converges to the corresponding stationary mass, without an aperiodicity assumption.
> <!-- bilingual-en:end -->

这里平均的是一族**分布或转移概率**。一条实际样本路径上的观测值平均是另一个随机对象，其收敛结果见 [[Markov时间平均收敛]]，不能用同一个公式替换。

在二状态确定性交替链中，若从状态 1 出发，则 $p_{11}^{(n)}$ 依次为 $1,0,1,0,\ldots$，所以逐步极限不存在；但它的前 $N$ 项算术平均趋于 $1/2=\pi_1$。

> [!question]- 自检
> 若 $p_{11}^{(n)}=1,0,1,0,\ldots$，能否说 $p_{11}^{(n)}\to1/2$？它的 Cesàro 平均呢？
>
> **答案：** 不能说逐步收敛；序列本身持续振荡。但前 $N$ 项的 Cesàro 平均收敛到 $1/2$。

## 来源与核验

- [MIT OCW 6.262, Chapter 5, Theorem 5.1.2](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/01d0892549619cb25d928f15ec7230ed_MIT6_262S11_chap05.pdf#page=9)：核对不可约正常返链的转移概率 Cesàro 平均与平稳质量的关系。
