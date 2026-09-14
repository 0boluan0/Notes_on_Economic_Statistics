---
aliases:
  - "有限状态 CTMC 的转移半群等于矩阵指数 e 的 tQ 次方"
  - CTMC matrix exponential
  - P(t) equals exp(tQ)
  - 生成矩阵的矩阵指数
student_os: knowledge-atom
atom_id: PROB-CTMC-008
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
  - "[[矩阵指数]]"
  - "[[常系数线性系统的矩阵指数解]]"
related:
  - "[[Kolmogorov前后向方程]]"
leads_to:
  - "[[CTMC均匀化]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 有限状态 CTMC 的转移半群等于矩阵指数 e 的 tQ 次方
<!-- bilingual-en:start -->
*The transition semigroup of a finite-state CTMC is the matrix exponential $e^{tQ}$*
<!-- bilingual-en:end -->

> [!summary] 从瞬时率到有限时长概率
> 对有限状态、时间齐次 CTMC，
> $$
> P(t)=e^{tQ}:=I+tQ+\frac{t^2Q^2}{2!}+\frac{t^3Q^3}{3!}+\cdots.
> $$
> 这是矩阵指数，不是对 $Q$ 的每个元素分别取标量指数。它是初值问题 $P'(t)=QP(t)=P(t)Q$、$P(0)=I$ 的唯一解。
> <!-- bilingual-en:start -->
> For a finite generator, the matrix exponential converts infinitesimal rates into finite-time transition probabilities. Entrywise exponentiation is a different and incorrect operation.
> <!-- bilingual-en:end -->

有限维保证幂级数收敛并可逐项求导。即使 $Q$ 不可对角化，$e^{tQ}$ 仍由幂级数定义；对角化、Jordan、Schur 或均匀化只是不同计算方法。

可数状态下的 $Q$ 可能是无界算子，矩阵乘法和逐项级数未必可交换或收敛到所需的 minimal semigroup。因此本结论不能仅凭“每行和为零”机械外推到无限矩阵。

> [!example] 一个吸收跳转
> 若
> $$Q=\begin{pmatrix}-\lambda&\lambda\\0&0\end{pmatrix},$$
> 则
> $$P(t)=\begin{pmatrix}e^{-\lambda t}&1-e^{-\lambda t}\\0&1\end{pmatrix}.$$
> 对角元的标量指数碰巧出现，但右上角来自整个矩阵指数，不是 $e^{\lambda t}$。

> [!question]- 自检
> $Q$ 不可对角化时，是否无法定义 CTMC 的 $P(t)$？
>
> **答案：** 否。有限维矩阵指数始终由幂级数定义；不可对角化只影响某些计算捷径。

## 来源与核验

- [Ward Whitt, Continuous-Time Markov Chains, Theorem 3.2](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=9)：核对有限状态转移函数的矩阵指数表示和唯一性。
- [[矩阵指数]]：复用矩阵指数对任意有限方阵均存在的线性代数身份。
- [Cambridge Applied Probability notes](https://www.statslab.cam.ac.uk/~ps422/notes-new2015.pdf)：核对无限状态生成元与 minimal semigroup 的额外边界。
