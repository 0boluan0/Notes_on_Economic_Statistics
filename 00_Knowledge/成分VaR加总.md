---
aliases:
  - "风险函数一次正齐次且可微时，成分 VaR 之和等于组合 VaR"
  - Component VaR Euler add-up
  - 成分 VaR 的 Euler 加总
student_os: knowledge-atom
atom_id: RM-VAR-043
atom_set: var-es-backtesting
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[成分VaR]]"
  - "[[全微分]]"
related:
  - "[[边际VaR]]"
  - "[[一致风险度量]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 风险函数一次正齐次且可微时，成分 VaR 之和等于组合 VaR
<!-- bilingual-en:start -->
*When a risk functional is positively homogeneous of degree one and differentiable, component VaRs sum to portfolio VaR*
<!-- bilingual-en:end -->

> [!summary] 精确加总来自 Euler 定理
> 若组合风险函数 $\rho(x)$ 对头寸一次正齐次，并在所考察头寸 $x$ 处可微，则 Euler 定理把总风险精确分解为各坐标的“头寸乘边际风险”。这些项正是成分 VaR。
>
> <!-- bilingual-en:start -->
> If portfolio risk $\rho(x)$ is positively homogeneous of degree one in positions and differentiable at the portfolio $x$, Euler's theorem decomposes total risk exactly into position times marginal risk for each coordinate. Those terms are the component VaRs.
> <!-- bilingual-en:end -->

一次正齐次是指

$$
\rho(\lambda x)=\lambda\rho(x),
\qquad \lambda>0.
$$

在可微点，Euler 定理给出

$$
\rho(x)
=\sum_i x_i\frac{\partial\rho(x)}{\partial x_i}
=\sum_i\operatorname{CVaR}_i(x).
$$

两个条件分工不同：正齐次把整体缩放与风险缩放联系起来，可微性使各坐标偏导在当前点可用。缺少任一条件，都不能从上述普通梯度公式直接推出精确加总。

<!-- bilingual-en:start -->
The conditions play different roles. Positive homogeneity links scaling of positions to scaling of risk, while differentiability makes the coordinate derivatives available at the current portfolio. Without either condition, exact add-up does not follow from this ordinary-gradient formula.
<!-- bilingual-en:end -->

例如 $x=(2,3)$，若边际 VaR 分别为 $(4,1)$，且 $\rho$ 在该点满足上述条件，则成分 VaR 为 $(8,3)$，组合 VaR 必为 11。这个结论是当前组合的 Euler 分摊，不等于分别删除两项头寸后的有限变化。

<!-- bilingual-en:start -->
For example, if $x=(2,3)$ and marginal VaRs are $(4,1)$, then under the stated conditions component VaRs are $(8,3)$ and portfolio VaR is 11. This is an Euler allocation at the current portfolio, not the finite effect of deleting each position.
<!-- bilingual-en:end -->

> [!question]- 自检
> 只知道风险函数可微，是否足以保证成分 VaR 加总为组合 VaR？
>
> **答案：** 不足。还需要风险函数对头寸一次正齐次。
>
> <!-- bilingual-en:start -->
> **Self-check:** Is differentiability alone enough to make component VaRs sum to portfolio VaR?
>
> **Answer:** No. The risk functional must also be positively homogeneous of degree one in positions.
> <!-- bilingual-en:end -->

## 边界

- 分位点跳变或其他不可微点需要不同的分摊理论，不能把任意次梯度自动当作唯一成分。
- 精确加总是一条函数结构结论，不证明风险模型、相关性或尾部估计正确。

<!-- bilingual-en:start -->
- Quantile jumps and other non-differentiable points require a different allocation theory; an arbitrary subgradient is not automatically a unique component allocation.
- Exact add-up is a structural property of the functional, not validation of the risk model, dependence, or tail estimate.
<!-- bilingual-en:end -->

## 来源与核验

- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle*](https://arxiv.org/abs/0708.2542)：核对一次正齐次、可微条件与 Euler 风险分摊公式。
