---
aliases:
  - "MinT在保持线性相干与无偏性的约束下最小化全部协调预测误差方差之和"
  - "MinT minimizes the sum of reconciled forecast-error variances subject to linear coherence and unbiasedness preservation"
  - "Minimum trace reconciliation"
  - "MinT reconciliation"
student_os: knowledge-atom
atom_id: TS-COMB-010
atom_set: forecast-combination-reconciliation-scenarios
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[预测组合、层级协调与情景.canvas]]"
requires:
  - "[[预测协调]]"
  - "[[层级预测相干性]]"
  - "[[组合误差协方差]]"
leads_to:
  - "[[MinT协方差边界]]"
  - "[[相干不等于准确]]"
contrasts_with:
  - "[[单层协调方法]]"
---

# MinT在保持线性相干与无偏性的约束下最小化全部协调预测误差方差之和
<!-- bilingual-en:start -->
*MinT minimizes the sum of reconciled forecast-error variances subject to linear coherence and unbiasedness preservation*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $\hat y_h$ 是所有节点的 base forecasts，$W_h=\operatorname{Var}(y_{T+h}-\hat y_h)$ 是相应 $h$ 步预测误差协方差。在线性协调、base forecasts 无偏且 $W_h$ 可逆的理论条件下，MinT 取
> $$
> \tilde y_h=S(S'W_h^{-1}S)^{-1}S'W_h^{-1}\hat y_h,
> $$
> 使协调预测误差协方差的迹——也就是全部节点误差方差之和——最小。
> <!-- bilingual-en:start -->
> Given base forecasts $\hat y_h$ and their $h$-step error covariance $W_h$, MinT is the linear reconciliation that preserves coherence and unbiasedness and minimises the trace of the reconciled-error covariance, under the theoretical conditions of unbiased base forecasts and an invertible $W_h$.
> <!-- bilingual-en:end -->

写成 $\tilde y_h=M\hat y_h$、$M=S(S'W_h^{-1}S)^{-1}S'W_h^{-1}$ 时，保持无偏性的约束为 $MS=S$。在 base forecasts 无偏、$W_h$ 为真实正定误差协方差的 oracle 问题中，协调误差协方差 $V_h=MW_hM'$ 满足
$$
W_h-V_h=(I-M)W_h(I-M)'\succeq0.
$$
因此 $V_{h,ii}\le W_{h,ii}$：每个节点的**总体误差方差**，也就是无偏条件下的总体 MSE，都不会高于该节点的 base forecast。trace 是这一结果的汇总目标，不是允许拿某个节点的总体方差去换另一个节点。更一般地，在同一 oracle 设定和线性无偏协调类内，MinT 也最小化任意固定正定权重矩阵定义的期望二次损失。
<!-- bilingual-en:start -->
Writing $\tilde y_h=M\hat y_h$, unbiasedness preservation requires $MS=S$. With unbiased base forecasts and the true positive-definite $W_h$, the reconciled-error covariance $V_h=MW_hM'$ satisfies $W_h-V_h=(I-M)W_h(I-M)'\succeq0$. Hence every node's population error variance—and its population MSE under unbiasedness—is no larger than that of its base forecast. Trace summarises this oracle result; it is not permission to trade one node's population variance against another. Within the same linear unbiased oracle problem, MinT also minimises expected quadratic loss for any fixed positive-definite weighting matrix.
<!-- bilingual-en:end -->

这不等于“每一次实现都逐节点变好”。以 $T=A+B$、$W_h=I$ 为例，令
$$
S=\begin{bmatrix}1&1\\1&0\\0&1\end{bmatrix},\qquad
q=\begin{bmatrix}1\\-1\\-1\end{bmatrix},\qquad
M=I-\frac{qq'}{3}.
$$
此时 $V_h=M$，三个节点的总体误差方差都从 $1$ 降至 $2/3$。但若某一次 base error 恰为 $e=(0,1,1)'$，协调后 $Me=(2/3,1/3,1/3)'$：总量节点在这一次由误差 $0$ 变成 $2/3$。单次误差变坏与总体方差下降可以同时成立。
<!-- bilingual-en:start -->
This is not a claim that every realised node error improves. For $T=A+B$ and $W_h=I$, let $S=\begin{bmatrix}1&1\\1&0\\0&1\end{bmatrix}$, $q=(1,-1,-1)'$, and $M=I-qq'/3$. Then $V_h=M$ and every population variance falls from 1 to $2/3$. Yet a realised base error $e=(0,1,1)'$ becomes $Me=(2/3,1/3,1/3)'$: the total happens to move from zero error to $2/3$. A worse realised error is compatible with a lower population variance.
<!-- bilingual-en:end -->

$W_h^{-1}$ 使误差尺度和跨节点相关共同决定调整。若某个节点的 base forecast 精度高，协调通常较不愿大幅移动它；若几个节点的错误高度相关，也不能把它们当作独立证据重复计数。$W_h=k_hI$ 时公式退化为 OLS reconciliation；对角方差、结构缩放和完整协方差则对应不同近似。
<!-- bilingual-en:start -->
The inverse covariance weights forecast precision and cross-node dependence jointly. OLS reconciliation arises when $W_h$ is proportional to the identity; variance scaling, structural scaling, and full-covariance approaches encode progressively richer approximations.
<!-- bilingual-en:end -->

“optimal”指在上述模型、损失和已知 $W_h$ 的问题内最优，不是数据有限时的无条件外样本保证。实践中最难的部分恰是 [[MinT协方差边界|估计 $W_h$]]；应把协方差估计方法视为 pipeline 的一部分并通过 rolling origins 检验。
<!-- bilingual-en:start -->
“Optimal” is conditional on the stated linear problem, loss, unbiasedness assumptions, and covariance matrix. In practice, estimating $W_h$ is part of the model and must itself survive rolling-origin evaluation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 在 oracle 条件下，MinT 能否保证每个节点都比 base forecast 更好？
>
> **答案：** 若“更好”指总体误差方差，能：$V_h\preceq W_h$ 保证每个对角元不升。若指某一次实现的绝对或平方误差，不能；而现实中的 $W_h$ 还需估计，偏差、近似协方差、额外约束和其他损失也会使 oracle 保证不再直接适用。

## 来源与核验

- Wickramasuriya, Athanasopoulos & Hyndman（2019），[Optimal forecast reconciliation through trace minimization](https://robjhyndman.com/papers/mint.pdf)：核对 coherence、无偏约束、最小均方误差、oracle 方差支配结果与闭式解。
- Athanasopoulos, Hyndman, Kourentzes & Panagiotelis（2024），[Forecast reconciliation: A review，§3.5](https://robjhyndman.com/papers/hf_review.pdf)：核对该节转述的 Panagiotelis et al.（2021）正定二次损失结果，以及协方差估计不确定性不包含在 oracle 结论内。
- [Hyndman & Athanasopoulos, FPP3 §11.3](https://otexts.com/fpp3/reconciliation.html)：核对 $SGS=S$、$V_h=SGW_hG'S'$、trace 目标、公式及常用 $W_h$ 近似。
