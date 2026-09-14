---
aliases:
  - "VaR 与 ES 的联合识别函数给出分位覆盖和尾部均值的成对校准矩"
  - VaR-ES joint identification
  - VaR 与 ES 联合可识别性
student_os: knowledge-atom
atom_id: RM-VAR-010
atom_set: var-es-backtesting
atom_type: identification-result
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[ES定义]]"
  - "[[VaR回测损益口径]]"
related:
  - "[[风险模型验证边界]]"
  - "[[Christoffersen条件覆盖]]"
  - "[[VaR-ES联合评分]]"
leads_to:
  - "[[超越均值不足以验证ES]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 与 ES 的联合识别函数给出分位覆盖和尾部均值的成对校准矩
<!-- bilingual-en:start -->
*A joint identification function for VaR and ES supplies paired calibration moments for quantile coverage and tail mean*
<!-- bilingual-en:end -->

> [!summary] ES 的校准方程要与对应 VaR 一起写
> 在常用的丰富分布类上，ES 没有一般的单变量识别函数；同一置信水平、持有期和信息集下的 $(\mathrm{VaR},\mathrm{ES})$ 二元组却可联合识别。第一条矩检查分位覆盖，第二条把 ES 与该分位以上的超额损失连接起来。
>
> <!-- bilingual-en:start -->
> ES generally lacks a stand-alone identification function on rich distribution classes, while the same-horizon, same-level $(\mathrm{VaR},\mathrm{ES})$ pair is jointly identifiable. One moment checks the quantile and the other links ES to excess loss above that quantile.
> <!-- bilingual-en:end -->

采用上尾损失口径，令 $q_t$ 是条件 $\alpha$ 分位数，$e_t$ 是对应 ES。一个联合识别函数是

<!-- bilingual-en:start -->
Under the upper-tail loss convention, let $q_t$ be the conditional $\alpha$-quantile and $e_t$ the corresponding ES. One joint identification function is
<!-- bilingual-en:end -->

$$
V_1(q_t,e_t,L_t)
=\mathbf 1\{L_t\le q_t\}-\alpha,
$$

$$
V_2(q_t,e_t,L_t)
=e_t-q_t-\frac{(L_t-q_t)_+}{1-\alpha}.
$$

正确预测在常规唯一性与可积条件下满足

<!-- bilingual-en:start -->
Under the usual uniqueness and integrability conditions, correct forecasts satisfy
<!-- bilingual-en:end -->

$$
E\!\left[
\begin{pmatrix}
V_1\\V_2
\end{pmatrix}
\middle|\mathcal F_{t-1}
\right]
=
\begin{pmatrix}0\\0\end{pmatrix}.
$$

$V_1$ 固定分位数，$V_2$ 才能在这个分位数基础上识别 ES。只留下 $V_2$ 却不处理 $q_t$，并没有得到 ES 的一般单变量校准方程。若要把条件矩做成可执行检验，还必须预先选择 $\mathcal F_{t-1}$-可测的工具变量，并使用适合时间依赖的协方差估计。

<!-- bilingual-en:start -->
$V_1$ identifies the quantile, allowing $V_2$ to identify ES relative to it. Dropping $V_1$ does not create a general stand-alone ES equation. An executable conditional test must also pre-specify instruments measurable at $\mathcal F_{t-1}$ and use covariance inference compatible with time dependence.
<!-- bilingual-en:end -->

> [!question]- 自检
> 只检验样本平均 $V_2=0$，但把 $q_t$ 当作不需验证的给定阈值，是否完成了 VaR–ES 联合校准？
>
> **答案：** 没有。必须同时处理分位覆盖矩 $V_1$；否则 ES 矩建立在可能错误的阈值上。
>
> <!-- bilingual-en:start -->
> **Self-check:** Does testing only the sample mean condition $V_2=0$, while treating $q_t$ as an unverified threshold, complete joint VaR–ES calibration?
>
> **Answer:** No. The quantile moment $V_1$ must also be addressed; otherwise the ES moment is conditioned on a potentially wrong threshold.
> <!-- bilingual-en:end -->

## 边界

- 分位点有概率质量时，简单覆盖等式需要广义分位或随机化 tie rule。
- 联合识别支持绝对校准矩；它不等于用联合评分比较两个模型。
- 未拒绝矩条件不能证明完整尾部分布正确。

<!-- bilingual-en:start -->
- Probability mass at the quantile requires a generalized or randomized tie treatment.
- Joint identification supports absolute calibration moments; it is distinct from comparative joint scoring.
- Failure to reject does not prove the full tail distribution correct.
<!-- bilingual-en:end -->

## 来源与核验

- Fissler & Ziegel, [*Higher Order Elicitability and Osband's Principle*](https://doi.org/10.1214/16-AOS1439)：核对 VaR–ES 二元组的联合识别结构。
- Nolde & Ziegel, [*Elicitability and Backtesting: Perspectives for Banking Regulation*](https://arxiv.org/abs/1608.05498)：核对识别矩、传统校准回测与比较回测的区别。
- Bayer & Dimitriadis, [*Regression-Based Expected Shortfall Backtesting*](https://doi.org/10.1093/jjfinec/nbaa013)：核对外部只提交 ES 的检验仍通过联合分位数—ES 回归结构完成识别。
