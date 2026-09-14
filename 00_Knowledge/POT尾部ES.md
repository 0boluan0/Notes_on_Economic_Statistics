---
aliases:
  - 连续GPD尾部的ES等于目标分位数加上该点以上的平均超额且有限性要求形状小于一
student_os: knowledge-atom
atom_id: RM-EVT-023
atom_type: method
status: source-checked
requires:
  - "[[POT分位数]]"
  - "[[ES定义]]"
  - "[[广义Pareto矩存在条件]]"
related:
  - "[[ES严格超越均值边界]]"
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
---

# 连续GPD尾部的ES等于目标分位数加上该点以上的平均超额且有限性要求形状小于一
<!-- bilingual-en:start -->
*ES for a continuous GPD tail is its target quantile plus mean excess at that point, and is finite only when shape is below one*
<!-- bilingual-en:end -->

当目标分位 $v_\alpha$ 严格高于 POT 阈值，拟合的尾部连续且 $\xi<1$ 时，[[ES定义|ES]] 可以按“超过分位数之后的平均损失”计算。这个平均损失包含分位数本身；只算平均超额会少一整段 $v_\alpha$。
<!-- bilingual-en:start -->
When the target quantile $v_\alpha$ lies strictly above the POT threshold, the fitted tail is continuous and $\xi<1$, [[ES定义|ES]] is the average loss beyond that quantile. It includes the quantile itself; averaging only the excess omits $v_\alpha$.
<!-- bilingual-en:end -->

在阈值 $u$ 处，超额采用尺度 $\beta>0$、形状 $\xi$ 的 GPD。令 $v=v_\alpha>u$，由 [[GPD阈值稳定性]]，进一步超过 $v$ 的超额仍是 GPD，尺度变成 $\beta_v=\beta+\xi(v-u)>0$。其均值为 $\beta_v/(1-\xi)$，因而
<!-- bilingual-en:start -->
At threshold $u$, model excesses by a GPD with scale $\beta>0$ and shape $\xi$. Let $v=v_\alpha>u$. [[GPD阈值稳定性|GPD threshold stability]] gives a GPD for excesses beyond $v$, with scale $\beta_v=\beta+\xi(v-u)>0$. Its mean is $\beta_v/(1-\xi)$, yielding:
<!-- bilingual-en:end -->

$$
\operatorname{ES}_\alpha
=E[L\mid L>v]
=v+\frac{\beta+\xi(v-u)}{1-\xi}
=\frac{v+\beta-\xi u}{1-\xi},\qquad \xi<1.
$$

也可以从 [[ES定义|上尾分位积分]] 推导：把 [[POT分位数]] 的 $v_q$ 从 $q=\alpha$ 积到 1。积分中的 $(1-q)^{-\xi}$ 仅在 $\xi<1$ 时可积。$\xi=0$ 给出 $ES_\alpha=v+\beta$；$\xi\ge1$ 则该 GPD 尾模型的 ES 为 $+\infty$，不能把同一个代数分式算出的负数或奇点当作有效 ES。
<!-- bilingual-en:start -->
Alternatively, integrate the [[POT分位数|POT quantile]] from $q=\alpha$ to 1 in the [[ES定义|upper-quantile definition of ES]]. The factor $(1-q)^{-\xi}$ is integrable only for $\xi<1$. At $\xi=0$, $ES_\alpha=v+\beta$. For $\xi\ge1$, this GPD tail model has $ES=+\infty$; a negative value or singularity from the algebraic fraction is not a valid ES.
<!-- bilingual-en:end -->

## 沿用同一损失模型
<!-- bilingual-en:start -->
*Continue with the same loss model*
<!-- bilingual-en:end -->

给定 $u=100$ 万元、$p_u=0.05,\xi=0.2,\beta=10$ 万元，[[POT分位数]] 已得到 $v_{0.99}=118.986483\ldots$ 万元。在这个分位数处，尺度为 $13.797297\ldots$ 万元，平均超额为 $17.246621\ldots$ 万元：
<!-- bilingual-en:start -->
For $u=100$ and $\beta=10$ in CNY 10,000, with $p_u=0.05,\xi=0.2$, the [[POT分位数|POT quantile]] is $v_{0.99}=118.986483\ldots$. Scale at that quantile is $13.797297\ldots$ and mean excess is $17.246621\ldots$ in the same units:
<!-- bilingual-en:end -->

$$
ES_{0.99}=118.986483\ldots+
\frac{10+0.2(118.986483\ldots-100)}{0.8}
=136.233104\ldots\ \text{万元}.
$$

分位数约 119 万元，尾部平均损失约 136.2 万元，平均超额约 17.25 万元，三者不可混称。计算保持同一组未舍入参数；这不证明现实尾部的均值已经被准确识别。若参数区间跨过 $\xi=1$，至少必须报告有限 ES 对尾形状的不稳定性。
<!-- bilingual-en:start -->
The quantile is about CNY 1.19 million, tail mean about CNY 1.362 million, and mean excess about CNY 172,500. Keep these three quantities distinct. Calculations use one unrounded parameter set; they do not establish accurate identification of the real-world tail mean. If an uncertainty interval crosses $\xi=1$, report at least the instability of finite ES with respect to tail shape.
<!-- bilingual-en:end -->

此处的严格超越条件均值简写来自模型尾部的连续性；原始经验分布是离散的，不能把这一步原封不动搬过去，见 [[ES严格超越均值边界]]。拟合出 $\xi\ge1$ 是该外推模型不产生有限 ES 的结论，不是数据已经证明真实世界损失均值无限。
<!-- bilingual-en:start -->
The strict-exceedance shortcut here uses continuity of the fitted tail. It cannot be transferred unchanged to a discrete empirical distribution; see [[ES严格超越均值边界|the ES strict-exceedance boundary]]. Fitting $\xi\ge1$ means the extrapolation model has no finite ES, not that the data have proved an infinite real-world loss mean.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，*Extreme Value Theory*，PDF 第 25、28 页](https://www.columbia.edu/~mh2078/QRM/EVT_MasterSlides.pdf#page=28)：已重开并目视 GPD 阈值稳定性、均值超额和 ES 公式；本页按超额均值及分位积分分别核算有限性条件。
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|第 13 章计算题 2]]：同一组参数与单位；独立复算 $136.233104\ldots$ 万元，区分原课逐步舍入值与高精度计算。
<!-- bilingual-en:start -->
- Haugh, PDF pp. 25 and 28, was reopened and visually checked for threshold stability, mean excess and ES. The finite-mean restriction was checked through both excess means and quantile integration.
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论|Chapter 13, Calculation Question 2]], provides the parameters and units. The independently calculated CNY 1,362,331.04 distinguishes full-precision evaluation from staged rounding in the course record.
<!-- bilingual-en:end -->
