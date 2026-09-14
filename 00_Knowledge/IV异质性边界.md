---
aliases:
  - "处理效应异质时，二元工具变量的 Wald 比率在额外假设下识别服从者的 LATE，而非自动识别 ATE"
  - "工具变量异质处理效应的 LATE 边界"
  - "Local average treatment effect boundary for a binary instrument"
student_os: knowledge-atom
atom_id: ECON-IV-012
atom_set: endogeneity-iv-gmm
atom_type: estimand-boundary
status: source-checked
mastery_state: unassessed
part_of: "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[IV比率估计]]"
  - "[[工具外生与排除]]"
related:
  - "[[工具变量有效条件]]"
  - "[[识别与估计]]"
---

# 处理效应异质时，二元工具变量的 Wald 比率在额外假设下识别服从者的 LATE，而非自动识别 ATE
<!-- bilingual-en:start -->
*With heterogeneous treatment effects, a binary-instrument Wald ratio identifies the compliers' LATE under additional assumptions rather than automatically identifying the ATE*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 令二元工具为 $Z_i\in\{0,1\}$，二元处理为 $D_i\in\{0,1\}$，$D_i(z)$ 是工具取值为 $z$ 时的潜在处理。在下面的联合独立、排除、非零第一阶段、单调性以及一致性/无干扰条件下，
> $$
> \frac{E[Y_i\mid Z_i=1]-E[Y_i\mid Z_i=0]}
> {E[D_i\mid Z_i=1]-E[D_i\mid Z_i=0]}
> =E[Y_i(1)-Y_i(0)\mid D_i(1)>D_i(0)].
> $$
> 右侧是因工具从 0 变 1 而接受处理的“服从者”平均效应，即 LATE（局部平均处理效应），不是自动等于全体的 ATE（平均处理效应）。
>
> <!-- bilingual-en:start -->
> Under instrument independence, exclusion, relevance, and monotonicity, the binary-instrument Wald ratio identifies the average treatment effect for compliers—units whose treatment status is changed by the instrument. It does not automatically identify the population average treatment effect.
> <!-- bilingual-en:end -->

这里必须同时维持五类条件：

1. **联合独立性：**
   $$
   (Y_i(0),Y_i(1),D_i(0),D_i(1))\perp Z_i.
   $$
   “近似随机”是支持这一假设的制度论证，不是定理中的替代条件。
2. **排除限制：**
   $$
   Y_i(d,z)=Y_i(d)
   $$
   对所有相关 $d,z$ 成立，即工具不直接改变结果。
3. **非零第一阶段：**
   $$
   E[D_i\mid Z_i=1]-E[D_i\mid Z_i=0]>0.
   $$
4. **单调性：** $D_i(1)\ge D_i(0)$ 对每个人成立，即不存在 defier；方向必须由制度机制支持，而不是由数据自动验证。
5. **一致性与无干扰：** 实际观察到的处理和结果等于相应潜在值，且他人的工具或处理不会改变本人的潜在处理或结果。

<!-- bilingual-en:start -->
The result requires more than instrument relevance: as-if random assignment of the instrument, exclusion, a nonzero first stage, monotonicity with no defiers, and a stable-treatment/no-interference setup. Monotonicity is a substantive claim about how the instrument changes participation, not a pattern established by the first-stage regression alone.
<!-- bilingual-en:end -->

若工具只在给定协变量 $W$ 后近似随机，上面的原始无条件 Wald 比率一般不再直接成立；应先识别条件于 $W$ 的局部效应，再按相应第一阶段权重聚合，或使用与该条件结构相匹配的饱和 IV/2SLS。把 $W$ 只当作普通线性控制而不说明权重，可能改变实际汇总的服从者人群。

“局部”指的是由这个具体工具推动的那群人。换一个工具，即使都有效，也可能推动不同的服从者，从而识别不同的 LATE。若处理效应异质，这些局部效应可以不同；因此不同工具得到不同系数不必自动解释为某一个估计出了错。
<!-- bilingual-en:start -->
Local means local to the margin shifted by this particular instrument. Different valid instruments can move different complier groups and therefore identify different LATEs when treatment effects are heterogeneous.
<!-- bilingual-en:end -->

只有再加上处理效应同质，或直接成立
$$
E[Y(1)-Y(0)\mid D(1)>D(0)]=E[Y(1)-Y(0)],
$$
或有可信的运输结构，LATE 才能等同于目标总体的 ATE。反过来，LATE 也不因“局部”而无用：若政策正好改变与工具相同的参与边际，它可能是直接相关的政策参数；关键是把人群和工具机制说清楚。
<!-- bilingual-en:start -->
Equating LATE with ATE needs further structure, such as homogeneous effects or a valid transport argument. A local effect can still be policy-relevant when the policy moves the same participation margin; the estimand and complier population must simply be stated explicitly.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 两个都满足排除限制的二元工具给出不同 Wald 比率，是否一定说明至少一个工具无效？
>
> **答案：** 不一定。若处理效应异质，两个工具可能推动不同服从者群体，从而识别不同 LATE；仍需分别审查各自的外生性、排除限制和单调性。

## 来源与核验

- Imbens and Angrist (1994), [“Identification and Estimation of Local Average Treatment Effects”](https://www.nber.org/papers/t0118), *Econometrica* 62(2), 467–475：原始证明有效工具本身不足以识别有意义的平均效应，并在额外参与单调性条件下识别由工具改变参与状态者的局部平均效应。
- MIT OpenCourseWare 14.387, [Instrumental Variables: Causal Effects in a Heterogeneous World](https://ocw.mit.edu/courses/14-387-applied-econometrics-mostly-harmless-big-data-fall-2014/02fdeb479fb5683f9d723db7b289a9d0_MIT14_387F14_Causaleffects.pdf)：逐项核验工具独立、排除限制、单调性、第一阶段与 LATE 定理。
