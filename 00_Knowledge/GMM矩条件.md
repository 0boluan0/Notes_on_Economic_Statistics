---
aliases:
  - "GMM 通过最小化样本矩条件的加权距离估计使总体矩为零的参数"
  - "广义矩估计的矩条件准则"
  - "Moment-condition criterion of generalized method of moments"
student_os: knowledge-atom
atom_id: ECON-IV-014
atom_set: endogeneity-iv-gmm
atom_type: estimator-definition
status: source-checked
mastery_state: unassessed
part_of: "[[内生性与工具变量.canvas|内生性与工具变量]]"
requires:
  - "[[识别与估计]]"
  - "[[样本正交与总体外生性]]"
related:
  - "[[2SLS投影]]"
  - "[[工具变量有效条件]]"
leads_to:
  - "[[GMM权重矩阵]]"
  - "[[过度识别检验边界]]"
---

# GMM 通过最小化样本矩条件的加权距离估计使总体矩为零的参数
<!-- bilingual-en:start -->
*Generalized method of moments estimates a parameter that sets population moments to zero by minimizing the weighted distance of their sample analogues*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 设单个观测给出 $q$ 维矩函数 $m(W_i,\theta)$，理论和识别设计声称真参数 $\theta_0$ 满足
> $$
> E[m(W_i,\theta_0)]=0.
> $$
> 样本矩为
> $$
> \bar g_n(\theta)=\frac1n\sum_{i=1}^n m(W_i,\theta).
> $$
> GMM 选择
> $$
> \hat\theta=\arg\min_{\theta}\;
> \bar g_n(\theta)'W_n\bar g_n(\theta),
> $$
> 其中 $W_n$ 是正定或适当半正定的权重矩阵。它不是要求有限样本里的每个矩都精确为零，而是让它们按既定权重尽量接近零。
>
> <!-- bilingual-en:start -->
> GMM starts from population restrictions $E[m(W_i,\theta_0)]=0$, replaces them with sample averages, and chooses the parameter that minimizes their weighted quadratic distance from zero. In an overidentified sample the individual moments generally cannot all be set exactly to zero.
> <!-- bilingual-en:end -->

在线性工具变量模型 $y_i=x_i'\beta+u_i$ 中，一个典型矩条件是

$$
E[z_i(y_i-x_i'\beta_0)]=E[z_iu_i]=0.
$$

每个工具方向提供一个总体正交限制。2SLS 是线性 IV 的一种 GMM 实现；动态面板差分 GMM 则根据序列相关和外生性假设，用不同滞后构造更多矩条件。名称相同不代表矩条件自动相同，每一个滞后工具都必须由具体的数据生成过程支持。
<!-- bilingual-en:start -->
For linear IV, the moments are instrument–structural-error orthogonality conditions. 2SLS is one GMM implementation, while dynamic-panel GMM constructs lag moments from explicit assumptions about serial dependence and regressor exogeneity. The method's name never validates a proposed moment.
<!-- bilingual-en:end -->

一致性至少需要三层条件同时成立：

1. **矩条件正确：** 真参数确实使总体矩为零。
2. **识别成立：** $E[m(W_i,\theta)]=0$ 或相应总体准则在目标参数处有唯一解；只有“矩很多”不等于有足够秩。
3. **样本矩收敛：** 依赖结构、矩存在性和参数空间等正则条件足以让样本矩及准则一致逼近总体对象。

若矩条件无效，GMM 仍会返回一个让错误样本矩折中得最小的数，但它可能收敛到伪目标而非研究问题中的结构参数。若识别弱或不唯一，目标函数可以很平，常规渐近正态推断也会失效。
<!-- bilingual-en:start -->
Consistency needs valid population moments, unique identification, and a uniform law of large numbers or analogous regularity. Invalid moments can still produce a numerical minimizer, but that minimizer may converge to a pseudo-true compromise rather than the intended structural parameter. Weak or nonunique identification makes the criterion flat and conventional inference unreliable.
<!-- bilingual-en:end -->

条件矩 $E[u_i\mid z_i]=0$ 可以通过选取函数 $h(z_i)$ 推出多条无条件矩 $E[h(z_i)u_i]=0$，但增加函数也增加了需要维持的限制和有限样本负担。GMM 的价值是有纪律地组合已有经济或统计限制，不是从数据中无限制造工具。

> [!question]- 最小自检
> 优化器把 GMM 目标函数降得很低，是否已经证明所用矩条件在总体中正确？
>
> **答案：** 没有。低样本准则只说明这些样本矩在该参数附近相容；总体矩的有效性来自模型和识别设计，仍需来源与机制论证。

## 来源与核验

- Hansen (1982), [“Large Sample Properties of Generalized Method of Moments Estimators”](https://larspeterhansen.org/lph_research/large-sample-properties-of-generalized-method-of-moments-estimators/), *Econometrica* 50(4), 1029–1054：原始定义使样本总体正交条件接近零的估计量，并建立其一致性与渐近正态性条件。
- MIT OpenCourseWare 14.382, [Lecture 3: Structural Equations Models and GMM](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/f6b8cd36eee8df2b259c979a1fd2673e_MIT14_382S17_lec3.pdf)：核验样本矩、二次型准则、识别维数和权重矩阵的课程级推导。
- StataCorp, [`ivregress` manual](https://www.stata.com/manuals/rivregress.pdf)：核验线性 IV 中 $E[z_iu_i]=0$、样本矩准则和过度识别 GMM 实现。
- [[02_Economy/01_Econometrics/13_面板数据模型.md#4.2.3. 差分 GMM 方法|本地课程：差分 GMM]]：保留本地动态面板矩条件入口。
