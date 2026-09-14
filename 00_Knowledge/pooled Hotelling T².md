---
aliases:
  - "pooled Hotelling T² 在共同协方差假设下用合并组内协方差检验两个独立正态总体的均值向量差"
  - Pooled Hotelling T-squared uses a pooled within-group covariance to test the mean-vector difference between two independent normal populations with common covariance
  - 两独立样本 Hotelling T²
  - 共同协方差的多元两样本均值检验
student_os: knowledge-atom
atom_id: STAT-HOT-007
atom_set: hotelling-mean-inference
atom_type: theorem-procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[样本协方差Wishart律]]"
  - "[[正态均值协方差独立]]"
  - "[[Gaussian仿射闭包]]"
  - "[[Wishart抽样假设]]"
  - "[[Wishart秩与可逆性]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
related:
  - "[[单样本Hotelling T²]]"
  - "[[配对Hotelling T²]]"
  - "[[线性约束Hotelling T²]]"
---

# pooled Hotelling T² 在共同协方差假设下用合并组内协方差检验两个独立正态总体的均值向量差
<!-- bilingual-en:start -->
*Pooled Hotelling T-squared uses a pooled within-group covariance to test the mean-vector difference between two independent normal populations with common covariance*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设两组样本相互独立，组内分别 iid，且
> $$X_{1i}\sim N_p(\mu_1,\Sigma),\qquad
> X_{2j}\sim N_p(\mu_2,\Sigma),\qquad \Sigma\succ0.$$
> 令 $n_1,n_2\ge2$、$m=n_1+n_2-2$，并定义
> $$S_p=\frac{(n_1-1)S_1+(n_2-1)S_2}{m}.$$
> 检验 $H_0:\mu_1-\mu_2=\delta_0$ 使用
> $$T^2=(\bar X_1-\bar X_2-\delta_0)^T
> \left[S_p\left(\frac1{n_1}+\frac1{n_2}\right)\right]^{-1}
> (\bar X_1-\bar X_2-\delta_0).$$
> 若 $m\ge p$，则在 $H_0$ 下
> $$\frac{m-p+1}{pm}T^2\sim F_{p,m-p+1}.$$
> <!-- bilingual-en:start -->
> Pooling is exact because both independent groups estimate the same covariance matrix. The denominator degrees of freedom are $m-p+1=n_1+n_2-p-1$.
> <!-- bilingual-en:end -->

利用
$$
\left(\frac1{n_1}+\frac1{n_2}\right)^{-1}
=\frac{n_1n_2}{n_1+n_2},
$$
零差异情形也常写成
$$
T^2=\frac{n_1n_2}{n_1+n_2}
(\bar X_1-\bar X_2)^TS_p^{-1}(\bar X_1-\bar X_2).
$$
$m\ge p$ 等价于 $n_1+n_2>p+1$；这保证正定总体协方差下的 pooled within-group scatter 几乎必然满秩。$p=1$ 时公式退化为共同方差两样本 pooled t 检验的平方。

> [!warning] 边界
> - 若 $\Sigma_1\ne\Sigma_2$，$S_p$ 不再估计一个共同协方差，显示的精确 F 律失效；本卡不承诺把同一公式继续使用，也不把某个未说明的“稳健修正”当作自动替代。
> - 配对资料不是两独立样本；应先用 [[配对Hotelling T²|差向量化约]]。
> - $m=p$ 虽刚好可逆并留下一个分母自由度，但估计可能极不稳定；代数可逆不等于实际可靠。
> - 两组各自近似正态但存在跨组依赖时，独立样本的方差与精确校准仍不成立。

> [!question]- 自检
> 若两组协方差不同但样本量很大，能否仍引用本卡的精确 F 分布？
>
> **答案：** 不能。共同协方差是 pooled Wishart 校准的模型前提；不等协方差需要明确选择并验证另一套近似、置换或稳健方法。

## 来源与核验

- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.4. 两个独立总体均值向量比较|多元统计课程 §1.4]]：核对共同协方差、$S_p$、$T^2$ 与 F 转换，并核对不等协方差边界。
- [Penn State STAT 505, Lesson 7, §§7.1.12–7.1.15 and §7.2.7](https://online.stat.psu.edu/stat505/Lesson07)：核对独立两样本设计、pooled 统计量、精确自由度与不等协方差边界。
- [Stanford STATS 305C, *Hotelling T2*](https://web.stanford.edu/class/stats305c/lectures/Hotelling_T2.html)：独立核对 two-sample model、pooled covariance、$m=n_1+n_2-2$ 与 $F_{p,m-p+1}$ 校准。
