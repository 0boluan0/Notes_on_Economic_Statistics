---
aliases:
  - "单样本 Hotelling T² 用样本协方差逆度量样本均值与目标向量的距离并在经典正态条件下作精确 F 检验"
  - One-sample Hotelling T-squared measures a sample-mean departure with the inverse sample covariance and has an exact F calibration under the classical normal model
  - 单样本 Hotelling T² 检验
  - 单总体均值向量检验
student_os: knowledge-atom
atom_id: STAT-HOT-001
atom_set: hotelling-mean-inference
atom_type: theorem-procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[样本协方差Wishart律]]"
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
implies:
  - "[[Hotelling投影极值]]"
  - "[[Hotelling置信椭球]]"
related:
  - "[[线性约束Hotelling T²]]"
  - "[[配对Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

# 单样本 Hotelling T² 用样本协方差逆度量样本均值与目标向量的距离并在经典正态条件下作精确 F 检验
<!-- bilingual-en:start -->
*One-sample Hotelling T-squared measures a sample-mean departure with the inverse sample covariance and has an exact F calibration under the classical normal model*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 设 $X_1,\ldots,X_n\overset{iid}{\sim}N_p(\mu,\Sigma)$，其中 $\Sigma\succ0$ 且 $n>p$。用分母 $n-1$ 定义样本协方差 $S$。检验
> $$H_0:\mu=\mu_0$$
> 时，
> $$T^2=n(\bar X-\mu_0)^TS^{-1}(\bar X-\mu_0),$$
> 并且在 $H_0$ 下
> $$F=\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.$$
> 因而水平 $\alpha$ 的检验在 $F>F_{p,n-p}(1-\alpha)$ 时拒绝 $H_0$。
> <!-- bilingual-en:start -->
> The statistic is a covariance-adjusted squared distance. Iid multivariate normality, positive-definite covariance, and $n>p$ give the displayed exact finite-sample F pivot.
> <!-- bilingual-en:end -->

$T^2$ 用 $S^{-1}$ 缩小高变异方向的偏离、放大低变异方向的偏离，所以它不是各坐标平方差的简单相加。精确 F 律来自两块同时成立的 Gaussian 抽样事实：$\bar X$ 与 $S$ 独立，且 $(n-1)S$ 服从 Wishart 分布。似然比检验也给出同一拒绝顺序，因为
$$
\Lambda^{2/n}=\left(1+\frac{T^2}{n-1}\right)^{-1}
$$
随 $T^2$ 严格递减。

当 $p=1$ 时，$T^2=t_{n-1}^2$，而 $F_{1,n-1}$ 正是 $t_{n-1}^2$ 的分布；这核对了它确为一元 t 检验的多元推广。

> [!warning] 边界
> - 若 $n\le p$，在 $\Sigma\succ0$ 的正态样本下 $S$ 仍几乎必然奇异，不能把伪逆代入后继续声称上面的 F 律；精确秩门槛复用 [[Wishart秩与可逆性]]。
> - 非正态、依赖或异质分布下，上面的有限样本 F 等式一般不成立。在 $H_0$ 下，固定 $p$、iid、有限二阶矩且总体协方差正定时，$T^2\Rightarrow\chi_p^2$ 可作大样本校准；这不是有限样本等式，也不能修复 $p$ 随 $n$ 增长或样本协方差奇异的问题。
> - “没有拒绝”只表示数据不足以排除 $\mu_0$，不证明 $\mu=\mu_0$。

> [!question]- 自检
> 为什么只知道每个 $X_i$ 有同一个均值和协方差，还不足以得到精确 F 分布？
>
> **答案：** 精确校准需要正态样本下的 Wishart 抽样律以及 $\bar X\perp\!\!\!\perp S$；相同的一、二阶矩本身不决定 $S$ 的完整分布。

## 来源与核验

- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.2. 均值向量的假设检验|多元统计课程 §1.2]]：核对统计量、F 转换与拒绝域；§§1.4、1.7 核对似然比单调关系与固定维数大样本近似。
- [Penn State STAT 505, Lesson 7, §§7.1.1–7.1.3](https://online.stat.psu.edu/stat505/Lesson07)：核对单样本问题、精确 F 校准与检验步骤。
- [NIST/SEMATECH e-Handbook, Hotelling's T-squared](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm)：独立核对一元 t 到多元 $T^2$ 的推广及精确 F 分布。
- [Stanford STATS 305C, *One sample problem*](https://web.stanford.edu/class/stats305c/lectures/Onesample.html)：核对 Wishart、似然比与 $T^2$ 的关系。
