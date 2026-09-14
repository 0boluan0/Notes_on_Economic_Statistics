---
aliases:
  - "相同 ACF 不能保证相同尾部与非线性依赖"
  - Matching ACF does not determine the process
student_os: knowledge-atom
atom_id: TS-STAT-020
atom_set: stationarity-ergodicity-spectrum
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[自协方差与ACF]]"
related:
  - "[[Gaussian过程均值协方差决定性]]"
  - "[[联合高斯独立判据]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
---

# 相同 ACF 不能保证相同尾部与非线性依赖
<!-- bilingual-en:start -->
*Matching the ACF does not guarantee matching tails or nonlinear dependence*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> ACF 只记录标准化二阶线性依赖。两个过程可有完全相同的 ACF，却有不同边际分布、极端风险、高阶矩或条件结构；只有在 Gaussian 等额外假设下，均值与协方差才足以确定全部有限维分布。
> <!-- bilingual-en:start -->
> The ACF records standardized second-order linear dependence. Processes with identical ACFs can differ in marginal laws, extremes, higher moments, or conditional structure; mean and covariance determine the full finite-dimensional law only under extra assumptions such as Gaussianity.
> <!-- bilingual-en:end -->

两个层次的反例：Gaussian 白噪声与 Rademacher 白噪声都有 $\rho(h)=0$（$h\ne0$），但尾部完全不同。进一步令 $X_t=\varepsilon_t\varepsilon_{t-1}$，其中 $\varepsilon_t$ 独立标准正态；$X_t$ 也有白噪声式 ACF，但相邻平方满足
$$
\operatorname{Cov}(X_t^2,X_{t-1}^2)=E[\varepsilon_t^2]E[\varepsilon_{t-1}^4]E[\varepsilon_{t-2}^2]-1\cdot1=3-1=2,
$$
所以存在 ACF 看不到的非线性依赖。
<!-- bilingual-en:start -->
Gaussian and Rademacher white noise share the same zero nonzero-lag ACF but have different tails. More strongly, $X_t=\varepsilon_t\varepsilon_{t-1}$ with IID standard-normal $\varepsilon_t$ also has a white-noise ACF, while $\operatorname{Cov}(X_t^2,X_{t-1}^2)=3-1=2$. The adjacent squares are therefore dependent even though the linear ACF is zero.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个零均值、单位方差过程的 ACF 在每个滞后都相同，是否已能推出它们的尾部风险和条件依赖也相同？
>
> **答案：** 不能。ACF 只保留二阶线性结构；边际尾部、高阶矩与非线性条件依赖都可以不同。

## 来源与核验

- [MIT OCW 6.450, Chapter 7, Theorem 7.5.2 and following example](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf#page=19)：核对 Gaussian 与离散系数过程可共享协方差而拥有不同完整分布。
- [[01_Math/06_时间序列分析/03_平稳时间序列模型.md#1.2. 三种‘没有关系’的辨析|课程：白噪声与非线性可预测性]]：核对序列不相关不排除非线性条件依赖。
<!-- bilingual-en:start -->
- MIT 6.450 was checked for processes sharing covariance while differing in full distribution.
- The local time-series example was checked for serial uncorrelatedness coexisting with nonlinear predictability.
<!-- bilingual-en:end -->
