---
aliases:
  - "自协方差去除均值而自相关一词有两种常见口径"
  - Autocovariance versus autocorrelation
  - ACF normalization
student_os: knowledge-atom
atom_id: TS-STAT-008
atom_set: stationarity-ergodicity-spectrum
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[宽平稳定义]]"
related:
  - "[[自协方差核条件]]"
  - "[[ACF信息边界]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# 自协方差去除均值而自相关一词有两种常见口径
<!-- bilingual-en:start -->
*Autocovariance removes the mean, while “autocorrelation” has two common conventions*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> 对均值为 $\mu$ 的实值宽平稳过程，
> $$
> \gamma(h)=E[(X_{t+h}-\mu)(X_t-\mu)]
> $$
> 是自协方差。时间序列中的 ACF 通常指标准化自协方差 $\rho(h)=\gamma(h)/\gamma(0)$；部分工程课程则把 raw second moment
> $$
> R(h)=E[X_{t+h}X_t]=\gamma(h)+\mu^2
> $$
> 称为 autocorrelation function。非零均值时三者不能混写。
> <!-- bilingual-en:start -->
> For a real WSS process with mean $\mu$, autocovariance is $\gamma(h)=E[(X_{t+h}-\mu)(X_t-\mu)]$. In time-series work the ACF usually means $\rho(h)=\gamma(h)/\gamma(0)$, whereas some engineering texts call the raw second moment $R(h)=E[X_{t+h}X_t]$ the autocorrelation function. These differ when the mean is nonzero.
> <!-- bilingual-en:end -->

$\rho(h)$ 无量纲并满足 $\rho(0)=1$，前提是 $\gamma(0)>0$。若过程退化为常数，$\gamma(0)=0$，标准化 ACF 的除法没有定义；此时不能机械写成 1。
<!-- bilingual-en:start -->
$\rho(h)$ is dimensionless and has $\rho(0)=1$ only when $\gamma(0)>0$. A degenerate constant process has zero variance, so the normalized ACF is undefined rather than mechanically equal to one.
<!-- bilingual-en:end -->

> [!question]- 自检
> $E[X_t]=2$ 且 $\gamma(3)=1$ 时，raw correlation $R(3)$ 是多少？
>
> **答案：** $R(3)=\gamma(3)+\mu^2=1+4=5$；它不是标准化 ACF。

## 来源与核验

- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|随机过程课程稿]]：核对 raw 相关函数与去均值互协方差的课程口径。
- [[01_Math/06_时间序列分析/lecture.pdf#page=89|时间序列讲义 Stationarity]]：核对 $\rho_h=\gamma_h/\gamma_0$ 的 ACF 口径。
- [MIT OCW 6.450, Chapter 7, Sections 7.2 and 7.5](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf)：核对 covariance 与非零均值分解。
<!-- bilingual-en:start -->
- The two local courses were checked for their different correlation conventions.
- MIT 6.450 was checked for the covariance-centering identity.
<!-- bilingual-en:end -->
