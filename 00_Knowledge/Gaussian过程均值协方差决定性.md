---
aliases:
  - "Gaussian 过程的有限维分布由均值与协方差完全决定"
  - Gaussian process determined by mean and covariance
student_os: knowledge-atom
atom_id: TS-STAT-017
atom_set: stationarity-ergodicity-spectrum
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[联合Gaussian]]"
  - "[[有限维分布]]"
related:
  - "[[Gaussian宽严平稳等价]]"
  - "[[联合高斯独立判据]]"
  - "[[ACF信息边界]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
---

# Gaussian 过程的有限维分布由均值与协方差完全决定
<!-- bilingual-en:start -->
*The finite-dimensional distributions of a Gaussian process are determined by its mean and covariance*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> Gaussian 过程要求任意有限向量 $(X_{t_1},\ldots,X_{t_k})$ 联合 Gaussian。每个这样的向量的分布由均值向量和协方差矩阵完全决定，所以过程的均值函数 $m(t)$ 与协方差核 $K(s,t)$ 共同决定全部有限维分布。
> <!-- bilingual-en:start -->
> A Gaussian process has jointly Gaussian finite vectors. Each such vector is completely determined by its mean vector and covariance matrix, so the process mean function and covariance kernel determine all finite-dimensional laws.
> <!-- bilingual-en:end -->

这是一项 Gaussian 特例，不是“任何过程都由前两个矩决定”。非 Gaussian 过程可以共享同一均值和协方差，却有不同偏度、尾部、联合极端和非线性依赖。
<!-- bilingual-en:start -->
This is a Gaussian exception, not a general rule. Non-Gaussian processes can share the same first two moments while differing in skewness, tails, joint extremes, or nonlinear dependence.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个过程的均值函数和协方差核完全相同，何时这些信息足以推出它们的所有有限维分布也相同？
>
> **答案：** 当两者都是 Gaussian 过程时可以。Gaussian 有限向量由均值向量和协方差矩阵决定；一般非 Gaussian 过程则不足。

## 来源与核验

- [[01_Math/05_随机过程/02_随机过程的概念和分类.docx|随机过程课程稿]]：核对任意有限时点联合多元正态的课程定义。
- [MIT OCW 6.450, Chapter 7, Sections 7.3 and 7.5](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf)：核对 Gaussian finite vector 及均值协方差决定联合分布。
<!-- bilingual-en:start -->
- The local course definition was checked for joint Gaussianity at every finite set of times.
- MIT 6.450 was checked for determination of Gaussian finite-dimensional laws by means and covariances.
<!-- bilingual-en:end -->
