---
aliases:
  - "实 Gaussian 过程宽平稳时也严平稳"
  - Gaussian WSS implies strict stationarity
  - Gaussian strict and weak stationarity equivalence
student_os: knowledge-atom
atom_id: TS-STAT-018
atom_set: stationarity-ergodicity-spectrum
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[严平稳定义]]"
  - "[[宽平稳定义]]"
  - "[[Gaussian过程均值协方差决定性]]"
related:
  - "[[宽平稳非严平稳反例]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
---

# 实 Gaussian 过程宽平稳时也严平稳
<!-- bilingual-en:start -->
*A real Gaussian process that is wide-sense stationary is also strictly stationary*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 对实 Gaussian 过程，宽平稳给出常均值和只依赖时差的协方差。任意有限向量共同平移后，其均值向量与协方差矩阵不变；Gaussian 联合分布由这两者完全决定，因此有限维分布不变，过程严平稳。
> <!-- bilingual-en:start -->
> For a real Gaussian process, WSS keeps every shifted finite vector's mean vector and covariance matrix unchanged. Because those quantities determine a jointly Gaussian law, every finite-dimensional distribution is shift-invariant and the process is strictly stationary.
> <!-- bilingual-en:end -->

反方向也成立：Gaussian 变量都有有限二阶矩，所以严平稳 Gaussian 过程满足“严平稳 + 有限二阶矩 ⇒ 宽平稳”。这项等价不能推广到任意非 Gaussian 过程；复 Gaussian 情形还需留意 pseudo-covariance 的约定。
<!-- bilingual-en:start -->
The converse also holds because Gaussian variables have finite second moments. The equivalence is not valid for arbitrary non-Gaussian processes, and complex Gaussian conventions may additionally require pseudo-covariance information.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么一般过程的宽平稳不能推出严平稳，而实 Gaussian 过程可以？
>
> **答案：** 宽平稳只保留平移前后的均值和协方差；对联合 Gaussian 有限向量，这两者已决定全部联合分布。

## 来源与核验

- [MIT 6.011 complete notes, Section 9.3](https://ocw.mit.edu/courses/6-011-introduction-to-communication-control-and-signal-processing-spring-2010/a6bddaee5966f6e73450e6fe79ab0566_MIT6_011S10_notes.pdf)：直接核对 Gaussian WSS 推出 SSS。
- [MIT OCW 6.450, Chapter 7, Theorem 7.5.1](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf#page=18)：核对 Gaussian stationarity 的常均值与 lag-only covariance 判据。
<!-- bilingual-en:start -->
- MIT 6.011 Section 9.3 directly states the Gaussian WSS-to-SSS implication.
- MIT 6.450 Theorem 7.5.1 was checked for the Gaussian mean-and-covariance stationarity criterion.
<!-- bilingual-en:end -->
