---
aliases:
  - "Wold 分解允许确定性成分而纯非确定平稳过程才由单边创新展开"
  - Wold decomposition
  - Purely nondeterministic process
  - Wold representation boundary
  - 沃尔德分解
student_os: knowledge-atom
atom_id: TS-ARMA-011
atom_set: arma-modeling
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[宽平稳定义]]"
  - "[[白噪声二阶定义]]"
related:
  - "[[创新]]"
  - "[[ARMA无限MA表示]]"
  - "[[谱峰解释边界]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# Wold 分解允许确定性成分而纯非确定平稳过程才由单边创新展开
<!-- bilingual-en:start -->
*Wold decomposition allows a deterministic component; only the purely nondeterministic part is a one-sided innovation process*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 一个零均值、离散时间宽平稳过程可正交分解为
> $$X_t=D_t+\sum_{j=0}^{\infty}\psi_j\varepsilon_{t-j},$$
> 其中 $D_t$ 是可由无限过去线性预测的确定性成分，第二项是纯非确定性成分，$\sum_j\psi_j^2<\infty$，$\varepsilon_t$ 是一步线性创新。
> <!-- bilingual-en:start -->
> Wold decomposition writes a covariance-stationary process as an orthogonal sum of a linearly deterministic component and a purely nondeterministic one-sided moving average of innovations.
> <!-- bilingual-en:end -->

这里的“创新”与“可预测”都是**线性投影**概念。令
$$\mathcal H_{t-1}=\overline{\operatorname{span}}\{X_{t-1},X_{t-2},\ldots\},$$
则一步创新是 $\varepsilon_t=X_t-P_{\mathcal H_{t-1}}X_t$，与整个过去闭线性空间正交；确定性部分属于可由任意遥远过去线性恢复的交空间。除非另加 Gaussian 等条件，$P_{\mathcal H_{t-1}}X_t$ 不等于任意非线性信息集下的条件期望。

如果过程纯非确定，则 $D_t=0$，整个过程都有单边创新表示。一般平稳过程却可能保留完全可预测的周期或其他确定性成分；不能把“宽平稳”直接改写成“全部由白噪声驱动的 MA($\infty$)”。

因果 ARMA 是 Wold 纯非确定部分的一类有限参数化：其 $\psi_j$ 受有理传递函数约束并通常几何衰减。Wold 定理本身不保证有限阶 ARMA，也不把确定性成分消失作为免费结论。
<!-- bilingual-en:start -->
Here prediction means orthogonal projection onto the closed linear span of past observations, not an arbitrary nonlinear conditional expectation. If the process is purely nondeterministic, the deterministic component vanishes. Stationarity alone does not guarantee this. A causal ARMA is a finite-parameter rational subclass of the purely nondeterministic Wold component; Wold's theorem does not imply a finite ARMA order.
<!-- bilingual-en:end -->

> [!question]- 自检
> “每个宽平稳过程都是白噪声的单边 MA($\infty$)”少了什么条件？
>
> **答案：** 少了纯非确定性；一般 Wold 分解还包含可由无限过去预测的确定性成分。

## 来源与核验

- [Berkeley Stat 153, Lecture 16](https://www.stat.berkeley.edu/~bartlett/courses/153-fall2010/lectures/16.pdf)：核对 Wold 分解的 deterministic 与 purely nondeterministic 两部分。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对线性创新表示及其与 ARMA 的关系。
