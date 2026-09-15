---
student_os: knowledge-atom
atom_id: 71c13e1d-a0fe-40c0-ae66-2056c0e72534
status: source-checked
aliases:
  - "均方误差等于估计量方差加偏差的平方"
  - "Mean squared error equals estimator variance plus squared bias"
---

# 均方误差等于估计量方差加偏差的平方

<!-- bilingual-en:start -->
*Mean squared error equals estimator variance plus squared bias*
<!-- bilingual-en:end -->

> [!summary] 核心
> 对固定目标 $\theta$ 和二阶矩有限的估计量 $T$，$E[(T-\theta)^2]=\operatorname{Var}(T)+(E[T]-\theta)^2$。偏差项必须平方，两个分量均非负。
>
> <!-- bilingual-en:start -->
> For a fixed target and an estimator with finite second moment, MSE equals variance plus squared bias. Both components are nonnegative.
> <!-- bilingual-en:end -->
^core

令 $m=E[T]$，把误差写为 $(T-m)+(m-\theta)$，展开平方后取期望：
$$E[(T-\theta)^2]=E[(T-m)^2]+2(m-\theta)E[T-m]+(m-\theta)^2.$$
因为 $E[T-m]=m-m=0$，交叉项消失，剩下[[方差]]与[[估计量偏差|偏差]]平方。偏差 2、方差 4 给出 MSE $2^2+4=8$，可优于偏差 0、方差 16 的无偏估计量。

<!-- bilingual-en:start -->
Add and subtract the estimator’s expectation, expand the square, and take expectations. The cross term vanishes because the centred estimator has mean zero. Bias 2 and variance 4 give MSE 8, better than an unbiased estimator with variance 16.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*PSI Lecture 1 — Foundations*（slides 66，PDF p.66）：支持本条在课程中的定义、条件与用途。

<!-- bilingual-en:start -->
*The EC400 lecture supplies the course context and notation. Additional cited references support the stated definitions, assumptions, or boundaries; worked arithmetic and direct implications are checked explicitly.*
<!-- bilingual-en:end -->
