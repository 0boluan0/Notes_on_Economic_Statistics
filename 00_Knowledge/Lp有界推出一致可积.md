---
aliases:
  - '若随机变量族的某个 $p>1$ 阶绝对矩一致有界，则该族一致可积'
  - "A uniform Lp bound with p greater than one implies uniform integrability"
student_os: knowledge-atom
atom_id: PROB-MG-023
atom_set: martingales-stopping
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[一致可积]]"
related:
  - "[[L1有界不推一致可积]]"
  - "[[Markov不等式]]"
part_of:
  - "[[鞅与停时.canvas]]"
---

# 若随机变量族的某个 $p>1$ 阶绝对矩一致有界，则该族一致可积
<!-- bilingual-en:start -->
*A uniform bound on some absolute $p$th moment with $p>1$ implies uniform integrability*
<!-- bilingual-en:end -->

> [!summary] 常用充分条件
> 若存在 $p>1$ 与 $C<\infty$，使
> $$\sup_iE|X_i|^p\le C,$$
> 则随机变量族 $\{X_i\}$ 一致可积。
> <!-- bilingual-en:start -->
> If $\sup_iE|X_i|^p\le C<\infty$ for some $p>1$, then the family is uniformly integrable.
> <!-- bilingual-en:end -->

## 为什么高一阶的矩能压住尾部
<!-- bilingual-en:start -->
*Why the higher moment controls the tail*
<!-- bilingual-en:end -->

在事件 $\{|X_i|>K\}$ 上，

$$
|X_i|\le\frac{|X_i|^p}{K^{p-1}}.
$$

因此

$$
\sup_iE\!\left[|X_i|\mathbf1_{\{|X_i|>K\}}\right]
\le \frac{C}{K^{p-1}}
\longrightarrow0.
$$

右侧的界与 $i$ 无关，正好提供 [[一致可积]] 定义所需的共同阈值控制。
<!-- bilingual-en:start -->
On the tail event, $|X_i|\le |X_i|^p/K^{p-1}$. The resulting bound $C/K^{p-1}$ is uniform in the family index and tends to zero.
<!-- bilingual-en:end -->

$p>1$ 是这条证明的关键。只知道 $p=1$ 时分母不随 $K$ 增长，不能推出尾部趋零；对应反例见 [[L1有界不推一致可积]]。
<!-- bilingual-en:start -->
The strict inequality $p>1$ is essential for this argument. At $p=1$, the bound no longer decays with the threshold.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $\sup_nE[X_n^2]\le9$，怎样直接给出一致可积尾部的上界？
> <!-- bilingual-en:start -->
> If $\sup_nE[X_n^2]\le9$, what bound controls the uniform-integrability tail?
> <!-- bilingual-en:end -->
>
> **答案：** 取 $p=2$，有
> $$\sup_nE[|X_n|\mathbf1_{\{|X_n|>K\}}]\le9/K\to0.$$
> <!-- bilingual-en:start -->
> **Answer:** Taking $p=2$ gives the bound $9/K$, which tends to zero.
> <!-- bilingual-en:end -->

## 来源与核验

- [MIT OCW 18.445, Lecture 18, pp. 2–4](https://ocw.mit.edu/courses/18-445-introduction-to-stochastic-processes-spring-2015/e131f4c516e7f67ae27487d6a80f41d0_MIT18_445S15_lecture18.pdf#page=3)：核对统一 $L^p$ 界（$p>1$）推出一致可积及其尾部估计。
