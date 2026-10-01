---
student_os: knowledge-atom
atom_id: c826ad79-e17e-4102-8561-4b5cac2083df
status: needs-review
aliases:
  - "后验赔率等于先验赔率乘以证据的似然比"
  - "Bayes rule in odds form"
  - "Posterior odds"
---

# 后验赔率等于先验赔率乘以证据的似然比

<!-- bilingual-en:start -->
*Posterior odds equal prior odds multiplied by a likelihood ratio*
<!-- bilingual-en:end -->

当有关分母非零，且 $0<P(A)<1$ 时，
$$\frac{P(A\mid B)}{P(A^c\mid B)}=\frac{P(A)}{P(A^c)}\frac{P(B\mid A)}{P(B\mid A^c)}.$$
由 $A,A^c$ 的两个 Bayes 公式相除得到，证据的归一化分母抵消。

<!-- bilingual-en:start -->
With nonzero denominators and a nondegenerate prior, dividing the two Bayes formulas cancels the evidence normaliser and multiplies prior odds by the likelihood ratio.
<!-- bilingual-en:end -->
概率 $p$ 对应赔率 $o=p/(1-p)$，反向为 $p=o/(1+o)$。先验 0.1、两种状态下的信号概率 0.8 与 0.2，给先验赔率 $1/9$、似然比 4、后验赔率 $4/9$，后验概率为 $4/13$。似然比大于 1 提高赔率，不是直接把概率乘同样倍数。

<!-- bilingual-en:start -->
Convert probabilities to odds and back before and after multiplication. In the default example, odds 1/9 multiplied by likelihood ratio four give odds 4/9 and probability 4/13. A likelihood ratio multiplies odds, not probability.
<!-- bilingual-en:end -->

**关联：** [[Bayes法则]] · [[似然函数]]

## 来源与核验

- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=59|EC400 PSI Lecture 2，slide 59]]：支持赔率恒等式及似然比方向；slide 58 给数值模型。

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=59|EC400 PSI Lecture 2, slide 59]]: Supports the odds identity and evidence direction; slide 58 provides the example.
<!-- bilingual-en:end -->

- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 正式笔记]]：保留本课的完整算例与讲解语境。

<!-- bilingual-en:start -->
- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 course note]] retains the full worked examples and lecture context.
<!-- bilingual-en:end -->
