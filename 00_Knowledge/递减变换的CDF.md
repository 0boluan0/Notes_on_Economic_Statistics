---
student_os: knowledge-atom
atom_id: 9d431e6c-099e-4b54-863e-4c047ab930df
status: needs-review
aliases:
  - "严格递减变换的 CDF 使用逆像处的左极限补概率"
  - "CDF of a decreasing transformation"
---

# 严格递减变换的 CDF 使用逆像处的左极限补概率

<!-- bilingual-en:start -->
*The CDF of a decreasing transformation uses the complementary left limit at the inverse*
<!-- bilingual-en:end -->

若 $Y=g(X)$ 且 $g$ 在支持范围上严格递减，对逆函数可定义的 $y$，
$$F_Y(y)=P(X\ge g^{-1}(y))=1-F_X(g^{-1}(y)-).$$
$F_X(a-)=P(X<a)$。只有在该点没有质量时，才能把左极限换成 $F_X(a)$；范围外还需补上 0 或 1。

<!-- bilingual-en:start -->
A decreasing transform reverses the inequality. Its CDF uses the complement of the source CDF’s left limit at the inverse value. The ordinary CDF can replace that limit only when there is no atom at the threshold.
<!-- bilingual-en:end -->
若 $P(X=0)=P(X=1)=1/2$、$Y=-X$，则 $F_Y(0)=1-F_X(0-)=1$；错误地减 $F_X(0)$ 会得到 $1/2$。连续均匀情形中 $Y=3-2X$ 则可直接用 $1-F_X((3-y)/2)$。

<!-- bilingual-en:start -->
An equal-mass variable at zero and one transformed by negation has CDF value one at zero. Using the ordinary source CDF there incorrectly gives one half. The simpler expression works for a continuous uniform source.
<!-- bilingual-en:end -->

**关联：** [[累积分布函数]] · [[随机变量的分布变换]]

## 来源与核验

- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=40|EC400 PSI Lecture 2，slide 40]]：支持递减方向与连续情形；左极限修正由 slide 9 的 CDF 定义直接推出。

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 2 - Statistics I.pdf#page=40|EC400 PSI Lecture 2, slide 40]]: Supports decreasing transforms; the left-limit correction follows from the CDF definition on slide 9.
<!-- bilingual-en:end -->

- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 正式笔记]]：保留本课的完整算例与讲解语境。

<!-- bilingual-en:start -->
- [[04_PSI Lecture 2 - 随机变量、条件分布与独立性|EC400 PSI Lecture 2 course note]] retains the full worked examples and lecture context.
<!-- bilingual-en:end -->
