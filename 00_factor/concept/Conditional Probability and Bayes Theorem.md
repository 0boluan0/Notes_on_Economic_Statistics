---
aliases: [Conditional Probability and Bayes Theorem, Conditional Probability, Bayes Theorem, 条件概率与贝叶斯定理]
tags: [concept, discrete-mathematics, probability]
---
# Conditional Probability and Bayes Theorem

当 $\Pr(B)>0$ 时，
$$
\Pr(A\mid B)=\frac{\Pr(A\cap B)}{\Pr(B)}.
$$
若 $H_1,\ldots,H_k$ 构成划分且分母非零，则
$$
\Pr(H_i\mid E)=
\frac{\Pr(E\mid H_i)\Pr(H_i)}{\sum_j\Pr(E\mid H_j)\Pr(H_j)}.
$$

条件化改变样本空间的权重；$\Pr(A\mid B)$ 与 $\Pr(B\mid A)$ 一般不同。

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
