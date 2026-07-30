---
aliases: [Harmonic Series Divergence, 调和级数发散证明, 调和级数发散]
tags: [proof, calculus, discrete-mathematics]
---
# Harmonic Series Divergence

## 目标

证明 $\sum_{n=1}^{\infty}1/n$ 发散。

## 构造

把首项后的项按长度为 $1,2,4,8,\ldots$ 分组：

$$
1+\frac12
+\left(\frac13+\frac14\right)
+\left(\frac15+\cdots+\frac18\right)+\cdots.
$$

## 推导

第 $k$ 个二进制分组包含 $2^{k-1}$ 项，每一项至少为 $1/2^k$，所以该组和至少为

$$
2^{k-1}\cdot\frac1{2^k}=\frac12.
$$

因此取到第 $m$ 组的部分和至少为 $1+m/2$，随 $m\to\infty$ 无界。

## 结论

调和级数发散，说明 $a_n\to0$ 只是级数收敛的必要条件，不是充分条件。

## 关联卡片

- [[Series]]
- [[Series Convergence Tests]]

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
