---
aliases: [Sine Limit by Squeeze Theorem, Sine Limit, 三角极限夹逼证明, 正弦极限]
tags: [proof, calculus]
---
# Sine Limit by Squeeze Theorem

## 假设

$0<\theta<\pi/2$，并使用弧度制。

## 构造

在单位圆中比较内接三角形、扇形与外接三角形面积，可得

$$
\frac12\sin\theta
<\frac12\theta
<\frac12\tan\theta.
$$

## 推导

乘以 $2$，再除以正数 $\theta$：

$$
\frac{\sin\theta}{\theta}<1,
\qquad
1<\frac{\tan\theta}{\theta}
=\frac{\sin\theta}{\theta\cos\theta}.
$$

第二个不等式等价于

$$
\cos\theta<\frac{\sin\theta}{\theta}.
$$

所以

$$
\cos\theta<\frac{\sin\theta}{\theta}<1.
$$

当 $\theta\to0^+$ 时两端都趋于 $1$，由夹逼定理，中间项趋于 $1$。函数 $\sin\theta/\theta$ 为偶函数，因此左极限相同。

## 结论

$$
\lim_{\theta\to0}\frac{\sin\theta}{\theta}=1.
$$

## 关联卡片

- [[Limit]]
- [[Derivative]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
