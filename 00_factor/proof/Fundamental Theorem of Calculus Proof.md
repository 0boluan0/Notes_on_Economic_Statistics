---
aliases: [Fundamental Theorem of Calculus Proof, FTC Proof, 微积分基本定理证明]
tags: [proof, calculus]
---
# Fundamental Theorem of Calculus Proof

## FTC I：累积函数的导数

假设 $f$ 在 $[a,b]$ 连续，定义 $A(x)=\int_a^xf(t)\,dt$。对允许的 $h\ne0$，

$$
\frac{A(x+h)-A(x)}{h}
=\frac1h\int_x^{x+h}f(t)\,dt.
$$

由积分平均值定理，存在位于 $x$ 与 $x+h$ 之间的 $c_h$，使右侧等于 $f(c_h)$。当 $h\to0$ 时，$c_h\to x$；连续性给出 $f(c_h)\to f(x)$，故 $A'(x)=f(x)$。

## FTC II：用原函数计算定积分

若 $F'=f$，则由 FTC I，$A'=f$。因此 $(F-A)'=0$；由平均值定理，$F-A$ 在区间上为常数。因为 $A(a)=0$，有 $A(x)=F(x)-F(a)$。令 $x=b$：

$$
\int_a^bf(x)\,dx=F(b)-F(a).
$$

## 边界

上述证明使用了 $f$ 的连续性；更弱条件下仍有推广，但不属于本课程结论。

## 关联卡片

- [[Fundamental Theorem of Calculus]]
- [[Definite Integral]]
- [[Antiderivative]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
