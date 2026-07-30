---
aliases: [Mean Value Theorem Proof, 平均值定理证明, 拉格朗日中值定理证明]
tags: [proof, calculus]
---
# Mean Value Theorem Proof

## 假设

$f$ 在 $[a,b]$ 上连续、在 $(a,b)$ 上可导，且 $a<b$。

## 目标

证明存在 $c\in(a,b)$ 使

$$
f'(c)=\frac{f(b)-f(a)}{b-a}.
$$

## 构造

令端点割线为

$$
\ell(x)=f(a)+\frac{f(b)-f(a)}{b-a}(x-a),
$$

并定义 $g(x)=f(x)-\ell(x)$。

## 推导

$g$ 继承连续性和可导性，并且 $g(a)=g(b)=0$。由 [[Rolle's Theorem]]，存在 $c\in(a,b)$ 使 $g'(c)=0$。而

$$
g'(c)=f'(c)-\frac{f(b)-f(a)}{b-a},
$$

所以目标等式成立。

## 结论

曲线上至少存在一条切线与端点割线平行。

## 关联卡片

- [[Lagrange Mean Value Theorem]]
- [[Rolle's Theorem]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM "" WHERE (contains(file.path, "01_Math/") OR contains(file.path, "02_Economy/") OR contains(file.path, "03_Computer_Science/")) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
