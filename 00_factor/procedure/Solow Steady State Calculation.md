---
aliases:
- Solow Steady State Calculation
- 索罗稳态计算
- 索罗模型稳态计算步骤
tags:
- procedure
- economics
type: procedure
---
# Solow Steady State Calculation

## 输入

- 生产函数 $y=f(k)$。
- 储蓄率 $s$。
- 人口增长率 $n$。
- 折旧率 $\delta$。
- 若题目给技术进步，再加入技术进步率 $g$。

## 输出

- 稳态人均资本 $k^*$。
- 稳态人均产出 $y^*$。
- 参数变化对稳态水平的影响。

## Step 1：写资本积累方程

无技术进步时：

$$
\dot{k}=sf(k)-(n+\delta)k
$$

有劳动增进型技术进步时，常写成：

$$
\dot{k}=sf(k)-(n+g+\delta)k
$$

## Step 2：令资本变化为 0

稳态条件是：

$$
\dot{k}=0
$$

所以：

$$
sf(k^*)=(n+\delta)k^*
$$

或有技术进步时：

$$
sf(k^*)=(n+g+\delta)k^*
$$

## Step 3：代入生产函数求 $k^*$

若 $f(k)=k^\alpha$：

$$
sk^{\alpha}=(n+\delta)k
$$

整理得到：

$$
k^*=\left(\frac{s}{n+\delta}\right)^{\frac{1}{1-\alpha}}
$$

## Step 4：求 $y^*$

把 $k^*$ 代回生产函数：

$$
y^*=f(k^*)
$$

## Step 5：解释比较静态

- $s$ 上升：$k^*$ 和 $y^*$ 上升。
- $n$ 上升：$k^*$ 和 $y^*$ 下降。
- $\delta$ 上升：$k^*$ 和 $y^*$ 下降。
- $g$ 上升：有效劳动人均资本稳态下降，但长期人均产出增长率提高。

## 检查点

- 题目问的是总量、人均变量，还是有效劳动人均变量。
- 技术进步是否进入盈亏平衡投资项。
- 生产函数是否已经写成人均形式。
- 只要问长期人均增长率，必须看是否有技术进步。

## 来自课程位置

- [[06_经济增长理论#3.3 模型的稳态分析]]

## 关联卡片

- [[Solow Model]]
- [[Steady State Analysis]]
- [[Solow Model Interpretation]]
- [[Growth Theory-hub]]

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
