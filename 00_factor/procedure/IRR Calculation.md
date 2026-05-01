---
aliases:
- IRR Calculation
- 内部收益率计算
- 内部收益率计算步骤
tags:
- procedure
- finance
---
# IRR Calculation

## 输入

- 项目现金流序列 $CF_0,CF_1,\dots,CF_n$。
- 容忍误差 $\epsilon$。
- 搜索区间或初始猜测。

## 输出

- 使 $NPV(r)=0$ 的 $IRR$。
- 是否存在唯一 IRR 的提示。

## Step 1：检查现金流符号

- 常规项目：先流出、后流入，符号变化一次。
- 非常规项目：现金流正负多次切换，可能出现多个 IRR。
- 全正或全负现金流通常没有可解释的 IRR。

## Step 2：写出 NPV 函数

$$
NPV(r)=\sum_{t=0}^{n}\frac{CF_t}{(1+r)^t}
$$

IRR 是让 $NPV(r)=0$ 的 $r$。

## Step 3：用二分法求解

1. 选区间 $[r_l,r_u]$，并确认 $NPV(r_l)$ 与 $NPV(r_u)$ 异号。
2. 取中点 $r_m=(r_l+r_u)/2$。
3. 计算 $NPV(r_m)$。
4. 若 $|NPV(r_m)|<\epsilon$，停止。
5. 若 $NPV(r_l)$ 与 $NPV(r_m)$ 异号，令 $r_u=r_m$。
6. 否则令 $r_l=r_m$。
7. 重复直到收敛。

## Step 4：用牛顿法复核（可选）

$$
r_{k+1}=r_k-\frac{NPV(r_k)}{NPV'(r_k)}
$$

其中：

$$
NPV'(r)=-\sum_{t=1}^{n}\frac{tCF_t}{(1+r)^{t+1}}
$$

牛顿法更快，但初始值不好时可能发散。

## Step 5：解释结果

- $IRR>r_{\text{required}}$：独立项目可接受。
- $IRR<r_{\text{required}}$：独立项目拒绝。
- 多重 IRR、互斥项目排序冲突时，回到 [[NPV Calculation]]。

## 检查点

- $r>-1$，否则折现公式失去经济意义。
- 二分法只在端点异号时可靠。
- 求解器给出一个数不代表 IRR 唯一。

## 常见错误

- 把 IRR 当作项目创造的金额。
- 没检查现金流符号变化次数。
- 互斥项目只选 IRR 高的项目。
- 用错误的区间方向手动逼近。

## 来自课程位置

- [[05_投资项目资本预算]]

## 关联卡片

- [[Internal Rate of Return]]
- [[Net Present Value]]
- [[NPV Calculation]]
- [[Investment Decisions]]
- [[Capital Budgeting Decision Map]]

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
