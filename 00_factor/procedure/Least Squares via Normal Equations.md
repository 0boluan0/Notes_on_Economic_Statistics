---
aliases:
- Least Squares via Normal Equations
- Normal Equations Workflow
- 最小二乘正规方程求解
tags:
- procedure
- 线性代数
---
# Least Squares via Normal Equations

>[!note] 何时使用
> - 题目要求 best fit、projection onto column space、least squares，或者给出一个通常无精确解的过定方程组。
> - 默认前提是你想先走课程里的标准路线：`误差正交 -> normal equations -> projection`。

## Step 1. 先判断你在解什么

- 明确原题是在问 `Ax=b` 的精确解，还是在问 best approximation。
- 若题目出现“无解但求最接近”“best fit line”“projection onto Col(A)”等信号，就切到 least squares。

## Step 2. 写出误差正交条件

- 设投影点为 $p=A\hat{x}$，残差为 $e=b-p$。
- 写出核心条件：残差必须与列空间正交，即
  $$
  A^T(b-A\hat{x})=0.
  $$

## Step 3. 建立并求解正规方程

- 把正交条件整理成
  $$
  A^TA\hat{x}=A^Tb.
  $$
- 计算 $A^TA$ 与 $A^Tb$，解出 $\hat{x}$。
- 若 $A^TA$ 不可逆，就不要硬算 inverse，改走 [[Pseudoinverse]] 或 QR 路线。

## Step 4. 回到投影与残差

- 用 $\hat{x}$ 写出投影点：
  $$
  p=A\hat{x}.
  $$
- 写出残差：
  $$
  e=b-p.
  $$
- 若题目要求验证，检查 $A^Te=0$。

## Step 5. 把答案写成题目真正要的形式

- 若题目要 least squares solution，就给出 $\hat{x}$。
- 若题目要 closest point / projection，就给出 $p$。
- 若题目要 projection matrix，写出
  $$
  P=A(A^TA)^{-1}A^T
  $$
  并说明它把 $b$ 投到 $\operatorname{Col}(A)$ 上。

## 输出检查

- 我有没有先说明这是“最佳逼近”而不是“精确求解”。
- 我有没有把正规方程解释成“误差对列空间正交”。
- 我有没有区分未知量 $\hat{x}$、投影点 $p$、残差 $e$ 这三个对象。

## 关联卡片

- [[Least Squares]]
- [[Orthogonal Projection]]
- [[Projection Matrix]]
- [[Pseudoinverse]]

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
