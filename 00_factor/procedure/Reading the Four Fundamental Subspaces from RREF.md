---
aliases:
- Reading the Four Fundamental Subspaces from RREF
- Read Four Fundamental Subspaces from RREF
- 从 rref 读取四个基本子空间
tags:
- procedure
- 线性代数
---
# Reading the Four Fundamental Subspaces from RREF

## 它是做什么的
- 该流程用于从矩阵 A 与它的行最简形 rref 中，系统读出四个基本子空间的基、维数和所在空间。

## 输入
- 原矩阵 $A\in \mathbb{R}^{m\times n}$。
- 它的 rref，或至少它的 pivot 列 / free 列信息。

## 输出
- `C(A)`、`N(A)`、`C(A^T)`、`N(A^T)` 的一组基与维数解释。

## Step 1
- 先对 A 做 row reduction，标出 pivot 列与 free 列。
- pivot 个数就是 `rank(A)=r`。

## Step 2
- 读取 [[Column Space]] 时：
  - 回到**原矩阵 A**；
  - 取原矩阵中对应 pivot 的那些列；
  - 它们构成 `C(A)` 的一组基。

## Step 3
- 读取 [[Row Space]] 时：
  - 直接取 rref 中所有非零行；
  - 它们构成 `C(A^T)` 的一组基。

## Step 4
- 读取 [[Null Space]] 时：
  - 解 `Ax=0`；
  - 把 free variables 设为参数；
  - 每次让一个参数为 1、其余为 0，就得到一个 special solution；
  - 所有 special solutions 构成 `N(A)` 的一组基。

## Step 5
- 读取 [[Left Nullspace]] 时：
  - 解 `A^Ty=0`；
  - 对 `A^T` 重复同样流程；
  - 得到的 special solutions 构成 `N(A^T)` 的一组基。

## Step 6
- 最后做两条一致性检查：
  - `dim C(A)=dim C(A^T)=rank(A)=r`
  - `dim N(A)=n-r`，`dim N(A^T)=m-r`

## 常见错误
- 用 rref 的 pivot 列直接当 `C(A)` 的基。列空间的基必须从**原矩阵**取。
- 以为四个基本子空间都活在同一个空间里。`C(A)` 与 `N(A^T)` 活在 $\mathbb{R}^m$，`C(A^T)` 与 `N(A)` 活在 $\mathbb{R}^n$。
- 只记维数，不检查正交关系。

## 关联卡片
- [[Column Space]]
- [[Row Space]]
- [[Null Space]]
- [[Left Nullspace]]
- [[Matrix Rank]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.outlinks, this.file.link)
)
SORT file.mtime DESC
```
