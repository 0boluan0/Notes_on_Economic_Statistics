---
aliases:
- Gram-Schmidt Orthogonalization
- Gram-Schmidt
- Gram-Schmidt process
- 格拉姆-施密特正交化
tags:
- procedure
- 线性代数
---
# Gram-Schmidt Orthogonalization

## 它是做什么的
- 该流程用于把一组线性无关向量改造成同一子空间中的正交组或标准正交组。

## 输入
- 一组线性无关向量 $a_1,\dots,a_n$。

## 输出
- 一组正交向量 $u_1,\dots,u_n$，以及标准正交向量 $q_1,\dots,q_n$。

## Step 1
- 令 $u_1=a_1$，再单位化得到 $q_1=u_1/\|u_1\|$。

## Step 2
- 对第 $k$ 个向量，减去它在前面所有正交方向上的投影：
$$
u_k=a_k-\sum_{j=1}^{k-1}\operatorname{proj}_{u_j}(a_k).
$$

## Step 3
- 若只需要正交组，到此结束；若需要标准正交组，再做 $q_k=u_k/\|u_k\|$。

## Step 4
- 把所有 $q_k$ 按列排成矩阵，就得到 QR 分解中的 Q。

## 常见错误
- 忘记每一步都要减去“对所有已有方向”的投影，而不是只减去最近一个。
- 输入向量若线性相关，某一步会得到零向量，流程就会停住。

## 关联卡片
- [[Orthogonality]]
- [[Orthogonal Matrix]]
- [[Projection Matrix]]

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
