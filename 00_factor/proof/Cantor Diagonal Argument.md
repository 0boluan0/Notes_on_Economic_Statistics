---
aliases: [Cantor Diagonal Argument, Cantor Diagonalization, 康托尔对角论证]
tags: [proof, discrete-mathematics]
type: proof
---
# Cantor Diagonal Argument

## 假设

假设所有无限二进制序列都能列成 $s_0,s_1,s_2,\ldots$。

## 构造

定义新序列 $d$，令其第 $n$ 位与 $s_n$ 的第 $n$ 位相反：
$$
d_n=1-(s_n)_n.
$$

## 推导

对每个 $n$，$d$ 与 $s_n$ 至少在第 $n$ 位不同，所以 $d\ne s_n$。因此 $d$ 不在假设的完整列表中，与“列表包含所有序列”矛盾。

## 结论

无限二进制序列集合不可数；等价地，$\mathcal P(\mathbb N)$ 不可数。

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
