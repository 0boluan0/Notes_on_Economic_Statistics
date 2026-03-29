---
aliases:
- Jordan Form
- Jordan Canonical Form
- 若尔当标准形
- Jordan 标准形
tags:
- concept
- 线性代数
---
# Jordan Form

>[!note] 它是什么
> - 「Jordan Form」是指把矩阵化为若尔当块对角形式的标准表示，是对角化失败时的替代结构。
>
>[!note] 最小可检索信息
> - 定义：任何复矩阵都与若干 Jordan block 组成的块对角矩阵相似。
> - 符号/公式：$A=PJP^{-1}$。
> - 最小例子：若矩阵只有一个特征值且只有一个特征向量，就会出现大小大于 1 的 Jordan block。
>
## 关键性质
- Jordan form 说明“缺失的特征向量”会以广义特征向量和上超对角 1 的形式出现。
- 它控制矩阵幂和矩阵指数中的多项式因子。

## 关联卡片
- [[Similar Matrix]]
- [[Diagonalization]]
- [[Matrix Exponential]]

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
