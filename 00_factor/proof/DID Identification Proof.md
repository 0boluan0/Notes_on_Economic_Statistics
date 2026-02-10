---
aliases:
- DID识别证明
- Difference-in-Differences proof
- Parallel trends proof
- 双重差分识别
- DID Identification Proof
- DID
tags:
- proof
---
# DID Identification Proof

## 假设
- **平行趋势**：
  $E[Y_{it}(0)|G_i=1,Post_t=1]-E[Y_{it}(0)|G_i=1,Post_t=0]=E[Y_{it}(0)|G_i=0,Post_t=1]-E[Y_{it}(0)|G_i=0,Post_t=0]$
- **无预期效应** 与 **无溢出效应**。

## 推导
设处理组指示为 $G_i$，政策后指示为 $Post_t$，处理发生于 $G_i \times Post_t$。

观察结果：
$Y_{it} = D_{it}Y_{it}(1) + (1-D_{it})Y_{it}(0)$

两组两期的差上加差：
$$
\begin{aligned}
DID &= [E(Y|G=1,Post=1)-E(Y|G=1,Post=0)]\\
&\quad-[E(Y|G=0,Post=1)-E(Y|G=0,Post=0)]
\end{aligned}
$$

代入潜在结果并使用平行趋势，可得：
$DID = E[Y_{it}(1)-Y_{it}(0)|G_i=1,Post_t=1] = ATT$

## 结论
- 在平行趋势成立下，DID 识别处理组平均处理效应（ATT）。

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
