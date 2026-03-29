---
aliases:
- Difference-in-Differences
- Difference in Differences
- DiD
- DID
- 双重差分
- 差分中的差分
tags:
- concept
---
# DID

>[!note] 它是什么
> - Difference-in-Differences（DID）是一种准实验因果识别方法，通过比较处理组与对照组在处理前后的变化差异来估计处理效应。
>
>[!note] 最小可检索信息
> - 两组两期估计量：
>   $\text{DID}=(\bar Y_{T,post}-\bar Y_{T,pre})-(\bar Y_{C,post}-\bar Y_{C,pre})$
> - 回归形式：
>   $Y_{it}=\alpha+\beta(Treat_i\times Post_t)+\gamma Treat_i+\delta Post_t+\varepsilon_{it}$
>   其中 $\beta$ 为 DID 效应。
> - 关键识别：平行趋势（未处理潜在结果的趋势相同）。
>
## 关联卡片
- [[DID Framework]]
- [[DID Estimation Steps]]
- [[DID Diagnostics]]
- [[DID Identification Proof]]
- [[DID Writing Template]]

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
