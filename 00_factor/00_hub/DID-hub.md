---
aliases:
- DID-hub
- DID hub
- Difference-in-Differences hub
- 双重差分知识地图
tags:
- hub
- econometrics
- causal-inference
---
# DID-hub

## 这组卡解决什么

DID 用处理组与对照组的“前后变化差异”识别政策或事件的因果效应。核心不是回归式，而是 [[Parallel Trends]] 是否可信。

## 学习路线

1. 定义入口：[[DID]]、[[ATT]]、[[Parallel Trends]]。
2. 使用判断：[[DID Framework]]。
3. 估计流程：[[DID Estimation Steps]]。
4. 识别证明：[[DID Identification Proof]]。
5. 有效性诊断：[[DID Diagnostics]]。
6. 写作表达：[[DID Writing Template]]。

## 前置知识

- [[Panel Data Model]]
- [[Fixed Effects Model]]
- [[Endogeneity]]
- [[Hausman Test]]

## 什么时候不要硬用 DID

- 处理前趋势已经明显分叉。
- 对照组被政策溢出影响。
- 处理组和对照组样本构成在政策前后变化很大。
- 错位处理时间和异质处理效应很强，却只用简单 TWFE。

## 课程笔记入口

- [[13_面板数据模型]]

## 交付物导航

| 你要完成的事 | 入口卡 | 输出 |
| --- | --- | --- |
| 判断能否识别 | [[DID Framework]]、[[DID Diagnostics]] | 假设与诊断清单 |
| 写出估计式 | [[DID Estimation Steps]] | 回归式、固定效应与聚类层级 |
| 解释识别逻辑 | [[DID Identification Proof]] | ATT 的反事实分解 |
| 写入报告 | [[DID Writing Template]] | 可直接替换占位符的段落 |

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
