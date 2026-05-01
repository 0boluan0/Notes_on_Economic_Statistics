---
aliases:
- DID
- DiD
- Difference-in-Differences
- Difference in Differences
- 双重差分
- 差分中的差分
tags:
- concept
- econometrics
- causal-inference
---
# DID

## 先记一句话

DID 用处理组和对照组的前后变化差异，扣掉共同时间趋势，估计处理效应。

## 它是什么

两组两期估计量：

$$
DID=(\bar Y_{T,post}-\bar Y_{T,pre})-(\bar Y_{C,post}-\bar Y_{C,pre})
$$

回归形式：

$$
Y_{it}=\alpha+\beta(Treat_i\times Post_t)+\gamma Treat_i+\delta Post_t+u_{it}
$$

其中 $\beta$ 是 DID 效应。

## 解决什么判断

它回答：“政策后处理组相对对照组多出来的变化，能否解释为政策因果效应？”

## 最小例子

某省实施最低工资政策，其他省未实施。比较实施省政策前后工资变化，再减去未实施省同期变化，就是 DID。

## 易混点

- DID 不是自动因果识别，关键是 [[Parallel Trends]]。
- DID 识别的常是 [[ATT]]，不是所有人群的平均处理效应。
- 面板固定效应只是估计方式，不能替代识别假设。

## 来自课程位置

- [[13_面板数据模型]]

## 关联卡片

- [[DID Framework]]
- [[DID Estimation Steps]]
- [[DID Diagnostics]]
- [[DID Identification Proof]]
- [[DID Writing Template]]
