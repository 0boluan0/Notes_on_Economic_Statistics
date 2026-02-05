---
aliases:
- TARCH
tags:
- concept
---
>[!note] TARCH ,门限GARCH效应 Threshold GARCH
> TARCH通过在方差方程中引入一个针对负残差的指示变量来实现非对称效应。以TARCH(1,1)为例，其形式可写为：
> 
$> h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \lambda_1d_{t-1}\epsilon_{t-1}^2 + \beta_1 h_{t-1},$
> 
> ==其中$d_{t-1}$是一个哑变量==，当$\epsilon_{t-1} < 0$时$d_{t-1}=1$，当$\epsilon_{t-1} \ge 0$时$d_{t-1}=0$。也就是说，$\epsilon_{t-1}^2$项会根据$\epsilon_{t-1}$的符号被赋予不同的系数：$如果前一期是负冲击，则方差方程中实际影响是(\alpha_1+\lambda_1)\epsilon_{t-1}^2；如果前一期是正冲击，则影响是\alpha_1 \epsilon_{t-1}^2（因为这时d_{t-1}=0，额外项不起作用）。$

## 相关链接

- 基础模型：[[ARCH]], [[GARCH]]
- 相关模型：[[EGARCH]]
- 现象：杠杆效应（负冲击比正冲击引起更大的波动）
