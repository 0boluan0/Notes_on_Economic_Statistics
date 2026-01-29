
>[!note] EGARCH 指数GARCH模型 Exponential GARCH
>指数GARCH模型采用对数形式的方差方程，形式例如EGARCH(1,1)：
> 
> $$\ln h_t = \alpha_0 + \alpha_1 \left(\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}\right) + \lambda_1 \left|\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}\right| + \beta_1 \ln h_{t-1}.$$
> 
> 这里，引入$\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}$作为标准化的残差（通常称为_z-score_，表示上一期残差相对于其标准差的大小和方向），这样做有两个好处：其一，使用$\ln h_t$确保了预测的$h_t$永远为正（因为指数的输出总是正的），不需要像标准GARCH那样对参数非负作约束；其二，通过$\alpha_1$乘以标准化残差和$\lambda_1$乘以残差的绝对值，相当于把残差的符号和幅度分离来影响$\ln h_t$，从而实现非对称效果。

## 相关链接

- 基础模型：[[ARCH]], [[GARCH]]
- 相关模型：[[TARCH]]
- 现象：杠杆效应（负冲击比正冲击引起更大的波动）
> 
