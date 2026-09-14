---
aliases:
  - "简单指数平滑的 alpha 控制几何权重衰减与响应噪声取舍"
  - SES geometric weights
  - Simple exponential smoothing alpha
  - 简单指数平滑权重
student_os: knowledge-atom
atom_id: TS-ETS-002
atom_set: exponential-smoothing-ets
atom_type: method-mechanism
status: source-checked
mastery_state: unassessed
related:
  - "[[时间序列分解与预测]]"
  - "[[SES-ARIMA受限等价]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# 简单指数平滑的 alpha 控制几何权重衰减与响应噪声取舍
<!-- bilingual-en:start -->
*In simple exponential smoothing, alpha controls geometric weight decay and the responsiveness-noise trade-off*
<!-- bilingual-en:end -->

> [!summary] 原子机制
> 无趋势、无季节时，simple exponential smoothing（SES）用
> $$\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad \hat y_{t+h|t}=\ell_t$$
> 更新水平。$\alpha$ 越大，历史权重衰减越快，模型对新变化响应越快，也越容易追随短期噪声。
> <!-- bilingual-en:start -->
> SES updates a single level state. A larger alpha makes past weights decay faster, increasing responsiveness to recent changes while also increasing sensitivity to short-run noise.
> <!-- bilingual-en:end -->

把递推反复展开，有限样本的一步预测为
$$
\hat y_{T+1|T}
=\sum_{j=0}^{T-1}\alpha(1-\alpha)^j y_{T-j}
+(1-\alpha)^T\ell_0.
$$
第 $j$ 个滞后观测的权重是 $\alpha(1-\alpha)^j$，相邻权重之比恒为 $1-\alpha$；这就是“指数”平滑的含义。不能漏掉末尾的初始水平项：只有样本较长且 $0<\alpha\le1$ 时，$(1-\alpha)^T\ell_0$ 才通常很小。

$\alpha=1$ 时，$\ell_t=y_t$，所有未来点预测等于最后观测，即 naïve forecast。$\alpha$ 接近零时，水平变化极慢；在边界 $\alpha=0$，数据完全不能更新 $\ell_0$。因此“较小 $\alpha$ 更稳定”不是无条件优点：若真实水平发生变化，它会产生持续滞后；相反，较大 $\alpha$ 会更快吸收变化，但也会把一次异常冲击写入当前水平。

SES 的 forecast function 对所有 $h\ge1$ 都是同一个 $\ell_T$。所以它只适合没有需要外推的趋势或季节结构的序列；平滑后的曲线看起来好看，并不能弥补模型缺少趋势与季节机制。
<!-- bilingual-en:start -->
The finite-sample expansion includes the initial-state remainder $(1-\alpha)^T\ell_0$. At $\alpha=1$, SES reduces to the naive forecast; at $\alpha=0$, the level never learns from the data. All horizons share the flat forecast $\ell_T$, so SES is unsuitable when a trend or seasonal component must be extrapolated.
<!-- bilingual-en:end -->

> [!question]- 自检
> 把 $\alpha$ 从 $0.2$ 提高到 $0.8$，会怎样改变两期前观测的权重与转折后的反应？
>
> **答案：** 两期前权重从 $0.2(0.8)^2=0.128$ 降到 $0.8(0.2)^2=0.032$；新数据占比更大，所以转折后反应更快，但短期噪声也更容易改变水平。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.1](https://otexts.com/fpp3/ses.html)：核对几何权重、包含 $\ell_0$ 的有限样本展开、flat forecast 与 $\alpha=1$ 的 naïve 边界。
- Brown, R. G. (1959), *Statistical Forecasting for Inventory Control*：SES 的经典来源；历史归属由 FPP3 Chapter 8 交叉核对。
