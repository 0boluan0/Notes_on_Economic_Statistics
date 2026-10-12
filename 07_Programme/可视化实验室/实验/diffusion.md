<!-- visual-lab:generated:start -->
# 扩散加噪与精确 Gaussian 去噪

<!-- bilingual-en:start -->
*Gaussian diffusion & conditional uncertainty*
<!-- bilingual-en:end -->

用已知 Gaussian 数据先验，精确计算加噪观测对应的后验均值、方差、噪声预测与 score。

> [!question] 先预测，再操作
> 即使知道精确的去噪条件均值，为什么仍不能恢复每次加噪前的原样本？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/diffusion.mp4]]

约 40 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 40 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 信号与噪声，先分别缩放再相加
- **0:09** 看到 xₜ 后，反过来推断可能的 x₀
- **0:25** 去噪估计，不是恢复唯一真值
- **0:31** 这个精确小模型解释去噪的条件推断

<!-- bilingual-en:start -->
- **0:00** Scale signal and noise before adding them.
- **0:09** Condition on the noisy observation to infer the clean variable.
- **0:25** A denoised estimate does not determine a unique original.
- **0:31** This exact toy model isolates conditional denoising.
<!-- bilingual-en:end -->

## 亲手实验

<!-- bilingual-en:start -->
*Try it yourself*
<!-- bilingual-en:end -->

在下方展开“影片参数与延伸对照”，选择“载入本片的模型与数据”，先核对影片中的一个对象或运算，再改变一个条件。其他对照用于继续探索。

<!-- bilingual-en:start -->
Expand the film inputs below and load the model and data used in this film. Check one object or calculation, then change one condition. The other comparisons provide further exploration.
<!-- bilingual-en:end -->

<!-- visual-lab:film:end -->
```visual-lab
id: diffusion
height: 980
```

## 观察路线

1. 先固定先验和观测，降低 ᾱ，观察条件方差与先验靠近。
2. 选择标准正态先验，手算后验均值 √ᾱxₜ 和方差 1−ᾱ。
3. 移动观测 xₜ，检查后验均值改变而条件方差不变。
4. 比较预测噪声与边际 score，核对 E[ε|xₜ]=−√(1−ᾱ) score(xₜ)。

## 公式与适用条件

$$
x_t=\sqrt{\bar\alpha}x_0+\sqrt{1-\bar\alpha}\epsilon,\quad v_t=\bar\alpha\sigma^2+1-\bar\alpha,\quad E[x_0|x_t]=\mu+\frac{\sqrt{\bar\alpha}\sigma^2}{v_t}(x_t-\sqrt{\bar\alpha}\mu),\quad \operatorname{Var}(x_0|x_t)=\frac{\sigma^2(1-\bar\alpha)}{v_t}
$$

- x₀~N(μ,σ²)，ε~N(0,1)，二者独立；ᾱ∈(0,1)。
- ᾱ 是累计信号比例，不是某一步噪声方差 βₜ。
- 本页计算 x₀|xₜ；论文反向训练推导中的 xₜ₋₁|xₜ,x₀ 是另一个条件分布。

## 回到知识点

- [[00_Knowledge/条件期望投影|条件期望投影]]：精确后验均值最小化条件平方误差，但未消除条件方差，不能逐样本恢复原始噪声。
- [[00_Knowledge/正态分布|正态分布]]：Gaussian 先验和独立 Gaussian 噪声的尺度约定；作为解析去噪参照而非图像生成器。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/vae|VAE 的重参数化与 KL 代价]] · [[07_Programme/可视化实验室/实验/flow|可逆变换如何搬运概率密度]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/sampling|抽样均值、大数与中心极限定理]]

## 来源与核验

- [Ho et al. · Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239) — §2, Eqs. (2), (4), (6)–(7)。支持：前向 Gaussian 加噪与累计信号比例；本页 x₀|xₜ 使用另行指定 Gaussian 数据先验推导，不把它混同论文的 xₜ₋₁|xₜ,x₀。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
