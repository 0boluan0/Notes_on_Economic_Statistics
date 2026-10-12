<!-- visual-lab:generated:start -->
# VAE 的重参数化与 KL 代价

<!-- bilingual-en:start -->
*Gaussian reparameterization & ELBO terms*
<!-- bilingual-en:end -->

在完全可解析的一维模型里，分开看一个随机样本、期望重构项与 Gaussian KL。

> [!question] 先预测，再操作
> 改变采样噪声 ε，为什么当前解析训练目标可以完全不变？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/vae.mp4]]

约 42 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 42 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 把随机性放在一个独立的噪声变量里
- **0:12** 抽样点的变化，可以沿确定性路径求导
- **0:20** 重建与先验匹配，分别计算再相加
- **0:33** 改噪声样本，和改分布参数，是两种操作

<!-- bilingual-en:start -->
- **0:00** Put randomness in a separate noise variable.
- **0:12** The sampled value has a differentiable parameter path.
- **0:20** Reconstruction and prior matching are distinct terms.
- **0:33** Changing a draw differs from changing its distribution.
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
id: vae
height: 980
```

## 观察路线

1. 先设 μ=v=0，核对 KL 为零，再分别改变均值与方差。
2. 固定所有分布参数，只移动 ε，区分样本 z 与解析期望。
3. 改变解码系数 w，观察重构项的均值误差与方差惩罚同时变化。
4. 比较 β=0、1、4 的目标；只有 β=1 是这个指定模型的标准负 ELBO。

## 公式与适用条件

$$
z=\mu+e^{v/2}\epsilon,\quad\mathrm{KL}(q\|p)=\frac12(\mu^2+e^v-1-v),\quad L_\beta=\frac12[(x-w\mu)^2+w^2e^v]+\beta\,\mathrm{KL}
$$

- q(z|x)=N(μ,exp v)，先验 N(0,1)，似然 x|z=N(wz,1)。
- 目标省略统一常数 ½log(2π)，没有隐藏的方差估计。
- μ 和 v 是当前一个观测的手动变分参数，没有训练共享 encoder 网络。

## 回到知识点

- [[00_Knowledge/正态分布|正态分布]]：明确 exp(logvar) 是方差，exp(logvar/2) 是标准差；不把标量 Gaussian 例子当作完整 VAE 训练。
- [[00_Knowledge/链式法则|链式法则]]：固定噪声 ε 后重参数化样本对均值、log variance 的路径导数。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/autoencoder|线性自编码器与秩一重构]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]] · [[07_Programme/可视化实验室/实验/diffusion|扩散加噪与精确 Gaussian 去噪]] · [[07_Programme/可视化实验室/实验/flow|可逆变换如何搬运概率密度]]

## 来源与核验

- [Kingma & Welling · Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114) — §2.2 variational bound; §2.4 Gaussian reparameterization; Appendix B。支持：重参数化、ELBO 与对角 Gaussian 对标准正态的解析 KL；本页是一维指定似然的解析特例。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
