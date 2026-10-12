<!-- visual-lab:generated:start -->
# GAN 生成器为什么会遇到梯度饱和

<!-- bilingual-en:start -->
*GAN generator objectives under a frozen discriminator*
<!-- bilingual-en:end -->

冻结一个 sigmoid 判别器，对照原始 minimax 与非饱和目标的精确梯度和真实更新轨迹。

> [!question] 先预测，再操作
> 判别器确信样本是假的时，哪一种生成器目标仍提供较大的梯度？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/gan.mp4]]

约 39 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 39 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先冻结判别器，只移动一个生成点
- **0:04** 同一个低分，会产生不同强度的更新
- **0:11** 沿各自梯度走，比较实际位置更新
- **0:30** 这段只隔离生成器损失的影响

<!-- bilingual-en:start -->
- **0:00** Freeze the discriminator and move one generated point.
- **0:04** The same low score gives different generator gradients.
- **0:11** Follow the two actual update sequences.
- **0:30** This isolates generator objectives under a frozen discriminator.
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
id: gan
height: 980
```

## 观察路线

1. 先选强烈拒绝预设，比对两个目标在同一点的梯度大小。
2. 观察参数沿负梯度更新，而不是沿梯度本身移动。
3. 把 k 设为零，确认即使 −log D 不小，也没有关于 θ 的梯度。
4. 延长更新轨迹，说明固定判别器为何不保证生成点停在真实示例 x=2。

## 公式与适用条件

$$
D(\theta)=\sigma(k\theta+b),\quad L_s=\log(1-D),\quad L_n=-\log D,\quad L_s'=-kD,\quad L_n'=k(D-1)
$$

- 判别器 D(x)=σ(kx+b) 全程冻结；真实参考点固定为 x=2。
- 生成器 G=θ 是一个点质量，不含潜噪声或复杂网络。
- 使用精确梯度下降和稳定 softplus 计算，不对 D 人为截断。

## 回到知识点

- [[00_Knowledge/链式法则|链式法则]]：sigmoid 判别器与两个生成目标的复合导数；零判别器斜率说明损失数值与参数梯度不同。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/logistic|概率、阈值与交叉熵]] · [[07_Programme/可视化实验室/实验/backprop|反向传播把局部导数串起来]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/vae|VAE 的重参数化与 KL 代价]]

## 来源与核验

- [Goodfellow et al. · Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661) — §3 Adversarial nets; Eq. (1) and non-saturating alternative。支持：原始 minimax 目标及生成器改用最大化 log D 的梯度动机；本页冻结判别器以隔离差异。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
