<!-- visual-lab:generated:start -->
# Bellman 备份如何传播未来奖励

<!-- bilingual-en:start -->
*Finite MDPs & value iteration*
<!-- bilingual-en:end -->

在有回路的三个状态中计算每次 Bellman 备份，核对精确最优价值与可计算误差界。

> [!question] 先预测，再操作
> 冒险失败会回到原状态时，为什么不能只把一次成功概率乘上奖励？

<!-- visual-lab:film:start -->
## 动画：分步观察

<!-- bilingual-en:start -->
*Animation: follow the mechanism*
<!-- bilingual-en:end -->

![[07_Programme/可视化实验室/media/rl-bellman.mp4]]

约 51 秒，中英字幕。可暂停、点击进度条定位或全屏观看。

<!-- bilingual-en:start -->
About 51 seconds, with Chinese and English captions. Pause, seek on the timeline or watch full screen.
<!-- bilingual-en:end -->

- **0:00** 先把奖励与转移概率分开读
- **0:09** 把后续价值折现，再回传到当前状态
- **0:16** 每次都用上一轮的价值做下一轮更新
- **0:41** 取最大值，才从评估走向最优控制

<!-- bilingual-en:start -->
- **0:00** Rewards and transition probabilities are different quantities.
- **0:09** Discount future value before backing it up.
- **0:16** Synchronous backups use the previous value vector.
- **0:41** The maximum chooses the best available action.
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
id: rl-bellman
height: 980
```

## 观察路线

1. 先读清奖励在执行动作时收到：到 S₁ 之后的奖励 3 属于下一步。
2. 把迭代次数设为 1，再设为 2，观察奖励如何向前传播。
3. 比较安全行动的 1 与重复冒险固定点，解释哪个更大。
4. 查看实际误差和 Bellman 残差界；γ 接近 1 时界可能很松。

## 公式与适用条件

$$
V_{k+1}(S_0)=\max\{1,c+\gamma[pV_k(S_1)+(1-p)V_k(S_0)]\},\quad V^*(S_1)=3,\quad V^*(S_0)=\max\left\{1,\frac{c+3\gamma p}{1-\gamma(1-p)}\right\}
$$

- S₀ 可选安全：立即奖励 1 后终止；或冒险：立即奖励 c，以 p 到 S₁、以 1−p 回 S₀。
- S₁ 唯一动作奖励 3 后终止；终止状态价值 0。所有奖励时间均按动作时刻计。
- γ<1；同步更新全部使用上一轮值，初始为零。环境转移概率已知。

## 回到知识点

- [[00_Knowledge/条件期望|条件期望]]：价值函数按当前状态条件化未来折扣回报；有限已知转移把条件期望化为加权求和。

[[07_Programme/可视化实验室/00_实验库入口|回到实验库]] · [[07_Programme/可视化实验室/实验/rl-policy|策略梯度如何改变行动概率]] · [[07_Programme/可视化实验室/实验/gradient|梯度下降何时收敛]] · [[07_Programme/可视化实验室/实验/bayes|先验与数据怎样形成后验]]

## 来源与核验

- [Sutton & Barto · Reinforcement Learning（作者教材草稿）](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) — Ch.3 Bellman equations; §4.4 Value Iteration, Eq. (4.10)。支持：有限 MDP、折扣价值与 Bellman 最优备份。实验中的三状态 MDP 和解析式为独立构造。

实验使用明确给定的公式、合成数据或固定小型示例。代码、可复现检查和逐轮修订记录见 [[07_Programme/可视化实验室/01_使用与维护|使用与维护]]。
<!-- visual-lab:generated:end -->
