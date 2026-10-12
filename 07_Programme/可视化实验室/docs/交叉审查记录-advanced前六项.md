# 六项进阶实验的独立交叉审查

2026-10-06，教学内容代理独立阅读 `advanced-math.js` 和 `advanced-content.js` 的前六项实现与解释，覆盖 `positional`、`gnn`、`embeddings`、`autoencoder`、`vae`、`gan`。随后将原先执行的 66 项验算保存为 [可重复运行的检查脚本](../tests/cross-advanced.test.js)，并为发现的两个问题补充 4 项回归检查。最新运行共 **70 项检查通过**。

<!-- bilingual-en:start -->
On 6 October 2026, the teaching-content reviewer independently read the first six advanced models and their explanations. The original sixty-six checks were saved as a reproducible script, followed by four regression checks for the two review findings. The latest run passed all seventy checks.
<!-- bilingual-en:end -->

## 独立检查方法

| 实验 | 不依赖原实现期望值的检查 | 结果与解释边界 |
|---|---|---|
| 正弦位置编码 | 不同频率对数与底数下核对范数平方等于频率对数；比较两对绝对位置不同、相对位移相同的内积 | 符合共同平移不变性；这不意味着单个坐标平移后不变，也不是所有位置编码共有性质。 |
| 图传播 | 无自环、无残差的二部星图手写两步序列；链图/星图核对对称归一化算子保持 √degree 向量 | 两步周期与归一化固定方向正确；该例不能宣称所有图传播单调平滑或总会收敛。 |
| 嵌入几何 | 长度 2 与 1/2 的非单位向量，核对内积等于余弦；零向量余弦必须未定义 | 两个教学/显示问题见下一节；修复后回归通过。 |
| 线性自编码器 | 利用正交分解，核对 gain=0 与 gain=2 的重建误差相等、与 gain=1 的差等于平均投影平方；最佳误差核对样本协方差分母修正 79/80 | 实际平均平方误差与特征值口径一致；没有把样本协方差除数 n−1 直接当每观测 MSE。 |
| VAE | 9 组 μ/log-variance 下，对两个变分参数分别做中心差分 | 解析总目标梯度与数值导数一致；检查的是当前给定的标量模型和 β 口径。 |
| GAN | 9 组生成位置和判别斜率，包含负、零、正斜率；分别对饱和与非饱和目标做中心差分 | 两种生成器目标的梯度符号与链式法则正确；固定判别器切片不代表完整对抗训练收敛。 |

本脚本验证独立恒等式、解析特例与有限差分，不把曲线取样点数量当作独立科学检查数量。检查依赖项目实际模型导出，因此修改模型后可直接重跑。

<!-- bilingual-en:start -->
The script tests independent identities, analytic special cases and finite differences. Sampled plot points are not counted as separate scientific claims. It calls the actual exported models and can be rerun after a revision.
<!-- bilingual-en:end -->

## 发现、修复与复核

| 发现 | 数学反例或后果 | 已落盘修复 | 回归证据 |
|---|---|---|---|
| `embeddings` 原说明将单位长度写成内积与余弦相等的必要条件 | 两个非零向量长度为 2 和 1/2 时范数乘积为 1，仍相等；正交时二者也同为零而不要求单位长度 | 内容改为“两个向量都单位化时，余弦相似度与内积必相同；一般情形不能互换。” | 数值反例保留在脚本；回归检查旧必要条件文字已移除。 |
| 任一向量为零时余弦指标正确未定义，但第 4 项仍显示数值夹角 | 零向量没有几何方向；构造参数 θ 不能作为两向量的实际夹角显示 | 第 4 项在零长度时显示“零向量：未定义”，并移除角度单位；非零情况仍报夹角 | 分别测试 a=0、b=0、二者均为0，共3种边界。 |

问题先提交模型维护代理和主代理；模型维护代理修正内容与指标后，本审查代理读取实际文件并重跑正式脚本。两处均在当前实现中确认，不以口头接受建议视为修复完成。

<!-- bilingual-en:start -->
Both findings were reported to the model maintainer and parent agent. After the maintainer changed the content and metric, the reviewer read the updated files and reran the saved script. Acceptance is based on the actual changes, not merely agreement with the suggestions.
<!-- bilingual-en:end -->

## 复跑与证据限制

在 `07_Programme/可视化实验室` 目录运行：

```sh
node tests/cross-advanced.test.js
```

最新输出：

```json
{"suite":"independent cross-review: six advanced models","reviewed":["positional","gnn","embeddings","autoencoder","vae","gan"],"independentChecks":70,"status":"passed"}
```

本次交叉审查不声称额外查看了图形截图，也不替代原作者的来源核验、全部控件与预设测试、主代理实际视觉审阅或原生 Obsidian 交互验收。六项的剩余适用条件保留在实验解释和来源账本中。

<!-- bilingual-en:start -->
This review makes no additional screenshot claim. It complements, rather than replaces, source verification, full control and preset tests, rendered inspection and native Obsidian interaction checks. Experiment explanations and source ledgers retain the remaining assumptions and limits.
<!-- bilingual-en:end -->
