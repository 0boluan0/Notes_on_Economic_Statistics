# 2026-10-15 评估证据

本目录只收录本轮从当前实验室得到的截图、影片抽帧和观察记录。主结论见 [[07_Programme/可视化实验室/docs/教学体验评估与改造优先级-2026-10-15|评估报告]]。

## 覆盖方式

60 部影片各抽取约 36% 与 72% 时点，图片上保留实际时间。抽帧用于观察构图、对象和信息组织，不能证明整段运动都流畅。double-descent 和 calibration 的初次抽帧落在淡入淡出附近；补查后能看到稳定的曲线和点图，因此没有把过渡帧误判成缺图。

全部教学配置与影片均检查；实际交互操作覆盖 13 项，详见核查范围 JSON。浏览器截图为 1280×720。原生 Obsidian 截图裁去无关侧栏，仅保留实验笔记区域；截图没有修改图中内容。原生完整截图临时保存在 /tmp/visual-lab-audit/native-originals，本目录保留用于评估的实验面板。

## 影片抽帧索引

| 图版 | 覆盖主题 |
|---|---|
| [影片抽帧 01](影片抽帧-01.jpg) | tangent、linear、projection、gradient、sampling |
| [影片抽帧 02](影片抽帧-02.jpg) | confidence、testing、bayes、regression、overfit |
| [影片抽帧 03](影片抽帧-03.jpg) | pca、bootstrap、knn、logistic、svm |
| [影片抽帧 04](影片抽帧-04.jpg) | tree、ensemble、boosting、kmeans、gmm |
| [影片抽帧 05](影片抽帧-05.jpg) | kde、modelselection、metrics、likelihood、activations |
| [影片抽帧 06](影片抽帧-06.jpg) | neuron、mlp、backprop、initialization、optimizers |
| [影片抽帧 07](影片抽帧-07.jpg) | regularization、normalization、convolution、pooling、residual |
| [影片抽帧 08](影片抽帧-08.jpg) | attention、positional、gnn、embeddings、autoencoder |
| [影片抽帧 09](影片抽帧-09.jpg) | vae、gan、flow、diffusion、rl-bellman |
| [影片抽帧 10](影片抽帧-10.jpg) | rl-policy、double-descent、fairness、svd、biasvariance |
| [影片抽帧 11](影片抽帧-11.jpg) | kernel、entropy、softmax、calibration、rnn |
| [影片抽帧 12](影片抽帧-12.jpg) | lstm、contrastive、jacobian、uncertainty、causal |

补查：[双降稳定段](影片细查-double-descent.jpg) · [校准稳定段](影片细查-calibration.jpg)。时长、抽帧时间、音轨、编码和文件哈希见 [影片抽帧清单](影片抽帧清单.json)。

## 当前界面与操作证据

| 文件 | 所观察的步骤或动作 |
|---|---|
| [01-实验库入口.jpg](01-实验库入口.jpg) | 首页目录、筛选、三个精选入口 |
| [02-反传阶段.jpg](02-反传阶段.jpg) | 进入反向阶段时的全部导数与布局 |
| [03-理解检验.jpg](03-理解检验.jpg) | 完成 w 微扰核对后进入理解检验 |
| [04-Obsidian影片.png](04-Obsidian影片.png) | 原生笔记内定位至约 19 秒 |
| [05-Obsidian嵌入.png](05-Obsidian嵌入.png) | 影片之后进入独立的内嵌实验 |
| [06-Obsidian反传滚动.png](06-Obsidian反传滚动.png) | 切换反向步骤，再在内嵌长图中滚动 |
| [07-注意力读取.jpg](07-注意力读取.jpg) | 进入 Value 乘权重的读取步骤 |
| [08-卷积乘加.jpg](08-卷积乘加.jpg) | 进入九项乘积阶段 |
| [09-卷积单项核对.jpg](09-卷积单项核对.jpg) | 选择第 3 行第 3 列，看到 1×1 与累计和 3 |
| [10-归一化分组.jpg](10-归一化分组.jpg) | 进入统计轴比较；随后切换为 LayerNorm |
| [11-Bootstrap重抽样.jpg](11-Bootstrap重抽样.jpg) | 把本轮重抽样进度移到末尾 |
| [12-优化器轨迹.jpg](12-优化器轨迹.jpg) | 点击前进一步，更新次数由 60 变为 70 |
| [13-PCA重构.jpg](13-PCA重构.jpg) | 进入一维投影与重建 |
| [14-RNN共享梯度.jpg](14-RNN共享梯度.jpg) | 进入共享参数各时间贡献柱图 |
| [15-因果组分.jpg](15-因果组分.jpg) | 比较观察组与总体的 Z 构成 |
| [16-Softmax换例.jpg](16-Softmax换例.jpg) | 修改 z₁ 后切换步骤，进入新的预设 |
| [17-Bellman迭代.jpg](17-Bellman迭代.jpg) | 第 2 步载入 1 轮，再前进一步到 2 轮 |
| [18-扩散条件分布.jpg](18-扩散条件分布.jpg) | 进入已知 Gaussian 模型的条件分布 |
| [19-置信区间重抽样.jpg](19-置信区间重抽样.jpg) | 更换随机种子，生成新的一组区间 |

Softmax 的参数变更有切换前后完整辅助功能树记录；attention 与 normalization 的文本记录为动作后差异快照。它们只用于复核当时的变化，不替代完整数学验证。

