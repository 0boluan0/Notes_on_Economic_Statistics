# Obsidian 内交互的兼容性核验

Mac mini 核验：2026-10-06，**Obsidian 1.13.7 / Academic**；2026-10-07，**Obsidian 1.14.4 / Academic**。以下各节直至“MacBook 故障反馈”均为 Mac mini 的原生界面证据。MacBook 的独立核验及覆盖边界见末节。

| 路径 | 实测结果 | 结论 |
|---|---|---|
| 已安装 HTML Reader 1.0.15 | 本地源码的 Balance 和 LowRestricted 分支均删除脚本，Unrestricted 是全局模式 | 不修改它的全局安全策略 |
| Markdown 中 `iframe src="data:text/html;base64,…"` | Live Preview 与 Reading 均出现 `about:blank`；探针内容和按钮没有渲染 | 原生净化流程不保留该 data URL |
| Markdown 中 `iframe` 的 app://local、file://、相对路径三种本地地址 | 三者均显示 `chrome-error://chromewebdata/` | 这些原生本地文件入口不能承载实验 |
| 核心 Web Viewer 的本地 file:// 地址 | 地址被当作搜索文本，转为搜索引擎页面 | 不把核心 Web Viewer 当作本地 HTML 文件阅读器 |
| 专用 Visual Knowledge Lab 插件 | 已在真实 Obsidian 运行独立标签页、Live Preview 和 Reading 嵌入，并实际改变参数 | 原生交互路径通过下述测试 |

## 启用状态与实际交互证据

用户明确批准“允许退出受限模式，只启用实验室插件”后，先以截图确认关闭 **Restore 33 previously installed plugins**，再退出 Restricted mode。启用后重新读取 `.obsidian/community-plugins.json`，名单只有 `visual-knowledge-lab`；其他 33 个插件保持停用。HTML Reader 的设置没有改变。

实际操作包括：

- 通过命令面板打开独立 Obsidian 标签页，iframe 报告 `ready / library`，公式、图表、导航和控件均渲染。
- 切线实验把 x 从 1 改为 2：精确导数从 2 变为 11，前向差分从 3.75 变为 14.25，中心差分从 2.25 变为 11.25；SVG 线条及纵轴范围同时改变。状态栏收到实际 `state` 消息。
- 在同一原生标签页切换 PCA，选择标准化情境：第一特征值由 16.4615 变为 1.8842，每点重建误差由 0.9199 变为 0.1148；三张图的坐标标签从中心化坐标改为标准化坐标。
- 打开真实实验笔记 `实验/tangent.md`，在 Live Preview 和 Reading 均成功渲染交互。Reading 中点击“拐点：两种差商重合”，导数变为 −1，前向和中心差分均为 −0.75，误差图与局部图同步更新。
- 在笔记嵌入内点击“来源与双链 → 导数”，受限消息处理成功在新 Obsidian 标签页打开 `00_Knowledge/导数.md`。
- 目录白名单已改由固定 `assets/catalog.json` 提供；新增实验不需要改插件源代码。受控来源链接和复制当前实验双链的宿主处理已安装并重新加载，接口单元测试通过。

原生交互证据覆盖上述代表性实验与嵌入路径，并不等价于未来新增实验已经逐一完成图形与教学审核。新增图形仍需各自的数值、教学、视觉和交互验收。

## 后续扩展的复验步骤

1. 只启用 Visual Knowledge Lab，保持其他插件的原有状态。
2. 打开带有 `visual-lab` 代码块的实验说明笔记，切换到 Reading。
3. 状态栏必须从加载状态变为由 iframe 实际发来的“实验已就绪”。
4. 改变切线实验的 x，核对导数和差商；仅改变 h 时，精确导数应保持不变，差商与误差随之更新；再切换至 PCA 或置信区间实验并操作参数。
5. 点击进入知识原子，确认只打开允许的本地 Markdown 目标；检查参数/数据和 SVG 下载。
6. 在独立 Obsidian 标签页再次完成实验切换；返回笔记后确认嵌入仍可操作。

静态接口验证命令：`node tests/obsidian-plugin.test.js`。它覆盖配置拒绝规则、CSP 早于实验脚本插入、沙箱权限、发送窗口身份与消息标识校验、笔记路径范围及未知消息拒绝。该结果本身不证明上面的原生交互步骤已经通过。

## 本轮重做的原生复验（1.14.4）

在真实 Obsidian 中载入最终60项实验库，分步教学界面可交互；没有用普通浏览器代替宿主验收。

- 切线第三步点击“间隔减半”：h从0.5到0.25，割线斜率3.75→2.8125，切线斜率仍为2，P/Q与图形同步变化。截图 `quality/obsidian-tangent.png`。
- 反向传播载入可手算情境得到w梯度−2；随后点击“v=0”反例，w、b及上游梯度为0，v梯度为−0.7616，隐藏值0.7616仍可见。
- 因果注意力选择C：A/B/C权重为0.2483/0.2483/0.5035，D被遮挡为0，输出(0.7517,0.7517)。改选A后权重为(1,0,0,0)，输出(1,0)。
- “实验笔记”成功打开真实 `实验/attention.md`，其Live Preview嵌入重新发出ready；在这个新嵌入中点击第四步，D权重归零、输出正确更新。
- 在该嵌入的“来源与双链”点击“缩放点积注意力”，成功打开 `00_Knowledge/缩放点积注意力.md`。
- `.obsidian/community-plugins.json`仍只有 `visual-knowledge-lab`。原有其他社区插件没有恢复。

普通浏览器另外完成60项全控件/预设、183步骤及320/390/768/1200宽度的交互回归；报告见 `tests/browser-report.json`。窄浏览器结果不等价于iOS或Android Obsidian实机认证。导出代码与数据结构已检查，但本轮未完成系统下载文件落盘的全链路复验；不把“已生成导出文件”提示当成磁盘落盘证据。

最终回归报告对应构建标识 `acb4a64bff7509f9e147adf2799e3d8026ffbd25fdb4e56ea31ae21e771af25d`，60项实验、6477项浏览器检查，失败与页面错误均为0。最终原生首页截图见 [Obsidian 实验库首页](quality/obsidian-library.png)。保留正式源码、当前离线包、伴读材料和核验记录；本轮临时科学计算环境、试验脚本、重复临时截图及失败的HTML兼容性探针已清理，回收约300MB。

## MacBook 故障反馈与启动修复（1.0.1）

用户反馈 MacBook 的 Live Preview 和 Reading 均只显示 `visual-lab` 代码块。社区市场搜索无结果是因为本插件没有上架，不能据此判断本地安装是否存在。此前原生操作证据仅来自 Mac mini，不构成 MacBook 验收。

检查中独立复现了目录延迟到达的启动故障：`onload` 在读取目录失败后未注册任何入口；已加载插件的命令也会在同一读取失败后直接退出。1.0.1 保留入口注册，并让命令打开现有的错误与重载界面。新增回归先出现三个失败，修复后通过。这项独立缺陷不是下述 MacBook 故障定位的证据。

2026-10-07 通过屏幕共享直接检查 MacBook M4 Max（主机名 HUAWEI-Matebook-Fold，Obsidian 1.14.4）的 Academic 库：

- `.obsidian/plugins/visual-knowledge-lab/` 已有 `main.js`、`manifest.json`、`styles.css`，manifest 为 1.0.1；安装目录并未缺失。远端 `main.js` SHA-256 为 `8e7142863dd3a393a8bc21a7c89c3bbde359b4e85052b5a6ceb1aa279c422b45`，与源文件一致。配置目录仍为默认 `.obsidian`。
- MacBook 的 Installed plugins 列表刷新前只有 33 项，未识别实验室插件；执行刷新后变为 34 项，Visual Knowledge Lab 1.0.1 出现且开关关闭。通过界面启用后命令入口和笔记处理器恢复。本次恢复没有复制插件文件或修改插件代码，没有改变其他插件的启用选择。
- 在 MacBook 独立实验室标签中打开 `sampling`，选择“偏斜总体，小样本”：n 从 30 到 3，理论标准误从 0.1826 到 0.5774，直方图和 Gamma 密度线同时改变，宿主显示“参数已更新”。
- 点击“实验笔记”在 MacBook 新标签中打开 `实验/sampling.md`，Live Preview 内嵌图成功渲染并报告就绪。

这次现场证据将问题定位为运行中的插件列表未刷新，且识别后尚未启用。不能据此断言 iCloud 不同步隐藏目录。MacBook 的 Reading 模式操作、Bootstrap 参数操作及导出落盘尚未在本轮验证；用户开始使用远程电脑输入消息后已停止远程操作。

跨设备使用：等待安装文件同步到当前配置目录，在 **Settings → Community plugins → Installed plugins** 点击刷新，再启用 Visual Knowledge Lab；这里使用的是已安装列表，不是 **Browse** 社区市场。对应官方说明见 [Community plugins](https://obsidian.md/help/community-plugins) 和 [Configuration folder](https://obsidian.md/help/configuration-folder)。若已加载的代码块显示读取错误，目录到齐后重开笔记或使用“重新加载”重试。


## 全库动画与同参数复现（2026-10-08）

当前 60 个实验笔记均原生嵌入 H.264 MP4。Mac mini 的 Obsidian 1.14.4 / Academic 已实测卷积笔记：Live Preview 与 Reading 都加载视频，播放从 0 秒推进到 5 秒，可暂停；全屏后点击进度条从结尾定位到 19 秒。在 Reading 嵌入内点击第三段动画的“终点”，步幅变为 2、补零变为 1，图形与输出读数同步变为 4×4，宿主显示“参数已更新”。

<!-- bilingual-en:start -->
All 60 experiment notes now embed H.264 MP4s natively. On Mac mini, Obsidian 1.14.4 loaded the convolution film in Live Preview and Reading. Playback advanced, paused, entered full screen and sought from the end to 19 seconds. Loading the third comparison endpoint inside the Reading embed set stride to 2 and padding to 1; the plot and readout changed to 4×4 and the host acknowledged the update.
<!-- bilingual-en:end -->

本日连接 MacBook 的屏幕共享停在重新登录界面，未进入远端桌面；因此未确认本批新增视频在 MacBook 的同步完成和播放。上节 2026-10-07 的插件恢复记录仍是已有证据，不能替代本批视频验收。iOS、Android 与导出落盘本轮也未复测。完整边界见 [[07_Programme/可视化实验室/docs/全库动画验收|全库动画验收]]。

<!-- bilingual-en:start -->
Today's MacBook screen-sharing connection required a new login, so synchronization and playback of the new films on that device were not verified. The previous plugin recovery is historical evidence, not acceptance of these films. iOS, Android and export-to-disk were not retested.
<!-- bilingual-en:end -->

## 机制画面与当前成片复验（2026-10-08 质量重修）

本节对应构建 `b14b4d4146d0102a3ad2e57378b3bad95a62a67ffbe544cdb7d74556aa290f5c`。此前构建号与测试数量保留为历史记录；当前完整记录见 [[07_Programme/可视化实验室/docs/全库动画验收|全库动画验收]]。

- Mac mini / Obsidian 1.14.4 的 Reading 嵌入实际选择卷积第九项；输入与核的对应格同步高亮，累计和为 3。把窗口列从 1 移至 0 后，窗口、输出及累计和同步变为 0，宿主收到参数更新。
- 已打开的播放器曾继续保留旧的约 43 秒成片；离开该笔记再重新打开后，加载当前约 48 秒成片。这是本次现场观察，不把它概括为所有缓存或同步问题的成因。
- 当前成片在 Reading 实际播放、暂停、全屏、定位到约 21 秒，显示第九项乘积与累计结果；Live Preview 实际从 0 秒播放到结束。修正初始淡入后，视频首帧已有完整主图。
- 该次截图已由最新复核图替换；当前证据见末节。该轮没有改动已安装插件或其他插件的启用状态。

<!-- bilingual-en:start -->
The current build was exercised in native Obsidian on Mac mini. Selecting a convolution term and moving the window updated the paired cells and sum. Reopening the note released an older buffered video and loaded the current rendition. Reading mode verified playback, pause, full screen and seeking; Live Preview played from start to finish. The opening frame contains the complete figure. No installed plugin or unrelated plugin setting was changed in this revision.
<!-- bilingual-en:end -->

当前新版本的 MacBook 同步与播放未重新确认，iOS、Android 和导出落盘也未复测。2026-10-07 的 MacBook 插件恢复记录不能替代这些检查。

## 独立分镜成片复验（2026-10-10）

本次最终构建为 `cb37a70929c17bde5b236ddf3ddcd80c8fce14ef55febf3e575c01d3270573c9`。上文的启用名单、构建号和影片时长属于各自日期的历史记录。

Mac mini / Obsidian 1.14.4 的 Reading 模式实际播放新版 61 秒卷积影片，暂停、全屏并定位至 31 秒，画面显示九项乘积与累计和 3。交互区点击“载入本片的模型与数据”，选中第九项后读数为 `1 × 1 = 1 · 前 9 项之和 3`；输出列由 1 改为 0 后，输入高亮、输出位置与累计和同时更新为 0，宿主报告参数更新。Live Preview 实际播放同一新版影片，从 0 秒到结束。

[当前原生播放画面](quality/obsidian-convolution-manim.png) · [当前交互画面](quality/obsidian-mechanism.png)

这轮没有安装或重新启停插件，也没有修改其他插件的状态。两套浏览器回归覆盖 60 项、共 14,710 项检查，0 失败；具体范围与单独保留的全库结构保护失败见 [[07_Programme/可视化实验室/docs/全库动画验收|当前复核记录]]。MacBook、iOS、Android 与导出落盘未在本轮操作，不能以同一路径或历史恢复推断当前版本已同步。

<!-- bilingual-en:start -->
The new convolution film was played, paused, viewed full screen and sought to 31 seconds in native Obsidian on Mac mini. Loading its exact inputs and changing the output column updated the corresponding cells and sum. Live Preview played the new film from start to finish. Plugin states were preserved. Other devices and export-to-disk were not tested in this revision.
<!-- bilingual-en:end -->
