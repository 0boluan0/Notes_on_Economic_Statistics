# 前向与反向传播试点的构建

入口是 [[07_Programme/可视化实验室/实验/backprop|前向与反向传播实验]]。

- `assets/dl-math.js`：同一组参数的前向值和解析梯度。
- `story-backprop-00-model.js`：干预、同时更新、分支求和。
- `story-backprop-03-causal.js`：影片与默认交互共用的 SVG 场景；曲线、点位、数值由同一计算产生。
- `story-backprop-04-player.js`：阶段、局部查看、连续播放、暂停与更新。
- `story-backprop-02-film.js`：20 段中文讲稿、动作区间、测得的音轨时长与章节。
- `story-backprop.js` 与 Anatomy 几何：展开后的逐项核算区。

## 构建与验证

在实验室目录执行：

```sh
node build.mjs
node tests/backprop-causal.test.js
node tests/backprop-pilot.test.js
node tests/dl.test.js
```

用本地 HTTP 服务打开 `tests/backprop-causal.html` 与 `tests/backprop-pilot.html`，点击检查按钮；原生嵌入仍需在 Obsidian 的实验笔记内核对。

## 生成有声影片

使用微软 `zh-CN-XiaoxiaoNeural` 中文神经语音，语速设为 −8%。生成器通过 [edge-tts 7.3.1](https://github.com/rany2/edge-tts) 获取逐段 MP3 与句子时间戳。需要 uv、Node、已安装的 `@napi-rs/canvas`、FFmpeg，以及系统 Arial Unicode 字体。模块路径通过 `NODE_PATH` 指向已安装依赖即可。首次生成配音需联网；命中缓存后可离线重绘，成片始终可离线播放。

```sh
node anatomy/backprop/render-causal-film.cjs /tmp/backprop-causal-render
```

脚本按讲稿与声线参数的哈希缓存逐段音频，并读取句子时间戳。每段先讲清观看位置，再启动动作，结束后留出观察时间；连续渲染共享 SVG 场景的 24 fps 帧，加入 AAC 讲解，验证完整解码、音轨、画幅和时长。20 张复核图用于检查结果，不用来合成幻灯片。语音为合成讲解；交互本身由用户控制，不自动播放语音。

检查生成影片、复核图和媒体信息后，将库外输出的 `backprop.mp4` 替换到 `media/`，配套 `backprop-timeline.json` 放入 `animations/`，再执行：

```sh
node animations/export-films.cjs
node build-notes.mjs backprop
node build.mjs
node tests/structure.test.js
```

`animations/render-films.py` 保留此共享场景影片，避免批量任务覆盖。影片更新后，关闭再打开实验笔记刷新媒体；交互中的“重新加载”刷新 HTML。

保留当前影片、源码、时间线、报告与少量当前截图。音频分段与渲染缓存放在库外。Anatomy 的 MIT 许可见 `../LICENSE`。
