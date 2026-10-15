# 数学动画

60 个实验笔记各嵌入一份当前 MP4。反向传播为 约 386 秒、1080p、24 fps，包含中文合成讲解音轨与中文字幕，连续渲染共享交互场景；构建见 [[07_Programme/可视化实验室/anatomy/backprop/README|反向传播构建说明]]。其余 59 项沿用 30 fps、H.264/yuv420p、中英字幕、无音轨版本，其中 Bootstrap 为 81 秒。观看成片无需安装 Manim。

<!-- bilingual-en:start -->
Each of the 60 experiment notes embeds one current MP4. Backpropagation is a 386-second, 1080p, 24 fps film with Chinese synthetic narration and Chinese captions, rendered continuously from the shared interactive scene. The other 59 films retain their silent, 30 fps, H.264/yuv420p versions with Chinese and English captions. Bootstrap remains 81 seconds. Playback does not require Manim.
<!-- bilingual-en:end -->

反向传播试点的构建与更新见 [[07_Programme/可视化实验室/anatomy/backprop/README|共享场景影片说明]]。

## 分镜、数据与交互

`narrative_neural.py`、`narrative_geometry.py`、`narrative_statistics.py`、`narrative_learning.py` 保存各主题的分镜：先固定一个具体对象，再展示运算、结果及条件。`narrative.py` 提供文字、公式、坐标、单元格和时间线工具。数学变换保持对象身份与坐标含义；例如 SVD 的正交部分使用角度旋转，伸缩部分按奇异值变化。

`export-narratives.cjs` 从既有 JavaScript 模型导出完整参数、数据及必要对照。影片读取这些计算结果，并对额外展开的算式执行一致性断言。它没有另一套随机样本生成器。`*-timeline.json` 记录实际章节时间、检查点和参数；`export-films.cjs` 汇总为 `film-manifest.json`，并生成交互区的“载入本片的模型与数据”。`film-specs.cjs` 仅维护额外的交互起止对照。

<!-- bilingual-en:start -->
Four domain-specific Python files define each topic's sequence. The shared renderer supplies typography, coordinates, addressable cells and timestamps. Numerical payloads come from the existing JavaScript models; additional arithmetic expansions are checked against those results. Timelines record actual chapter times and parameters. The metadata exporter also creates the interactive control that restores each film's exact inputs. The older comparison specifications now supply optional interactive comparisons only.
<!-- bilingual-en:end -->

## 渲染与检查

在实验室目录、下述 Manim Python 环境中执行。工作目录必须在库外：

```sh
python animations/render-films.py --work /tmp/visual-lab-narratives --review review --workers 3
python animations/render-films.py --work /tmp/visual-lab-narratives --workers 3
node animations/export-films.cjs
node build-notes.mjs
node build.mjs
python tests/narrative.test.py
python tests/film_renderer.test.py
node tests/structure.test.js
```

先查看各检查点，再渲染正式影片。`--ids attention svd` 只重做指定主题。正式渲染检查编码、完整解码、时间线和边界后，原位替换对应 MP4，并提取首帧与全部检查点。每部影片成功后清除其片段与排版缓存。边界检查不能发现所有遮挡，必须查看实际编码画面；旋转等连续变换还需检查中间帧。

浏览器检查页为 `tests/browser.html?autorun` 和 `tests/films-browser.html?autorun`。最终需在 Obsidian 笔记中实际播放、暂停、定位，并操作同一参数实例。修改画面后，旧截图与旧检查结果不能证明新版本合格。完成后仅保留当前影片、源码、时间线、报告与少量当前检查图。当前结果见 [[07_Programme/可视化实验室/docs/全库动画验收|全库动画验收]]。

<!-- bilingual-en:start -->
Review checkpoints before the final render. The renderer validates encoding, full decoding, timing and frame bounds, publishes each film in place, and extracts the first frame and all checkpoints. Bounds checks do not detect every overlap; inspect encoded images and intermediate motion frames. Run both browser suites against the final bundle, then verify playback and interaction inside Obsidian. Retain current artifacts and compact evidence, and remove temporary renders and caches.
<!-- bilingual-en:end -->

## Bootstrap 机制动画

`bootstrap.py` 用 Manim 0.21.0 渲染约 81 秒的 Bootstrap 短片。读取 `assets/math.js` 的真实样本、六份重抽记录及 500 个均值；不另写随机抽样模型。字幕为中英双语，无音轨。公式用 MathTypst 排版。

<!-- bilingual-en:start -->
`bootstrap.py` renders an approximately 81-second Bootstrap film in Manim 0.21.0. It reads the actual sample, six resampling traces and 500 means from `assets/math.js`, without a second random model. Chinese and English captions are rendered into the silent video; formulas use MathTypst.
<!-- bilingual-en:end -->

运行环境保留在本机的 `uv` 工具目录中，不放入 iCloud 库。初次准备需要 `uv tool install 'manim==0.21.0' --with typst`；macOS 构建 pycairo 还需已有的 Cairo 与 pkgconf。无需完整 TeX 安装，也不调用模型 API。

<!-- bilingual-en:start -->
The authoring environment lives in the local `uv` tool directory, outside iCloud. Install it with `uv tool install 'manim==0.21.0' --with typst`; building pycairo on macOS also requires Cairo and pkgconf. No full TeX installation or model API is used.
<!-- bilingual-en:end -->

在实验室目录执行：

```sh
manim -qh --fps 30 --media_dir /tmp/visual-lab-manim animations/bootstrap.py Bootstrap
ffmpeg -i /tmp/visual-lab-manim/videos/bootstrap/1080p30/Bootstrap.mp4 -c copy -movflags +faststart media/bootstrap.mp4
node build-notes.mjs
```

每次渲染前自动检查均值、分位数、样本身份、重复计数和条件标准差。脚本会输出实际分镜时间 `bootstrap-timeline.json` 到临时渲染目录。独立运行 `python animations/bootstrap.py` 也能执行数值检查，Python 需使用上述 Manim 环境。

<!-- bilingual-en:start -->
Every render first checks means, quantiles, sample identities, multiplicities and conditional standard deviation. The script writes actual scene timestamps to `bootstrap-timeline.json` in the temporary render directory. Running `python animations/bootstrap.py` with the same environment performs the numerical checks without rendering.
<!-- bilingual-en:end -->

发布文件只有 `media/bootstrap.mp4`；原生视频通过 [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap 实验笔记]] 嵌入。改变参数使用笔记下方的交互实验。更新时核对实际关键帧、原生播放及实验中的相同参数，完成后清除临时目录中的旧清晰度版本、片段和排版缓存。保留成片、源码和必要的当前验收图片。

<!-- bilingual-en:start -->
The distributed film is `media/bootstrap.mp4`, embedded natively in the [[07_Programme/可视化实验室/实验/bootstrap|Bootstrap experiment note]]. Use the interactive experiment below it to change parameters. For updates, inspect actual frames, native playback and matching experiment parameters, then remove temporary renditions, partial clips and typesetting caches. Retain the current film, source and necessary current review images.
<!-- bilingual-en:end -->

制作参考：[Math-To-Manim 的分镜与视觉审查思路](https://github.com/HarleyCoops/Math-To-Manim/blob/main/docs/JEV_DESIGN_MAP.md)、[Manim MathTypst](https://docs.manim.community/en/stable/reference/manim.mobject.text.typst_mobject.MathTypst.html)。当前场景为本地编写，未运行 Math-To-Manim 的自动生成链。
