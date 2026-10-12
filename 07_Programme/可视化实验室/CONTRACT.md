# 可视化实验室实现契约

本项目是可离线运行的通用知识实验库，当前收录 **60 项实验**，涵盖数学统计、机器学习、深度学习、表示与生成、基础补充、序列与推断。数量是当前构建结果，不是接口上限。每项实验都有独立的 Obsidian 入口，使用明确模型、合成数据或固定小例子；不记录个人掌握情况。

<!-- bilingual-en:start -->
This offline library currently contains sixty experiments across mathematics, statistics, machine learning and deep learning. That count describes the current build, not an interface limit. Every experiment has an independent Obsidian entry and uses an explicit model, synthetic data or a fixed small example. It does not record personal mastery.
<!-- bilingual-en:end -->

## 权威文件与构建

| 对象 | 权威位置与用途 |
|---|---|
| 数学模型与合法参数 | `assets/math.js` 中的 `LabMath.schemas`，以及各 `*-math.js` 的注册项 |
| 教学内容与控件 | `assets/content.js` 及各 `*-content.js`；控件默认值、范围和选项须与 schema 一致 |
| 当前实验目录 | 构建生成的 `assets/catalog.json`；记录 id、标题、分组、笔记路径与来源 URL |
| 知识原子关联 | `docs/*atom-map.json`；每项 `{path,title,support}` 必须经过正文语义核对 |
| 渲染与交互 | `assets/app.js`、`assets/style.css` |
| 独立离线入口 | `实验室.html`；包含当前全部模型、内容、样式、KaTeX 与字体 |
| 源模板 | `index.html`；扩展组由构建脚本注入，不用源模板代替完整发行包验收 |
| Obsidian 入口 | `00_实验库入口.md` 与 `实验/<id>.md`；原生插件读取固定离线包 |
| 构建证据 | `build-manifest.json`；记录输入及离线 HTML 的 SHA-256 |

在项目目录运行 `node build.mjs`，同步生成目录、原子链接脚本、单文件离线包和构建清单。随后运行 `node build-notes.mjs` 更新实验笔记与总入口的受管区块。新增组还需加入这两个构建器的组列表；不能仅创建文件就认为已接入。生成器拒绝覆盖没有受管标记的已有笔记；受管区块之外的用户文字必须保留。

<!-- bilingual-en:start -->
Run `node build.mjs` from this directory to regenerate the catalog, atom links, self-contained HTML and build manifest. Then run `node build-notes.mjs` to update managed note blocks. New groups must also be registered with both builders. Unmanaged notes and user text outside managed blocks must be preserved.
<!-- bilingual-en:end -->

## 数学接口

`assets/math.js` 同时暴露浏览器全局 `LabMath` 与 CommonJS 导出。扩展模型通过 `LabMath.register(id, schema, model)` 注册唯一 id；扩展脚本在基础内核之后加载。`LabMath.compute(id, params)` 校验参数、补齐默认值，返回 `{metrics, plots, notes, data, params}`。同一 id、合法参数和种子必须可复现；不得以图形需要为由伪造优化结果或把未定义量替换为有限值。

<!-- bilingual-en:start -->
The base module exposes `LabMath` in browsers and through CommonJS. Extensions register a unique id, schema and model after the base module loads. `compute` validates parameters, fills defaults and returns metrics, plots, notes, data and validated parameters. Identical inputs must reproduce the result. A display requirement cannot justify a fabricated optimum or a finite substitute for an undefined quantity.
<!-- bilingual-en:end -->

| 字段或接口 | 约定 |
|---|---|
| 数值 schema | `[default, min, max, integer?]`；范围闭区间，数值须有限；`integer=true` 时只接受整数 |
| 枚举 schema | `[default, [options]]`；严格匹配值及其类型 |
| `validate(id, params)` | 缺少的已知参数取默认值；非法已知值抛错；只返回 schema 中的键，不依赖额外键产生行为 |
| `metrics` | `[{label,value,digits?,suffix?}]`；value 是数值或说明字符串，未定义/无穷必须可辨识 |
| `plots` | 图对象数组，规范见下文 |
| `notes` | 中文动态说明数组，解释当前配置下的退化、边界、数值限制与分母 |
| `data` | 可复算的原始样本、中间量和结果；若用 `null` 表达未定义量，应另附语义标志，避免与零混淆 |
| `params` | `compute` 添加的完整、已校验参数 |
| 共用工具 | RNG、分布函数、最小二乘及 `helpers`；具体导出以 `math.js` 为准 |

有限样本结果、理论期望、优化目标和真正拟合的参数必须分开命名。写明损失是求和还是平均、方差分母、截距是否惩罚、训练与测试随机性。对奇异、零支持、零向量、重根、梯度发散和不收敛应提供真实边界；必要时显示说明字符串或中断曲线，不能默默截成“正常”结果。

<!-- bilingual-en:start -->
Distinguish finite-sample outcomes, theoretical expectations, objectives and genuinely fitted parameters. State loss scaling, variance denominators, intercept penalties and the sources of training or test randomness. Singularities, zero support, zero vectors, repeated roots and divergence require explicit handling rather than a silently regularized display.
<!-- bilingual-en:end -->

## 图形接口与颜色语义

通常图对象为 `{title,xLabel,yLabel,series,...options}`。每条序列有 `type`、`label`、`points`，并可设数值 `color` 调色板索引。空 label 隐藏该项图例，适合合并多条重复拟合曲线。颜色必须配合标签、虚实线、点形或注释；仅改变颜色不能承担全部信息。

<!-- bilingual-en:start -->
A Cartesian plot contains a title, axis labels and series. Series carry a type, label and points, with an optional numeric palette index. An empty label omits a repeated legend entry. Labels, line styles, point shapes or annotations must supplement color.
<!-- bilingual-en:end -->

| `type` | `points` 与选项 |
|---|---|
| `line` | `[[x,y],…]`，可用 `null` 断开；支持 `dashed`、像素 `width`、浅色 `halo`、可选面积 `fill` |
| `scatter` | `[[x,y],…]`；`radius` 为像素，`shape:'diamond'` 可突出查询点或中心；普通序列由渲染器使用圆/叉区分 |
| `segments` | `[[x1,y1,x2,y2],…]`；支持 `dashed`、`width`、`halo`、`arrow:true` |
| `bars` | `[[x,y,width?],…]`；宽度为 x 轴数据单位；基线为零 |
| `heatmap` | `[[x,y,z],…]`；`cellWidth`、`cellHeight` 为数据单位；`zDomain:[min,max]`、`showValues` 控制色标范围与格内值 |

热图固定采用**低值蓝—中值浅色—高值橙红**，不随线条调色板变化。叠加二分类观测时，类别 0 用调色板 `color:2`（蓝），类别 1 用 `color:1`（橙）；图例说明颜色对应的数值或类别。矩阵热图应注明行列意义；分类编号不是连续测量单位。

<!-- bilingual-en:start -->
Heatmaps encode low values in blue, middle values in a pale tone and high values in orange-red. Binary overlays use palette index 2 for class zero and index 1 for class one. Labels must explain the encoded quantity, and matrix plots must identify rows and columns.
<!-- bilingual-en:end -->

| 图选项 | 约定 |
|---|---|
| `xDomain` / `yDomain` | `[min,max]`；显式范围须容纳有意义的有限点、柱/格边缘及必要标记；若刻意只画局部，标题和说明须明确 |
| `equalAspect:true` | 保持 x、y 一个单位的像素长度相同；渲染器会扩展显示范围，适合几何图 |
| `reverseY:true` | y 值由上向下增大，适合矩阵行；不自动改变数据的行列含义 |
| `xTickLabels` / `yTickLabels` | 按数值整数位置索引的文字标签数组；常配显式刻度使用 |
| `xTicks` / `yTicks` | 显式数值刻度；概率轴可用 0…1 刻度并给端点留显示边距 |
| `annotations` | `[{x,y,text}]`；文字位置用数据坐标，须检查裁切和相互遮挡 |
| `height` | 可选图高；窄宽布局仍需实际渲染检查 |
| `diagram` | `{nodes:[{label,value,operation},…]}`；按顺序连箭头，窄屏改纵向；图仍提供 `series:[]`，节点值来自实际计算 |

`diagram` 是计算步骤图，不是任意流程图编辑器。`content.animation:{key:'steps'}` 绑定已有数值滑块，每帧改变合法参数并重新计算真实迭代状态；动画键、范围与步长须匹配控件。不能以装饰性移动代替算法步骤，也不能从动画平滑程度推断收敛。

<!-- bilingual-en:start -->
A diagram is a sequential computation display, not a general graph editor. An animation binds an existing numeric control and recomputes the actual state at each valid step. Decorative motion is not an algorithm, and smooth playback is not evidence of convergence.
<!-- bilingual-en:end -->

## 教学内容与知识链接

各内容脚本合并到全局 `LabContent`，不覆盖其他组。每项包含 `title,en,group,summary,question,formula,controls,presets,steps,explanation,assumptions,pitfalls,challenges,sources,related`；`formula` 为 LaTeX，`animation` 与构建注入的 `atoms` 为可选项。每个合法参数都须有可理解的控件；默认值与 schema 一致。至少提供 **3 组有意义的预设、3 道附提示和完整推导的题目**。

<!-- bilingual-en:start -->
Content extensions merge into `LabContent` without replacing other groups. Each experiment supplies titles, its guiding question, formula, controls, presets, instructions, explanation, assumptions, pitfalls, worked challenges, sources and related ids. Every parameter needs an understandable control matching its schema. Provide at least three meaningful presets and three worked challenges with hints.
<!-- bilingual-en:end -->

来源条目为 `{label,url,section,support}`，说明具体支持的公式、条件或解释。重开教材、大学课程、官方数值文档或原始论文核验，记录在本组 `docs/*来源核验.md`。原子映射必须打开目标正文核对；不存在恰当目标时保留空映射，不能用近似关键词强接。实验入口提供可复用解释，实际掌握、学习计划或个人证据仍属于私有学习记录。

<!-- bilingual-en:start -->
Every source identifies a precise section and the claim it supports. Reopen authoritative sources and record the review in the group ledger. Atom mappings require reading the target's actual content; an empty mapping is preferable to a misleading keyword association. Personal progress and evidence remain in private learning records.
<!-- bilingual-en:end -->

## 插件与离线边界

插件只读取固定的 `实验室.html` 与 `assets/catalog.json`，以隔离 iframe 承载实验；具体安装、消息白名单与权限见 [插件说明](obsidian-plugin/README.md)。数学计算、图形、公式与字体均不依赖页面加载时的网络请求；点击原始来源会由宿主打开已列入目录的 HTTPS 页面。代码块不接受任意 HTML 路径、URL 或脚本。

<!-- bilingual-en:start -->
The plugin reads a fixed bundle and catalog into an isolated iframe. Calculations, charts, formulas and fonts work without page-load network requests. Explicit source navigation opens a catalog-listed HTTPS page through the host. Code blocks cannot select arbitrary HTML, scripts or URLs.
<!-- bilingual-en:end -->

## 验收与证据

每项实验要检查默认值、全部预设、控件边界、图表、动态说明、公式、挑战答案和精确双链。独立计算应针对恒等式、解析例子、有限差分或其他实现，不能只把当前实现抄成期望值。来源核验、数值检查和实际视觉审阅分别记录；每轮先记录具体问题，再记录修改与复核结果，不能从文件存在或首页截图推定全部完成。

<!-- bilingual-en:start -->
Acceptance covers defaults, every preset, parameter boundaries, charts, notes, formulas, worked answers and exact links. Independent checks use identities, analytic examples, finite differences or a separate implementation. Source, numerical and rendered reviews are distinct evidence; each revision needs its actual finding, change and follow-up result.
<!-- bilingual-en:end -->

当前原生集成验收范围是 **Obsidian 桌面端**。浏览器窄宽检查不等于移动端验收，manifest 允许移动端也不是测试证据。SVG、绘图 CSV、包含原始结果的 JSON、配置恢复及逐步播放应按各自实际能力验证。Python 教学 Notebook 的覆盖范围以 [Python 说明](python/README.md) 和执行报告为准；部分基础实验的 Notebook 不能代表 60 项都有 Python 教学版本。完整测试与人工验收记录见项目 `tests/` 和 `docs/`。

<!-- bilingual-en:start -->
Native integration acceptance currently covers Obsidian desktop. Narrow browser layouts and mobile-enabled manifest metadata do not establish mobile acceptance. Verify exports, configuration restoration and playback according to their actual interfaces. Python notebook coverage is defined by its own README and execution reports, not inferred from the sixty-item browser catalog.
<!-- bilingual-en:end -->
