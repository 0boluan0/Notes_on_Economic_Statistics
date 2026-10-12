# Obsidian 内交互支持

本插件把笔记阅读视图中的 `visual-lab` 代码块渲染成可操作实验，也提供独立的 Obsidian 标签页。当前离线包包含 **60 项实验**；插件每次读取固定 `assets/catalog.json` 获取合法编号，不在代码中限定实验数量。图形、公式、代码和字体来自仓库内 `07_Programme/可视化实验室/实验室.html`。

<!-- bilingual-en:start -->
This plugin renders interactive experiments inside reading-view code blocks and a dedicated Obsidian tab. The current offline bundle contains sixty experiments. Valid ids are loaded from the fixed catalog rather than a hard-coded experiment list. Charts, formulas, code and fonts come from the local self-contained bundle.
<!-- bilingual-en:end -->

```visual-lab
id: tangent
height: 900
```

`id` 接受目录中的实验编号或 `library` 总目录，省略时打开总目录。`height` 为 400–2000 的整数，默认 900。只接受这两个字段，各自最多一次；可以留空行和以 `#` 开头的注释。框内可继续切换实验、调整参数、操作有真实步骤的动画，并导出结果。

<!-- bilingual-en:start -->
The id is a catalog entry or `library`, which is the default. Height is an integer between 400 and 2000, defaulting to 900. Only these two unique fields are accepted; blank lines and comment lines beginning with `#` are allowed. Each frame supports navigation, controls, available algorithm-step animations and exports.
<!-- bilingual-en:end -->

启用后，在命令面板选择 **Visual Knowledge Lab: 打开可视化实验室**。笔记中的“在独立标签页打开”保留当前实验编号；“复制实验链接”复制当前实验笔记的 Obsidian 双链。双链不包含当前滑块值；保留具体配置请导出 JSON。修改离线包后，点击“重新加载”读取最新目录和 HTML。

<!-- bilingual-en:start -->
Use the command palette to open the laboratory. Opening a dedicated tab preserves the selected experiment id. Copying an experiment link copies its Obsidian note link, not the current slider values; use JSON export to preserve a configuration. Reload reads the latest catalog and HTML bundle.
<!-- bilingual-en:end -->

## 安装与维护

保留项目在固定的 `07_Programme/可视化实验室/` 路径，确认其中同时存在 `实验室.html` 和 `assets/catalog.json`。插件源文件是本目录的 `main.js`、`manifest.json`、`styles.css`。在用户选择安装时，把这三个文件复制到 `.obsidian/plugins/visual-knowledge-lab/`，再通过 **Settings → Community plugins → Visual Knowledge Lab** 启用。当前 manifest 声明最低 Obsidian 版本为 1.5.0；这只是加载条件，不代表每个历史版本均经过测试。

<!-- bilingual-en:start -->
Keep the project at its fixed vault path with both the HTML bundle and catalog present. Installation copies `main.js`, `manifest.json` and `styles.css` to `.obsidian/plugins/visual-knowledge-lab/`, after which the user enables the plugin in Community plugins. The manifest's minimum version of 1.5.0 is a loading requirement, not evidence of testing every historical release.
<!-- bilingual-en:end -->

本插件是随库保存的自定义插件，没有上架社区市场。iCloud 同步安装文件后，在另一台设备打开 **Settings → Community plugins → Installed plugins**，先点标题旁的刷新图标，再在这个已安装列表中搜索 **Visual Knowledge Lab** 并启用；不要进入 **Browse** 搜索。同步磁盘文件并不保证已经运行的 Obsidian 立即刷新插件列表。若启用了自定义配置目录，应以 **Files and links → Override configuration folder** 为准。

<!-- bilingual-en:start -->
This custom plugin is stored with the vault and is not listed in the community marketplace. After iCloud syncs its installed files, refresh the Installed plugins list on the other device, find Visual Knowledge Lab there, and enable it. Do not search through Browse. File synchronization does not guarantee that a running Obsidian instance has refreshed its plugin list. If a custom configuration folder is active, use the folder specified under Files and links.
<!-- bilingual-en:end -->

升级时替换这三个安装文件，再停用并重新启用插件；安装副本应与源文件逐字节一致。只更新实验内容时，在项目目录执行 `node build.mjs`，再执行 `node build-notes.mjs` 更新受管笔记，随后重新加载实验。卸载时由用户停用并删除插件安装目录；实验笔记与项目文件仍保留。

<!-- bilingual-en:start -->
Upgrade by replacing the three installed files and toggling the plugin off and on. Installed copies should match the source bytes. Content changes require rebuilding the bundle, updating managed notes and reloading the frame. Removing the plugin installation leaves experiment notes and project files intact.
<!-- bilingual-en:end -->

Obsidian 处于 Restricted mode 时第三方插件不会运行。退出该模式及其他插件的恢复应遵循用户明确选择。本插件不修改 Restricted mode、HTML Reader 全局设置或其他插件启用状态，也不建立学习记录数据库；独立标签页的恢复状态仅保存当前实验编号。

<!-- bilingual-en:start -->
Community plugins do not run in Restricted mode. Any change to that mode or other plugins remains the user's choice. This plugin does not alter those settings or create a learning-record database. Its dedicated view restores only the selected experiment id.
<!-- bilingual-en:end -->

## 沙箱与宿主接口

实验 iframe 使用 `sandbox="allow-scripts allow-downloads"`，没有 `allow-same-origin`；实验代码不能直接读取 Obsidian 父文档或使用仓库 API。脚本运行前注入 CSP：默认拒绝资源，允许内联脚本/样式、本地嵌入字体与图像，禁止网络连接、对象、子框架与表单提交。摄像头、麦克风、定位和 iframe 剪贴板权限被关闭。插件宿主本身仍运行在 Obsidian 插件环境中；隔离边界是 iframe 及以下受限消息接口。

<!-- bilingual-en:start -->
The iframe permits scripts and downloads but not same-origin access. It cannot directly read the Obsidian parent document or call vault APIs. A policy inserted before scripts blocks network connections, objects, nested frames and form submission. Camera, microphone, location and iframe clipboard permissions are disabled. The host plugin still runs with Obsidian plugin capabilities; the isolation boundary is the iframe and its restricted message bridge.
<!-- bilingual-en:end -->

所有消息必须来自当前 iframe 的 `contentWindow`，且带 `source: 'visual-knowledge-lab'`。只处理以下类型，其他消息忽略。

<!-- bilingual-en:start -->
Messages must originate from the current iframe window and carry the expected source tag. Only the following message types are handled.
<!-- bilingual-en:end -->

| 消息 | 宿主行为与限制 |
|---|---|
| `ready` / `state` | 只接受目录中的 id，更新临时状态栏；页面 load 本身不算实验已就绪 |
| `open-note` | 只打开 `00_Knowledge/` 或本项目下现有 `.md` 文件；拒绝绝对路径、上溯、协议、查询及编码路径 |
| `open-source` | URL 必须与该实验在固定目录中列出的来源完全一致，为无账号密码的 HTTPS；由宿主打开外部页面，iframe 仍无网络权限 |
| `copy-link` | 宿主从目录重建实验笔记双链并检查路径；不接受 iframe 提供的任意剪贴板文本 |

SVG、绘图 CSV 和 JSON 由用户点击后生成本地下载，不上传结果。复制双链通过宿主写入经过校验的文本。没有任意文件读取、写入、命令执行或通用 URL 打开的消息接口；固定离线包仍是需要维护者审阅的本地代码。

<!-- bilingual-en:start -->
SVG, plotted-data CSV and JSON exports are local downloads initiated by the user. The host copies only a validated note link. The bridge provides no arbitrary file, command or URL interface. The fixed bundle remains local code that maintainers must review.
<!-- bilingual-en:end -->

## 验证范围与故障定位

**Mac mini 已完成下述原生集成检查；MacBook 的 Obsidian 1.14.4 已实测命令入口、抽样实验参数更新及 Live Preview 笔记嵌入。** 两台设备的具体测试范围见 `../docs/Obsidian兼容性.md`，不能把单台设备的结果扩大为所有设备均已通过。manifest 的 `isDesktopOnly:false` 仅允许移动端尝试加载；移动端触控、下载、剪贴板和生命周期尚不能据此视为验收通过。普通浏览器的宽窄布局检查也不能替代移动端原生检查。HTML Reader 或其他 HTML 容器各自有脚本策略，不属于这个插件的已验收运行路径。

<!-- bilingual-en:start -->
The Mac mini has undergone native integration checks. On the MacBook running Obsidian 1.14.4, the command entry, sampling parameter updates and Live Preview note embedding have also been verified. The compatibility record identifies each device's exact coverage. The mobile-enabled manifest permits loading attempts but does not establish mobile acceptance for touch, downloads, clipboard access or lifecycle behavior. Browser viewport tests and other HTML containers are separate environments.
<!-- bilingual-en:end -->

在项目目录运行 `node tests/obsidian-plugin.test.js`，检查目录与配置校验、CSP 顺序、沙箱权限、消息发送者、来源白名单及笔记路径。静态和模拟宿主检查不代替真实桌面端的实验切换、参数变化、图形、公式、导出与双链操作；实际观察和修订记录保存在项目 `docs/`。

<!-- bilingual-en:start -->
The plugin test checks configuration, policy ordering, sandboxing and message validation. Static and mocked-host tests do not replace actual desktop interaction, rendering, export and navigation checks. Project review ledgers retain those observations and revisions.
<!-- bilingual-en:end -->

加载失败时先读框内错误：缺少文件时检查固定路径与离线包；“未知实验编号”时检查当前目录并重新构建；停在“页面已载入，等待实验初始化…”表示尚未收到有效就绪消息。不要用放宽全局 HTML 脚本限制代替检查固定包与插件状态。

<!-- bilingual-en:start -->
Use the frame's error message first: verify fixed files for missing assets, rebuild the catalog for unknown ids, and distinguish page load from an actual ready message. Diagnose the bundle and plugin state rather than broadly weakening another container's script policy.
<!-- bilingual-en:end -->

## 相关实现依据

- [项目实现契约](../CONTRACT.md)
- [Obsidian 官方 Markdown 后处理文档](https://github.com/obsidianmd/obsidian-developer-docs/blob/main/en/Plugins/Editor/Markdown%20post%20processing.md)
- [Obsidian 官方插件 API](https://github.com/obsidianmd/obsidian-api/blob/master/obsidian.d.ts)
