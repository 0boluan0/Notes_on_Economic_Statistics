---
date: 2026-06-21
aliases:
  - PaperQuay
  - 开源 AI 论文工作台
tags:
  - tools
  - literature-management
  - pdf
  - ai
  - open-source
---

# PaperQuay

PaperQuay 是一个 local-first、开源的桌面 AI 论文工作台，把文献管理、PDF 阅读、全文翻译、结构化概览、研究笔记、Zotero 导入、Agent 操作和本地 RAG 集成在同一个应用中。

## 基本信息

- 项目地址：[WangQrkkk/PaperQuay](https://github.com/WangQrkkk/PaperQuay)
- 定位：面向研究生、研究人员和重度论文阅读用户的跨平台桌面应用
- 支持平台：Windows、macOS、Linux
- 技术栈：Electron、React、TypeScript/Vite、PDF.js、Tiptap/ProseMirror、SQLite/sql.js、sqlite-vec
- 数据模式：local-first；论文库、笔记和本地 RAG 索引主要保存在本地 SQLite 数据库中
- AI 接口：用户自行配置 OpenAI-compatible endpoint、模型与 API Key
- PDF 结构解析：可选接入 MinerU
- 元数据补全：OpenAlex，失败时可回退到 Crossref
- 许可证：AGPL-3.0-only
- 主要语言：TypeScript
- GitHub Stars：218（查询日期：2026-06-21）
- 仓库最近更新：2026-06-20
- 当前最新版本：[PaperQuay v0.1.23](https://github.com/WangQrkkk/PaperQuay/releases/tag/app-v0.1.23)
- 发布日期：2026-06-13

> [!note]
> Zotero 是可选导入来源，不是运行 PaperQuay 的必要依赖。PaperQuay 会建立自己的本地文献库。

## 项目介绍（What it does）

PaperQuay 试图解决论文工作流在多个工具间频繁切换的问题：PDF 阅读器负责原文，翻译工具负责理解，ChatGPT 负责总结，Zotero 负责文献管理，笔记软件又负责知识整理。它把这些环节放入同一个桌面工作区，并保持原文页面、结构块、翻译、批注、笔记和论文元数据之间的关联。

### 主要能力

- **本地文献库**：导入 PDF，设置存储目录，维护分类、嵌套子分类、标签、收藏和元数据。
- **PDF 阅读与标注**：基于 PDF.js 阅读 PDF，记录阅读时间和位置，支持高亮、批注及带标注 PDF 导出。
- **结构化解析与翻译**：使用 MinerU 将 PDF 转为带页面区域关联的结构块；可预先翻译并缓存全文结构块。
- **论文概览**：通过 AI 生成背景、研究问题、方法、实验设置、主要发现、结论与局限等筛选字段。
- **研究笔记**：Tiptap 富文本笔记支持标题、列表、任务、代码、表格、图片、公式、双链、标签和论文引用。
- **内联关联**：可使用 `[[note]]` 链接笔记、`#tag` 组织主题、`@paper` 引用论文。
- **Agent 工作区**：辅助批量重命名、元数据补全、智能打标、标签清理、自动分类和论文总结，并展示工具调用结果供用户确认。
- **Zotero 导入**：从包含 `zotero.sqlite` 的本地 Zotero 数据目录导入 collections、tags 和可用 PDF 附件。
- **本地 RAG**：围绕论文和解析块建立本地检索索引，回答中可显示并跳转到对应结构块或 PDF 页面。
- **备份与恢复**：可通过 WebDAV 备份和恢复文献库、笔记库及本地 RAG 数据库。

### 适合谁

- 希望把论文导入、精读、翻译、总结和笔记集中到一个桌面应用中的用户。
- 希望继续利用 Zotero 现有文献库，但不想让新工具强依赖 Zotero 的用户。
- 希望自己选择 OpenAI-compatible 模型服务，而不是被绑定到固定 AI 平台的用户。
- 重视本地数据存储，同时接受可选外部 API 服务的用户。

## 安装方法（Installation）

### 方案一：下载安装包（普通用户推荐）

打开 [Releases 页面](https://github.com/WangQrkkk/PaperQuay/releases/latest)，在最新版本的 **Assets** 中按系统选择安装包。

| 系统 | 当前 v0.1.23 安装包 | 选择说明 |
|---|---|---|
| Windows x64 | `PaperQuay-0.1.23-win-x64.exe` 或 `.msi` | 一般优先使用 `.exe`；需要 MSI 部署时选择 `.msi` |
| macOS Apple Silicon | `PaperQuay-0.1.23-mac-arm64.dmg` | 适用于 M1、M2、M3、M4 等 Apple 芯片 |
| macOS Intel | `PaperQuay-0.1.23-mac-x64.dmg` | 适用于 Intel Mac |
| Debian / Ubuntu x64 | `PaperQuay-0.1.23-linux-amd64.deb` | 使用系统软件包安装 |
| 其他常见 Linux x64 | `.AppImage` 或 `.tar.gz` | AppImage 适合便携运行；tar.gz 适合手动部署 |

安装完成后直接启动 PaperQuay。普通用户不需要安装 Node.js。

### 方案二：从源码运行（开发者）

前置条件：

- Node.js 18 或更新版本
- npm
- Windows、macOS 或 Linux

```bash
git clone https://github.com/WangQrkkk/PaperQuay.git
cd PaperQuay
npm install
npm run dev
```

常用构建命令：

```bash
# 构建前端
npm run build

# 预览构建后的 Web 资源
npm run preview

# 构建桌面安装包
npm run electron:build
```

### 可选外部服务

- **OpenAI-compatible API**：用于论文概览、翻译、问答、Agent 与 RAG 生成。
- **MinerU API**：用于 PDF 结构化解析；已有本地解析缓存时不一定需要。
- **OpenAlex / Crossref**：用于 DOI、题名等元数据补全；需要联网。
- **WebDAV**：用于数据库备份与恢复。

## 首次使用（First run）

1. 启动 PaperQuay，打开 **Settings**。
2. 选择默认论文存储目录，并确认导入模式和文件命名规则。
3. 拖入 PDF，或使用导入按钮选择文件。
4. 在导入确认窗口检查标题、作者、年份、期刊、DOI、摘要、关键词和重复提示。
5. 确认后，让 PaperQuay 将 PDF 写入设定的存储目录并建立本地文献记录。
6. 在左侧栏建立分类和子分类，把论文拖入合适的 collection，并添加标签或收藏。
7. 打开一篇论文，确认 PDF 阅读、详情面板和阅读进度记录正常。
8. 若使用 AI 功能，在 Settings 中填写 OpenAI-compatible endpoint、API Key、model 和必要的运行参数。
9. 若使用结构化解析和全文块级翻译，配置 MinerU API Key。
10. 若已有 Zotero 文献库，选择包含 `zotero.sqlite` 的 Zotero data directory，再执行导入。
11. 若需要远程备份，配置 WebDAV，并先进行一次可恢复性测试。

> [!warning]
> API Key、私人 PDF、解析结果、笔记数据库和备份文件都不应提交到 Git 仓库。配置第三方 AI、MinerU 或 WebDAV 服务前，应先确认其隐私政策和数据传输范围。

## 后续使用（Daily usage）

### 推荐工作流

1. **导入**：拖入新 PDF。
2. **校对元数据**：检查 DOI、作者、年份、venue 和重复项。
3. **组织**：加入分类、标签和收藏。
4. **结构化解析**：需要块级阅读、翻译或 RAG 时，用 MinerU 生成带页面区域关联的结构块。
5. **快速筛选**：生成 paper overview，先读研究问题、方法、结果和局限。
6. **精读与翻译**：在 PDF 中阅读、选择翻译，或使用已缓存的全文块级翻译。
7. **批注**：高亮、写批注，并通过页面位置回看。
8. **写研究笔记**：用 `[[note]]`、`#tag` 和 `@paper` 建立笔记与论文之间的关系。
9. **调用 Agent**：对选中的论文执行重命名、元数据清理、分类、打标或总结；提交批量修改前检查调用结果。
10. **维护与备份**：定期检查阅读进度、清理标签，并验证 WebDAV 备份可恢复。

### Zotero 共存方式

- PaperQuay 会把 `zotero.sqlite` 复制为临时只读工作文件，不直接修改原始 Zotero 数据库。
- Zotero collections 会导入为 PaperQuay 本地分类。
- 可用的本地 PDF 附件会复制到 PaperQuay 设置的论文存储目录。
- 导入后数据进入 PaperQuay 自己的本地文献库；当前定位不是持续的双向同步。

### 数据与隐私

- 文献库、笔记和本地 RAG 索引存储在本地 SQLite 数据库。
- 导入的 PDF 位于用户指定的论文存储目录。
- 启用 WebDAV 后，数据库备份会上传到用户配置的远程服务器。
- 启用 AI、MinerU 或在线元数据服务时，相关请求可能离开本机；敏感论文应先评估服务条款。

## 常见问题与排错（Troubleshooting）

### macOS 提示无法打开或来源不明

- 先确认安装包来自官方 Releases，并选对 `arm64` 或 `x64` 架构。
- 项目路线图仍列有“完善签名后的 macOS 发布流程”，当前版本可能触发 Gatekeeper。
- 在确认来源和文件可信后，可到 macOS **系统设置 → 隐私与安全性** 查看阻止原因和系统提供的打开选项。
- 不要对来源不明的副本直接关闭系统安全保护。

### AI 翻译、概览、问答或 Agent 不工作

- 检查 endpoint 是否为所选服务要求的 OpenAI-compatible 地址。
- 检查 API Key、model 名称、余额、网络代理和服务商限流。
- 先用较短文本测试基础调用，再测试整篇论文或 RAG。
- 不同模型服务对参数和上下文长度的支持不同；出现 400 类错误时先减少自定义参数。

### MinerU 解析失败

- 确认已配置有效的 MinerU API Key。
- 检查网络、额度、PDF 是否损坏或被加密。
- 重新打开论文前先确认旧解析任务已经结束；v0.1.23 已改进 PDF.js worker 和渲染任务清理。

### Zotero 导入找不到数据

- 选择的是 Zotero **data directory**，其中应包含 `zotero.sqlite`，不是只选择某个 PDF 附件目录。
- 先关闭正在进行数据库写入的 Zotero 操作，再重试导入。
- 确认 PDF 附件在 Zotero 本地可用；仅有云端占位记录时可能无法复制文件。
- PaperQuay 不修改原始 Zotero 数据库，但导入结果保存在 PaperQuay 自己的本地库中。

### PDF 切换、关闭或重新打开时报 worker / render-task 错误

- 先升级到 v0.1.23 或更新版本，该版本专门改进了 PDF.js 生命周期清理。
- 完全关闭并重启 PaperQuay，再重新打开 PDF。
- 若只在某个文件复现，先在其他 PDF 阅读器中确认该文件可正常打开。
- 仍可复现时，记录操作步骤、系统版本、PaperQuay 版本和问题 PDF 特征后提交 GitHub Issue；含隐私内容的 PDF 不要公开上传。

### 元数据补全失败或速度慢

- 检查 OpenAlex / Crossref 网络访问。
- 优先补全 DOI；只有标题时匹配更容易产生歧义。
- 批量查询不稳定时，可配置 OpenAlex premium API key 或 `mailto` polite-pool email。
- 导入确认页中应人工复核自动匹配的作者、年份和 venue。

### WebDAV 备份无法恢复

- 检查服务器地址、账号权限、目标路径和可用空间。
- 首次配置后立即执行一次备份和测试恢复，不要等到数据丢失后才验证。
- 迁移设备前同时保留本地数据库、PDF 存储目录和远端备份。

## 参考来源（References）

- [PaperQuay GitHub 仓库](https://github.com/WangQrkkk/PaperQuay)
- [英文 README](https://github.com/WangQrkkk/PaperQuay/blob/main/README.md)
- [中文 README](https://github.com/WangQrkkk/PaperQuay/blob/main/README.zh-CN.md)
- [PaperQuay v0.1.23 Release](https://github.com/WangQrkkk/PaperQuay/releases/tag/app-v0.1.23)
- [全部 Releases](https://github.com/WangQrkkk/PaperQuay/releases)
- [Security Policy](https://github.com/WangQrkkk/PaperQuay/blob/main/SECURITY.md)
- [License](https://github.com/WangQrkkk/PaperQuay/blob/main/LICENSE)
