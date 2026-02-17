---
date: 2026-02-17
aliases: []
tags: [tools, ai-workflow, paper]
---

# Paper2Any

从论文 PDF/截图/文本到图表、流程图、PPT 等可编辑内容的一体化工作流工具。

## 基本信息

- 仓库链接：[OpenDCAI/Paper2Any](https://github.com/OpenDCAI/Paper2Any)
- 官方文档链接：[README_CN](https://github.com/OpenDCAI/Paper2Any/blob/main/README_CN.md) / [README](https://github.com/OpenDCAI/Paper2Any/blob/main/README.md)
- 适用场景：当你需要把论文或技术材料快速转成可编辑科研图、演示文稿、流程图或答辩文本时。

## 项目作用（What it does）

Paper2Any 解决的是“学术内容到可交付产物”的自动化问题：把论文内容（PDF/图片/文本）转成可编辑的多模态输出，减少手工画图、排版与整理时间。

主要输出类型包括：

- 科研图：模型架构图、技术路线图、实验数据图
- Drawio 图：论文/文本到可编辑 Drawio 流程图、结构图
- 演示文稿：Paper2PPT（论文到 PPT）
- 版式保留转换：PDF2PPT（PDF 转可编辑 PPTX）
- 图片转演示文稿：Image2PPT
- 论文回复辅助：Paper2Rebuttal（rebuttal 草稿与修订响应）

## 项目特点（Why it is useful）

- 多模态输入：支持 PDF、图片、文本等多种输入形式
- 多工作流统一入口：同一项目内覆盖绘图、PPT、转换、rebuttal 等能力
- 可编辑输出：重点产物支持 PPTX/SVG/Drawio 等可二次编辑格式
- 双入口使用：既可用 Web 前端，也可直接跑 CLI 脚本
- Docker 推荐部署：官方优先提供 `docker compose` 快速启动路径
- 配置弹性：可选 `.env`，也可在部分 CLI 场景下直接通过参数传入 API 信息
- 数据可持久化：Docker 场景下 `./outputs`、`./models` 挂载到宿主机，重启后数据仍保留

## 使用方法（第一次安装使用）

### A. Docker（推荐，首次）

```bash
# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. 后端环境变量（用于 API Key/模型配置）
cp fastapi_app/.env.example fastapi_app/.env

# 3. 构建并启动
docker compose up -d --build
```

首次启动后访问：

- 前端：`http://localhost:3000`
- 后端健康检查：`http://localhost:8000/health`

首次说明：

- 首次构建通常较慢（会安装系统与 Python 依赖）
- 前端配置是构建期生效，改动后通常需要再次 `docker compose up -d --build`

### B. 本地安装（Linux 为主，补充 Windows 入口）

Linux 首次安装命令（推荐使用 conda 隔离环境）：

```bash
# 0. 创建并激活环境
conda create -n paper2any python=3.11 -y
conda activate paper2any

# 1. 克隆仓库
git clone https://github.com/OpenDCAI/Paper2Any.git
cd Paper2Any

# 2. 基础依赖
pip install -r requirements-base.txt
pip install -e .

# 3. Paper2Any 相关依赖
pip install -r requirements-paper.txt || pip install -r requirements-paper-backup.txt
conda install -c conda-forge tectonic -y
pip install doclayout_yolo --no-deps
sudo apt-get update
sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
```

本地首次运行（Web 前后端）：

```bash
# 终端 1：启动后端
cd fastapi_app
uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
# 终端 2：启动前端
cd frontend-workflow
npm install
npm run dev
```

访问：

- `http://localhost:3000`

Windows 入口（简要）：

- 官方文档提供原生 Windows 安装路径，建议优先参考 `README_CN` 的“Windows 安装”章节执行（包含 `requirements-win-base.txt`、Inkscape 与可选 vLLM 安装）。

## 使用方法（后续启动使用）

### A. Docker（后续）

```bash
# 常规重启
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down
```

```bash
# 代码或 .env 变更后重建
docker compose up -d --build

# 更新到上游新版本并重建
git pull && docker compose up -d --build
```

### B. 本地（后续）

常规启动流程：

```bash
# 1. 激活环境
conda activate paper2any

# 2. 启动后端（终端 1）
cd fastapi_app && uvicorn main:app --host 0.0.0.0 --port 8000
```

```bash
# 3. 启动前端（终端 2）
cd frontend-workflow && npm run dev
```

CLI 方式（无需同时启动 Web 前后端）的代表命令：

```bash
# 示例 1：Paper2Figure
python script/run_paper2figure_cli.py \
  --input paper.pdf \
  --graph-type model_arch \
  --api-key sk-xxx
```

```bash
# 示例 2：Paper2PPT
python script/run_paper2ppt_cli.py \
  --input paper.pdf \
  --api-key sk-xxx \
  --page-count 15
```

## 快速排错与注意事项

- 端口占用（3000/8000）：若前端或后端起不来，先检查端口是否被其他进程占用（如 `lsof -i :3000`、`lsof -i :8000`）
- API Key 未配置：常见现象是工作流请求报鉴权错误、生成任务直接失败或返回上游模型服务错误
- Docker 修改后不生效：改了前端相关配置或 `.env` 后，优先执行 `docker compose up -d --build`
- Linux 图形/PDF 依赖缺失：若出现图形转换、PPT/PDF 处理失败，优先检查 `inkscape`、`libreoffice`、`poppler-utils`、`wkhtmltopdf` 是否已安装

## 参考来源

- https://github.com/OpenDCAI/Paper2Any
- https://github.com/OpenDCAI/Paper2Any/blob/main/README_CN.md
- https://github.com/OpenDCAI/Paper2Any/blob/main/README.md
