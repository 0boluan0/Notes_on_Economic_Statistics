---
date: 2026-02-17
aliases: []
tags: [tools, open-source, github]
---
	
# Paper2Any

Turn paper/text/topic into editable research figures, technical route diagrams, and presentation slides.

## 基本信息
- 输入链接：https://github.com/OpenDCAI/Paper2Any
- 来源类型：GitHub 项目
- GitHub 仓库：[OpenDCAI/Paper2Any](https://github.com/OpenDCAI/Paper2Any)
- 官方网站：[http://dcai-paper2any.nas.cpolar.cn/](http://dcai-paper2any.nas.cpolar.cn/)
- 主要语言：Python
- GitHub Stars：1621
- 开源协议：Apache License 2.0
- 最近更新：2026-02-17
- 网页标题：Paper2Any

## 项目介绍（What it does）
- 核心目标：基于公开资料自动汇总项目定位、安装和使用路径。
- 仓库简介：Turn paper/text/topic into editable research figures, technical route diagrams, and presentation slides.
- 主题标签：agent, ai, aippt, editable-pptx, langgraph, paper2slides, ppt-generator

## 安装方法（Installation）
- 参考章节：Requirements, 🐧 Linux Installation, 1. Create Environment & Install Base Dependencies
- We recommend using Conda to create an isolated environment (Python 3.11).

## 首次使用（First run）
- 参考章节：3. Build + run, Running Without Supabase, Start backend API
- docker compose up -d --build
- Open:
- Frontend: http://localhost:3000
- Backend health: http://localhost:8000/health

## 后续使用（Daily usage）
- 参考章节：4. System dependencies (Ubuntu example), Step 1: Copy Example Files, Workflow-level defaults
- sudo apt-get update
- sudo apt-get install -y inkscape libreoffice poppler-utils wkhtmltopdf
- PAPER2PPT_DEFAULT_MODEL=gpt-5.1
- PAPER2PPT_DEFAULT_IMAGE_MODEL=gemini-3-pro-image-preview

## 常见问题与排错（Troubleshooting）
- 信息不足：未找到 FAQ / troubleshooting 章节。
- TODO: 增补常见报错、日志位置与修复路径。

## 参考来源（References）
- https://github.com/OpenDCAI/Paper2Any
- https://github.com/OpenDCAI/Paper2Any/blob/main/README.md
- http://dcai-paper2any.nas.cpolar.cn/
