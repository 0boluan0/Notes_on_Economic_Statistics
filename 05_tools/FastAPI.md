---
date: 2026-02-17
aliases: []
tags: [tools, open-source, github]
---

# FastAPI

FastAPI framework, high performance, easy to learn, fast to code, ready for production

## 基本信息
- 输入链接：https://fastapi.tiangolo.com
- 来源类型：GitHub 项目
- GitHub 仓库：[fastapi/fastapi](https://github.com/fastapi/fastapi)
- 官方网站：[https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- 主要语言：Python
- GitHub Stars：95174
- 开源协议：MIT License
- 最近更新：2026-02-17
- 最新发布：[0.129.0](https://github.com/fastapi/fastapi/releases/tag/0.129.0)
- 发布时间：2026-02-12
- 网页标题：FastAPI
- 页面描述：FastAPI framework, high performance, easy to learn, fast to code, ready for production

## 项目介绍（What it does）
- 核心目标：基于公开资料自动汇总项目定位、安装和使用路径。
- 仓库简介：FastAPI framework, high performance, easy to learn, fast to code, ready for production
- 主题标签：api, async, asyncio, fastapi, framework, json, json-schema, openapi

## 安装方法（Installation）
- 参考章节：Requirements, Installation
- FastAPI stands on the shoulders of giants:
- * <a href="https://www.starlette.dev/" class="external-link" target="_blank">Starlette</a> for the web parts.
- * <a href="https://docs.pydantic.dev/" class="external-link" target="_blank">Pydantic</a> for the data parts.
- Create and activate a <a href="https://fastapi.tiangolo.com/virtual-environments/" class="external-link" target="_blank">virtual environment</a> and then install FastAPI:
- 推荐命令示例：
```console
$ pip install "fastapi[standard]"

---> 100%
```

## 首次使用（First run）
- 参考章节：Run it
- Run the server with:
- $ fastapi dev main.py
- ╭────────── FastAPI CLI - Development mode ───────────╮
- │ │
- 推荐命令示例：
```console
$ fastapi dev main.py

 ╭────────── FastAPI CLI - Development mode ───────────╮
 │                                                     │
 │  Serving at: http://127.0.0.1:8000                  │
 │                                                     │
 │  API docs: http://127.0.0.1:8000/docs               │
 │                                                     │
 │  Running in development mode, for production use:   │
 │                                                     │
 │  fastapi run                                        │
 │                                                     │
 ╰─────────────────────────────────────────────────────╯

INFO:     Will watch for changes in these directories: ['/home/user/code/awesomeapp']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [2248755] using WatchFiles
INFO:     Started server process [2248757]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

## 后续使用（Daily usage）
- 参考章节：**Typer**, the FastAPI of CLIs, Example upgrade, Without fastapi-cloud-cli
- If you are building a <abbr title="Command Line Interface">CLI</abbr> app to be used in the terminal instead of a web API, check out <a href="https://typer.tiangolo.com/" class="external-link" target="_blank">**Typer**</a>.
- **Typer** is FastAPI's little sibling. And it's intended to be the **FastAPI of CLIs**. ⌨️ 🚀
- Now modify the file main.py to receive a body from a PUT request.
- Declare the body using standard Python types, thanks to Pydantic.

## 常见问题与排错（Troubleshooting）
- 信息不足：未找到 FAQ / troubleshooting 章节。
- TODO: 增补常见报错、日志位置与修复路径。

## 参考来源（References）
- https://fastapi.tiangolo.com
- https://github.com/fastapi/fastapi
- https://github.com/fastapi/fastapi/blob/master/README.md
- https://fastapi.tiangolo.com/
