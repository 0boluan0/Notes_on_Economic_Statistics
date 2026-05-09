---
date: 2026-05-04
aliases: []
tags: [tools, open-source, github]
---

# StatsPAI: The Agent-Native Causal Inference & Econometrics Toolkit for Python

Agent-native causal inference & econometrics for Python: 900+ functions across DiD / IV / RD / synth / DML / Bayesian / causal discovery / structural — one `import statspai as sp` for both AI agents and human researchers. Numerically aligned with Stata & R.

## 基本信息
- 输入链接：https://github.com/brycewang-stanford/StatsPAI
- 来源类型：GitHub 项目
- GitHub 仓库：[brycewang-stanford/StatsPAI](https://github.com/brycewang-stanford/StatsPAI)
- 主要语言：Python
- GitHub Stars：150
- 开源协议：MIT License
- 最近更新：2026-05-04
- 最新发布：[StatsPAI 1.12.1 — sp.citation() + Zenodo DOI](https://github.com/brycewang-stanford/StatsPAI/releases/tag/v1.12.1)
- 发布时间：2026-05-01

## 项目介绍（What it does）
- 核心目标：基于公开资料自动汇总项目定位、安装和使用路径。
- 仓库简介：Agent-native causal inference & econometrics for Python: 900+ functions across DiD / IV / RD / synth / DML / Bayesian / causal discovery / structural — one `import statspai as sp` for both AI agents and human researchers. Numerically aligned with Stata & R.

## 安装方法（Installation）
- 参考章节：Installation
- pip install statspai
- With optional dependencies:
- pip install statspai[plotting] # matplotlib, seaborn
- pip install statspai[fixest] # pyfixest for high-dimensional FE
- 推荐命令示例：
```bash
pip install statspai
```
```bash
pip install statspai[plotting]    # matplotlib, seaborn
pip install statspai[fixest]      # pyfixest for high-dimensional FE
```

## 首次使用（First run）
- 参考章节：Quick Start — 60 seconds
- pip install statspai, then run any of the four canonical causal-inference exercises below. StatsPAI ships the classic teaching datasets bundled under sp.datasets — Callaway–Sant'Anna mpdta, Card (1995) returns-to-schooling, Abadie–Diamond–Hainmueller California Prop 99, Lee (2008) Senate RD, LaLonde / NSW–DW, Angrist–Krueger (1991) QOB, Basque terrorism, German reunification — so every snippet runs **offline** with no data wrangling.
- import statspai as sp
- sp.datasets.list_datasets() # name / design / n_obs / paper / expected_main
- 推荐命令示例：
```python
import statspai as sp

sp.datasets.list_datasets()   # name / design / n_obs / paper / expected_main
```

## 后续使用（Daily usage）
- 参考章节：Smart Workflow Engine *(unique to StatsPAI — no other package has these)*, Quick Example, v0.6.0 (2026-04-05) — Complete Econometrics Toolkit + Smart Workflow Engine
- Function | Description
- recommend() | Given data + research question → recommends estimators with reasoning, generates workflow, provides .run()
- compare_estimators() | Runs multiple methods (OLS, matching, IPW, DML, ...) on same data, reports agreement diagnostics
- assumption_audit() | One-call test of ALL assumptions for any method, with pass/fail/remedy for each

## 常见问题与排错（Troubleshooting）
- 信息不足：未找到 FAQ / troubleshooting 章节。
- TODO: 增补常见报错、日志位置与修复路径。

## 参考来源（References）
- https://github.com/brycewang-stanford/StatsPAI
- https://github.com/brycewang-stanford/StatsPAI/blob/main/README.md
