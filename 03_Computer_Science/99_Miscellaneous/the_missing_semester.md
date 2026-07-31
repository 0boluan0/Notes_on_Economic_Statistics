---
aliases:
  - "The Missing Semester"
  - "计算机教育中缺失的一课"
---

# The Missing Semester

> [!summary] 使用这份课程笔记
> 这门课不教新的编程语言，而是补上每天围绕代码发生的工作：命令行、编辑器、数据整理、版本控制、调试、构建和安全。
> 必须掌握的共享解释已在下文嵌入；每讲对应的练习仍保存在 `the_missing_semester_exercises/`。
> <!-- bilingual-en:start -->
> This course does not teach another programming language. It fills in the everyday work around code: shells, editors, data wrangling, version control, debugging, builds, and security.
> The shared explanations you need are embedded below; exercises for each lecture remain in `the_missing_semester_exercises/`.
> <!-- bilingual-en:end -->

## 第 1 讲 课程概览与 shell
<!-- bilingual-en:start -->
*Lecture 1: Course Overview and the Shell*
<!-- bilingual-en:end -->

命令行把程序组合成数据流，是随后所有工具的共同入口。
<!-- bilingual-en:start -->
The command line composes programs into data flows and serves as the common entry point for every tool that follows.
<!-- bilingual-en:end -->

![[Shell、数据整理与命令行环境#Shell 模型]]

## 第 2 讲 Shell 工具和脚本
<!-- bilingual-en:start -->
*Lecture 2: Shell Tools and Scripting*
<!-- bilingual-en:end -->

从“能运行命令”进入可重复脚本，关键是参数边界、输入输出和失败处理。
<!-- bilingual-en:start -->
Moving from commands that merely run to repeatable scripts requires clear argument boundaries, explicit input and output, and deliberate failure handling.
<!-- bilingual-en:end -->

![[Shell、数据整理与命令行环境#stdin、stdout、stderr]]

![[Shell、数据整理与命令行环境#shell 引用与变量]]

![[Shell、数据整理与命令行环境#脚本可靠性]]

## 第 3 讲 编辑器 (Vim)
<!-- bilingual-en:start -->
*Lecture 3: Editors (Vim)*
<!-- bilingual-en:end -->

本讲的重点是形成键盘驱动的编辑模型：移动、选择、操作、重复与宏。编辑器选择不形成独立知识文件；需要时回到 [MIT Missing Semester Editors](https://missing.csail.mit.edu/2020/editors/) 和本讲练习。
<!-- bilingual-en:start -->
This lecture develops a keyboard-driven editing model built from movement, selection, operations, repetition, and macros. Editor choice does not warrant a separate knowledge file; return to [MIT Missing Semester: Editors](https://missing.csail.mit.edu/2020/editors/) and the lecture exercises when needed.
<!-- bilingual-en:end -->

## 第 4 讲 数据整理
<!-- bilingual-en:start -->
*Lecture 4: Data Wrangling*
<!-- bilingual-en:end -->

把标准输入输出接起来后，搜索、过滤、排序、聚合和格式转换才能形成可检查的管线。
<!-- bilingual-en:start -->
Connecting standard input and output lets searching, filtering, sorting, aggregation, and format conversion form an inspectable pipeline.
<!-- bilingual-en:end -->

![[Shell、数据整理与命令行环境#文本处理]]

## 第 5 讲 命令行环境
<!-- bilingual-en:start -->
*Lecture 5: Command-Line Environments*
<!-- bilingual-en:end -->

长期任务和远程工作需要理解作业控制、signals、terminal multiplexer 与环境配置。
<!-- bilingual-en:start -->
Long-running jobs and remote work require an understanding of job control, signals, terminal multiplexers, and environment configuration.
<!-- bilingual-en:end -->

![[Shell、数据整理与命令行环境#作业控制与终端复用]]

![[Shell、数据整理与命令行环境#环境与 dotfiles]]

## 第 6 讲 版本控制 (Git)
<!-- bilingual-en:start -->
*Lecture 6: Version Control (Git)*
<!-- bilingual-en:end -->

Git 的命令只有放回对象图、工作区、暂存区和提交历史中才不会混乱。
<!-- bilingual-en:start -->
Git commands become coherent only when placed in the model of an object graph, working tree, staging area, and commit history.
<!-- bilingual-en:end -->

![[Git 版本控制#Git 数据模型]]

![[Git 版本控制#工作区、暂存区与提交]]

![[Git 版本控制#分支、合并与回退]]

## 第 7 讲 调试及性能分析
<!-- bilingual-en:start -->
*Lecture 7: Debugging and Profiling*
<!-- bilingual-en:end -->

先稳定复现和缩小范围，再用证据定位根因；性能问题先测量而不是猜测。
<!-- bilingual-en:start -->
First reproduce a failure reliably and reduce its scope, then use evidence to locate the root cause. Measure performance problems before guessing about them.
<!-- bilingual-en:end -->

![[测试、调试、异常与断言#调试循环]]

## 第 8 讲 元编程
<!-- bilingual-en:start -->
*Lecture 8: Metaprogramming*
<!-- bilingual-en:end -->

构建系统和依赖管理把输入、产物和环境写成可复现关系；CI 只是在干净环境中重复这些检查。
<!-- bilingual-en:start -->
Build systems and dependency management encode reproducible relationships among inputs, artifacts, and environments; continuous integration repeats those checks in a clean environment.
<!-- bilingual-en:end -->

![[构建、依赖与 CI#构建图、依赖解析与验证流水线]]

## 第 9 讲 安全和密码学
<!-- bilingual-en:start -->
*Lecture 9: Security and Cryptography*
<!-- bilingual-en:end -->

安全从威胁模型和最小权限开始；hash、MAC、签名与加密分别解决不同问题。
<!-- bilingual-en:start -->
Security begins with a threat model and least privilege. Hashes, message authentication codes, signatures, and encryption solve different problems.
<!-- bilingual-en:end -->

![[密码学原语与安全模型#密码学原语]]

![[密码学原语与安全模型#安全模型]]

## 第 10 讲 大杂烩
<!-- bilingual-en:start -->
*Lecture 10: Potpourri*
<!-- bilingual-en:end -->

本讲收集守护进程、FUSE、备份、API 与常见工具选择。只把反复独立使用的内容提升为知识文件，其余留作课程语境和练习。
<!-- bilingual-en:start -->
This lecture surveys daemons, FUSE, backups, APIs, and common tool choices. Only material with repeated independent use should become a knowledge file; the rest remains course context and practice.
<!-- bilingual-en:end -->

## 第 11 讲 提问&回答
<!-- bilingual-en:start -->
*Lecture 11: Questions and Answers*
<!-- bilingual-en:end -->

用这一讲检查自己能否从任务反推工具：数据流问题回到 shell，历史问题回到 Git，失败问题回到调试循环，风险问题回到威胁模型。
<!-- bilingual-en:start -->
Use this lecture to test whether you can infer the right tool from the task: data-flow problems return to the shell, history problems to Git, failures to the debugging loop, and risks to the threat model.
<!-- bilingual-en:end -->
