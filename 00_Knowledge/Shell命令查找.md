---
student_os: knowledge-atom
atom_id: CS-CLI-003
atom_type: execution-rule
aliases:
  - Bash命令查找区分函数内建命令缓存PATH与显式路径
  - Bash command lookup
status: source-checked
requires:
  - "[[环境变量]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# Bash命令查找区分函数内建命令缓存PATH与显式路径
<!-- bilingual-en:start -->
*Bash command lookup distinguishes functions, builtins, cached paths, PATH searches, and explicit paths.*
<!-- bilingual-en:end -->

在 Bash 普通模式中，完成解析和展开后，若简单命令的名称不含 `/`，先查同名 shell 函数，再查内建命令；都不是时，使用已缓存的外部命令路径或按 `PATH` 从左到右搜索可执行文件。名称含 `/` 时按给定路径执行，不搜索 `PATH`。
<!-- bilingual-en:start -->
In ordinary Bash mode, after parsing and expansion, a simple command name without `/` is looked up as a shell function and then a builtin. Otherwise, Bash uses a cached external-command pathname or searches `PATH` directories from left to right. A name containing `/` is used as a path without a `PATH` search.
<!-- bilingual-en:end -->

因此“`PATH` 决定运行哪个命令”只是外部命令搜索层的简写。[[Shell别名|Alias]] 在更早的读入阶段展开；POSIX 模式中的特殊内建命令还有不同优先级，不能把上述顺序当作所有 shell 的统一规则。
<!-- bilingual-en:start -->
Thus, “PATH determines the command” abbreviates only external-command search. [[Shell别名|Aliases]] expand earlier, while input is read. POSIX-mode special builtins have different precedence, so this sequence is not a universal rule for every shell.
<!-- bilingual-en:end -->

只读定位可用 `type -a command_name` 查看 Bash 所知的同名候选，结合 `command -v command_name` 查看解析结果；后者不保证返回文件路径，结果也可能是内建命令或别名。改变搜索路径前应先确认原因：把可由不可信用户写入的目录放到搜索前端会改变信任边界。
<!-- bilingual-en:start -->
Use `type -a command_name` to inspect same-named candidates known to Bash and `command -v command_name` to inspect resolution. The latter need not return a file path: its result can describe a builtin or alias. Before changing search paths, identify the cause; placing an untrusted writable directory first changes the trust boundary.
<!-- bilingual-en:end -->

## 来源与核验

[GNU Bash Manual §3.7.2, Command Search and Execution](https://www.gnu.org/software/bash/manual/html_node/Command-Search-and-Execution.html)：核对函数、内建命令、缓存、PATH 与含 `/` 名称；本机 `bash(1)` COMMAND EXECUTION 段逐项核对。
<!-- bilingual-en:start -->
[GNU Bash Manual §3.7.2: Command Search and Execution](https://www.gnu.org/software/bash/manual/html_node/Command-Search-and-Execution.html) supports function, builtin, cache, PATH, and explicit-path handling, cross-checked against installed `bash(1)`.
<!-- bilingual-en:end -->

[GNU Bash Manual, Bash POSIX Mode](https://www.gnu.org/software/bash/manual/html_node/Bash-POSIX-Mode.html)：用于限定特殊内建命令的模式差异；诊断用法另核对本机 `help type`、`help command`。
<!-- bilingual-en:start -->
[GNU Bash Manual: Bash POSIX Mode](https://www.gnu.org/software/bash/manual/html_node/Bash-POSIX-Mode.html) qualifies special-builtin precedence; diagnostic usage is cross-checked with installed `help type` and `help command`.
<!-- bilingual-en:end -->
