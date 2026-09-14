---
aliases:
  - "Bash大括号展开按文本模式生成多个词而不要求文件存在"
  - "Bash brace expansion"
student_os: knowledge-atom
atom_id: CS-SHELL-006
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell简单命令]]"
related:
  - "[[Shell文件名展开]]"
---

# Bash大括号展开按文本模式生成多个词而不要求文件存在

Bash 大括号展开按 `{a,b}` 或 `{1..3}` 这样的文本模式生成多个词，并保留列表顺序。它发生在变量等其他展开之前，不查询文件系统；这与[[Shell文件名展开]]匹配现存路径不同。本页讨论 Bash，不把此语法当作任意 `sh` 都支持的接口。
<!-- bilingual-en:start -->
Bash brace expansion generates multiple words from textual patterns such as `{a,b}` or `{1..3}`, preserving list order. It precedes parameter and other expansions and does not query the filesystem, unlike [[Shell文件名展开|pathname expansion]]. This is Bash syntax, not a guarantee for every `sh` implementation.
<!-- bilingual-en:end -->

```bash
printf '%s\n' report.{csv,json}
limit=3
printf '%s\n' {1..$limit}
```

前一条生成 `report.csv` 与 `report.json`，即使二者不存在。后一条不会生成 1、2、3：大括号处理时 `$limit` 尚未展开，最终得到字面形式 `{1..3}`。要使用运行时上界，应选择循环控制，不要用 `eval` 强行重新解释数据。
<!-- bilingual-en:start -->
The first command generates `report.csv` and `report.json` whether or not they exist. The second does not generate 1, 2, and 3: `$limit` is not expanded when braces are processed, so the final argument is the literal `{1..3}`. A runtime bound belongs in loop control, not in data reparsed with `eval`.
<!-- bilingual-en:end -->

引号可阻止大括号展开，例如 `'{a,b}'` 保持一个字面参数。大括号生成出的词仍可参与后续展开，例如 `*.{py,sh}` 先形成两个模式，再分别尝试匹配路径。
<!-- bilingual-en:start -->
Quoting can suppress brace expansion: `'{a,b}'` remains one literal argument. Generated words can still undergo later expansions; `*.{py,sh}` first produces two patterns and then attempts pathname matching for each.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Brace Expansion`；[官方对应章节](https://www.gnu.org/software/bash/manual/html_node/Brace-Expansion.html)：支持纯文本生成、顺序、引号和先于其他展开的规则。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，Curly braces：支持共享前后缀的课程用途；运行时上界反例由展开顺序推出。
<!-- bilingual-en:start -->
The local Bash manual supports textual generation, ordering, quoting, and precedence over other expansions. The course supplies the shared-prefix/suffix use case. The runtime-bound counterexample follows from that expansion order.
<!-- bilingual-en:end -->
