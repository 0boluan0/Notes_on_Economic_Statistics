---
aliases:
  - "Bash用前一命令的退出状态决定是否执行AND或OR列表的下一命令"
  - "Bash conditional command execution"
student_os: knowledge-atom
atom_id: CS-SHELL-015
atom_type: method
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[命令退出状态]]"
related:
  - "[[管道退出状态]]"
---

# Bash用前一命令的退出状态决定是否执行AND或OR列表的下一命令

在 Bash 中，`A && B` 仅当 A 返回 0 才执行 B；`A || B` 仅当 A 返回非零才执行 B。这是基于退出状态的短路执行，不是检查 stdout 的内容。用它表达“前一步满足约定条件，才做下一步”时，应先确认 A 的状态含义。
<!-- bilingual-en:start -->
In Bash, `A && B` runs B only when A returns zero; `A || B` runs B only when A returns nonzero. This is short-circuit execution based on status, not on standard-output contents. Before using it to guard a next step, establish what A's statuses mean.
<!-- bilingual-en:end -->

```bash
false && printf '%s\n' skipped
false || printf '%s\n' fallback
```

第一条不打印，第二条打印 `fallback`。`;` 没有同样的状态条件：在正常顺序执行到下一条的情况下，无论前一命令状态如何都会继续。AND 与 OR 列表的结果是实际最后执行命令的状态。
<!-- bilingual-en:start -->
The first line prints nothing; the second prints `fallback`. `;` adds no such status condition: when ordinary sequential execution reaches the next command, it proceeds regardless of the preceding status. An AND or OR list returns the status of its last executed command.
<!-- bilingual-en:end -->

不要把 `A && B || C` 当成通用的 if/else：即使 A 成功，只要 B 返回非零，C 仍会执行。例如 `true && false || printf '%s\n' C` 会打印 C。若 C 只应由 A 的条件决定，写显式的 `if A; then B; else C; fi`。
<!-- bilingual-en:start -->
`A && B || C` is not a general if/else substitute: even when A succeeds, a nonzero B still triggers C. For example, `true && false || printf '%s\n' C` prints C. If only A should choose the branch, write `if A; then B; else C; fi` explicitly.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`Lists` 与 `Compound Commands: if`；[官方命令列表章节](https://www.gnu.org/software/bash/manual/html_node/Lists.html)：支持短路条件、同优先级左结合、列表状态与 if 的分支规则。
- [Missing Semester 2020：Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)，`&&`、`||`、`;` 示例：支持课程中的条件执行语境；三段链反例由列表规则推出。
<!-- bilingual-en:start -->
The local Bash manual establishes short-circuit conditions, equal-precedence left association, list status, and `if` branching. The course supplies the basic operator examples; the three-command counterexample follows from those list rules.
<!-- bilingual-en:end -->
