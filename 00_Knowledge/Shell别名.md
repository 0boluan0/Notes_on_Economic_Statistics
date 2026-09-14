---
aliases:
  - "Bash别名是在读取命令时替换适用命令词的文本规则"
  - "Bash alias"
student_os: knowledge-atom
atom_id: CS-SHELL-021
atom_type: definition
status: source-checked
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
requires:
  - "[[Shell简单命令]]"
related:
  - "[[Shell函数]]"
  - "[[Shell命令查找]]"
---

# Bash别名是在读取命令时替换适用命令词的文本规则

Bash 别名是把适用的、未引用的命令词替换为预先定义文本的规则。替换发生在读取命令时，不是命令执行到该位置时；替换后的文本还会按 Shell 语法解释。它主要方便交互输入，不是独立程序，也没有函数那样自己的位置参数。
<!-- bilingual-en:start -->
A Bash alias replaces an applicable unquoted command word with predefined text. Expansion happens when the command is read, not when execution reaches it; the replacement is interpreted as shell syntax. Aliases primarily abbreviate interactive input. They are neither separate programs nor functions with their own positional parameters.
<!-- bilingual-en:end -->

在一个独立练习 Bash 中，可明确启用别名展开后逐行输入：
<!-- bilingual-en:start -->
In an isolated practice Bash, explicitly enable alias expansion and enter these on separate lines:
<!-- bilingual-en:end -->

```bash
shopt -s expand_aliases
alias hello='printf "%s\n" hello'
hello
```

最后一行打印 `hello`。默认非交互 Bash 不展开别名；定义与使用写在同一个已读入的命令行上，也可能尚不能使用新定义。别名值末尾若为空格，还可触发对下一个词的别名检查，这不等于为别名建立形参。
<!-- bilingual-en:start -->
The final line prints `hello`. Noninteractive Bash does not expand aliases by default. A new definition may also be unavailable to a use on the same already-read command line. A trailing blank in the alias value enables alias checking of the following word; it does not create alias parameters.
<!-- bilingual-en:end -->

如果需要判断条件、重排或重复使用调用实参，使用[[Shell函数]]，并在转发参数时使用 `"$@"`。不要依赖交互 Shell 中的别名在脚本里自动存在。
<!-- bilingual-en:start -->
Use a [[Shell函数|shell function]] when you need conditions or deliberate reuse and rearrangement of invocation arguments, forwarding them with `"$@"` where appropriate. Do not assume interactive aliases automatically exist and expand in scripts.
<!-- bilingual-en:end -->

## 来源与核验

- GNU Bash 3.2 本机 `bash(1)`，`ALIASES` 与内建 `alias`；[官方别名章节](https://www.gnu.org/software/bash/manual/html_node/Aliases.html)：支持读取时展开、非交互默认、末尾空格与无形参机制的边界。
<!-- bilingual-en:start -->
The local Bash manual supports read-time expansion, the noninteractive default, trailing-blank behavior, and the absence of a function-style argument mechanism. The example enables expansion explicitly so it does not rely on interactive defaults.
<!-- bilingual-en:end -->
