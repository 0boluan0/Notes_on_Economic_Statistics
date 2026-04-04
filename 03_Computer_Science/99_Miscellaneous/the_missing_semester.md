---
aliases:
  - The Missing Semester
  - 计算机教育中缺失的一课
tags:
  - computer-science
  - tools
  - shell
  - git
  - vim
  - security
---

# The Missing Semester

## 课程定位

>[!note]
> 这门课真正想补的不是某个单一工具，而是程序员的工作流能力。
> 它默认你已经会写代码，但还没有系统掌握：
> - 如何高效使用 shell 和命令行环境
> - 如何真正把编辑器变成生产工具
> - 如何用 Git、调试工具、构建系统和安全工具支撑长期工作
> - 如何让自己的环境、脚本和流程可复用、可迁移、可自动化

大学里的很多课程会教你算法、体系结构、操作系统、机器学习，但不会专门训练你如何高效使用每天都要碰的工具。于是很多人虽然会写程序，却仍然停留在“复制粘贴命令”“出问题再搜”“环境一换全崩”的阶段。`The Missing Semester` 想解决的正是这个问题。

它的核心精神不是“多背命令”，而是：

- 把命令行看成可编程环境，而不是黑窗口
- 把编辑器看成语言和工作流，而不是输入框
- 把 Git 看成数据模型，而不是神秘命令集合
- 把调试、性能分析、安全和构建看成工程习惯，而不是高级附属品

## 资料入口

- 官网首页：https://missing-semester-cn.github.io/
- 课程概览与 shell：https://missing-semester-cn.github.io/2020/course-shell/
- Shell 工具和脚本：https://missing-semester-cn.github.io/2020/shell-tools/
- 编辑器 (Vim)：https://missing-semester-cn.github.io/2020/editors/
- 数据整理：https://missing-semester-cn.github.io/2020/data-wrangling/
- 命令行环境：https://missing-semester-cn.github.io/2020/command-line/
- 版本控制 (Git)：https://missing-semester-cn.github.io/2020/version-control/
- 调试及性能分析：https://missing-semester-cn.github.io/2020/debugging-profiling/
- 元编程：https://missing-semester-cn.github.io/2020/metaprogramming/
- 安全和密码学：https://missing-semester-cn.github.io/2020/security/
- 大杂烩：https://missing-semester-cn.github.io/2020/potpourri/
- 提问&回答：https://missing-semester-cn.github.io/2020/qa/

## 如何使用这篇笔记

- 第一遍：按讲次顺序看，先建立整门课的地图。
- 第二遍：把每一讲里的示例命令自己在终端跑一遍。
- 第三遍：把常用内容吸收到自己的 dotfiles、脚本、Git 配置和编辑器里。

>[!note]
> 这一版采用“课堂讲义主线 + 本讲命令速查表 + 对应练习入口”的单层结构。每讲按主题顺序记录课堂概念、命令、例子和常见坑，避免时间戳层和重复总结层把信息打散。

如果只看不练，这门课几乎不会真正内化。它的价值高度依赖“把知识变成工作流”。

## 课程主线地图

| 讲次 | 主题 | 真正要学会的东西 |
| --- | --- | --- |
| 1 | 课程概览与 shell | 用 shell 作为文本接口操控计算机 |
| 2 | Shell 工具和脚本 | 用脚本和工具把重复劳动自动化 |
| 3 | 编辑器 (Vim) | 用模式化编辑提高读写代码效率 |
| 4 | 数据整理 | 用管道和小工具逐步变换数据 |
| 5 | 命令行环境 | 管理进程、会话、远程机器和配置 |
| 6 | 版本控制 (Git) | 把 Git 当成快照图和数据模型 |
| 7 | 调试及性能分析 | 系统定位错误和性能瓶颈 |
| 8 | 元编程 | 用构建、测试、依赖和 CI 管理工程流程 |
| 9 | 安全和密码学 | 理解常用安全工具背后的基本模型 |
| 10 | 大杂烩 | 补齐大量高频但零散的实用主题 |
| 11 | 提问&回答 | 回答工具选择与学习路径上的现实问题 |

## 第 1 讲 课程概览与 shell

资料：<https://missing-semester-cn.github.io/2020/course-shell/>

### 本讲主线

这一讲要你明白：为什么即使今天图形界面极强，shell 仍然是程序员最该掌握的接口之一。

GUI 适合覆盖高频、固定、被设计者预先考虑到的交互；shell 适合开放式组合、批量处理、自动化和远程工作。程序员需要的恰恰经常是后者。

### shell 是什么

shell 是一个命令解释器，也是一个编程环境。

它至少做三件事：

- 读取你输入的命令
- 解析命令、参数、重定向、管道和变量
- 找到对应程序并执行

因此 shell 不是“一个程序列表”，而是“组织程序协作的语言层”。

### 核心概念

- `command`：你要执行的程序
- `argument`：传给程序的参数
- `current working directory`：程序默认工作的目录
- `PATH`：shell 在哪些目录里搜索可执行程序
- `absolute path` vs `relative path`：绝对路径从根目录开始，相对路径相对于当前工作目录
- `.` 与 `..`：当前目录和父目录

### 最先要掌握的命令

```bash
pwd
cd /path/to/dir
cd ..
cd ~
ls
ls -l
man ls
which echo
echo $PATH
```

这些命令表面简单，但它们连着 shell 的最基本模型：你现在在哪、系统会去哪里找程序、一个程序又是如何被文档化的。

### 重定向与管道

shell 中最重要的抽象之一，是“程序读输入流、写输出流”。

- `>`：把标准输出写到文件
- `>>`：追加写入
- `<`：从文件读取作为输入
- `|`：把前一个程序的输出接到后一个程序的输入

这意味着你可以把很多小工具像积木一样串起来，而不是写一大坨程序。

>[!example] 典型工作流
>
> ```bash
> echo hello > hello.txt
> cat hello.txt
> cat < hello.txt > hello2.txt
> ls -l / | tail -n 1
> curl --head --silent google.com | grep -i content-length
> ```
>
> 这类命令真正重要的不是具体输出，而是你要形成直觉：
> 一个程序的输出通常都能成为下一个程序的输入。

### 权限与 root

`ls -l` 里最容易被忽视但非常重要的内容是权限位。

- 目录的 `x` 更多表示“可进入/可搜索”
- 文件的 `x` 表示“可执行”
- root 用户几乎不受限制，所以 `sudo` 很强也很危险

官网里专门用亮度文件的例子说明了一件很容易踩坑的事：`sudo echo 3 > brightness` 往往失败，不是因为 `echo` 没有 root 权限，而是因为重定向是 shell 先做的，写文件这一步不是由 `sudo echo` 完成的。

正确思路通常是：

```bash
echo 3 | sudo tee brightness
```

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `date` | 打印当前时间 | 第一讲开场用来说明 shell 可以直接调用程序 |
| `echo hello` | 打印文本参数 | 展示“命令 + 参数”的基本结构 |
| `echo "hello world" / echo 'hello'` | 带空格参数与 quoting | 说明引号决定 shell 如何切分参数 |
| `pwd` | 显示当前工作目录 | 建立路径坐标系 |
| `cd /path / cd .. / cd ~ / cd -` | 切换目录 | 目录导航、回 home、返回上一个目录 |
| `ls / ls -l / ls -lah` | 列目录内容与权限/大小信息 | 查看文件树、权限位、隐藏文件和可读大小 |
| `mv old new` | 移动或重命名文件 | 常用文件操作 |
| `cp src dst` | 复制文件 | 常用文件操作 |
| `rm file / rmdir dir` | 删除文件/空目录 | 常用文件操作，理解删除和目录状态 |
| `mkdir dir` | 创建目录 | 常用文件操作 |
| `man ls / cmd --help` | 查看命令文档 | 命令不会背时优先查官方帮助 |
| `which echo / type echo` | 查看命令解析到哪里 | 确认 shell 实际执行的是哪个程序/内建 |
| `echo $PATH` | 查看可执行文件搜索路径 | 解释 shell 为什么能直接找到某个命令 |
| `cat file / cat < file > out` | 查看文件或连接输入输出 | 引出标准输入/输出和重定向 |
| `tail -n 1` | 取最后若干行 | 配合 `ls -l / \| tail -n 1` 演示管道 |
| `curl --head --silent URL` | 抓 HTTP 头并安静输出 | 配合 `grep -i content-length` 演示网络输出进入管道 |
| `grep -i pattern` | 按模式过滤文本 | 作为管道右侧消费者 |
| `> / >> / < / \| / 2>` | 输出覆盖、追加、输入重定向、管道、错误重定向 | 标准流重接线是 shell 最核心抽象 |
| `sudo cmd` | 以 root 权限运行命令 | 权限提升不是对整行 shell 语法都生效 |
| `echo 3 \| sudo tee file` | 以 root 权限写文件 | 修复 `sudo echo 3 > file` 的重定向陷阱 |
| `chmod +x file` | 增加可执行权限 | 让脚本可直接执行，理解权限位和 shebang |
| `#!/bin/sh` | shebang 指定解释器 | 解释 shell 如何知道脚本该交给谁执行 |
| `xdg-open file / open file` | 用系统默认程序打开文件 | 讲义里提到的便捷文件打开方式 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/01_Course Shell Exercises|第 1 讲练习]]

## 第 2 讲 Shell 工具和脚本

资料：<https://missing-semester-cn.github.io/2020/shell-tools/>

### 本讲主线

第一讲让你学会手动使用 shell；这一讲让你学会把 shell 变成自动化工具。

当你发现自己总是在重复执行一串命令时，就应该开始考虑脚本、函数、别名和更合适的命令行工具。

### Shell 脚本的核心地位

shell 脚本并不是“比 Python 更高级”的语言，而是“更贴近命令行工作流”的语言。

它擅长的事包括：

- 调命令
- 连管道
- 重定向
- 批量处理文件
- 在系统环境里做轻量自动化

它不适合承担大型程序的复杂数据结构、复杂错误处理和长期维护逻辑。

### 变量、引号、控制流

#### 变量赋值

```bash
foo=bar
echo "$foo"
echo '$foo'
```

关键点：

- 赋值时等号两侧不能有空格
- 双引号会展开变量
- 单引号基本保持原样

#### 条件执行

```bash
false || echo "Oops, fail"
true && echo "Things went well"
false ; echo "This will always run"
```

关键点：

- `&&`：前一个成功才继续
- `||`：前一个失败才继续
- `;`：无论如何都继续

### 特殊变量

官网特别强调 Bash 的一组特殊变量，因为它们极常用：

- `$0`：脚本名
- `$1` 到 `$9`：位置参数
- `$@`：全部参数
- `$#`：参数个数
- `$?`：前一条命令退出码
- `$$`：当前脚本进程号
- `!!`：上一条完整命令
- `$_`：上一条命令最后一个参数

### 命令替换与进程替换

#### 命令替换

```bash
files=$(ls)
now=$(date)
```

shell 会先执行命令，再把输出结果替换进去。

#### 进程替换

```bash
diff <(ls foo) <(ls bar)
```

这类写法适合“某个工具想读文件，但你手头只有命令输出”的场景。

### 函数、脚本与 shebang

函数和脚本的关键差异：

- 函数运行在当前 shell 进程里，能直接改当前环境
- 脚本通常在新的进程里执行，不能直接修改父 shell 的当前目录
- 脚本可以用任意解释器，不局限于 shell

shebang 的作用是告诉系统用什么解释器执行脚本。

```bash
#!/usr/bin/env bash
#!/usr/bin/env python3
```

`/usr/bin/env` 的价值在于可移植性，它会根据 `PATH` 去找解释器。

### 查找与搜索工具

#### `find`

适合按文件树、属性、时间、权限等复杂条件搜索。

```bash
find . -name '*.tmp' -exec rm {} \;
find . -name '*.png' -exec magick {} {}.jpg \;
```

#### `fd`

官网推荐把它看成更现代、更友好的 `find` 替代品。

- 默认更快
- 语法更简单
- 默认行为更符合直觉

#### `locate`

基于数据库索引，速度快，但结果可能不够新，也不如 `find` 灵活。

#### `grep` / `ripgrep`

- `grep`：经典文本搜索
- `rg`（ripgrep）：更快，更适合代码仓库

### 代码质量工具

shell 脚本很容易写出“能跑但很脆”的东西，所以课程推荐用 `shellcheck` 之类工具检查常见错误。

比如：

- 忘记给变量加引号
- 误用 `for file in $(ls)`
- 条件判断写法不稳

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `foo=bar` | 变量赋值 | 等号两侧不能有空格 |
| `"$foo" / '$foo'` | 变量展开与字面字符串 | 双引号展开变量，单引号基本不展开 |
| `mcd () { mkdir -p "$1"; cd "$1"; }` | shell 函数 | 函数运行在当前 shell，可直接改变目录 |
| `$0 $1 $@ $# $? $$ !! $_` | 脚本上下文特殊变量 | 参数读取、退出码检查、重复上一条命令 |
| `false \|\| echo fail / true && echo ok / cmd1 ; cmd2` | 基于退出码的流程控制 | shell 常用的短路执行方式 |
| `if [[ -f file ]]; then ... fi` | 条件判断 | 检查文件存在等场景 |
| `for f in ...; do ...; done` | 循环批处理 | 批量处理参数或文件集合 |
| `$(cmd)` | 命令替换 | 把命令输出嵌入参数或变量 |
| `diff <(ls foo) <(ls bar)` | 进程替换 | 把命令输出伪装成文件输入 |
| `*.sh / image.{png,jpg} / project/**/test/*.py` | glob 与 brace expansion | 批量生成路径参数 |
| `#!/usr/bin/env bash` | 可移植 shebang | 不要写死解释器位置 |
| `shellcheck script.sh` | 检查 shell 脚本常见错误 | 尤其是未加引号、脆弱循环、条件误写 |
| `find . -name "*.tmp" -exec rm {} \;` | 按文件属性递归查找并执行动作 | 比手工 `ls` 更适合复杂筛选 |
| `fd pattern` | 更现代的 find 替代 | 默认行为更贴近日常查找文件 |
| `locate name` | 基于索引的快速文件名查找 | 很快，但结果可能不是最新 |
| `grep -R pattern . / rg pattern` | 按内容搜索代码/文本 | `rg` 往往更快且默认忽略 `.git` |
| `history / Ctrl+R / fzf` | 搜索历史命令 | 降低重复命令的记忆负担 |
| `autojump / fasd` | 按历史快速跳目录 | 频繁目录切换的效率工具 |
| `tree / broot / nnn / ranger` | 目录树查看与终端文件管理 | 课程推荐的目录导航扩展工具 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/02_Shell Tools Exercises|第 2 讲练习]]

## 第 3 讲 编辑器 (Vim)

资料：<https://missing-semester-cn.github.io/2020/editors/>

### 本讲主线

这讲不是在教你一个“终端里的古老编辑器”，而是在教一种高效率的编辑思想。

程序员的大量时间并不花在连续输入文本上，而是花在：

- 阅读代码
- 跳转位置
- 选中对象
- 做小范围修改
- 在多个文件之间切换

Vim 之所以值得学，是因为它把这些高频动作设计成了一个可组合的语言。

### Vim 的哲学

#### 多模态

Vim 的一个核心假设是：阅读/导航/编辑是不同动作，不应该都用“插入字符”的同一套键盘语义。

常用模式包括：

- 正常模式
- 插入模式
- 替换模式
- 可视模式
- 命令模式

你大部分时间应该在正常模式，而不是插入模式。

#### 命令可组合

Vim 的接口像一种微型语言：

- 移动命令像“名词”
- 编辑命令像“动词”
- 次数和修饰语进一步改变它们的含义

例如：

- `dw`：删除一个词
- `d$`：删除到行尾
- `ci(`：改括号内内容
- `3w`：向前移动三个词

### 最基本的操作分层

#### 进入与退出

```vim
i
R
v
V
Ctrl-v
:
<Esc>
```

#### 文件级命令

```vim
:q
:w
:wq
:e filename
:ls
:help topic
```

#### 常用移动

```vim
h j k l
w b e
0 ^ $
gg G
/pattern
n N
f{char}
%
```

#### 常用编辑

```vim
x
d{motion}
c{motion}
y
p
u
Ctrl-r
o
O
```

### 真正高效的地方

Vim 的效率不来自“快捷键多”，而来自三个机制：

- 常见对象都可以被快速定位
- 动作和对象可以组合
- 肌肉记忆形成后，修改非常少依赖鼠标

>[!example] 典型编辑思路
>
> 假设你要把函数调用里的参数整体改掉：
>
> - 传统编辑器常见做法：鼠标选中、删除、输入
> - Vim 思路：`ci(`
>
> 你不再是“移动光标到每个字符”，而是“对语法对象做操作”。

### 你还需要掌握的扩展能力

- 搜索替换：`:%s/foo/bar/g`
- 窗口分屏：`:sp` / `:vsp`
- 宏：`q{寄存器}`、`@{寄存器}`
- 帮助系统：`:help`
- 学习资源：`vimtutor`

### 配置与生态

Vim 课程并不要求你一上来就折腾复杂插件生态，但强调两件事：

- 配置是值得投入的，因为编辑器是长期工具
- 不要盲目复制别人整套配置，先理解再吸收

最基本的长期动作包括：

- 写自己的 `~/.vimrc`
- 给 shell / readline / 浏览器开启 Vim 风格键位
- 逐步把高频操作映射到更顺手的方式

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `vim file` | 打开文件进入 Vim | 编辑器入口 |
| `i / a / o / O / Esc` | 进入插入、追加、新行插入、回到普通模式 | 最小可用编辑流程 |
| `:w / :q / :wq / :q!` | 保存、退出、保存退出、强制退出 | 文件级操作 |
| `h j k l` | 左下上右移动 | 普通模式基础导航 |
| `w b e` | 按单词前进/后退/到词尾 | 语义级移动 |
| `0 ^ $` | 到行首/首个非空字符/行尾 | 行内导航 |
| `gg / G / {line}G` | 到文件开头/结尾/指定行 | 跨文件大范围跳转 |
| `f{char} / t{char} / ; / ,` | 在行内按字符跳转并重复 | 快速定位局部文本 |
| `% / ( ) / { }` | 括号匹配、句子/段落移动 | 按结构移动 |
| `x / r{char}` | 删字符/替换字符 | 最小粒度编辑 |
| `d{motion} / c{motion} / y{motion}` | 删除/修改/复制一个范围 | operator + motion 组合思想 |
| `dd / cc / yy / p / P` | 行级删除/修改/复制/粘贴 | 高频编辑动作 |
| `u / Ctrl-r` | 撤销/重做 | 编辑回退 |
| `v / V / Ctrl-v` | 字符/行/块可视选择 | 显式选择区域 |
| `ci" / da( / yi{` | text object 操作 | 直接对引号/括号/代码块内部或整体编辑 |
| `/pattern / ?pattern / n / N` | 搜索并跳转匹配 | 定位文本 |
| `:%s/foo/bar/g` | 全文件替换 | 搜索替换主力命令 |
| `:sp / :vsp / Ctrl-w hjkl` | 分屏与窗口切换 | 多窗口编辑 |
| `:e file / :ls / :bN` | 打开文件、列 buffer、切 buffer | 理解 buffer 和 window 的区别 |
| `q{reg} ... q / @{reg}` | 录制和重放宏 | 处理重复但带模式的编辑任务 |
| `:help subject / vimtutor` | 查看帮助与完成入门教程 | 课程推荐的学习路径 |
| `~/.vimrc / :CtrlP` | 配置与插件入口 | 把高频能力固化到编辑器环境 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/03_Vim Exercises|第 3 讲练习]]

## 第 4 讲 数据整理

资料：<https://missing-semester-cn.github.io/2020/data-wrangling/>

### 本讲主线

现实里的数据往往不是“直接拿来就能分析”的，它通常是日志、CSV、命令输出、半结构化文本，需要经过一连串变换才变成有用信息。

数据整理的关键不是某一个工具，而是：

- 理解输入是什么
- 明白目标输出要长什么样
- 选择合适的小工具逐步把前者变成后者

### 核心工作流思想

课程用日志处理说明了一个非常典型的思路：

1. 先过滤噪音
2. 再提取感兴趣字段
3. 再排序、统计或聚合
4. 必要时把中间结果保存下来

这与 shell 的哲学完全一致：不要一上来就写大程序，先把小工具串起来。

### 正则表达式

正则是数据整理的基本功。至少要熟悉这些模式：

- `.`：任意单字符
- `*`：前一模式重复零次或多次
- `+`：前一模式重复一次或多次
- `[abc]`：字符集合
- `(A|B)`：或
- `^`：行首
- `$`：行尾

课程也特别提醒了正则的现实问题：

- 不同工具的正则方言有差异
- 贪婪匹配很容易匹配过头
- 正则非常强大，但并不总是最稳的解决方案

### 高频工具

#### `grep`

做过滤，找匹配行。

#### `sed`

做流式替换与简单重写，最经典语法是：

```bash
sed 's/REGEX/SUBSTITUTION/'
```

#### `awk`

按字段处理文本，适合：

- 取列
- 统计
- 条件筛选
- 简单汇总

#### `sort` / `uniq`

适合排序与去重统计。

#### `cut` / `paste`

适合按分隔符抽列、拼列。

#### `less`

适合把长输出变得可读。

### 远程数据整理

课程里一个非常好的习惯是：尽量在数据所在位置先过滤，再传输结果。

```bash
ssh myserver 'journalctl | grep sshd | grep "Disconnected from"' > ssh.log
```

这不只是“会用 ssh”，而是理解了：

- 网络也是成本
- 过滤应该尽量前置
- 管道可以跨机器工作

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `ssh myserver journalctl` | 在远端机器读取日志 | 先在数据源附近获取数据 |
| `grep sshd / grep "Disconnected from"` | 按文本模式逐层过滤 | 日志清洗第一步 |
| `sed -E 's/.../.../'` | 用扩展正则做替换和捕获组提取 | 从半结构化行中抽出用户名/IP 等字段 |
| `sort / uniq -c / sort -nk1,1` | 排序、计数、按数值排序 | 把清洗后的行变成频数统计 |
| `awk '{print $2}' / awk '$1 == 1 && $2 ~ /^c[^ ]*e$/ { ... }'` | 按字段提取、条件过滤、聚合 | 把文本当表格处理 |
| `BEGIN { ... } / END { ... }` | awk 开头/结尾动作 | 汇总统计 |
| `paste -sd+ \| bc -l` | 把多行数字拼成表达式并计算 | 在 shell 管道末端做数值运算 |
| `R --slave -e 'x <- scan(file="stdin", quiet=TRUE); summary(x)'` | 把管道结果送进 R 做统计摘要 | 命令行数据整理连接统计工具 |
| `gnuplot -p -e 'set boxwidth 0.5; plot "-" using 1:xtic(2) with boxes'` | 从 stdin 绘图 | 命令行可视化收尾 |
| `xargs cmd` | 把标准输入转回命令参数 | 批量卸载/批量处理文件 |
| `tr / sed y` | 字符级转换 | 大小写归一化等预处理 |
| `less` | 分页检查中间输出 | 长管道每一步都要验证 |
| `journalctl -b` | 按 boot 提取系统日志 | 课后练习中的启动日志对比 |
| `ffmpeg ... \| convert - -colorspace gray - \| gzip \| ssh host ...` | 二进制流也可走管道 | 摄像头图像处理后压缩并发到远端 |
| `tee copy.jpg` | 一边写文件一边继续向下游输出 | 在远端保存二进制副本并显示 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/04_Data Wrangling Exercises|第 4 讲练习]]

## 第 5 讲 命令行环境

资料：<https://missing-semester-cn.github.io/2020/command-line/>

### 本讲主线

前几讲教你“怎么执行命令”，这讲教你“怎么把命令行变成长期可用的工作环境”。

重点包括：

- 任务控制
- 终端多路复用
- 别名与 dotfiles
- SSH 与远程工作

### 任务控制

需要掌握的不是命令本身，而是信号模型。

常见信号与操作：

- `Ctrl-C`：通常发 `SIGINT`
- `Ctrl-\`：通常发 `SIGQUIT`
- `Ctrl-Z`：通常发 `SIGTSTP`
- `kill -TERM PID`：更优雅地请求终止
- `kill -KILL PID`：强制杀死，最后手段

需要形成的工作习惯：

- 优先优雅结束，不要默认 `kill -9`
- 学会区分前台、后台、挂起、终止
- 理解关闭终端时后台子进程可能一起死掉

### jobs / fg / bg / nohup / disown

这套组合决定你是否能自然地管理长任务。

```bash
jobs
fg %1
bg %1
nohup long-running-command &
disown
```

### tmux：把终端升级为工作空间

tmux 解决的是“一个终端窗口不够用，而且断线后不想丢工作”的问题。

它的核心对象层次：

- session
- window
- pane

最重要的价值：

- 多任务并行
- 分屏
- 断开后重连
- 远程工作更稳定

### 别名与 dotfiles

别名适合缩短高频命令，例如：

```bash
alias ll="ls -lh"
alias gs="git status"
alias v="vim"
```

但别名只是表层。更重要的是 dotfiles：

- `.bashrc`
- `.zshrc`
- `.vimrc`
- `.gitconfig`

课程真正鼓励的是：把自己的环境配置文本化、版本化、可迁移。

### SSH

SSH 不只是“远程登录”。

你应该把它看成：

- 远程 shell
- 文件传输能力
- 端口转发能力
- 公钥认证体系

长期习惯包括：

- 用公私钥登录，少用密码
- 配置 `~/.ssh/config`
- 理解 known_hosts 与 host key 的意义
- 远程长任务优先用 tmux

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `Ctrl-C / Ctrl-\ / Ctrl-Z` | 向前台进程发送 SIGINT/SIGQUIT/SIGTSTP | 任务控制的键盘入口 |
| `jobs / fg %1 / bg %1` | 查看任务、切回前台、后台继续运行 | 管理当前 shell 会话里的 job |
| `kill -TERM PID / kill -KILL PID / kill -0 PID` | 请求终止、强制杀死、只检测进程是否存在 | 优先优雅结束，`kill -0` 可写 `pidwait` |
| `pgrep -af pattern / pkill -f pattern` | 按名字查找或结束进程 | 避免手工复制 PID |
| `sleep 60 & / wait` | 后台启动并等待子进程结束 | 任务编排与课后练习 |
| `nohup cmd & / disown` | 让任务脱离终端挂断影响 | 单命令级别的不中断运行 |
| `tmux / tmux new -s name / tmux ls / tmux attach -t name` | 创建、列出、重连会话 | session 级别的上下文保留 |
| `tmux prefix + c / % / " / d` | 新建 window、左右/上下分屏、detach | 终端多路复用常用操作 |
| `alias dc=cd` | 定义命令别名 | 修正高频误输入或缩短长命令 |
| `source ~/.bashrc / source ~/.zshrc` | 重载 shell 配置 | 修改 dotfiles 后立即生效 |
| `PS1=...` | 自定义提示符 | 在 prompt 中显示环境/路径/分支等信息 |
| `ln -s source target` | 为 dotfiles 建符号链接 | 让仓库里的配置接回 home 目录固定路径 |
| `ssh user@host / ssh host cmd` | 登录远端或直接执行远端命令 | 把 shell 工作流延伸到远端机器 |
| `ssh-keygen -o -a 100 -t ed25519` | 生成较现代的 SSH 密钥对 | 推荐的密钥生成方式 |
| `ssh-agent / ssh-add` | 缓存解锁后的私钥 | 避免反复输入 passphrase |
| `ssh-copy-id vm` | 把公钥装到远端 `authorized_keys` | 启用免密码 SSH 登录 |
| `scp src vm:dst / rsync -avP src vm:dst` | 复制或增量同步文件 | 远端文件传输 |
| `ssh -L 9999:localhost:8888 vm` | 本地端口转发 | 从本机访问远端 Web 服务 |
| `~/.ssh/config` | 配置主机别名、用户、密钥和端口转发 | 让多主机 SSH 使用更干净 |
| `mosh vm` | 高延迟/网络切换下更稳的远程 shell | SSH 体验优化 |
| `sshfs vm:/path mountpoint` | 把远端目录挂载成本地文件系统 | 远端文件编辑/浏览 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/05_Command-line Environment Exercises|第 5 讲练习]]

## 第 6 讲 版本控制 (Git)

资料：<https://missing-semester-cn.github.io/2020/version-control/>

### 本讲主线

课程对 Git 的态度非常明确：不要从命令表学 Git，要从数据模型学 Git。

很多人用 Git 时像在背咒语，是因为只记命令，不理解底层对象和历史图。

### Git 的本质

Git 不是“改动列表工具”，而是“快照图管理工具”。

它把项目在某个时刻的完整状态记录成快照，并用提交图把这些快照串起来。

核心对象可以粗略理解为：

- blob：文件内容
- tree：目录结构
- commit：一次快照与元数据

提交不是“补丁”，而是“带父指针的快照节点”。

### 为什么这个模型重要

一旦你理解提交图是 DAG，很多命令就不再神秘：

- `branch`：本质是某个提交的可移动引用
- `HEAD`：当前所在位置
- `merge`：把两段历史合并
- `rebase`：在另一条基线上重放提交
- `checkout` / `switch`：移动工作位置

### 你至少要能清楚区分的东西

- 工作区
- 暂存区
- 提交历史
- 本地分支
- 远程分支

如果这几层混在一起，Git 就会永远显得诡异。

### 高频命令的正确理解

```bash
git status
git add
git commit
git log --all --graph --decorate
git diff
git checkout
git switch
git branch
git merge
git rebase
git stash
git reset
git reflog
git cherry-pick
git bisect
```

课程真正强调的不是“记住这些名字”，而是知道每个命令在操作哪一层状态。

### merge 与 rebase

- `merge`：保留分叉结构，生成合并提交
- `rebase`：改写提交基底，让历史看起来更线性

没有哪一个永远更高级。你需要理解两者在“历史表达方式”上的差异。

### reflog 的现实价值

Git 最让人安心的一点，是很多“看起来丢了”的东西其实没真丢。

`reflog` 能帮你找回：

- 被 reset 掉的提交
- 被误切换前的位置
- 一时找不到的历史状态

这也是为什么理解引用和提交对象很重要。

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `git init` | 初始化仓库 | 创建 `.git` 对象数据库和引用系统 |
| `git status` | 查看工作区和暂存区状态 | 日常确认当前改动处于哪一层 |
| `git add file` | 把工作区改动加入暂存区 | 准备进入下一次 commit |
| `git commit -m "msg"` | 从暂存区创建提交并移动当前分支 | Git 快照生成主动作 |
| `git log --all --graph --decorate --oneline` | 用图看所有分支历史 | 理解提交 DAG 和 ref 位置 |
| `git cat-file -p HASH` | 查看对象库里的 blob/tree/commit 内容 | 用 plumbing 命令直接观察数据模型 |
| `git branch name / git checkout name / git switch name` | 创建/切换分支 | 分支只是可移动指针 |
| `git checkout HASH` | 切到旧提交进入 detached HEAD | 临时查看历史快照 |
| `git diff / git diff --staged` | 比较工作区与暂存区、暂存区与 HEAD | 分清三层状态差异 |
| `git restore file` | 用历史或暂存区内容恢复工作区文件 | 撤销局部文件改动 |
| `git reset [--soft\|--mixed\|--hard] TARGET` | 移动分支并可选重置暂存区/工作区 | 理解引用移动与三层状态变化 |
| `git merge branch` | 创建合并提交或快进合并 | 把分叉历史重新汇合 |
| `git rebase base` | 把一串提交重放到新基底上 | 改写提交历史以线性化 |
| `git remote -v / git remote add origin URL` | 查看/添加远端 | 远端本质上是另一个仓库引用集合 |
| `git clone URL` | 复制远端仓库到本地 | 获得对象和远端跟踪分支 |
| `git fetch / git pull / git push` | 拉取对象、拉取并合并/变基、推送引用更新 | 远端协作主命令 |
| `git stash / git stash pop` | 临时收起未完成改动并恢复 | 切任务前保存现场 |
| `git blame file / git show COMMIT` | 追踪某行是谁改的、查看提交细节 | 课程网站练习里的历史追查 |
| `git bisect` | 二分定位引入 bug 的提交 | 版本历史本身也是调试工具 |
| `git reflog` | 查看 HEAD/ref 移动记录 | 从误 reset/误切换中找回状态 |
| `git config --global alias.graph ...` | 配置 Git 别名 | 把常用历史图命令固定下来 |
| `git config --global core.excludesfile ~/.gitignore_global` | 设置全局忽略文件 | 忽略 OS/编辑器临时文件 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/06_Version Control Exercises|第 6 讲练习]]

## 第 7 讲 调试及性能分析

资料：<https://missing-semester-cn.github.io/2020/debugging-profiling/>

### 本讲主线

这讲处理两个现实问题：

- 程序错了，怎么定位
- 程序慢了，怎么证明它为什么慢

很多人对这两件事的默认做法是“盲改”和“凭感觉优化”，课程要你彻底改掉这种习惯。

### 调试的层次

#### 打印与日志

打印调试依然有效，但日志更适合长期系统。

日志的价值在于：

- 可持久化
- 可分级
- 可过滤
- 很多问题发生时，日志可能已经包含线索

常见日志级别：

- DEBUG
- INFO
- WARN
- ERROR

### 调试器与系统调用追踪

有些问题不是“程序逻辑错了”，而是“程序和系统交互出了问题”。

这时就需要：

- `gdb` / `pdb` 等调试器
- `strace` / `dtruss` 看系统调用
- `lsof` 看打开文件
- `tcpdump` / `Wireshark` 看网络包
- 浏览器 DevTools 看前端和网络请求

### 静态分析

静态分析的价值在于：不运行代码，也能发现大量明显问题。

例如：

- 拼写错误
- 未定义变量
- 覆盖名字
- 风格与潜在 bug

这类工具包括语言自己的 linter、type checker、静态分析器等。

### Profiling：先测再优化

性能分析最重要的原则：

- 不要先猜
- 先测瓶颈
- 优化后再测

你通常会用到：

- wall-clock time
- CPU time
- 内存分配
- I/O 等待

典型工具包括：

- `time`
- `perf`
- 各语言 profiler
- benchmark 工具

### 微基准测试的坑

课程会提醒你：benchmark 很容易测错。

常见干扰项包括：

- 缓存预热
- 编译器优化
- 输入规模太小
- 测试环境不稳定
- 把启动成本和核心逻辑混在一起

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `journalctl / log show` | 查看系统日志 | 追踪 sudo、服务退出、系统事件 |
| `logger "msg"` | 向系统日志写一条消息 | 自造日志事件便于观察 |
| `python -m pdb script.py / import pdb; pdb.set_trace()` | 进入 Python 调试器 | 断点、单步、看变量和调用栈 |
| `gdb / lldb` | 调试 C/C++ 等程序 | 低层调试器 |
| `strace -p PID / ltrace ./prog` | 跟踪系统调用/库调用 | 定位程序卡在 OS/库边界的原因 |
| `shellcheck script.sh` | 静态检查 shell 脚本 | 课程示例里自动发现脆弱写法 |
| `pyflakes / mypy` | Python 静态分析/类型检查 | 运行前发现错误 |
| `time cmd` | 粗粒度测命令耗时 | 先看整体再深入 profile |
| `python -m cProfile -s tottime script.py` | 函数级 CPU profiling | 定位热点函数 |
| `kernprof -l -v script.py` | 行级 profiling | 定位具体慢在哪一行 |
| `python -m memory_profiler script.py` | 按行看内存增长 | 定位内存瓶颈 |
| `perf stat / perf record / perf report` | 硬件计数器与采样分析 | 观察循环数、缓存命中/缺失 |
| `hyperfine --warmup 3 "cmd1" "cmd2"` | 稳定做多次 benchmark 比较 | 减少预热和噪声影响 |
| `htop / top` | 交互式查看 CPU/进程状态 | 先判断资源瓶颈在哪 |
| `iotop` | 查看磁盘 I/O 热点进程 | I/O 慢时定位元凶 |
| `df -h / du -h / ncdu` | 看磁盘容量和目录占用 | 磁盘空间排查 |
| `free -h` | 查看内存占用概况 | 内存压力检查 |
| `lsof \| grep LISTEN` | 查看监听端口对应进程 | 定位端口被谁占用 |
| `ss -tulpn` | 查看网络连接/监听状态 | 网络层排查 |
| `stress -c 3 / taskset --cpu-list 0,2 stress -c 3` | 造 CPU 压力并限制 CPU 亲和性 | 资源限制与观测练习 |
| `rr / RevPDB` | 可逆调试 | 进阶调试练习 |
| `wireshark + http filter` | 抓包并过滤 HTTP 流量 | 观察 `curl ipinfo.io` 的请求与响应 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/07_Debugging and Profiling Exercises|第 7 讲练习]]

## 第 8 讲 元编程

资料：<https://missing-semester-cn.github.io/2020/metaprogramming/>

### 本讲主线

这里的 “元编程” 不是指语言层面的宏系统，而是指“围绕程序开发流程的编程”。

课程关注的是：

- 构建系统
- 依赖管理
- 测试
- 持续集成

它们看起来不像“写功能”，但在真实工程里经常比功能本身更决定效率。

### 构建系统

构建系统本质上是在声明三件事：

- 目标
- 依赖
- 规则

典型例子是 `make`。

它的关键价值不是“帮你执行命令”，而是：

- 知道依赖关系
- 知道哪些目标已经过期
- 只重建必要部分

这就是增量构建思想。

>[!example] 典型理解
>
> 如果 `paper.pdf` 依赖 `paper.tex` 和 `plot-data.png`，而 `plot-data.png` 又依赖脚本和数据文件，那么当你只改了 `paper.tex` 时，`make` 不应该重新生成图像。
>
> 构建系统的价值就在于“自动判断哪些步骤真的需要重跑”。

### 依赖管理

一个项目可能依赖：

- 解释器或编译器
- 系统包
- 语言生态里的库
- 外部服务和工具链

课程强调几个现实原则：

- 依赖必须有版本
- 版本升级必须理解兼容性
- 不同层级的包管理器要分清角色

### 语义化版本

课程明确提到 `semver` 的思路：

- patch：修 bug，不改 API
- minor：新增兼容功能
- major：破坏兼容

这套约定的价值，是让依赖关系能更有预测性地管理。

### 测试

测试不是“交作业前再跑一下”，而是给重构和协作提供安全边界。

你至少要理解几类测试：

- 单元测试
- 集成测试
- 回归测试
- 性能测试

好测试的目标不是追求数量，而是：

- 关键路径有保护
- 失败能快速定位
- 执行成本可接受

### 持续集成

CI 的核心不是“上云跑命令”，而是把“质量检查”从个人习惯变成系统流程。

典型 CI 会自动做：

- 安装依赖
- 构建
- 测试
- Lint
- 部署前检查

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `make / make target` | 按依赖规则构建目标 | 构建系统核心入口 |
| `target: deps
	command` | Makefile 规则格式 | 目标、依赖、命令三元组 |
| `$@ / $< / $^` | Make 自动变量 | 写通用构建规则 |
| `.PHONY: clean` | 声明伪目标 | `clean` 不是文件，而是动作名 |
| `make clean` | 清理构建产物 | 撤销构建并强制重新生成 |
| `git ls-files` | 列出 Git 跟踪文件 | 写 `clean` 目标时辅助判断哪些可删 |
| `^1.2.3 / ~1.2.3 / * / >=1.2,<2.0` | 依赖版本约束写法 | 理解 semver 与兼容范围 |
| `package-lock.json / Cargo.lock / Gemfile.lock` | 锁定依赖解析结果 | 保证未来和 CI 可复现 |
| `vendoring` | 把依赖源码纳入项目 | 提高可控性但增加同步成本 |
| `pytest / test target` | 自动化测试入口 | 把“别忘了测”变成机制 |
| `.git/hooks/pre-commit` | 提交前自动执行检查 | 用 Git hook 拒绝不可构建提交 |
| `GitHub Actions workflow` | 仓库事件触发 CI | push/PR 后自动构建、测试、lint、部署 |
| `shellcheck` | CI 中检查 shell 脚本 | 课程练习要求对所有 shell 文件跑它 |
| `proselint / write-good` | CI 中检查 Markdown 写作质量 | 自定义 Action 练习 |

### 对应练习

练习笔记：[[the_missing_semester_exercises/08_Metaprogramming Exercises|第 8 讲练习]]

## 第 9 讲 安全和密码学

资料：<https://missing-semester-cn.github.io/2020/security/>

### 本讲主线

这讲不是教你设计密码算法，而是教你理解常见安全工具背后的基本模型，避免在日常开发和使用中“会用但不懂”。

### 熵

熵衡量的是不确定性，也就是猜测难度。

课程借密码强调一个现实原则：

- 安全性不是“字符看起来复杂”
- 而是“攻击者可行搜索空间到底有多大”

40 比特熵大致能抵抗很多在线暴力猜测；
面对离线暴力破解，通常需要更高熵。

### 密码散列函数

散列函数把任意输入映射到固定长度输出。

安全散列至少要求：

- 确定性
- 难以反推原文
- 难以构造碰撞

课程用 Git 里的 SHA-1 说明散列的工程用途：

- 内容寻址
- 文件完整性校验
- 承诺机制

### KDF

密钥生成函数的作用是：

- 从密码导出密钥
- 抵抗暴力枚举

关键实践包括：

- 不存明文密码
- 加盐
- 使用慢一点的 KDF

### 对称加密与非对称加密

#### 对称加密

同一把密钥负责加密和解密。

适合：

- 文件加密
- 磁盘加密
- 会话内数据加密

#### 非对称加密

公钥与私钥分工不同。

适合：

- 安全分发加密能力
- 签名与验签
- SSH / PGP / 软件签名

课程这里最重要的不是数学细节，而是使用模型：

- 加密解决保密性
- 签名解决真实性与不可抵赖

### 现实安全习惯

课程最后强调的安全建议非常务实：

- 用密码管理器
- 对每个站点使用独立强密码
- 开启 2FA
- 用全盘加密
- 理解 SSH / Git 签名等工具背后的信任模型

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `log2(N)` | 估算搜索空间熵 | 比较 4 个随机单词口令和 8 位随机字母数字口令强度 |
| `sha256sum file` | 计算文件哈希 | 对比下载镜像和官方摘要 |
| `openssl aes-256-cbc -salt -in plain -out cipher` | 用对称加密加密文件 | 课程练习中的 AES 示例 |
| `openssl aes-256-cbc -d -in cipher -out plain` | 解密文件 | 验证加解密正确性 |
| `cat / hexdump / cmp` | 查看密文、比较原文与解密结果 | 验证密文不可读且解密后完全一致 |
| `ssh-keygen -o -a 100 -t ed25519` | 生成 SSH 密钥对 | 非对称加密实践入口 |
| `gpg --full-generate-key / gpg --encrypt / gpg --decrypt` | 生成 GPG 密钥、加密、解密 | 邮件加密练习 |
| `git commit -S / git show --show-signature` | 签名提交并验证签名 | 把签名机制接到 Git 工作流 |
| `git tag -s / git tag -v` | 签名 tag 并验证 | 发布对象签名 |
| `Password manager / 2FA / full-disk encryption` | 密码管理器、双因子认证、全盘加密 | 课程收尾强调的现实安全默认项 |
| `KDF + salt` | 从口令导出密钥并抵抗暴力猜测 | 理解为什么不能只做普通 hash |
| `public key / private key / certificate / TOFU` | 公私钥、证书、首次信任模型 | 理解“怎么确认这个公钥真属于对方” |

### 对应练习

练习笔记：[[the_missing_semester_exercises/09_Security and Cryptography Exercises|第 9 讲练习]]

## 第 10 讲 大杂烩

资料：<https://missing-semester-cn.github.io/2020/potpourri/>

### 本讲主线

有一类主题非常重要，但又很难单独撑起一整讲。大杂烩讲的就是这些“高频、零散、但长期很值”的内容。

### 课程覆盖的典型主题

- 键位映射
- 守护进程与 `systemd`
- `cron`
- FUSE
- 备份
- API
- 常见命令行标志习惯
- Markdown
- Docker / Vagrant / 虚拟机 / 云
- 交互式计算环境
- GitHub

### 你应该怎么理解这一讲

它不是要你把所有主题立刻精通，而是帮你建立一份“以后值得继续扩展的工具地图”。

几个最值得马上吸收的点：

#### 键位映射

如果你每天大量敲键盘，那么：

- 把 `Caps Lock` 改成 `Ctrl` 或 `Esc`
- 给高频动作设置合理映射

这种改动长期回报非常高。

#### 守护进程与定时任务

需要区分：

- 一次性命令
- 长期后台服务
- 定时任务

Linux 下常见组合：

- `systemd`：服务管理
- `cron`：定时执行

#### FUSE

课程用它提醒你：文件系统接口本身也可以被扩展，很多“像本地文件一样访问远端资源”的工具，底层其实就是在利用这类机制。

#### 容器、虚拟机与云

这部分最重要的不是工具品牌，而是抽象差异：

- Docker：更轻量，更贴近应用打包与运行环境
- VM：更完整隔离，更像整台机器
- Cloud：把计算、存储、网络按服务交付

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `systemctl status/start/stop/restart/enable SERVICE` | 管理 systemd 服务 | 把后台服务纳入系统生命周期管理 |
| `journalctl -u SERVICE` | 查看某个服务日志 | 配合 systemd 排查守护进程 |
| `sshfs host:/dir mountpoint` | 通过 FUSE 挂载远端目录 | 把远端资源当本地文件系统用 |
| `curl URL` | 调用 Web API | 把互联网服务接入 shell 自动化 |
| `jq '...'` | 查询 JSON | 从 API 响应中抽字段 |
| `--help / --version / -v / --quiet / - / --` | 常见命令行惯例 | 提升面对陌生 CLI 的迁移能力 |
| `xdg-open file / open file` | 用默认应用打开文件 | CLI 与 GUI 的连接点 |
| `Hammerspoon / Karabiner / AutoHotkey` | 桌面自动化与键位改造 | 把“减少高频摩擦”延伸到 GUI |
| `pandoc` | 文档格式转换 | Markdown 作为中间格式输出到 HTML/LaTeX/PDF |
| `Jupyter Notebook` | 交互式计算和展示 | 适合探索式数据分析但要注意复现和版本控制 |
| `Docker / VM / Vagrant / Cloud` | 隔离和复现开发环境 | 在不污染本机的前提下获得可复制环境 |
| `GitHub Issues / Pull Requests` | 代码协作和评审 | 把 Git 工作流接到协作平台 |
| `备份策略 / 云同步 / RAID` | 区分“同步/冗余”和真正备份 | 防止把单点复制误当备份 |

### 对应练习

>[!note]
> 官方第 10 讲没有单独课后练习章节，这里链接到占位练习笔记，后续可继续补自定义实践。
>
> 练习笔记：[[the_missing_semester_exercises/10_Potpourri Exercises|第 10 讲练习]]

## 第 11 讲 提问&回答

资料：<https://missing-semester-cn.github.io/2020/qa/>

### 本讲主线

Q&A 的价值不在于提供唯一答案，而在于把很多学习和工具选择问题拉回现实上下文。

### 课程里反复出现的判断逻辑

#### 先问是否真的需要某个层级的复杂度

比如更底层的操作系统知识、复杂性能工具、特殊编辑器配置，都不是默认人人都该立刻深入的主题。课程非常强调“按问题驱动学习”。

#### 不要把工具选择绝对化

例如：

- Python vs Bash：看任务复杂度、可维护性、与系统交互程度
- `apt` vs `pip`：看系统级安装还是语言生态、看隔离需求
- Docker vs VM：看隔离粒度和目标场景
- Vim vs Emacs vs VS Code：看你是否愿意投入相应学习成本

#### 优先建立稳定工作流，而不是追逐所有新工具

这门课的整体态度一直是：

- 先把最常用的工具真正用熟
- 再按痛点扩展
- 不要陷入“收藏工具却没有工作流升级”的假进步

### 课程给出的高价值现实建议

- 软件包尽量分清系统级和语言级
- 对语言级依赖优先使用隔离环境
- 不要混用多个包管理方案造成环境污染
- 浏览器、插件、编辑器和终端的选择都应服务于你的真实工作流
- 2FA、密码管理器、全盘加密这类安全实践值得尽早变成默认配置

### 本讲命令速查表

| 命令 / 工具 | 用途 | 课堂场景 / 关键语义 |
| --- | --- | --- |
| `source script.sh` | 在当前 shell 执行脚本 | 修改当前 shell 环境时用它，不同于 `./script.sh` |
| `./script.sh` | 在新进程运行脚本 | 不会直接改变父 shell 当前目录/环境 |
| `python -m venv .venv / source .venv/bin/activate` | 创建并激活 Python 虚拟环境 | 区分系统包和项目依赖 |
| `apt install / brew install / pip install` | 系统级包管理 vs 语言级包管理 | 按安装对象和隔离边界选工具 |
| `conda / virtualenv` | 语言环境与依赖隔离 | 减少项目间环境污染 |
| `history / history 1` | 查看命令历史 | 不同 shell 下统计高频命令 |
| `Vim / Emacs / VS Code Vim mode` | 编辑器选择与投入深度 | 不要宗派化，按工作流和学习成本选 |
| `Password manager / 2FA / browser extensions` | 日常安全工具栈 | 把第 9 讲建议落到真实工作流 |
| `pandas -> HTML/LaTeX export` | 按输出目标选择数据工具 | Q&A 中关于表格产出的一类回答 |
| `Linux distro / BSD / browser choice` | 生态选择 | 关注更新节奏、包管理、隐私和场景匹配 |

### 对应练习

>[!note]
> 官方第 11 讲是 Q&A，没有单独课后练习章节，这里链接到占位练习笔记，后续可继续补自定义实践。
>
> 练习笔记：[[the_missing_semester_exercises/11_Q&A Exercises|第 11 讲练习]]

## 全课主线回顾

如果把整门课压成几句话，我会这样记：

### 1. 命令行不是备用方案，而是开放式接口

GUI 负责常见路径，shell 负责组合、自动化和远程操作。

### 2. 小工具组合比大而全更强

管道、脚本、正则、文本处理工具、Git、SSH、tmux 都在体现同一个思想：把复杂任务拆成可组合组件。

### 3. 环境和流程也要工程化

dotfiles、构建系统、测试、CI、包管理、调试工具，都是让个人工作方式从“临时手工”走向“可维护系统”。

### 4. 工具能力最终要落到习惯

真正有用的不是“我知道 tmux / Vim / Git / shellcheck 的存在”，而是：

- 我会在真实工作里默认使用它们
- 我的环境是文本化、版本化的
- 我的排错与优化是基于证据的

## 最值得养成的习惯

- 遇到重复操作，先问能不能脚本化
- 遇到大量文本，先问能不能管道化
- 遇到 Git 困惑，先回到数据模型
- 遇到 bug，先收集证据再修改
- 遇到性能问题，先 profile 再优化
- 遇到环境问题，先文本化和版本化配置
- 遇到安全问题，优先采用成熟工具，不自己造轮子

## 建议实践清单

- 为自己的 shell 写一份最小可用配置文件
- 学会用 `ssh` + `tmux` 在远端机器稳定工作
- 让 `git log --graph --all --decorate` 成为默认观察历史的方式之一
- 至少完整学一遍 `vimtutor`
- 把 3 个真实的数据处理任务改写成管道
- 为一个小项目补上 `Makefile`、测试和简单 CI
- 使用密码管理器并开启关键账户的 2FA

## 最后判断

`The Missing Semester` 真正缺失的不是“某几个命令”，而是“把工具、环境和流程当成计算机教育正式内容”的视角。

如果你学完后只多记住了一些命令，这门课就学浅了。
如果你开始：

- 主动自动化重复劳动
- 把环境配置成自己的工作系统
- 用正确抽象理解 Git、Shell、Vim 和安全工具

那这门课才算真正发挥价值。
