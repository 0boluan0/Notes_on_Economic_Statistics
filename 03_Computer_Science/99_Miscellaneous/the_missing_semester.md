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

```text
date                                  # 打印当前时间；开场演示 shell 可以直接调用程序
echo hello                            # 打印参数；最基本的“命令 + 参数”结构
echo "hello world" / echo 'hello'     # 用引号把含空格内容当成一个参数；单/双引号展开规则不同
pwd                                   # 显示当前工作目录
cd /path / cd .. / cd ~ / cd -        # 切换目录；绝对路径、上级目录、家目录、返回上一个目录
ls / ls -l / ls -lah                  # 列目录；-l 看权限和元数据，-a 看隐藏文件，-h 让大小易读
mv old new                            # 移动或重命名文件
cp src dst                            # 复制文件
rm file / rmdir dir                   # 删除文件/空目录；rmdir 只能删空目录
mkdir dir                             # 新建目录
man ls / ls --help                    # 查官方帮助；不会背参数时先查这里
which echo / type echo                # 看 shell 实际会执行哪个程序/内建命令
echo $PATH                            # 查看可执行文件搜索路径；解释为什么命令能直接运行
cat file / cat < file > out           # 读文件，或把标准输入重定向到标准输出
tail -n 1                             # 取最后 1 行；常接在管道右侧截取结果
curl --head --silent URL              # 抓 HTTP 头且隐藏进度条；适合继续接文本处理命令
grep -i pattern                       # 按模式过滤文本；-i 忽略大小写
> / >> / < / | / 2>                   # stdout 覆盖、追加、stdin 重定向、管道、stderr 重定向
sudo cmd                              # 用 root 权限执行命令；注意它不会自动提升 shell 先处理的重定向
echo 3 | sudo tee brightness          # 让 tee 以 root 身份写文件；修复 sudo echo 3 > file 这类坑
chmod +x file                         # 给脚本增加可执行权限
#!/bin/sh                             # shebang；告诉系统这个脚本应该交给哪个解释器执行
xdg-open file / open file             # 用系统默认程序打开文件；Linux/macOS 分别常用 xdg-open/open
```

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

```text
foo=bar / echo "$foo" / echo '$foo'                 # 变量赋值与展开；双引号展开变量，单引号基本原样保留
mcd() { mkdir -p "$1" && cd "$1"; }                # shell 函数；把“建目录后进入”封成一个可复用命令
source script.sh                                    # 在当前 shell 里执行脚本；让函数/变量定义留在当前会话
$0 / $1 / $@ / $# / $? / $$ / $_                    # 特殊变量：脚本名、参数、参数列表、参数个数、退出码、进程号、上一条命令最后参数
true / false / cmd1 && cmd2 / cmd1 || cmd2          # 基于退出码做条件执行；0 表示成功，非 0 表示失败
cmd1 ; cmd2                                         # 无条件顺序执行两条命令
$(cmd)                                              # 命令替换；把命令输出嵌回另一个命令的参数位置
<(cmd)                                              # 进程替换；把命令输出伪装成一个可读“文件”
*.py / ? / foo.{py,sh} / **/*.py                    # glob 展开；单星、单字符、花括号枚举、递归匹配
diff <(ls foo) <(ls bar)                            # 比较两个命令输出；课堂里用它展示进程替换
find . -name '*.tmp' -delete                        # 按条件找文件并直接执行删除动作
find . -type f -name '*.py' -exec grep -H PATTERN {} \;  # 对 find 找到的每个文件执行命令；{} 是当前文件占位符
fd pattern                                          # 更现代的文件查找工具；默认行为通常比 find 更符合日常使用
locate pattern                                      # 查文件名索引；很快，但依赖后台数据库，不是实时遍历
grep -R pattern . / rg pattern                      # 递归搜索文本；rg 通常更快，默认排除很多无用目录
history / Ctrl-r / fzf                              # 命令历史搜索；Ctrl-r 反向增量搜索，fzf 做模糊选择
tree / broot / nnn / ranger                         # 快速看目录树或做交互式文件浏览
tldr cmd                                            # 看更偏“例子驱动”的简明帮助页；比 man 更适合快速回忆用法
shellcheck script.sh                                # 静态检查 shell 脚本；尤其能抓 quoting、空格、未定义变量等常见坑
```

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

```text
vim file                         # 打开文件进入 Vim
i / a / o / O / Esc              # 进入插入、光标后追加、下/上方开新行、回到普通模式
:w / :q / :wq / :q!              # 保存、退出、保存并退出、放弃修改强退
h j k l                          # 左 / 下 / 上 / 右移动
w / b / e                        # 按词移动；跳到下一个词首、上一个词首、当前/下一个词尾
0 / ^ / $                        # 到行首、首个非空字符、行尾
gg / G / {line}G                 # 到文件开头、文件结尾、指定行
f{char} / t{char} / ; / ,        # 行内字符跳转；; 重复上次查找，, 反向重复
% / ( / ) / { / }                # 括号匹配跳转、句子移动、段落移动
x / r{char}                      # 删除当前字符、把当前字符替换成指定字符
d{motion} / c{motion} / y{motion} # 对 motion 覆盖的范围做删除 / 修改 / 复制
dd / cc / yy / p / P             # 行级删除、修改、复制、粘贴；p 粘到后面，P 粘到前面
u / Ctrl-r                       # 撤销 / 重做
v / V / Ctrl-v                   # 字符级、行级、块级可视选择
ci" / da( / yi{                  # text object：改引号内部、删整个括号对象、复制花括号内部
/pattern / ?pattern / n / N      # 向下/向上搜索；n/N 跳到下一个/上一个匹配
:%s/foo/bar/g                    # 全文件替换；g 表示一行里所有匹配都替换
:sp / :vsp / Ctrl-w hjkl         # 水平/垂直分屏，以及窗口间移动
:e file / :ls / :bN              # 打开文件、列出 buffer、切到第 N 个 buffer
q{reg} ... q / @{reg}            # 录制宏到寄存器并重放；适合处理“重复但位置不同”的编辑任务
:help subject / vimtutor         # 查帮助文档 / 完成交互式入门教程
~/.vimrc / :CtrlP                # 配置文件与插件入口；CtrlP 是课堂示例里的模糊找文件插件
```

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

```text
ssh myserver journalctl                              # 远程取日志；数据整理常常从“命令输出”而不是手工下载文件开始
journalctl                                           # 查看 systemd 日志；课堂把它当作原始半结构化文本数据源
grep 'pattern' / grep -v 'pattern'                  # 按模式保留/排除行
sed -E 's/old/new/'                                 # 用扩展正则做替换；课堂里主要拿它做字段抽取和文本清洗
awk '{print $2}' / awk '$1 == 1 {print $2}'          # 按列打印、按条件过滤；适合处理“空白分隔的表格型文本”
sort / sort -n / sort -k2,2                         # 排序；字典序、数值序、按指定列排序
uniq -c                                             # 统计连续重复行次数；通常必须先 sort 再 uniq 才有意义
head -n 10 / tail -n 10                             # 取前/后若干行；常用来截 top-k 或快速抽查输出
cut -d, -f2                                         # 按分隔符取指定字段；适合简单 CSV/TSV
paste -sd+                                          # 把多行合并成一行并插入分隔符；常配合 bc 做求和
bc -l                                               # 命令行计算器；-l 打开数学库并支持浮点计算
tr 'A-Z' 'a-z'                                      # 字符级替换；常用于大小写归一化
xargs cmd                                           # 把 stdin 里的文本转成命令参数；把“文本流”重新变回“参数列表”
tee output.txt                                      # 一边继续向下游输出，一边把中间结果落盘
wc -l / wc -w / wc -c                               # 统计行数、词数、字节数
gnuplot                                             # 快速画图；课堂用它说明 shell 输出可以直接接可视化工具
```

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

```text
Ctrl-c / Ctrl-\ / Ctrl-z                # 发送 SIGINT / SIGQUIT / SIGTSTP；终止、退出并 core dump、挂起前台进程
sleep 1000 &                             # 把命令放到后台运行；shell 立即返回提示符
jobs                                     # 查看当前 shell 管理的作业
bg %1 / fg %1                            # 让作业在后台继续跑 / 拉回前台
kill -TERM %1 / kill -STOP %1            # 给进程/作业发信号；TERM 请求退出，STOP 强制暂停
nohup cmd &                              # 忽略 SIGHUP，终端断开后尽量继续运行
screen / tmux                            # 终端多路复用器；让远程会话可恢复
tmux new -s work                         # 新建一个命名 tmux 会话
Ctrl-b d                                 # 从当前 tmux 会话 detach；不杀掉里面的程序
tmux ls / tmux attach -t work            # 查看已有会话 / 重新接回指定会话
ssh user@host                            # 登录远程机器
ssh -p 2222 user@host                    # 指定端口连接远程机器
ssh -L 9999:localhost:8888 user@host     # 本地端口转发；把远端服务映射到本地端口
scp file user@host:/path                 # 在本地和远端之间复制文件
rsync -avP src/ user@host:dst/           # 增量同步目录；保留属性、显示进度、支持断点续传
~/.ssh/config                            # 给常用主机写别名、用户名、端口和转发规则
ssh-keygen / ssh-copy-id user@host       # 生成 SSH 密钥 / 把公钥装到远端实现免密登录
alias ll='ls -lah'                       # 用别名固化高频短命令
ln -s source target                      # 用软链接管理 dotfiles，避免多机配置靠手工复制
```

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

```text
git help <cmd> / man git-<cmd>                  # 查 Git 子命令文档
git init / git clone URL                        # 新建仓库 / 克隆已有仓库
git status                                      # 查看工作区、暂存区、当前分支状态
git add file / git add -p                       # 把修改加入暂存区；-p 可以按块交互式选择
git commit / git commit -m "msg"                # 基于暂存区生成新提交对象
git log --oneline --graph --decorate --all      # 看提交 DAG 和分支/HEAD 指针位置
git diff / git diff --staged                    # 比较工作区改动 / 暂存区相对 HEAD 的改动
git switch branch / git checkout branch         # 切换到已有分支
git switch -c feature / git checkout -b feature # 基于当前提交新建分支并切过去
git branch / git branch -d feature              # 列出分支 / 删除已合并分支
git merge feature                               # 把 feature 合并进当前分支；可能产生 merge commit
git rebase main                                 # 把当前分支提交重放到 main 顶上；会改写提交历史
git stash / git stash pop                       # 暂存未提交改动 / 恢复暂存改动
git reset --soft/--mixed/--hard <rev>           # 移动 HEAD/分支指针；三种模式分别保留到不同层，--hard 会丢工作区改动
git restore file / git restore --staged file    # 撤销工作区文件改动 / 把文件从暂存区移出
git reflog                                      # 查看 HEAD 曾经指向过哪里；误 reset/rebase 后常靠它找回提交
git remote -v / git fetch / git pull / git push # 查看远端、拉取对象、拉取并合并、推送
git cherry-pick <commit>                        # 把某个提交对应的补丁应用到当前分支
```

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

```text
print(...) / logging.debug(...)                  # 最基础的观察手段；日志比裸 print 更适合保留上下文和分级输出
python -m pdb script.py                          # 用 Python 调试器单步执行脚本
ipdb                                             # 更好用的交互式 Python 调试器
gdb / lldb                                       # C/C++ 等程序的调试器；断点、单步、看调用栈
strace -e openat cmd                             # 跟踪系统调用；定位程序到底在请求哪些文件/资源
ltrace cmd                                       # 跟踪动态库函数调用
journalctl -u service                            # 查看系统服务日志
dmesg                                            # 查看内核日志
shellcheck script.sh                             # 静态检查 shell 脚本常见问题
python -m cProfile -s tottime script.py          # Python CPU profiler；按耗时排序找热点
time cmd                                         # 粗略查看 real / user / sys 时间
hyperfine 'cmd1' 'cmd2'                          # 更稳定地 benchmark 多个命令，并自动做多轮统计
perf stat / perf record / perf report            # Linux 性能计数和采样分析
py-spy top --pid PID                             # 低侵入地查看 Python 进程热点函数
htop / top                                       # 查看进程 CPU/内存占用
iotop / iostat                                   # 查看 I/O 瓶颈
free -h / df -h                                  # 查看内存余量 / 磁盘空间
```

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

```text
make / make target                               # 运行默认目标 / 指定目标；make 会根据依赖判断哪些步骤需要重跑
make -n target                                   # 只打印将要执行的命令，不真正执行；调试 Makefile 很有用
.PHONY: clean                                    # 声明伪目标；避免同名文件干扰 make 的增量判断
python -m venv .venv / source .venv/bin/activate # 创建并激活隔离环境
pip install -r requirements.txt                  # 按依赖清单安装包
pip freeze > requirements.txt                    # 把当前环境版本导出成快照；注意这不等于精心维护过的依赖声明
poetry install / npm install / cargo build       # 不同语言生态里的依赖安装和构建入口
pytest / cargo test                              # 运行测试；把“是否还能工作”变成可重复检查的机器动作
tox                                              # 在多个环境矩阵下跑测试
git bisect start / git bisect good / git bisect bad # 用二分法定位是哪次提交引入 bug
GitHub Actions / CI                              # 持续集成；每次提交后自动跑构建、测试、检查
```
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

```text
sha256sum file / shasum -a 256 file         # 计算文件哈希；用于完整性校验
python -c "import hashlib; ..."            # 用 Python 哈希库演示“哈希函数是确定性映射”这一点
openssl rand -base64 32                    # 生成高熵随机字符串；适合做随机密钥/令牌
gpg -c file / gpg file.gpg                 # 用口令对文件做对称加密 / 解密
openssl enc -aes-256-cbc -salt -in plain -out cipher # 对称加密演示；课堂借它说明“同一把密钥负责加解密”
ssh-keygen -t ed25519                      # 生成非对称密钥对
gpg --full-generate-key                    # 生成 GPG 公私钥对
gpg --sign file / gpg --verify file.sig    # 数字签名 / 验签；验证“这确实是谁签的，且内容没被改”
age -r recipient -o file.age file          # 更现代的文件加密工具；比手写 openssl 参数更不容易踩坑
pwgen -s 20                                # 生成高熵随机密码
pass / 1password                           # 密码管理器；核心目标是“每个站点都用独立强密码”
2FA / U2F security key / TOTP              # 二因素认证；优先硬件 key 或 TOTP，不要只依赖短信验证码
```

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

```text
xdotool / AutoHotkey / Karabiner-Elements   # 桌面自动化、热键绑定、键位重映射；把重复 GUI 操作脚本化
jq '.field' data.json                       # 解析 JSON；比用 grep/sed 硬切 JSON 稳得多
python -m http.server 8000                  # 在当前目录临时起一个静态文件 HTTP 服务
python -m SimpleHTTPServer 8000             # Python 2 时代写法；旧环境里可能还会遇到
sshfs user@host:/path mountpoint            # 把远端目录挂成本地文件系统来用
rclone sync src remote:dst                  # 同步云盘/对象存储和本地目录
ffmpeg -i in.mp4 out.mp3                    # 媒体转码、抽音轨、改封装
convert in.png out.jpg / magick ...         # ImageMagick 图像转换；新版本常用 magick 入口
pandoc in.md -o out.pdf                     # 文档格式转换
tmux / mosh                                 # 远程工作更稳的会话工具；mosh 对高延迟/网络切换更友好
```

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

```text
man cmd / cmd --help / tldr cmd    # 先查文档再上网；完整手册、快速参数提示、示例导向速览各有侧重
apropos keyword                    # 不知道命令名时，按关键词搜索 manpage
shellcheck script.sh               # 写 shell 时先让工具扫一遍常见坑
rg pattern                         # 在代码库里快速全文搜索；通常比 grep 更适合日常开发
vimtutor                           # 系统化入门 Vim 的最短路径之一
git help <cmd> / git reflog        # 查 Git 子命令文档 / 在误操作后找回提交位置
tmux                               # 长任务和远程会话尽量放进可恢复的多路复用会话
alias / dotfiles / ~/.ssh/config   # 把常用环境配置沉淀下来，并让迁移到新机器时可复现
```

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
