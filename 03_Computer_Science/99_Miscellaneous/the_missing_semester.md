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

```bash
# 1) shell 会把你输入的“命令名 + 参数”交给程序执行。
date                    # 运行 date 程序并把当前时间打印到 stdout；课堂开场用它说明 shell 不是 GUI，而是“启动程序的接口”。
echo hello              # echo 会把参数原样打印出来；适合用来观察 shell 到底把哪些 token 当成了参数。
echo "hello world"      # 双引号把带空格的文本保成一个参数，而且会展开 $VAR、$(cmd) 这类替换。
echo 'hello world'      # 单引号几乎完全禁止 shell 展开；适合保存字面量文本。

# 2) 先建立“我现在在哪个目录”“路径怎么写”的坐标系。
pwd                     # print working directory：显示当前工作目录的绝对路径。
cd /usr/bin             # 进入一个绝对路径；绝对路径从根目录 / 开始。
cd ..                   # 回到父目录；.. 是“上一级目录”的路径别名。
cd ~                    # 回到当前用户的 home 目录。
cd -                    # 回到上一个工作目录；在两个目录间来回切换时特别好用。

# 3) 浏览目录和做最基本的文件操作。
ls                      # 列出当前目录下的文件名；默认不显示隐藏文件，也不展示权限细节。
ls -l                   # long listing：显示权限位、链接数、所有者、大小、修改时间。
ls -lah                 # -a 显示 . 开头隐藏文件；-h 把字节数转成 KB/MB 这种更容易读的单位。
mv old.txt new.txt      # 重命名文件；mv 的本质是“把路径 old.txt 移到 new.txt”。
cp src.txt dst.txt      # 复制文件内容到新路径；如果目标已存在会覆盖，要自己确认。
rm file.txt             # 删除普通文件；shell 里删除通常没有“回收站”这层保护。
rmdir empty_dir         # 只能删除空目录；如果目录里还有文件会失败。
mkdir notes             # 创建新目录；常配合 cd 一起用。

# 4) 不会背参数时，先问 shell 和命令自己的帮助系统。
man ls                  # 打开 ls 的手册页；这是最完整的官方说明。
ls --help               # 很多 GNU 命令都支持 --help，适合快速看参数含义。
which echo              # 在 $PATH 里查“echo 这个名字会解析到哪个可执行文件”；但 shell 内建命令不一定靠它就能看准。
type echo               # 让 shell 直接告诉你 echo 是 builtin、alias、function，还是外部程序；比 which 更贴近“shell 实际会怎么执行”。
echo "$PATH"            # 查看 PATH 搜索路径列表；shell 只有在这些目录里找到命令名，才能让你直接敲 `python`、`git` 这种短名字。

# 5) 这讲真正重要的抽象：stdin/stdout/stderr 可以被重定向，程序可以被管道串起来。
cat file.txt            # 把文件内容读出来写到 stdout；常用作“先看一下文件内容”的最小工具。
cat < file.txt          # 把 file.txt 接到 stdin，再交给 cat；这个例子是在强调“程序读的是输入流，不一定非得自己打开文件路径”。
cat < file.txt > out.txt # 同时重定向输入和输出：从 file.txt 读，写到 out.txt。
echo hello > out.txt    # 用 > 覆盖写 stdout；如果 out.txt 已存在，原内容会被替换掉。
echo hello >> out.txt   # 用 >> 追加写 stdout；不会清空原文件，而是接在末尾继续写。
ls /no/such/path 2> err.txt # 把 stderr 单独重定向到 err.txt；这解释了为什么“正常输出”和“错误输出”是两条不同的流。
ls -l / | tail -n 1     # 管道 `|` 把左边程序的 stdout 接到右边程序的 stdin；这里是先列根目录，再只保留最后一行。
curl --head --silent https://missing-semester-cn.github.io/ | grep -i content-length # curl 抓 HTTP 响应头，grep 过滤出 content-length；这是“网络输出也是普通文本流，可以继续进管道”的例子。
tail -n 1 file.txt      # 只看最后 1 行；如果不跟文件名，也可以从 stdin 接管道输入。
grep -i pattern file.txt # 过滤包含 pattern 的行；-i 忽略大小写。

# 6) 权限、sudo、脚本执行，这一组是第一讲最容易踩坑的地方。
sudo command            # 以 root 权限运行“这个外部命令本身”；注意不是把整行 shell 语法都变成 root。
sudo echo 3 > brightness # 这通常会失败：`>` 重定向是 shell 先执行的，shell 仍然是普通用户，所以写 brightness 这一步没有提权。
echo 3 | sudo tee brightness # 正确做法：让 tee 这个“真正负责写文件的程序”在 sudo 下运行。
chmod +x script.sh      # 给脚本加可执行位；没有 x 权限时，文件即使内容是脚本，也不能被当作程序直接运行。
#!/bin/sh               # shebang 写在脚本第一行，告诉内核“执行这个文件时应该用哪个解释器来跑它”。
./script.sh             # 当前目录通常不在 PATH 里，所以运行当前目录下脚本时要写 `./script.sh`，而不是只写 `script.sh`。
xdg-open report.pdf     # Linux 下用系统默认程序打开文件/URL；适合“我已经在 shell 里找到文件了，但想交给 GUI 程序看”。
open report.pdf         # macOS 对应的是 open。
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

```bash
# 1) 变量、引号、退出码：写 shell 脚本时，最容易出错的就是“shell 到底展开了什么”。
foo=bar                 # 变量赋值不能写成 `foo = bar`；等号两侧有空格会被 shell 当成“运行 foo 这个命令，参数是 = 和 bar”。
echo "$foo"             # 双引号里会展开变量，输出 bar；大多数时候都应该给变量加双引号，避免空格把一个值拆成多个参数。
echo '$foo'             # 单引号不展开变量，输出字面量 $foo；适合展示原始文本。
echo $?                 # 查看上一条命令的退出码；0 表示成功，非 0 表示失败。
true && echo ok         # 只有 true 成功时才执行右边命令；这就是“用退出码做控制流”。
false || echo fail      # 左边失败时才执行右边命令。
echo $$                 # 当前 shell 进程的 PID。
echo $_                 # 上一条命令的最后一个参数；交互式 shell 里偶尔很方便。

# 2) 脚本参数和函数：把“重复命令序列”收成可复用接口。
echo "$0"               # 脚本名或当前 shell 名称。
echo "$1"               # 第一个位置参数。
echo "$@"               # 全部位置参数；配合双引号时，会尽量保持“一个参数就是一个参数”。
echo "$#"               # 参数个数。
mcd() {
  mkdir -p "$1"         # -p 表示父目录不存在就一起创建；如果目录已存在也不报错。
  cd "$1"               # 立刻切进刚创建的目录；这个函数正是课程里“把常见两步操作封成一个命令”的例子。
}
source script.sh        # 在当前 shell 进程里执行脚本；如果脚本里定义了函数/变量，执行完后当前 shell 还能继续用。
./script.sh             # 另起一个子进程运行脚本；子进程里的变量/函数不会自动回写到当前 shell。

# 3) shell 的“把命令输出塞回命令参数里”的两种机制。
echo "Today is $(date)" # 命令替换：先运行 date，把它的 stdout 替换进这条 echo 命令的参数位置。
diff <(ls dir1) <(ls dir2) # 进程替换：把 `ls dir1` 和 `ls dir2` 的输出伪装成两个临时文件名，交给 diff 去比较。

# 4) globbing：这些模式是 shell 自己先展开的，不是程序收到后再解释。
ls *.py                 # 匹配当前目录下所有 .py 文件；如果没匹配到，具体行为取决于 shell 配置。
ls project?.md          # `?` 匹配恰好 1 个字符。
ls foo.{py,sh}          # 花括号展开成 foo.py 和 foo.sh；这是“枚举多个固定后缀”的短写法。
ls **/*.py              # 递归匹配子目录下的 .py 文件；是否默认启用取决于 shell，比如 zsh 和 bash 的配置不完全一样。

# 5) 文件查找、内容查找、历史查找：这讲的重点是“别什么都手写 for 循环扫目录”。
find . -name '*.tmp'    # 从当前目录递归找所有名字匹配 *.tmp 的路径。
find . -name '*.tmp' -delete # 直接删除查到的临时文件；这类命令执行前最好先去掉 -delete 干跑一遍确认结果。
find . -type f -name '*.py' -exec grep -H 'TODO' {} \; # 对每个匹配到的 Python 文件执行 grep；{} 会被当前文件路径替换，`\;` 表示 -exec 子命令到这里结束。
fd PATTERN              # find 的现代替代品之一；默认输出更干净，常见用法更短。
locate PATTERN          # 按系统维护的文件名索引查路径，速度很快；缺点是索引不是实时更新，刚创建的文件可能搜不到。
grep -R 'TODO' .        # 递归搜索文本内容；-R 表示递归进入子目录。
rg 'TODO' .             # ripgrep 通常比 grep -R 更快，而且默认尊重 .gitignore，噪音更少。
history                 # 打印 shell 历史命令。
Ctrl-r                  # 反向增量搜索历史命令；不是一条 shell 命令，而是 readline/zsh 的交互式快捷键。
fzf                     # 模糊筛选器；常和 history、find/rg 输出接起来做“边搜边选”。

# 6) 目录浏览和帮助系统。
tree                    # 树状展示目录结构；适合快速看项目长什么样。
broot                   # 交互式目录树浏览工具，比纯 tree 更适合“边看边跳转”。
nnn                     # 终端文件管理器，偏轻量。
ranger                  # 终端文件管理器，带多栏预览。
tldr tar                # 用“常见例子”快速回忆一个命令怎么用；比 man 短，但细节没有 man 全。
shellcheck script.sh    # 静态检查 shell 脚本；它尤其擅长抓“变量没加引号”“for file in $(ls) 这种坏模式”“条件测试写法不稳”。
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
# 1) 进入、保存、退出：Vim 最大的门槛是“我现在在哪个模式”。
vim file.txt          # 从 shell 打开文件；进入 Vim 后默认通常在 Normal mode，不是 Insert mode。
i                     # 在光标前进入 Insert mode，开始真正输入文本。
a                     # 在光标后进入 Insert mode；如果你想在当前字符后追加内容，比 i 更顺手。
o                     # 在当前行下方新开一行并进入 Insert mode。
O                     # 在当前行上方新开一行并进入 Insert mode。
Esc                   # 从 Insert/Visual 等模式回到 Normal mode；这是“继续做移动和编辑命令”的前提。
:w                    # 写盘保存当前 buffer。
:q                    # 退出当前窗口/缓冲区；如果有未保存改动会报错，防止误丢内容。
:wq                   # 保存并退出。
:q!                   # 放弃未保存改动并强制退出；只有你确认这些改动不要了才用。

# 2) 移动：Vim 高效的核心不是“插入文字快”，而是“在代码结构里移动快”。
h / j / k / l         # 左 / 下 / 上 / 右移动；Normal mode 下不要依赖方向键。
w                     # 跳到下一个 word 的开头。
b                     # 跳回上一个 word 的开头。
e                     # 跳到当前/下一个 word 的末尾。
0                     # 到行首第 0 列。
^                     # 到本行第一个非空白字符；写代码时通常比 0 更实用。
$                     # 到行尾。
gg                    # 到文件第一行。
G                     # 到文件最后一行。
42G                   # 跳到第 42 行；把 42 换成任意行号即可。
f,                    # 在当前行向右跳到下一个逗号；`,` 可以换成任意字符。
t)                    # 在当前行向右跳到右括号前一个字符；适合“停在目标字符前面继续编辑”。
;                     # 重复上一次 f/t/F/T 字符查找。
,                     # 反向重复上一次 f/t/F/T 字符查找。
%                     # 在匹配的括号/花括号/方括号之间跳转。
( / )                 # 按句子向前/向后移动；处理自然语言文本时更常用。
{ / }                 # 按段落向前/向后移动；处理按空行分隔的结构时很方便。

# 3) 编辑动作的“语法”：operator + motion / text object。
x                     # 删除光标下单个字符。
ra                    # 把光标下字符替换成 a；`r{char}` 是“一次性替换一个字符”。
dw                    # 删除从光标到下一个词首前的范围；这是 `d` + `w`。
cw                    # 修改一个 word；本质是先删掉 motion 覆盖范围，再进入 Insert mode。
yw                    # 复制一个 word 到寄存器。
dd                    # 删除整行。
cc                    # 修改整行。
yy                    # 复制整行。
p                     # 把寄存器内容粘贴到光标后/下一行。
P                     # 把寄存器内容粘贴到光标前/上一行。
u                     # 撤销上一次修改。
Ctrl-r                # 重做被撤销的修改。

# 4) 选择、text objects、搜索替换：这部分最能体现 Vim “编辑语言”的优势。
v                     # 字符级 Visual mode；先选一个区域，再对区域执行 d/y/c 等操作。
V                     # 行级 Visual mode。
Ctrl-v                # 块选择 Visual mode；适合多行列编辑。
ci"                   # change inside quotes：只改双引号内部文本，不碰外层引号。
da(                   # delete around parentheses：连同括号本身一起删掉整个括号对象。
yi{                   # yank inside braces：只复制花括号内部内容。
/pattern              # 向下搜索 pattern。
?pattern              # 向上搜索 pattern。
n                     # 跳到下一个匹配项；方向由上一次 `/` 或 `?` 决定。
N                     # 跳到上一个匹配项。
:%s/foo/bar/g         # 对整个文件做替换；`%` 是全文件范围，`g` 表示每行所有匹配都替换。

# 5) 多文件/多窗口/宏/帮助：课程强调的是“把编辑器当长期工作环境”，不只是会改一个文件。
:sp                   # 水平分屏打开一个窗口。
:vsp                  # 垂直分屏打开一个窗口。
Ctrl-w h/j/k/l        # 在不同窗口之间移动焦点。
:e other.txt          # 在当前窗口打开另一个文件。
:ls                   # 列出当前 Vim 会话里的所有 buffers。
:b2                   # 切换到编号为 2 的 buffer；编号来自 `:ls` 输出。
qa ... q              # 把一串操作录制到寄存器 a；中间的 `...` 是你真实执行的编辑动作。
@a                    # 重放寄存器 a 里的宏；适合“重复结构相同，但每次位置不同”的批量修改。
:help text-objects    # 查 Vim 自己的帮助文档；Vim 的文档非常系统。
vimtutor              # 课程推荐的入门练习；比纯看文档更适合第一次建立肌肉记忆。
~/.vimrc              # Vim 配置文件；把常用设置、快捷键、插件配置固化下来。
CtrlP                 # 课程里提到的模糊文件查找插件；如果要真正使用，需要先装插件并在 Vim 里触发它，而不是在 shell 里直接敲 `:CtrlP`。
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

```bash
# 1) 数据整理这讲的基本流程是：先拿到原始文本流，再逐步过滤、提取、排序、聚合。
ssh myserver journalctl          # 在远程机器上直接运行 journalctl，把日志文本通过 ssh 返回本地；这比“先手动拷贝日志文件再处理”更符合管道思路。
journalctl                       # 读取 systemd 日志；课堂把它当作“真实世界里杂乱但可流式处理的原始文本数据源”。

# 2) grep/sed/awk 是最常见的三层处理：按行过滤、按正则改写、按字段计算。
grep 'sshd' log.txt              # 只保留包含 sshd 的行；grep 默认按“整行是否匹配模式”过滤。
grep -v 'Disconnected' log.txt   # -v 表示反向过滤：把匹配到的行丢掉，保留没匹配到的行。
sed -E 's/.*Disconnected from //' # -E 启用扩展正则；`s/old/new/` 把每行里匹配到的 old 替换成 new，常用于“把无关前缀剥掉，只留下关心字段”。
awk '{print $2}'                 # awk 默认按空白切列；$2 是第二列，适合快速从“类表格文本”里抽字段。
awk '$1 == 1 {print $2}'         # 只有当第一列等于 1 时才打印第二列；awk 的优势就在“模式 + 动作”能直接写条件逻辑。

# 3) sort/uniq/head/tail 是做频数统计和 top-k 的常规后半段。
sort                             # 按字典序排序整行；如果后面要接 uniq，通常必须先 sort，让相同值挨在一起。
sort -n                          # 按数值排序；不加 -n 时，"10" 会排在 "2" 前面，因为那是字符串字典序。
sort -k2,2                       # 按第 2 列排序；`-k2,2` 表示排序键从第 2 列开始，到第 2 列结束。
uniq -c                          # 统计“连续重复行”的出现次数；如果输入没先 sort，同一个值分散在不同位置就不会被合并统计。
head -n 10                       # 只保留前 10 行；常用在 sort/uniq 之后截 top 结果。
tail -n 10                       # 只保留后 10 行；也常用于“先跑完整管道，再只看末尾结果”。

# 4) cut/paste/tr/wc/bc/xargs/tee 用来补上“字段切割、格式重排、计数、数值计算、参数化执行、保留中间结果”。
cut -d, -f2 data.csv             # 用逗号做分隔符，抽第 2 列；适合格式很规整的 CSV/TSV。
paste -sd+                       # 把多行合并成一行，并用 + 作为分隔符；常见套路是先生成 `1+2+3`，再交给 bc 求和。
bc -l                            # 命令行计算器；-l 加载数学库并开启浮点计算。
tr 'A-Z' 'a-z'                   # 按字符做一一映射替换；这里是把大写字母转成小写。
wc -l                            # 统计行数；如果前面接 grep，就可以直接变成“匹配到多少行”。
wc -w                            # 统计词数。
wc -c                            # 统计字节数。
tee output.txt                   # 把管道流一边继续往下游送，一边写一份到文件里；适合调试长管道时保留中间产物。
find . -name '*.csv' | xargs wc -l # xargs 把 stdin 里的路径列表变成命令参数，再批量交给 wc -l；这一步的核心是“把文本流重新转回参数列表”。

# 5) gnuplot 代表的是“shell 管道最后不一定只输出文本，也可以直接接可视化工具”。
gnuplot                          # 课堂用它说明：只要前面的数据整理管道把数据整理成合适格式，就可以直接进入绘图工具。
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
# 1) 任务控制和信号：先分清“前台/后台/挂起/终止”这几种状态。
Ctrl-c                  # 给前台进程发 SIGINT：请求中断当前运行中的程序；很多 CLI 程序会捕获它并做清理后退出。
Ctrl-\                  # 给前台进程发 SIGQUIT：通常会让程序退出并产生 core dump；比 Ctrl-c 更偏“调试/异常退出”语义。
Ctrl-z                  # 给前台进程发 SIGTSTP：不是结束程序，而是把它挂起，之后可以用 bg/fg 接着管理。
sleep 1000 &            # `&` 把命令直接放到后台作业里运行，shell 立刻把提示符还给你。
jobs                    # 查看当前 shell 管理的 job 列表；`%1` 这类 job 编号就从这里来。
bg %1                   # 让第 1 个挂起的 job 在后台继续跑。
fg %1                   # 把第 1 个 job 拉回前台；如果你需要继续交互输入，就必须 fg 回来。
kill -TERM %1           # 给 job 1 发 SIGTERM，请求它优雅退出；优先用 TERM，而不是一上来就 KILL。
kill -STOP %1           # 给 job 1 发 SIGSTOP，强制暂停；和 Ctrl-z 类似，但这是显式发信号。
nohup long_task &       # 忽略 SIGHUP 后在后台运行长任务；否则终端断开时，后台任务很可能跟着收到 hangup 信号。
disown %1               # 把 job 从当前 shell 的作业表里移除，降低“shell 退出时顺手把它带走”的风险。

# 2) tmux/screen：重点不是“多开几个 pane”，而是“远程断线后工作现场还在”。
tmux new -s work        # 新建一个名为 work 的 tmux session；建议给长期任务起有意义的 session 名。
Ctrl-b d                # tmux 默认前缀是 Ctrl-b；`Ctrl-b d` 表示 detach，断开当前终端和 session 的连接，但不杀 session 里的程序。
tmux ls                 # 列出当前有哪些 tmux sessions 还活着。
tmux attach -t work     # 重新接回 work 这个 session；断线重连后最常用。
screen                  # screen 是另一套终端多路复用器；课程主要是让你知道“这类工具解决的是会话持久化问题”。

# 3) SSH 和文件同步：远程工作不只是能登录，还要能稳定认证、传文件、做端口转发。
ssh user@host           # 登录远程机器；如果没写用户名，默认用本地当前用户名。
ssh -p 2222 user@host   # 指定远端 SSH 端口；当服务器不是监听 22 端口时就要加 -p。
ssh -L 9999:localhost:8888 user@host # 本地端口转发：访问本机 9999 时，流量通过 SSH 隧道转到远端机器看到的 localhost:8888；跑远端 Jupyter 时很常见。
scp file.txt user@host:/tmp/ # 复制单个文件到远端；scp 用法简单，但大量/重复同步时通常不如 rsync 高效。
rsync -avP src/ user@host:dst/ # 增量同步目录；-a 保留属性，-v 显示过程，-P 显示进度并支持部分传输续传。注意 `src/` 末尾这个 `/` 会影响“同步目录本身”还是“同步目录内容”。

# 4) SSH 配置、密钥登录、dotfiles：这部分是在把“临时能用”升级成“长期可维护”。
ssh-keygen -t ed25519   # 生成一对 SSH 公私钥；ed25519 是现在常用且推荐的密钥类型之一。
ssh-copy-id user@host   # 把你的公钥追加到远端 `~/.ssh/authorized_keys`，以后就可以用私钥登录，不必每次输密码。
~/.ssh/config           # SSH 客户端配置文件；可以在里面给主机起别名、指定 User/HostName/Port/IdentityFile/LocalForward。
alias ll='ls -lah'      # 把高频长命令缩成短别名；适合交互式 shell 里降低重复输入成本。
ln -s ~/.dotfiles/.vimrc ~/.vimrc # 用软链接把 dotfiles 仓库里的配置文件挂到 home 目录，避免“复制多份配置，最后不知道哪份才是最新的”。
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

```bash
# 1) 先看文档、再建仓库。课程对 Git 的核心建议是：把命令放回“提交图 + 对象模型”里理解。
git help status                # 查某个子命令的官方帮助。
man git-status                 # 同样是查文档，但走 manpage 入口。
git init                       # 在当前目录新建一个 Git 仓库；本质是创建 .git 目录和初始引用结构。
git clone URL                  # 把远端仓库对象、引用和工作区文件一起克隆下来。

# 2) 工作区 / 暂存区 / 提交历史：这三层一定要分开看。
git status                     # 看哪些文件在工作区改了、哪些已经进暂存区、当前分支相对上游分支是什么状态。
git add file.py                # 把 file.py 当前这份内容放进暂存区；注意 add 的是“此刻这份快照”，之后工作区继续改，不会自动更新暂存区。
git add -p                     # 交互式按 hunk 选择要不要暂存；适合把“同一文件里的多件事”拆成干净提交。
git diff                       # 看工作区相对暂存区的差异；如果你已经 git add 过，再改工作区，这里看的就是“未暂存的新改动”。
git diff --staged              # 看暂存区相对 HEAD 的差异；也就是“下一次 commit 会提交什么”。
git commit                     # 用暂存区内容生成一个新 commit 对象，并让当前分支引用前移到这个新提交。
git commit -m "message"        # 直接在命令行写提交信息；适合很短的消息。

# 3) 看提交 DAG 和引用：如果你看不懂 HEAD/branch 在哪，merge/rebase 就一定会混乱。
git log --oneline --graph --decorate --all # 把提交历史画成 DAG，并显示所有分支名、tag、HEAD 指向哪里；这是课程里非常推荐的看图方式。
git branch                     # 列出本地分支；分支本质上只是“指向某个 commit 的可移动名字”。
git branch -d feature          # 删除已经合并的 feature 分支；-d 会做安全检查，不会轻易删未合并分支。
git switch main                # 切到 main 分支，并把工作区更新成 main 指向的那个提交快照。
git switch -c feature          # 从当前提交新建 feature 分支并直接切过去。
git checkout branch            # 老命令也能切分支/切文件/切提交；课程会提它，但现在日常更建议把“切分支”优先写成 switch，语义更清楚。
git checkout -b feature        # checkout 的“新建并切分支”写法；和 `git switch -c feature` 类似。

# 4) 合并、重放、摘提交：这三种操作都在改提交图，但语义不同。
git merge feature              # 把 feature 分支的历史并入当前分支；如果两边都有新提交，通常会产生一个 merge commit 来保留“历史在这里分叉又汇合过”。
git rebase main                # 把当前分支相对分叉点之后的提交，重放到 main 当前最新提交后面；历史会变线性，但 commit hash 会变，因为那已经是“新提交对象”了。
git cherry-pick <commit>       # 只把某个单独提交对应的改动复制到当前分支；适合“我不要整条分支，只要其中一个修复”。

# 5) 临时搁置、撤销、救援：这组命令最危险，也最值得写清楚。
git stash                      # 把当前未提交改动临时收起来，让工作区回到干净状态；常用于“先切去别的分支处理急事”。
git stash pop                  # 把最近一次 stash 应用回来，并尝试从 stash 栈里移除；如果冲突了，要按 Git 提示处理。
git restore file.py            # 丢掉工作区里 file.py 相对暂存区/HEAD 的改动；如果这个改动还没 commit，执行前要确认真的不要了。
git restore --staged file.py   # 把 file.py 从暂存区移出去，但保留工作区内容；也就是“撤销 git add，但不撤销文件本身修改”。
git reset --soft <rev>         # 只移动 HEAD/当前分支到 <rev>，暂存区和工作区尽量保留；适合“提交做错了，想重新组织 commit，但文件改动还要留着”。
git reset --mixed <rev>        # 移动 HEAD，并把暂存区重置成 <rev>；工作区改动保留。这也是很多情况下 reset 的默认模式。
git reset --hard <rev>         # HEAD、暂存区、工作区都强行回到 <rev>；这会直接丢掉未提交工作区改动，除非你非常确定，否则不要拿它当常规撤销按钮。
git reflog                     # 查看 HEAD 和分支引用曾经指到过哪里；如果误 reset/rebase 后“提交好像不见了”，第一反应应该是先看 reflog。

# 6) 远端同步：把“本地提交图”与“远端引用”区分开。
git remote -v                  # 看远端别名和 URL，比如 origin 指到哪里。
git fetch                      # 从远端下载新对象并更新 remote-tracking branches，但不直接改你当前工作区。
git pull                       # fetch + 合并/重放；因为它会直接动当前分支历史和工作区，所以最好在理解 pull 策略后再用。
git push                       # 把本地分支的新提交推到远端对应引用；如果远端历史比你本地领先，push 会被拒绝，避免你覆盖别人提交。
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
# 1) 先把“观察程序状态”这件事做扎实：print 和 logging 不是幼稚办法，而是最便宜的第一层证据。
print(variable)                 # 在你怀疑某个变量值不对时，先把它直接打印出来；适合快速缩小问题范围。
logging.debug("x=%s", x)        # 日志比 print 更适合长期保留：可以分级别、带时间戳、带上下文，而且上线后也更容易统一收集。

# 2) 真正需要“停下来单步看状态”时，上调试器。
python -m pdb script.py         # 用 Python 标准库 pdb 启动脚本；可以下断点、单步执行、看调用栈和变量。
ipdb                            # IPython 风格的 pdb，交互体验更好；本质仍然是在做“暂停程序、检查现场”。
gdb ./a.out                     # C/C++ 程序常用调试器；用 break/run/next/step/backtrace 这一套看崩溃和调用链。
lldb ./a.out                    # LLVM 生态里的调试器，macOS 上尤其常见；角色和 gdb 类似。

# 3) 如果怀疑问题出在“程序和操作系统/动态库怎么交互”，就往系统调用和库调用层看。
strace -e openat ./cmd          # 只追踪 openat 相关系统调用；很适合查“程序到底在尝试打开哪个文件，为什么说找不到”。
ltrace ./cmd                    # 追踪动态库函数调用；比 strace 更贴近 libc/共享库这一层。
journalctl -u myservice         # 查看某个 systemd 服务的日志；排查“服务为什么起不来/为什么反复重启”时很常用。
dmesg                           # 查看内核 ring buffer；硬件、驱动、OOM、内核层报错往往先从这里看线索。
shellcheck script.sh            # 对 shell 脚本做静态检查；很多 bug 在运行前就能被它指出来。

# 4) profiling 的原则是：不要先靠直觉优化，先测热点在哪里。
python -m cProfile -s tottime script.py # 运行 Python profiler，并按函数自身耗时 tottime 排序；适合先定位“到底哪个函数最耗 CPU”。
time ./cmd                      # 粗略测一条命令的 real/user/sys 时间；real 是墙钟时间，user/sys 分别对应用户态和内核态 CPU 时间。
hyperfine './cmd1' './cmd2'     # 对两条命令做多轮 benchmark 并给出统计结果；比手写 `time` 跑一次更稳，因为它会处理多次采样和预热。
perf stat ./cmd                 # 看硬件/内核层性能计数的摘要，比如 cycles、instructions、cache misses。
perf record ./cmd               # 采样记录性能数据。
perf report                     # 交互式查看 perf record 采到的热点。
py-spy top --pid PID            # 不改代码、低侵入地看一个正在运行的 Python 进程当前热点函数；适合线上进程“不能轻易停下来但又要知道卡在哪”。

# 5) 最后别忘了“慢”不一定是 CPU，常见瓶颈还有内存、磁盘、网络。
htop                            # 交互式看进程 CPU/内存占用、线程、进程树；比 top 更适合日常排查。
top                             # 更基础的进程资源监视工具，几乎哪台机器都有。
iotop                           # 看哪个进程在疯狂读写磁盘；需要相应权限和内核支持。
iostat                          # 看设备级 I/O 吞吐、等待时间、利用率；适合判断是不是磁盘本身已经打满。
free -h                         # 查看内存余量、缓存/缓冲区占用；-h 让单位更易读。
df -h                           # 查看各文件系统剩余空间；很多“程序突然写不动/构建失败”其实就是磁盘满了。
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

```makefile
# 1) Makefile 的基本语法是“目标: 依赖” + “生成目标的命令”。
paper.pdf: paper.tex plot-data.png
	pdflatex paper.tex
# 上面这条规则的意思是：paper.pdf 依赖 paper.tex 和 plot-data.png；
# 只有当依赖比目标更新，或者目标还不存在时，make 才会执行下面那行命令去重建 paper.pdf。

plot-data.png: plot.py data.csv
	python plot.py -i data.csv -o plot-data.png
# 这条规则把“图像怎么从脚本和数据生成出来”也纳入依赖图；
# 所以当你只改 paper.tex 时，不应该无意义地重跑画图脚本。

# 2) clean 这类“动作型目标”要声明成 phony，否则同名文件会干扰 make 的判断。
.PHONY: clean
clean:
	rm -f paper.pdf plot-data.png
# 如果不写 .PHONY，而目录里刚好有个叫 clean 的文件，make 可能会误以为 clean 这个目标已经是最新的，于是什么都不做。

# 3) Make 自动变量：用它们可以把规则写得更泛化。
%.png: %.dot
	dot -Tpng $< -o $@
# $@ 代表当前目标名，这里是某个 .png 文件；
# $< 代表第一个依赖，这里是对应的 .dot 文件；
# $^ 代表所有依赖列表，适合“编译时把所有依赖都传给命令”的场景。
```

```bash
# 4) make 命令本身怎么用：这些是从 shell 里调用 make 的命令，所以应该放在 bash 代码块里，而不是 makefile 代码块里。
make                          # 运行默认目标；通常是 Makefile 里的第一个目标。
make paper.pdf                # 只构建指定目标；make 会沿依赖图递归检查哪些中间产物需要更新。
make -n paper.pdf             # dry-run：只打印将要执行哪些命令，不真的执行；调试规则和依赖判断时非常有用。
make clean                    # 执行上面声明的 clean 伪目标，删除构建产物，强制下次重新生成。
git ls-files                  # 列出 Git 正在跟踪的文件；写 clean 规则时可以借它判断“哪些产物不该误删源文件”。

# 5) Python/多语言依赖管理：重点是“环境隔离 + 版本可复现”，不是“能装上就行”。
python3 -m venv .venv         # 在项目目录下创建一个 Python 虚拟环境，避免把项目依赖直接污染到系统 Python。
source .venv/bin/activate     # 激活虚拟环境；之后 python/pip 会优先指向 .venv 里的解释器和包目录。
pip install -r requirements.txt # 按依赖清单安装包；这适合“别人 clone 你的项目后如何复现环境”。
pip freeze > requirements.txt # 把当前环境里已安装包和精确版本导出成快照；注意这不等于你已经设计好了合理依赖边界，只是把现状记录下来。
poetry install                # Python Poetry 生态里按 pyproject/lockfile 安装依赖。
npm install                   # Node 生态里安装 package.json 声明的依赖，并参考 package-lock.json 锁定解析结果。
cargo build                   # Rust 生态里的构建入口；Cargo 同时承担依赖解析和构建系统角色。
# package-lock.json / Cargo.lock / poetry.lock 不是“要手敲的命令”，而是锁定依赖解析结果的文件；
# 它们的作用是让“今天能装出来的依赖版本集合”在明天、别人机器、CI 里也尽量可复现。
# `^1.2.3`、`~1.2.3`、`>=1.2,<2.0` 这类版本约束写法属于 semver 语义的一部分：
# 你要记住的是“版本范围不是随便写的，major/minor/patch 的兼容性承诺会直接影响升级风险”。
# vendoring 指把依赖源码直接纳入仓库或项目树：好处是更可控，代价是升级和同步成本更高。

# 6) 测试、hook、CI：把“我记得手动检查”升级成“机器自动替我挡住低级错误”。
pytest                        # 运行 Python 测试；测试的价值不只是“这次过了”，更是给后续重构提供回归保护。
cargo test                    # 运行 Rust 测试。
tox                           # 在多个 Python 环境/依赖矩阵里跑测试；适合检查“不是只在我本机这个解释器版本下能过”。
git bisect start              # 开始二分定位引入 bug 的提交。
git bisect bad HEAD           # 标记当前提交是坏的。
git bisect good v1.0.0        # 标记某个已知旧提交是好的；之后 Git 会自动切到中间提交让你测试，从而快速缩小范围。
.git/hooks/pre-commit         # Git pre-commit hook 脚本路径；可以在提交前自动跑格式化/测试/lint，不通过就拒绝提交。
shellcheck script.sh          # CI 或 hook 里跑 shell 静态检查，尤其适合课程练习里的脚本。
proselint README.md           # 检查英文写作风格问题；课程练习里会让你把这类写作检查也接进 CI。
write-good README.md          # 另一个偏英文文风检查的工具。
# GitHub Actions 不是一条本地 shell 命令，而是一套“push/PR 事件触发 CI 工作流”的机制；
# 对应配置通常写在 `.github/workflows/*.yml`，核心目标是让构建、测试、lint 在远端自动执行。
```

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

```bash
# 1) 哈希：同一输入必然映射到同一摘要，所以它很适合做“完整性校验”和“内容寻址”。
sha256sum file.bin          # Linux 常见写法：计算 file.bin 的 SHA-256 摘要。
shasum -a 256 file.bin      # macOS 上常见写法：同样计算 SHA-256。
python3 -c "import hashlib; print(hashlib.sha256(b'hello').hexdigest())" # 用 Python 直接演示“固定输入 -> 固定摘要”的确定性。

# 2) 随机数和熵：密码/密钥安全不看“看起来复杂不复杂”，而是看攻击者要猜的空间有多大。
openssl rand -base64 32     # 生成 32 字节随机数并用 base64 打印；适合做高熵 token/secret。
pwgen -s 20                 # 生成 20 字符随机密码；-s 表示偏安全随机，而不是生成容易读的弱口令。

# 3) 对称加密：同一把密钥负责加密和解密，适合“我自己保存/传输一个文件”。
gpg -c secrets.txt          # 用口令对 secrets.txt 做对称加密，生成 secrets.txt.gpg。
gpg secrets.txt.gpg         # 解密 GPG 对称加密文件；会提示你输入口令。
openssl enc -aes-256-cbc -salt -in plain.txt -out cipher.bin # 课程用这类命令演示对称加密流程；重点是理解“算法 + 密钥 + 随机盐/IV 参数”这些组成件，而不是鼓励手工拼复杂 openssl 命令当长期方案。

# 4) 非对称密钥、签名、验签：公钥可以公开，私钥必须自己保管好。
ssh-keygen -t ed25519       # 生成 SSH 公私钥对；公钥放服务器，私钥留本地。
gpg --full-generate-key     # 交互式生成一套 GPG 公私钥；后续可以用于签名、验签、加密。
gpg --sign message.txt      # 用你的 GPG 私钥对文件签名；默认会生成签名产物。
gpg --verify message.txt.gpg # 用对应公钥验签；核心问题是“内容有没有被改，签名是不是来自那把私钥”。
age -r RECIPIENT -o file.age file.txt # 用 age 按接收方公钥加密文件；相比手写 openssl 参数，工具接口更不容易误用。

# 5) 密码管理和 2FA：这是课程最后最该落成日常习惯的部分。
pass                        # 基于 GPG + Git 的命令行密码管理器；适合把“每个站点一个强密码”变成可操作工作流。
# 1Password/Bitwarden 这类 GUI 密码管理器不是“课程要求你背的 shell 命令”，但它们和 pass 解决的是同一个现实问题：
# 每个站点都用独立强密码，并且由工具负责保存、同步、自动填充，而不是靠人脑记忆。
# TOTP 和 U2F/FIDO2 security key 也不是一条命令，而是两类二因素认证方案；
# 课程想强调的是：优先用 Authenticator App 或硬件安全密钥，不要只依赖短信验证码。
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

```bash
# 1) 把“重复 GUI 操作”和“键位不顺手”也工程化。
xdotool key Ctrl+L              # Linux 下模拟键盘/鼠标输入；适合把某些 GUI 重复动作半自动化。
# AutoHotkey 是 Windows 上常见的热键/脚本工具，Karabiner-Elements 是 macOS 上常见的键位重映射工具；
# 它们不是这段里要死记参数的 CLI，而是“如果高频按键不顺手，就值得重映射”的工具类别代表。

# 2) JSON、临时 HTTP 服务、文件系统挂载：这些是“大杂烩”里非常实用的工程小工具。
jq '.field' data.json           # 结构化读取 JSON 字段；比用 grep/sed 按文本硬切 JSON 稳得多，因为 jq 真正理解 JSON 语法树。
python3 -m http.server 8000     # 在当前目录临时起一个静态文件服务器；适合快速把本地目录通过 HTTP 暴露出来做测试。
python -m SimpleHTTPServer 8000 # Python 2 的老写法；现在主要是“读旧资料/旧环境时知道这是什么”。
sshfs user@host:/path mountpoint # 通过 FUSE 把远端目录挂载成本地目录；优点是“像访问本地文件一样访问远端文件”，缺点是网络抖动和权限语义要自己心里有数。
rclone sync src remote:dst      # 同步本地目录和云存储/远端后端；sync 语义通常是“让目标变得和源一致”，用前要确认会不会删目标端多余文件。

# 3) 媒体、图像、文档格式转换。
ffmpeg -i in.mp4 out.mp3        # 从视频里抽音频/做转码；ffmpeg 的强项是“几乎所有音视频格式都能接”，但参数很多，建议按任务查文档。
convert in.png out.jpg          # ImageMagick 旧入口：图片格式转换、缩放、裁剪等。
magick in.png out.jpg           # ImageMagick 新版更推荐的入口。
pandoc in.md -o out.pdf         # 文档格式转换；比如 Markdown -> PDF/HTML/docx。格式转换能不能高质量成功，常取决于模板、引用、字体和公式链路是否配好。

# 4) 远程交互体验。
tmux                            # 断线不丢 session。
mosh user@host                  # 比传统 ssh 更适合高延迟、弱网络、IP 漂移场景；代价是它和 ssh 的连接/转发模型不完全一样，不能无脑互换。
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
# 这讲是 Q&A，不是“新命令密集讲解课”，所以这段速查表更像一份“遇到问题时先走哪条工具路线”的备忘单。

# 1) 不会用某个命令时，先查本地文档，再去搜索引擎。
man tar                  # 最完整的本地手册；适合认真查参数、行为边界、退出码、文件格式说明。
tar --help               # 很多命令的快速帮助入口；适合先粗看“这个参数大概叫什么”。
tldr tar                 # 例子驱动的简明说明；适合“我知道要做什么，只是忘了常见写法”。
apropos archive          # 如果你连命令名都不确定，可以按关键词搜 manpage 摘要。

# 2) 写脚本、搜代码、学编辑器：先把这几类高回报基础工具稳定用起来。
shellcheck script.sh     # shell 脚本先跑静态检查，少靠肉眼猜 quoting/条件判断/未定义变量这些坑。
rg 'pattern' .           # 代码库全文搜索优先用 ripgrep；它快，而且默认会避开不少无意义路径。
vimtutor                 # 真想学 Vim，不要只背命令表，先完整做一遍交互式教程建立基本动作模型。

# 3) Git 和远程会话出问题时，优先回到“可恢复”和“可解释”的工具。
git help rebase          # 忘了子命令语义时查官方帮助，而不是硬背一串网上复制来的参数。
git reflog               # 历史引用救援入口；如果 reset/rebase 之后“提交好像丢了”，先看 reflog 再慌。
tmux                     # 远程长任务默认放进可恢复 session，别把几十分钟计算直接裸跑在一个会掉线的 SSH 窗口里。

# 4) 环境配置要沉淀成文本资产，而不是靠“这台机器我手动调过，所以大概能用”。
alias ll='ls -lah'       # 把高频命令缩成别名，降低日常重复输入成本。
~/.dotfiles              # 用 dotfiles 仓库集中管理 shell/editor/git 配置；真正重要的是“配置可版本化、可迁移、可回滚”。
~/.ssh/config            # 给常用远程机器写别名、默认用户、端口、IdentityFile、LocalForward，让 SSH 使用变成稳定接口而不是临时长命令。
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
