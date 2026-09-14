---
aliases:
  - 从一份日志理解Shell参数数据流与命令行环境
  - Shell arguments data pipelines and execution environments
---

# Shell 与数据管道：把一份日志变成能解释的统计结果

<!-- bilingual-en:start -->
*Shell and data pipelines: turn a log into an explainable result*
<!-- bilingual-en:end -->

这条路径串起 Missing Semester 第 1、2、4、5 讲。我们先统计一个小日志中的状态码，再解释为什么路径有空格、输入格式改变或上游失败，会让看起来相似的命令得到不同结果。最后把同一执行模型扩展到脚本、后台任务和远程机器。

<!-- bilingual-en:start -->
This path connects Missing Semester Lectures 1, 2, 4 and 5. Count status codes in a small log, then explain how spaces in a path, a changed format or an upstream failure alter the result. Extend the same execution model to scripts, background jobs and remote machines.
<!-- bilingual-en:end -->

[[the_missing_semester|逐讲笔记与练习]] · [[Shell、数据整理与命令行环境.canvas|主题关系图]] · [[知识原子.base|定义入口]]

> [!note] 练习环境
> 以下代码按 **Bash** 解释，不把它的默认展开规则套到 Zsh。可以在终端用 `/bin/bash --noprofile --norc` 进入练习 shell，结束时用 `exit` 返回；这不改变默认 shell 或配置文件。本页使用 macOS 自带 Bash 3.2 与常用文本工具核验。
>
> 从 Academic 库的根目录开始，样例是 [[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/fixtures/access log.tsv|access log.tsv]]。它是为这条路径编写的五条模拟记录，不是你的真实访问日志。下列统计只读样例、输出到终端。
> <!-- bilingual-en:start -->
> Run these examples in **Bash**, not by assuming its defaults apply to Zsh. `/bin/bash --noprofile --norc` starts a practice shell; `exit` returns without changing configuration. The examples were checked with macOS Bash 3.2 and its standard text tools. Start at the Academic vault root. The linked fixture contains five synthetic records, not personal logs, and the counting examples only read it.
> <!-- bilingual-en:end -->

## 1. 先弄清楚谁在解释这行字

<!-- bilingual-en:start -->
*1. Identify what interprets the command line*
<!-- bilingual-en:end -->

[[Shell]]是解释命令的程序；终端提供与它交互的窗口。你写下一行命令，Shell 先把语法、展开、参数和重定向处理好，然后调用内建命令、函数或外部程序。不能把“每行都启动一个外部程序”当成统一规则：`cd` 要改变当前 Shell 的目录，`echo` 在 Bash 中通常也是内建命令。

<!-- bilingual-en:start -->
A [[Shell|shell]] interprets commands; a terminal supplies the interaction surface. Syntax, expansions, arguments and redirections are processed before invoking a builtin, function or external program. Not every line starts an external executable: `cd` changes the current shell's directory, and Bash commonly handles `echo` as a builtin.
<!-- bilingual-en:end -->

用 `type -a echo` 看[[Shell命令查找]]的实际候选。`PATH` 参与外部程序查找，但不替代函数、内建命令、别名等机制。`$SHELL` 通常记录登录 Shell 的路径，也不能证明当前这层正在运行哪个解释器。

<!-- bilingual-en:start -->
Use `type -a echo` to inspect [[Shell命令查找|command lookup]]. `PATH` participates in finding external programs; it does not replace functions, builtins or alias expansion. `$SHELL` usually names the login shell and is not proof of the interpreter currently running.
<!-- bilingual-en:end -->

[[文件路径]]也有自己的解释起点：本页相对路径以库根为基准，不是以当前打开的笔记为基准。先用 `pwd` 确认位置，再给样例路径命名：

<!-- bilingual-en:start -->
A [[文件路径|relative path]] needs a starting directory. Here it is the vault root, not the folder of the open note. Check `pwd`, then name the fixture:
<!-- bilingual-en:end -->

```bash
log_file='03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/fixtures/access log.tsv'
```

## 2. 一个路径必须仍然是一个参数

<!-- bilingual-en:start -->
*2. Keep one path as one argument*
<!-- bilingual-en:end -->

先不用任何文件操作，只观察[[Shell简单命令|命令收到的参数]]：

<!-- bilingual-en:start -->
Observe the [[Shell简单命令|arguments a command receives]] without touching files:
<!-- bilingual-en:end -->

```bash
show_args() {
    printf 'argc=%s\n' "$#"
    for arg in "$@"; do
        printf '<%s>\n' "$arg"
    done
}
demo_path='access log.tsv'
show_args "$demo_path"
show_args $demo_path
```

在本页 Bash 默认设置下，第一遍收到一个参数 `access log.tsv`，第二遍收到 `access` 与 `log.tsv` 两个参数。原因是未引用的展开结果经过了[[Shell字段分割]]；若其中还含 `*`，可能继续触发[[Shell文件名展开]]。[[Shell引用|双引号]]保留展开后的参数边界，单引号则用于这里的字面值赋值。

<!-- bilingual-en:start -->
With the stated Bash defaults, the first call receives one argument, `access log.tsv`; the second receives `access` and `log.tsv`. The unquoted expansion underwent [[Shell字段分割|word splitting]], and a `*` could also trigger [[Shell文件名展开|filename expansion]]. [[Shell引用|Double quotes]] preserve the expanded argument boundary; single quotes protect the literal assignment here.
<!-- bilingual-en:end -->

`"$@"` 在函数中把原有各个[[Shell位置参数|位置参数]]分别转发，不是把全部内容揉成一个字符串。引号只解决 Shell 的这一层：程序仍可能把以 `-` 开头的参数解释成选项。处理任意文件名还要遵守目标程序的操作数约定，见[[文件名安全传参]]。

<!-- bilingual-en:start -->
`"$@"` forwards each [[Shell位置参数|positional argument]] separately. Quoting protects one parsing layer; a program may still interpret a leading hyphen as an option. Arbitrary filenames also require the operand conventions described in [[文件名安全传参|safe filename passing]].
<!-- bilingual-en:end -->

## 3. 先规定日志格式，再提取一列

<!-- bilingual-en:start -->
*3. State the log format before extracting a field*
<!-- bilingual-en:end -->

这份样例没有表头，每条记录由三个 **TAB 分隔**的字段组成：请求方法、路径、三位状态码；字段内部没有 TAB 或换行。第一条是 `GET`、`/`、`200`。先读五行，核对这个约定：

<!-- bilingual-en:start -->
The fixture has no header. Each record contains three **tab-separated** fields: method, path and a three-digit status. Fields contain no embedded tabs or newlines. The first record is `GET`, `/`, `200`. Inspect the five records against this contract:
<!-- bilingual-en:end -->

```bash
sed -n '1,5p' "$log_file"
awk -F '\t' '{print NR, NF, $3}' "$log_file"
```

第二条命令使用[[awk文本处理]]：`NR` 是当前累计记录数，`NF` 是字段数，`$3` 是第三字段。这里的 `$3` 被 Shell 单引号保护，交给 **awk** 解释，不是 Shell 的第三个参数。你应该看到五条记录的 `NF` 都为 `3`，状态依次是 `200, 404, 403, 200, 200`。

<!-- bilingual-en:start -->
In [[awk文本处理|awk]], `NR` counts records, `NF` counts fields and `$3` selects the third field. Shell single quotes pass `$3` to **awk** rather than expanding a shell positional parameter. All five records should have `NF = 3`, with statuses `200, 404, 403, 200, 200`.
<!-- bilingual-en:end -->

这也解释了为什么 `awk '{print $9}'` 不是通用的访问日志解析器：只有声明的格式确实把状态码放在第九个空白字段，它才回答那个问题。改变日志格式，或者字段里出现未被该分隔规则识别的引号与空白，列号就可能失去意义。[[字段切分不等于格式解析]]；JSON、包含引号规则的 CSV 等应使用理解该格式的解析器。

<!-- bilingual-en:start -->
`awk '{print $9}'` is not a universal access-log parser. It works for that task only when the specified format puts the status in the ninth whitespace-delimited field. A changed format or quoted whitespace can invalidate the field number. [[字段切分不等于格式解析|Field splitting is not format parsing]]; use a format-aware parser for JSON or quoted CSV.
<!-- bilingual-en:end -->

## 4. 沿数据流观察，直到得到频数

<!-- bilingual-en:start -->
*4. Follow the data through to its frequencies*
<!-- bilingual-en:end -->

[[标准流]]把输入、常规输出和诊断输出分开。[[Shell管道]]把前一段的标准输出接到后一段的标准输入，传递的是字节，不会自动把一行变成一个命令参数。对这份固定样例，每一步都可以独立读懂：

<!-- bilingual-en:start -->
[[标准流|Standard streams]] separate input, ordinary output and diagnostics. A [[Shell管道|pipeline]] connects stdout to stdin; it carries bytes rather than automatically turning lines into command arguments. Each stage below is inspectable:
<!-- bilingual-en:end -->

```bash
awk -F '\t' '{print $3}' "$log_file"
awk -F '\t' '{print $3}' "$log_file" | LC_ALL=C sort
awk -F '\t' '{print $3}' "$log_file" | LC_ALL=C sort | uniq -c
awk -F '\t' '{print $3}' "$log_file" | LC_ALL=C sort | uniq -c | LC_ALL=C sort -k1,1nr -k2,2n
```

第一步抽取状态码；第二步把相同状态码放在一起；第三步统计每个相邻段；第四步按第一列计数降序、第二列状态码升序排列。最终是 `3 200`、`1 403`、`1 404`。具体填充空格由工具决定，不是结果的一部分。

<!-- bilingual-en:start -->
Extract statuses, group equal values by sorting, count adjacent runs, then sort by descending count and ascending status. The result is `3 200`, `1 403`, `1 404`; tool-specific padding is immaterial.
<!-- bilingual-en:end -->

顺序不能随意交换：[[相邻行去重|uniq 只看相邻行]]，所以直接对原状态序列做 `uniq -c` 会得到两个分离的 `200` 组。[[文本排序键选择]]中的 `-n` 是数值比较；去掉它，字符顺序会把 `10` 排在 `2` 前面。明确第二个排序键，还说明了同频率时怎样展示，不能把工具的回退规则猜成“保留输入顺序”。

<!-- bilingual-en:start -->
Order matters: [[相邻行去重|uniq groups only adjacent duplicates]], so the unsorted status stream contains separate `200` groups. [[文本排序键选择|Choosing a numeric sort key]] differs from character comparison, which puts `10` before `2`. A second key specifies tie behavior rather than guessing that original order is preserved.
<!-- bilingual-en:end -->

本例只处理 ASCII 状态码，并为排序明确 `LC_ALL=C`。若处理姓名、语言字符或带地区小数格式的值，要明确各阶段的[[文本处理的locale边界|locale 约定]]，需要相同等值判定的阶段应使用兼容规则，而不只固定最后一个工具。用排序把完整值聚拢后再计数的方法见[[排序分组频数统计]]。

<!-- bilingual-en:start -->
This example uses ASCII statuses and specifies `LC_ALL=C` for sorting. Names, language characters or regional numeric formats need an explicit [[文本处理的locale边界|locale contract]] for each relevant stage, not only the final command. Stages requiring the same equality relation need compatible rules. See [[排序分组频数统计|sorting-based frequency counting]] for that method.
<!-- bilingual-en:end -->

## 5. 没有输出、没有匹配和执行失败不是同一件事

<!-- bilingual-en:start -->
*5. Distinguish empty output, no match and execution failure*
<!-- bilingual-en:end -->

[[grep行筛选]]按模式选行；`grep -F` 把模式当固定字符串，`grep -E` 使用扩展正则。[[正则表达式]]不是 Shell 通配符，匹配语法还依赖[[正则方言边界|工具与版本]]。从日志删除前缀时，[[sed替换]]的正则可能因[[贪婪匹配跨越分隔符]]吃掉两个重复标记之间本来想保留的内容，必须用反例检验它。

<!-- bilingual-en:start -->
[[grep行筛选|grep]] selects lines; `-F` treats the pattern literally and `-E` selects extended regular expressions. [[正则表达式|Regexes]] are not shell globs, and their syntax depends on the [[正则方言边界|tool and version]]. A [[sed替换|sed substitution]] may remove intended content because [[贪婪匹配跨越分隔符|greedy matching crosses a delimiter]]. Test such boundaries with counterexamples.
<!-- bilingual-en:end -->

对本例，可以搜索不存在的路由，观察[[grep退出状态]]：无匹配通常返回 `1`，它是一个可预期的查询结果，不等于文件读取失败。[[命令退出状态]]要结合具体命令解释；Shell 的条件把零当作成功，只是选择控制路径的约定。

<!-- bilingual-en:start -->
Searching for an absent route illustrates [[grep退出状态|grep's statuses]]: no match normally returns one, an expected query outcome rather than a read failure. Interpret [[命令退出状态|exit statuses]] through each command's contract; the shell uses zero to select its success path.
<!-- bilingual-en:end -->

另一个陷阱更隐蔽：前段失败，后段仍可能成功。用下面这个没有文件副作用的实验观察[[管道退出状态]]：

<!-- bilingual-en:start -->
An upstream failure can coexist with downstream success. Observe [[管道退出状态|pipeline status]] without file side effects:
<!-- bilingual-en:end -->

```bash
false | cat
printf 'default=%s\n' "$?"
(
    set -o pipefail
    false | cat
    printf 'pipefail=%s\n' "$?"
)
```

默认得到 `0`，因为末段 `cat` 正常读到输入结束；启用 `pipefail` 的子 Shell 得到 `1`。选项让失败可见，但不会决定如何解释失败，也不会撤销已输出的内容。对生产数据做统计时，既检查退出状态，也检查格式与统计总数；不要把某个选项当作完整的[[测试判据]]。

<!-- bilingual-en:start -->
The default status is zero because `cat` successfully reads end-of-input; the `pipefail` subshell reports one. The option exposes a failure but neither interprets it nor retracts emitted output. Validate exit status, format and totals; no shell option substitutes for a complete [[测试判据|oracle]].
<!-- bilingual-en:end -->

## 6. 验证数据，再让结果离开程序

<!-- bilingual-en:start -->
*6. Validate records before releasing a result*
<!-- bilingual-en:end -->

可以把格式检查与计数放进一次[[awk条件聚合]]。下面只接受三个 TAB 字段、非空的方法和路径、`100` 到 `599` 形式的状态码；这只是本例的数据约定，不保证每个三位码都是某个协议版本已注册的状态。整个输入中一旦发现违规记录，结束时不输出频数并返回非零。

<!-- bilingual-en:start -->
Combine validation and counting in one [[awk条件聚合|awk aggregation]]. This accepts three tab-separated fields, nonempty method and path, and a status shaped like `100` through `599`. That is a fixture contract, not a registry of protocol-defined statuses. A malformed record prevents the final counts from being emitted and produces nonzero status.
<!-- bilingual-en:end -->

```bash
(
    export LC_ALL=C
    set -o pipefail
    awk -F '\t' '
        NF != 3 || $1 == "" || $2 == "" || $3 !~ /^[1-5][0-9][0-9]$/ {
            bad = 1
            next
        }
        { count[$3]++ }
        END {
            if (bad) exit 2
            for (code in count) print count[code], code
        }
    ' "$log_file" | sort -k1,1nr -k2,2n
)
```

关联数组的遍历顺序不作输出承诺，所以最后仍显式排序。此例允许空文件得到空结果；如果任务要求至少一条有效记录，就应把它加进契约与 `END` 检查，而不是让使用者自行猜测。文件读取或排序失败也仍须由调用者按退出状态处理。

<!-- bilingual-en:start -->
Associative-array traversal has no promised output order, so sorting remains explicit. An empty file intentionally produces no counts; a minimum-record requirement would need an additional contract and `END` check. Callers must still handle file-reading or sorting failures through exit status.
<!-- bilingual-en:end -->

如果接下来要把结果写回文件，先理解[[Shell重定向]]与[[重定向顺序]]。`sed ... input.txt > input.txt` 会在处理器读文件前截断目标；它不是“读完后再覆盖”。安全工作流应先[[临时目录安全创建|安全创建临时工作区]]，在其中保存候选结果、验证后再按明确目标替换；这里不执行对源文件的覆盖。

<!-- bilingual-en:start -->
Before saving results, understand [[Shell重定向|redirection]] and its [[重定向顺序|ordering]]. `sed ... input.txt > input.txt` truncates the file before processing; it is not read-then-overwrite. A safe workflow [[临时目录安全创建|safely creates a temporary workspace]], then stages and validates output before replacing an explicit target. No source overwrite is performed here.
<!-- bilingual-en:end -->

## 7. 把交互命令变成脚本时，重新确认环境边界

<!-- bilingual-en:start -->
*7. Recheck environment boundaries when making a script*
<!-- bilingual-en:end -->

[[Shell函数]]可复用一组命令，在调用它的环境中执行；例如从子 Shell 调用时，改动就发生在子 Shell。[[Shell脚本]]把命令存入文件，对这个文件再区分[[脚本执行与载入|单独执行与 source 载入]]：它们决定变量和工作目录的变化是否留在当前 Shell。子进程不回写父 Shell 的目录与变量，但仍可能改动共享文件，不能把“独立环境”误读成安全沙箱。

<!-- bilingual-en:start -->
A [[Shell函数|function]] reuses commands in its calling environment; a subshell call therefore changes subshell state. A [[Shell脚本|script]] stores commands in a file. For that file, [[脚本执行与载入|execution versus sourcing]] determines whether variable and directory changes remain in the current shell. A child cannot rewrite its parent's shell state but can still modify shared files; an independent environment is not a sandbox.
<!-- bilingual-en:end -->

[[Shebang]]为直接执行的脚本指定解释器；显式运行 `bash script.sh` 时，解释器已经由命令选定。直接执行还需要合适的[[Unix权限位|执行与目录搜索权限]]。这些不同条件正是第 1 讲 `./semester` 与 `sh semester` 对照题要你观察的东西。

<!-- bilingual-en:start -->
A [[Shebang|shebang]] chooses the interpreter for direct script execution; `bash script.sh` already selects it explicitly. Direct execution also requires suitable [[Unix权限位|execution and directory-search permissions]]. These are the distinct conditions in Lecture 1's `./semester` versus `sh semester` exercise.
<!-- bilingual-en:end -->

配置通过[[环境变量]]与[[环境变量导出]]传给后续命令；`export` 不会永久写进配置文件。交互中的[[Shell别名]]也不应成为脚本依赖。需要持久配置时，明确[[Bash启动文件选择|哪个启动文件会被读取]]，按[[dotfiles配置管理]]保留可复现的配置与机器差异；秘密不进入 Git 历史。

<!-- bilingual-en:start -->
[[环境变量|Environment variables]] and [[环境变量导出|export]] pass configuration to later commands without permanently editing configuration files. Interactive [[Shell别名|aliases]] should not be hidden script dependencies. Identify [[Bash启动文件选择|which startup file is read]], then manage portable settings and host differences through [[dotfiles配置管理|dotfiles]]. Keep secrets out of Git history.
<!-- bilingual-en:end -->

脚本重复执行也需要可说明的结果。[[脚本幂等性]]要求对约定的目标状态重复应用等效于执行一次，不只是“看起来没坏”。例如每次都给配置末尾追加同一行，会积累副本；每次写入同一条规范化设置才可能满足该设置层面的幂等要求。

<!-- bilingual-en:start -->
[[脚本幂等性|Idempotence]] means repeated application has the same specified state effect as one application, not merely an absence of obvious damage. Appending the same configuration line accumulates copies; setting one normalized value can be idempotent at that configuration layer.
<!-- bilingual-en:end -->

## 8. 后台和远程都不改变责任归属

<!-- bilingual-en:start -->
*8. Background and remote execution still need explicit ownership*
<!-- bilingual-en:end -->

[[Unix进程]]是程序执行实例，可能正在运行、等待或暂停；[[Shell作业]]是 Shell 管理的一组相关进程，例如一个管道。[[前台与后台作业]]关心的是终端交互与等待方式，不是“有没有在计算”。[[作业状态切换|暂停后用 bg 恢复]]仍需查看作业状态，不能把 `Ctrl-Z` 当成结束。

<!-- bilingual-en:start -->
A [[Unix进程|process]] is a program execution instance that may be running, waiting or stopped; a [[Shell作业|job]] groups related processes such as a pipeline. [[前台与后台作业|Foreground and background]] concern terminal interaction and waiting, not whether computation occurs. [[作业状态切换|Resuming a stopped job]] requires checking its state; `Ctrl-Z` is not termination.
<!-- bilingual-en:end -->

[[Unix信号]]传达请求或事件。[[正常终止与强制终止]]的区别在于进程能否处理请求与清理，不能用“发送命令成功”证明任务已经退出。等待属于自己 Shell 的任务时，用[[等待子进程]]取得完成状态；按名称匹配的 `pkill` 则可能误中同名的其他任务，所以本路径不执行终止命令。

<!-- bilingual-en:start -->
[[Unix信号|Signals]] convey requests or events. [[正常终止与强制终止|Graceful and forced termination]] differ in handling and cleanup; successfully sending a signal does not establish completion. [[等待子进程|Wait for managed child processes]] to obtain their status. Name-based termination can match unrelated tasks, so no termination commands are executed in this path.
<!-- bilingual-en:end -->

[[后台作业断连边界|后台运行不保证断连后存活]]。tmux 这类[[终端复用]]工具把可连接的交互客户端与持有终端的服务端分开；[[tmux会话保持边界]]说明重新接入要求服务端与相应会话仍存在且可访问。原任务是否仍在运行要另外检查，也不能据此保证主机重启或任务失败后自动恢复。

<!-- bilingual-en:start -->
[[后台作业断连边界|Backgrounding does not guarantee survival after disconnection]]. A [[终端复用|terminal multiplexer]] such as tmux separates its attachable client from the server retaining terminals. [[tmux会话保持边界|tmux reattachment]] requires an existing, accessible server and corresponding session. Check the original task's state separately; this is not automatic recovery from host restart or task failure.
<!-- bilingual-en:end -->

[[SSH远程执行]]还增加了本地与远端两个解释位置。`ssh host 'producer | filter' | summary` 中，前面的管道由远端 Shell 解释，最后的 `summary` 在本地运行。主机名 `host` 在这里是示意，不执行连接。[[SSH公钥认证]]由服务器核对公钥对该账户的授权，并验证用户持有对应私钥；[[SSH主机认证]]则帮助你判断连接的是哪台服务器。两种身份核验不能互相替代。

<!-- bilingual-en:start -->
[[SSH远程执行|Remote execution]] adds local and remote interpretation. In `ssh host 'producer | filter' | summary`, the quoted pipeline is remote and `summary` runs locally; `host` is illustrative, and no connection is made. [[SSH公钥认证|User public-key authentication]] checks the key's authorization for the account and proves possession of the corresponding private key. [[SSH主机认证|Server authentication]] establishes a different identity.
<!-- bilingual-en:end -->

第 5 讲的[[SSH本地端口转发]]练习则要区分三个位置：本机监听地址、SSH 服务器、服务器去连接的目标地址。转发目标中的 `localhost` 从服务器一端解释，不是浏览器所在机器；先画清这三个位置，再填写端口配置。

<!-- bilingual-en:start -->
Lecture 5's [[SSH本地端口转发|local-forwarding exercise]] distinguishes the local listener, SSH server and destination reached from that server. A destination of `localhost` is interpreted remotely, not at the browser's machine. Establish these locations before entering port settings.
<!-- bilingual-en:end -->

## 来源与核验

- MIT Missing Semester 2020：[Course Shell](https://missing.csail.mit.edu/2020/course-shell/)、[Shell Tools](https://missing.csail.mit.edu/2020/shell-tools/)、[Data Wrangling](https://missing.csail.mit.edu/2020/data-wrangling/)、[Command-line Environment](https://missing.csail.mit.edu/2020/command-line/)：支持课程范围与命令、流、文本处理、作业及远程工作的衔接；TSV 数据、逐段输出和验证例子为本路径自建。
- 本机 Apple 分发的 `bash(1)`（GNU Bash 3.2）手册中 *EXPANSION*、*REDIRECTION*、*COMMAND EXECUTION*、*COMMAND EXECUTION ENVIRONMENT*：实际核对分词、命令替换、重定向顺序、内建/外部命令与父子环境边界。精确工具条件及线上主来源分别保存在共享原子末尾。

<!-- bilingual-en:start -->
MIT's four lecture pages establish the scope and sequence. The TSV fixture, stage outputs and validation example are constructed for this path. Apple's bundled Bash 3.2 manual was reopened for expansion, redirection, command lookup and execution-environment semantics. Individual atoms record exact utility conditions and primary online sources.
<!-- bilingual-en:end -->
