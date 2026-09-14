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

第一次学第 1、2、4、5 讲，先读 [[Shell与数据管道|Shell 与数据管道]]：用同一份小日志贯穿参数、格式、统计和失败处理，再延伸到脚本与远程环境。下面按课程顺序嵌入完整共享解释，供继续阅读和回查。
<!-- bilingual-en:start -->
For Lectures 1, 2, 4 and 5, start with [[Shell与数据管道|Shell and Data Pipelines]], following one small log through arguments, formats, counting and failure handling, then extending the model to scripts and remote environments. The complete shared explanations below follow lecture order for further reading and reference.
<!-- bilingual-en:end -->

[[Missing Semester Course Atlas.canvas|全课程路线]] · [[Shell、数据整理与命令行环境.canvas|Shell 与数据整理关系图]]

## 第 1 讲 课程概览与 shell
<!-- bilingual-en:start -->
*Lecture 1: Course Overview and the Shell*
<!-- bilingual-en:end -->

命令行把程序组合成数据流，是随后所有工具的共同入口。
<!-- bilingual-en:start -->
The command line composes programs into data flows and serves as the common entry point for every tool that follows.
<!-- bilingual-en:end -->

先分清解释器、命令与路径。命令名不一定表示外部文件；找到了文件，也还要判断路径搜索和权限是否允许访问。
<!-- bilingual-en:start -->
Distinguish the interpreter, commands and paths. A command name need not denote an external file; locating a file still leaves path-search and permission requirements to check.
<!-- bilingual-en:end -->

![[Shell]]

![[Shell简单命令]]

![[文件路径]]

![[Shell命令查找]]

![[Unix权限位]]

再看程序之间传递什么。流传输字节，退出状态报告这次执行的结果；重定向连接流，管道组合程序。重定向写在同一行不代表它们可以交换顺序。
<!-- bilingual-en:start -->
Next separate what programs exchange: streams carry bytes, while exit status reports an execution outcome. Redirections connect streams and pipelines compose programs; redirections on one line are not necessarily interchangeable.
<!-- bilingual-en:end -->

![[标准流]]

![[Shell重定向]]

![[重定向顺序]]

![[Shell管道]]

![[命令退出状态]]

[[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/01_Course Shell Exercises|第 1 讲练习]]

## 第 2 讲 Shell 工具和脚本
<!-- bilingual-en:start -->
*Lecture 2: Shell Tools and Scripting*
<!-- bilingual-en:end -->

从“能运行命令”进入可重复脚本，关键是参数边界、输入输出和失败处理。
<!-- bilingual-en:start -->
Moving from commands that merely run to repeatable scripts requires clear argument boundaries, explicit input and output, and deliberate failure handling.
<!-- bilingual-en:end -->

先沿着“一行文字如何变成参数”阅读。引用、字段分割和文件名展开决定参数边界；大括号展开生成文字，并不检查文件是否存在。这里只讨论注明的 Bash 规则。
<!-- bilingual-en:start -->
Follow how command text becomes arguments. Quoting, word splitting and filename expansion determine boundaries; brace expansion generates text without checking file existence. Apply the stated Bash rules only in their specified context.
<!-- bilingual-en:end -->

![[Shell引用]]

![[Shell字段分割]]

![[Shell文件名展开]]

![[Bash大括号展开]]

![[Shell位置参数]]

命令替换取得输出文字，进程替换提供流的访问路径，二者不是同一种临时文件机制。要把任意文件名交给下一条命令，仍须同时保护参数边界和目标程序的选项边界。
<!-- bilingual-en:start -->
Command substitution captures output text; process substitution exposes a stream through a path. They are not one temporary-file mechanism. Passing arbitrary filenames still requires both argument boundaries and the receiving program's option conventions.
<!-- bilingual-en:end -->

![[命令替换]]

![[进程替换]]

![[文件名安全传参]]

控制流再用退出状态决定下一步。管道的整体状态如何计算，与前面已经输出了多少内容是两个问题；不要把一条短表达式误当作完整的成功、失败分支。
<!-- bilingual-en:start -->
Control flow uses statuses to choose the next step. A pipeline's aggregate status is separate from how much output has already escaped; a short conditional expression is not automatically a complete success/failure branch.
<!-- bilingual-en:end -->

![[Shell条件执行]]

![[管道退出状态]]

最后把命令装进函数或脚本，明确谁解释、在哪个环境执行。先有这个模型，才谈临时文件与重复运行的可靠性。
<!-- bilingual-en:start -->
Package commands into functions or scripts only after identifying their interpreter and execution environment. That model comes before reliable temporary-file handling and repeat execution.
<!-- bilingual-en:end -->

![[Shell函数]]

![[Shell脚本]]

![[脚本执行与载入]]

![[Shebang]]

![[Shell别名]]

![[临时目录安全创建]]

![[脚本幂等性]]

[[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/02_Shell Tools Exercises|第 2 讲练习]]

## 第 3 讲 编辑器 (Vim)
<!-- bilingual-en:start -->
*Lecture 3: Editors (Vim)*
<!-- bilingual-en:end -->

本讲的重点是形成键盘驱动的编辑模型：移动、选择、操作、重复与宏。编辑器选择不形成独立知识文件；需要时回到 [MIT Missing Semester Editors](https://missing.csail.mit.edu/2020/editors/) 和本讲练习。
<!-- bilingual-en:start -->
This lecture develops a keyboard-driven editing model built from movement, selection, operations, repetition, and macros. Editor choice does not warrant a separate knowledge file; return to [MIT Missing Semester: Editors](https://missing.csail.mit.edu/2020/editors/) and the lecture exercises when needed.
<!-- bilingual-en:end -->

[[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/03_Vim Exercises|第 3 讲练习]]

## 第 4 讲 数据整理
<!-- bilingual-en:start -->
*Lecture 4: Data Wrangling*
<!-- bilingual-en:end -->

把标准输入输出接起来后，搜索、过滤、排序、聚合和格式转换才能形成可检查的管线。
<!-- bilingual-en:start -->
Connecting standard input and output lets searching, filtering, sorting, aggregation, and format conversion form an inspectable pipeline.
<!-- bilingual-en:end -->

先规定希望输出什么、输入实际是什么，再选择工具。正则是匹配语言，不是通用结构化数据解析器；模式看起来相同，也可能因方言与重复规则不同而改变含义。
<!-- bilingual-en:start -->
Specify the desired output and actual input format before choosing tools. A regular expression is a matching language, not a universal structured-data parser; dialect and repetition rules can change the meaning of similar-looking patterns.
<!-- bilingual-en:end -->

![[数据整理]]

![[正则表达式]]

![[正则方言边界]]

![[贪婪匹配跨越分隔符]]

筛选、替换、分字段与聚合承担不同职责。没有匹配是查询结果，不应自动等同于工具执行出错；切出了第三列，也不表示理解了文件格式。
<!-- bilingual-en:start -->
Selection, substitution, field splitting and aggregation have different responsibilities. No match is a query outcome, not automatically a utility error; extracting a third field does not prove the format was parsed correctly.
<!-- bilingual-en:end -->

![[grep行筛选]]

![[grep退出状态]]

![[sed替换]]

![[awk文本处理]]

![[字段切分不等于格式解析]]

![[awk条件聚合]]

完成统计还需要明确等值、顺序和计数对象。uniq 可以直接对相邻组计数；若要得到全局频数，则需先保证同一完整值的记录都已归在一起，例如按兼容的完整值规则排序。数值键、并列顺序和 locale 都是结果契约的一部分。回到 [[Shell与数据管道#4. 沿数据流观察，直到得到频数|日志例子的逐段输出]]，检查每一次转换。
<!-- bilingual-en:start -->
Counting requires explicit equality, ordering and counting units. uniq can count adjacent groups directly; global frequencies require all records of the same complete value to be grouped together, for example by a compatible full-value sort. Numeric keys, tie order and locale belong to the result contract. Inspect every transformation in the [[Shell与数据管道#4. 沿数据流观察，直到得到频数|log example's stage outputs]].
<!-- bilingual-en:end -->

![[文本排序键选择]]

![[相邻行去重]]

![[排序分组频数统计]]

![[文本处理的locale边界]]

[[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/04_Data Wrangling Exercises|第 4 讲练习]]

## 第 5 讲 命令行环境
<!-- bilingual-en:start -->
*Lecture 5: Command-Line Environments*
<!-- bilingual-en:end -->

长期任务和远程工作需要理解作业控制、signals、terminal multiplexer 与环境配置。
<!-- bilingual-en:start -->
Long-running jobs and remote work require an understanding of job control, signals, terminal multiplexers, and environment configuration.
<!-- bilingual-en:end -->

先解释为什么“在这个窗口能运行”不保证换一个窗口或机器仍然相同。环境变量传给子程序，启动文件决定配置何时加载，dotfiles 则管理可复现设置和机器差异；第 1 讲的[[Shell命令查找]]继续复用同一个解释。
<!-- bilingual-en:start -->
First explain why a command working in one window need not work identically elsewhere. Environment variables pass configuration to child programs, startup files determine loading, and dotfiles manage reproducible settings and host differences. Reuse Lecture 1's same [[Shell命令查找|command-lookup explanation]].
<!-- bilingual-en:end -->

![[环境变量]]

![[环境变量导出]]

![[Bash启动文件选择]]

![[dotfiles]]

![[dotfiles配置管理]]

再区分进程与 Shell 管理的作业。暂停、恢复、后台运行和终止各改变不同状态；发出一个请求，不等于已经拿到任务完成的证据。
<!-- bilingual-en:start -->
Distinguish a process from a shell-managed job. Stopping, resuming, backgrounding and terminating change different states; sending a request does not establish completion.
<!-- bilingual-en:end -->

![[Unix进程]]

![[Shell作业]]

![[前台与后台作业]]

![[作业状态切换]]

![[Unix信号]]

![[正常终止与强制终止]]

![[等待子进程]]

后台运行改变终端交互和 Shell 的等待方式，但不自动保证断连后继续。断开终端后是否继续，需要另查信号、输入输出和会话安排；tmux 重连要求相应服务端与会话存在且可访问，原任务的运行或完成状态还须另查。
<!-- bilingual-en:start -->
Backgrounding changes terminal interaction and shell waiting behavior without guaranteeing survival after disconnection. Survival also depends on signals, streams and session arrangements. tmux reattachment needs an existing, accessible server and session; the original task's running or completion state requires a separate check.
<!-- bilingual-en:end -->

![[后台作业断连边界]]

![[终端复用]]

![[tmux会话保持边界]]

最后才跨到远端：确定命令在哪台机器解释，分别验证服务器身份与用户身份，再画清端口转发的监听端、SSH 服务器和目标端。不能把这些都简写成“SSH 已连通”。
<!-- bilingual-en:start -->
Then cross to a remote machine: locate command interpretation, verify server and user identities separately, and distinguish the listener, SSH server and forwarding destination. These are not interchangeable claims that SSH is connected.
<!-- bilingual-en:end -->

![[SSH远程执行]]

![[SSH主机认证]]

![[SSH公钥认证]]

![[SSH本地端口转发]]

[[03_Computer_Science/99_Miscellaneous/the_missing_semester_exercises/05_Command-line Environment Exercises|第 5 讲练习]]

## 第 6 讲 版本控制 (Git)
<!-- bilingual-en:start -->
*Lecture 6: Version Control (Git)*
<!-- bilingual-en:end -->

Git 的命令只有放回对象图、工作区、暂存区和提交历史中才不会混乱。
<!-- bilingual-en:start -->
Git commands become coherent only when placed in the model of an object graph, working tree, staging area, and commit history.
<!-- bilingual-en:end -->

本讲原子的完整关系图：[[Git 版本控制.canvas|Git 版本控制 Canvas]]。
<!-- bilingual-en:start -->
Integrated map of this lecture's atoms: [[Git 版本控制.canvas|Git Version Control Canvas]].
<!-- bilingual-en:end -->

先建立不会随命令名称变化的数据模型：对象保存状态，commit 用父指针形成历史，branch 与 HEAD 只是给对象图命名的引用。
<!-- bilingual-en:start -->
Begin with the model that does not change with command names: objects store states, commits form history through parent links, and branches plus HEAD merely name positions in that graph.
<!-- bilingual-en:end -->

![[Git 对象图]]

![[Git commit 快照与父指针]]

![[Git 分支引用]]

![[Git HEAD 与分离状态]]

日常编辑再落到三个并存状态。暂存区不是含糊的“待处理区”，而是下一次提交的候选 tree；任何 diff 都必须说清两个比较端点。
<!-- bilingual-en:start -->
Everyday editing then falls into three coexisting states. The index is not a vague waiting area but the candidate tree for the next commit, and every diff must name its two endpoints.
<!-- bilingual-en:end -->

![[Git 三棵状态]]

![[Git 暂存区]]

![[git diff 比较端点]]

整合与恢复都在修改同一张图，但风险取决于提交是否已经共享。merge、rebase、reset、revert 和 reflog 不能按“撤销命令清单”来背，而要判断它们创建对象、移动引用还是覆盖本地层。
<!-- bilingual-en:start -->
Integration and recovery both change the same graph, but their risk depends on whether commits have been shared. Merge, rebase, reset, revert, and reflog should not be memorised as an undo-command list; determine whether each creates objects, moves references, or overwrites local layers.
<!-- bilingual-en:end -->

![[Git merge 与 rebase]]

![[Git reset 三种模式]]

![[Git revert 与公开历史]]

![[Git reflog]]

![[Git 冲突解决]]

最后补上两个边界：Git 的历史保留会让误提交的秘密长期存在，而同一个本地历史却仍不能替代独立备份。
<!-- bilingual-en:start -->
Finish with two boundaries: Git's historical retention can preserve an accidentally committed secret, while that same local history still cannot replace an independent backup.
<!-- bilingual-en:end -->

![[Git 历史中的 secret]]

![[本地仓库与异地备份]]

## 第 7 讲 调试及性能分析
<!-- bilingual-en:start -->
*Lecture 7: Debugging and Profiling*
<!-- bilingual-en:end -->

先稳定复现和缩小范围，再用证据定位根因；性能问题先测量而不是猜测。
<!-- bilingual-en:start -->
First reproduce a failure reliably and reduce its scope, then use evidence to locate the root cause. Measure performance problems before guessing about them.
<!-- bilingual-en:end -->

![[调试]]

![[最小失败复现]]

![[调试假设检验]]

课堂从打印、日志与调试器进入观察。先写下假设会带来的可观察差异，再决定在哪里打印或暂停；若程序抛出异常，traceback 提供调用路径，结合[[异常传播]]解释它，而不是把最后一行出现的位置自动当作根因。MIT 6.100L 的[[03_Computer_Science/03_MIT 6.100L/12_测试与失败处理|回文调试例子]]使用同一组原子，修复后保留[[回归测试]]。
<!-- bilingual-en:start -->
The lecture uses prints, logs and debuggers as observation tools. Predict a difference before deciding where to print or pause. A traceback gives a call path interpreted through [[异常传播|exception propagation]], not automatic proof that its final location is the root cause. MIT 6.100L's [[03_Computer_Science/03_MIT 6.100L/12_测试与失败处理|palindrome example]] shares these atoms and retains [[回归测试|regression checks]].
<!-- bilingual-en:end -->

功能偏差与资源问题使用同一种“提出假设—收集证据”的思路，但观测指标不同。性能剖析本身就能定位热点；先声明任务、输入、时间或内存指标，不需要先凭直觉猜出耗时的函数。
<!-- bilingual-en:start -->
Functional and resource problems share hypothesis-driven investigation but require different measurements. Profiling itself can locate hotspots; first state the task, input and resource metric rather than guessing the expensive function.
<!-- bilingual-en:end -->

![[性能剖析]]

![[剖析自身与累计时间]]

![[计时与渐近分析]]

本节工具层依据：[The Missing Semester 2020 — Debugging and Profiling](https://missing.csail.mit.edu/2020/debugging-profiling/)，调试器、性能剖析与资源监测部分；原子末尾分别给出精确语义的核验来源。
<!-- bilingual-en:start -->
Tool-level source: [The Missing Semester 2020 — Debugging and Profiling](https://missing.csail.mit.edu/2020/debugging-profiling/), especially debuggers, profiling and resource monitoring. Each atom records its precise semantic sources.
<!-- bilingual-en:end -->

## 第 8 讲 元编程
<!-- bilingual-en:start -->
*Lecture 8: Metaprogramming*
<!-- bilingual-en:end -->

本讲把三层责任串起来：构建系统声明项目内部目标与依赖，包管理器解析外部组件，CI 在 runner 上按事件重跑项目入口。它能暴露声明缺口，但干净 runner、锁文件与 cache 都不等于可复现性证明。
<!-- bilingual-en:start -->
This lecture connects three responsibilities: a build system declares internal targets and dependencies, a package manager resolves external components, and CI reruns the project entry point on event-triggered runners. This can expose missing declarations, but a clean runner, a lockfile, and a cache are not proofs of reproducibility.
<!-- bilingual-en:end -->

本讲原子的完整关系图：[[构建、依赖与 CI.canvas|构建、依赖与 CI Canvas]]。
<!-- bilingual-en:start -->
Integrated map of this lecture's atoms: [[构建、依赖与 CI.canvas|Build, Dependencies, and CI Canvas]].
<!-- bilingual-en:end -->

先读 Make 的具体机制：规则把目标、先决条件与配方分开；请求目标后只展开它的传递依赖。时间戳只是 GNU Make 的失效策略，正确的增量和并行执行仍依赖完整的边；phony 与 pattern rule 是两种不同的规则扩展。
<!-- bilingual-en:start -->
Begin with Make's concrete mechanisms: a rule separates target, prerequisites, and recipe, and a requested target expands only its transitive dependencies. Timestamps are GNU Make's freshness strategy; correct incremental and parallel execution still require complete edges. Phony and pattern rules extend ordinary rules in different ways.
<!-- bilingual-en:end -->

![[构建规则三要素]]

![[传递依赖递归构建]]

![[Make 目标过期判断]]

![[构建依赖完整性]]

![[Make phony target]]

![[Make 模式规则]]

再沿外部依赖走完整条链：版本号表达兼容承诺，manifest 给 resolver 范围，lockfile 保存一次精确解析。锁住包版本只是控制部分输入；可复现性是逐字节输出判据，密封性则是隔离隐式输入的设计属性。
<!-- bilingual-en:start -->
Next follow the external-dependency chain: version numbers communicate compatibility promises, a manifest gives the resolver ranges, and a lockfile records one exact resolution. Locking package versions controls only some inputs. Reproducibility is a byte-identical output criterion, whereas hermeticity is a design property that isolates implicit inputs.
<!-- bilingual-en:end -->

![[构建系统与包管理器]]

![[语义化版本]]

![[版本要求范围]]

![[锁文件解析结果]]

![[锁文件与可复现构建]]

![[可复现构建]]

![[密封构建]]

最后把同一项目入口放进 CI workflow。先辨认 trigger、job、runner 与 step，再区分 CI 和部署、fresh runner 和 hermetic build、cache 和 artifact。本地 pre-commit hook 可以复用同一入口取得早期反馈，却不能替代共享 CI；正式管线还要显式声明检查与 artifact 交接，失败时再按提交、工具链、环境与依赖图逐项缩小范围。
<!-- bilingual-en:start -->
Finally place the same project entry point in a CI workflow. Distinguish triggers, jobs, runners, and steps; then distinguish CI from deployment, a fresh runner from a hermetic build, and a cache from an artifact. A local pre-commit hook may reuse that entry point for early feedback but cannot replace shared CI. The formal pipeline must also declare its checks and artifact hand-offs before failures are narrowed across the revision, toolchain, environment, and dependency graph.
<!-- bilingual-en:end -->

![[CI 工作流结构]]

![[CI 与部署发布]]

![[本机与 CI 统一入口]]

![[pre-commit 与 CI]]

![[干净 CI runner 的边界]]

![[CI 缓存与 artifact]]

![[CI 验证管线]]

![[本机与 CI 差异诊断]]

![[增量构建失效诊断]]

## 第 9 讲 安全和密码学
<!-- bilingual-en:start -->
*Lecture 9: Security and Cryptography*
<!-- bilingual-en:end -->

安全从威胁模型开始，再按目标区分 hash、MAC、签名、认证加密和口令存储。主题关系见 [[密码学原语与安全模型.canvas|密码学原语与安全模型]]。
<!-- bilingual-en:start -->
Security begins with a threat model, then separates hashes, MACs, signatures, authenticated encryption, and password storage by goal. See the [[密码学原语与安全模型.canvas|cryptographic primitives and security models map]] for their relationships.
<!-- bilingual-en:end -->

先明确资产、攻击者能力与信任边界；否则“安全”没有可检验的对象。
<!-- bilingual-en:start -->
First make assets, adversary capabilities, and trust boundaries explicit; otherwise “secure” has no testable object.
<!-- bilingual-en:end -->

![[威胁模型]]

随后按需要的性质选择原语，不把“检测变化”“认证来源”和“隐藏内容”混为一谈。
<!-- bilingual-en:start -->
Then choose a primitive by the required property rather than conflating change detection, source authentication, and secrecy.
<!-- bilingual-en:end -->

![[密码学哈希]]

![[密码学哈希的认证边界]]

![[消息认证码]]

![[MAC 的来源边界]]

![[数字签名]]

![[数字签名与身份绑定]]

![[签名时点与密钥状态]]

![[认证加密]]

原语选对之后还要满足它的调用边界，并给低熵口令使用专门的离线猜测防护。
<!-- bilingual-en:start -->
After choosing the right primitive, satisfy its invocation boundary and give low-entropy passwords dedicated resistance to offline guessing.
<!-- bilingual-en:end -->

![[GCM IV 唯一性]]

![[认证解密先验证后使用]]

![[认证不等于防重放]]

![[口令哈希存储]]

![[pepper 与验证器分离]]

最后回到系统层：原语验证成功不能外推为系统安全；密钥、身份、权限、调用顺序、更新和恢复流程仍要分别验证。
<!-- bilingual-en:start -->
Finally return to the system level: successful primitive-level verification does not establish system security; keys, identities, permissions, call order, updates, and recovery still need separate validation.
<!-- bilingual-en:end -->

![[原语安全不等于系统安全]]

![[RPO 与 RTO]]

![[备份密钥恢复]]

![[备份恢复需演练验证]]

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
