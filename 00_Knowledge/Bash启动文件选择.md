---
student_os: knowledge-atom
atom_id: CS-CLI-004
atom_type: execution-rule
aliases:
  - Bash按登录与交互方式选择启动文件而不是每次读取全部配置
  - Bash startup file selection
status: source-checked
requires:
  - "[[环境变量]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# Bash按登录与交互方式选择启动文件而不是每次读取全部配置
<!-- bilingual-en:start -->
*Bash selects startup files by login and interactive mode rather than reading every configuration file each time.*
<!-- bilingual-en:end -->

以通常名为 `bash` 的调用为准：登录调用读取 `/etc/profile`，再从 `~/.bash_profile`、`~/.bash_login`、`~/.profile` 依次选择第一个存在且可读的文件；交互式非登录调用读取 `~/.bashrc`。普通非交互脚本通过环境中的 `BASH_ENV` 指定启动文件，不按上述交互规则自动读取 `.bashrc`。
<!-- bilingual-en:start -->
For normal invocation as `bash`, login invocation reads `/etc/profile`, then the first existing readable file among `~/.bash_profile`, `~/.bash_login`, and `~/.profile`. An interactive non-login invocation reads `~/.bashrc`. An ordinary non-interactive script uses the environment's `BASH_ENV`, not automatic `.bashrc` loading under the interactive rule.
<!-- bilingual-en:end -->

这些是有条件的执行规则：`--noprofile`、`--norc` 可以禁读；以 `sh` 名称启动、POSIX 模式及被远程 shell daemon 调用还有专门规则。登录和交互是两条不同轴，不能仅凭“打开了终端窗口”推定全部配置会执行。
<!-- bilingual-en:start -->
These are conditional execution rules. `--noprofile` and `--norc` suppress loading; invocation as `sh`, POSIX mode, and remote-shell-daemon invocation have additional rules. Login and interactive status are separate dimensions: opening a terminal window does not imply that all configuration files run.
<!-- bilingual-en:end -->

如果 `.bash_profile` 显式加载 `.bashrc`，登录 Bash 才会通过这条桥梁取得其中配置。编辑配置只影响之后真正读取它的调用；也可在当前 shell 明确[[脚本执行与载入|加载经过审查的文件]]，但这会执行文件中的命令，不是只导入无害文本。
<!-- bilingual-en:start -->
A login Bash obtains `.bashrc` settings through an explicit load from `.bash_profile` if one is present. Editing a file affects invocations that subsequently read it. [[脚本执行与载入|Explicitly loading a reviewed file]] in the current shell executes its commands; it is not a harmless text import.
<!-- bilingual-en:end -->

## 来源与核验

[GNU Bash Manual §6.2, Bash Startup Files](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html)：核对登录、交互非登录、BASH_ENV、sh 与远程调用分支；同时完整读取本机 `bash(1)` INVOCATION 对应段落。
<!-- bilingual-en:start -->
[GNU Bash Manual §6.2: Bash Startup Files](https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html) supports login, interactive non-login, BASH_ENV, sh, and remote-invocation branches, cross-checked with the corresponding installed `bash(1)` INVOCATION section.
<!-- bilingual-en:end -->
