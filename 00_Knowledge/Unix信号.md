---
student_os: knowledge-atom
atom_id: CS-CLI-010
atom_type: definition
aliases:
  - Unix信号是通知进程事件并按处置规则改变执行的机制
  - Unix signal
status: source-checked
requires:
  - "[[Unix进程]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# Unix信号是通知进程事件并按处置规则改变执行的机制
<!-- bilingual-en:start -->
*A Unix signal notifies a process of an event and affects execution according to its disposition.*
<!-- bilingual-en:end -->

信号可由内核事件、终端操作或获准的进程发送产生。进程对信号可能采用默认动作、忽略或安装处理函数；默认动作也不全是终止，还可能暂停或忽略。`SIGKILL`、`SIGSTOP` 不能被捕获或忽略。
<!-- bilingual-en:start -->
Signals can originate from kernel events, terminal actions, or authorized senders. A process can use the default disposition, ignore a signal, or install a handler. Defaults include stopping or ignoring as well as termination. `SIGKILL` and `SIGSTOP` cannot be caught or ignored.
<!-- bilingual-en:end -->

在常见终端设置下，`Ctrl-C` 对应 `SIGINT`，`Ctrl-Z` 对应 `SIGTSTP`；终端驱动向前台进程组发送它们，并非总由 shell 发给一个 PID。名称中的 `kill` 也只是发送信号接口，不意味着所选信号一定终止进程。
<!-- bilingual-en:start -->
Under common terminal settings, `Ctrl-C` generates `SIGINT` and `Ctrl-Z` generates `SIGTSTP`. The terminal driver sends them to the foreground process group, not invariably through the shell to one PID. An interface named `kill` sends signals; its name does not imply that every selected signal terminates a process.
<!-- bilingual-en:end -->

发送成功不等于业务处理或清理已完成。发送者必须有适当权限，并准确区分单个 PID、进程组与 shell jobspec；具体终止决策见[[正常终止与强制终止]]。
<!-- bilingual-en:start -->
Successful sending does not establish completion of application handling or cleanup. The sender needs appropriate permission and must distinguish a PID, process group, and shell jobspec. See [[正常终止与强制终止|termination judgment]] for shutdown decisions.
<!-- bilingual-en:end -->

## 来源与核验

[OpenBSD signal(3)](https://man.openbsd.org/signal.3)：核对默认动作、处理、忽略以及 KILL/STOP 例外；[kill(2)](https://man.openbsd.org/kill.2) 支持目标和发送权限边界。
<!-- bilingual-en:start -->
[OpenBSD signal(3)](https://man.openbsd.org/signal.3) supports dispositions and KILL/STOP exceptions; [kill(2)](https://man.openbsd.org/kill.2) supports targeting and permission boundaries.
<!-- bilingual-en:end -->

[OpenBSD termios(4), Special Characters](https://man.openbsd.org/termios.4#Special_Characters)：支持 `ISIG` 条件下向前台进程组生成终端信号。
<!-- bilingual-en:start -->
[OpenBSD termios(4): Special Characters](https://man.openbsd.org/termios.4#Special_Characters) supports terminal signals to the foreground process group under the `ISIG` condition.
<!-- bilingual-en:end -->
