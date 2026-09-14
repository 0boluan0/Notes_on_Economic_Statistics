---
student_os: knowledge-atom
atom_id: CS-CLI-015
atom_type: boundary
aliases:
  - tmux分离client后可保留存活server上的会话但不保证任务完成或跨重启恢复
  - Tmux session persistence boundaries
status: source-checked
requires:
  - "[[终端复用]]"
  - "[[后台作业断连边界]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# tmux分离client后可保留存活server上的会话但不保证任务完成或跨重启恢复
<!-- bilingual-en:start -->
*Detaching a tmux client can preserve sessions on a surviving server, but guarantees neither task completion nor recovery across a reboot.*
<!-- bilingual-en:end -->

通常配置下，detach 或 SSH 连接中断只断开显示会话的 client，tmux server 及窗格内程序可继续存在。重新连接到同一主机上可访问的同一 server／socket 后，才能 attach 到仍存在的会话。
<!-- bilingual-en:start -->
Under normal configuration, detaching or losing an SSH connection disconnects the displaying client while the tmux server and pane programs can survive. Reattachment requires reaching the same accessible server or socket on the same host and an existing session.
<!-- bilingual-en:end -->

关键是 tmux 运行在哪里：在远端 tmux 中启动远端任务，才能把本地 SSH client 的断开与任务终端分离；仅在本地 tmux 中运行 SSH，不自动保留 SSH 断开后的远端程序。
<!-- bilingual-en:start -->
Location matters: start a remote task inside remote tmux to separate its terminal from the local SSH client. Running SSH inside local tmux alone does not automatically preserve remote programs after SSH disconnects.
<!-- bilingual-en:end -->

Server 被终止、主机重启、会话被关闭、程序失败或资源耗尽都可能结束工作。`exit-unattached` 等选项也能改变无人连接时的行为。Tmux 不是 checkpoint、备份或作业调度器；attach 成功只是会话可访问，任务完成仍须查看退出状态和产物。
<!-- bilingual-en:start -->
Server termination, host reboot, session closure, program failure, or resource exhaustion can end the work. Options such as `exit-unattached` also change behavior without attached clients. Tmux is neither checkpointing, backup, nor a scheduler. Successful attachment proves accessibility, while completion still requires checking status and outputs.
<!-- bilingual-en:end -->

## 来源与核验

[Tmux official manual, Description](https://man.openbsd.org/tmux.1#DESCRIPTION)：支持 detach、意外断连与 server/client 分离；同手册 `exit-unattached`、`exit-empty` 条目支持配置和 server 生命周期边界。跨主机位置判断是由该进程模型推出的应用条件。
<!-- bilingual-en:start -->
[Tmux official manual: Description](https://man.openbsd.org/tmux.1#DESCRIPTION) supports detachment, disconnection, and server/client separation. Its `exit-unattached` and `exit-empty` entries qualify the lifecycle. The host-location judgment follows from this process model.
<!-- bilingual-en:end -->
