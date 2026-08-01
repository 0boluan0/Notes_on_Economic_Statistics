---
aliases:
  - The Missing Semester Lecture 5 Exercises
  - Missing Semester Command Line Environment Exercises
tags:
  - computer-science
  - tools
  - exercises
  - command-line
  - tmux
  - ssh
  - the-missing-semester
---

# 第 5 讲 命令行环境 练习
<!-- bilingual-en:start -->
*Lecture 5: Command-Line Environment Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 5 讲 命令行环境]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/command-line/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//command-line-solution

## 任务控制
<!-- bilingual-en:start -->
*Job Control*
<!-- bilingual-en:end -->

1. 运行 `sleep 10000`。
   - 用 `Ctrl-Z` 挂起
   - 用 `bg` 让它在后台继续运行
   - 用 `pgrep` 查找 pid
   - 用 `pkill` 结束它，不要手动输入 pid
<!-- bilingual-en:start -->

&nbsp;
**1.** Run `sleep 10000`. Suspend it with `Ctrl-Z`, resume it in the background with `bg`, locate its PID with `pgrep`, and terminate it with `pkill` without typing the PID manually.<br>
<!-- bilingual-en:end -->
2. 让一个进程结束后再开始另一个进程。
   - 先尝试 `sleep 60 &` 后使用 `wait`
   - 然后思考在另一个 bash 会话中为什么失效
   - 写一个 bash 函数 `pidwait`，输入 pid，循环等待到进程结束
   - 需要使用 `sleep` 避免空转浪费 CPU
<!-- bilingual-en:start -->

&nbsp;
**2.** Arrange for one process to start only after another has finished. First run `sleep 60 &` and use `wait`; then explain why this does not work from a separate Bash session. Write a Bash function named `pidwait` that accepts a PID and repeatedly checks until that process exits, using `sleep` to avoid wasting CPU in a busy loop.<br>
<!-- bilingual-en:end -->

## 终端多路复用
<!-- bilingual-en:start -->
*Terminal Multiplexing*
<!-- bilingual-en:end -->

1. 完成一个 `tmux` 教程，并继续学习基本自定义：
   - 入门教程：https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/
   - 自定义步骤：https://www.hamvocke.com/blog/a-guide-to-customizing-your-tmux-conf/
<!-- bilingual-en:start -->

&nbsp;
**1.** Complete a `tmux` tutorial and learn the basics of customization. Use https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/ for the introduction and https://www.hamvocke.com/blog/a-guide-to-customizing-your-tmux-conf/ for configuration.<br>
<!-- bilingual-en:end -->

## 别名
<!-- bilingual-en:start -->
*Aliases*
<!-- bilingual-en:end -->

1. 创建一个 `dc` 别名，让误输 `dc` 时也能执行 `cd`。
<!-- bilingual-en:start -->

&nbsp;
**1.** Create a `dc` alias so that mistyping `cd` as `dc` still changes directories.<br>
<!-- bilingual-en:end -->
2. 执行：

```sh
history | awk '{$1="";print substr($0,2)}' | sort | uniq -c | sort -n | tail -n 10
```

   找出你最常用的十条命令，并考虑为它们写更短的别名。
   如果你用的是 ZSH，把 `history` 换成 `history 1`。
<!-- bilingual-en:start -->

&nbsp;
**2.** Run the pipeline shown above to find your ten most frequently used commands, then consider defining shorter aliases for them. If you use Zsh, replace `history` with `history 1`.<br>
<!-- bilingual-en:end -->

## 配置文件
<!-- bilingual-en:start -->
*Dotfiles*
<!-- bilingual-en:end -->

1. 为自己的 dotfiles 新建一个目录，并开启版本控制。
<!-- bilingual-en:start -->

&nbsp;
**1.** Create a directory for your dotfiles and put it under version control.<br>
<!-- bilingual-en:end -->
2. 至少把一个配置文件放进去，例如 shell 配置，先从自定义 `$PS1` 开始。
<!-- bilingual-en:start -->

&nbsp;
**2.** Add at least one configuration file, such as your shell configuration; begin by customizing `$PS1`.<br>
<!-- bilingual-en:end -->
3. 建立一套新机器快速安装 dotfiles 的方法。
   最简单的方案是写一个脚本调用 `ln -s`，也可以使用专用工具。
<!-- bilingual-en:start -->

&nbsp;
**3.** Create a way to install your dotfiles quickly on a new machine. The simplest approach is a script that calls `ln -s`, although you may use a dedicated dotfile manager.<br>
<!-- bilingual-en:end -->
4. 在新的虚拟机上测试安装脚本。
<!-- bilingual-en:start -->

&nbsp;
**4.** Test the installation script in a fresh virtual machine.<br>
<!-- bilingual-en:end -->
5. 把你现有的所有配置迁移进这个仓库。
<!-- bilingual-en:start -->

&nbsp;
**5.** Migrate all of your existing configuration into this repository.<br>
<!-- bilingual-en:end -->
6. 将这个仓库发布到 GitHub。
<!-- bilingual-en:start -->

&nbsp;
**6.** Publish the repository on GitHub.<br>
<!-- bilingual-en:end -->

## 远端设备
<!-- bilingual-en:start -->
*Remote Machines*
<!-- bilingual-en:end -->

1. 准备一台 Linux 虚拟机。
<!-- bilingual-en:start -->

&nbsp;
**1.** Set up a Linux virtual machine.<br>
<!-- bilingual-en:end -->
2. 检查 `~/.ssh/` 是否已有密钥对；若没有，用 `ssh-keygen -o -a 100 -t ed25519` 生成。
<!-- bilingual-en:start -->

&nbsp;
**2.** Check whether `~/.ssh/` already contains a key pair. If not, generate one with `ssh-keygen -o -a 100 -t ed25519`.<br>
<!-- bilingual-en:end -->
3. 在 `~/.ssh/config` 中增加一个 `Host vm` 配置，包含：
   - 用户名
   - 主机 IP
   - `IdentityFile`
   - `LocalForward 9999 localhost:8888`
<!-- bilingual-en:start -->

&nbsp;
**3.** Add a `Host vm` entry to `~/.ssh/config` containing the username, host IP address, `IdentityFile`, and `LocalForward 9999 localhost:8888`.<br>
<!-- bilingual-en:end -->
4. 使用 `ssh-copy-id vm` 复制公钥到服务器。
<!-- bilingual-en:start -->

&nbsp;
**4.** Copy your public key to the server with `ssh-copy-id vm`.<br>
<!-- bilingual-en:end -->
5. 在虚拟机里运行 `python -m http.server 8888`，通过本机 `http://localhost:9999` 访问它。
<!-- bilingual-en:start -->

&nbsp;
**5.** Run `python -m http.server 8888` inside the virtual machine and access it locally at `http://localhost:9999`.<br>
<!-- bilingual-en:end -->
6. 修改 `sshd_config`：
   - 禁用密码认证
   - 禁用 root 登录
   - 重启 `ssh` 服务并重新测试
<!-- bilingual-en:start -->

&nbsp;
**6.** Edit `sshd_config` to disable password authentication and root login, restart the `ssh` service, and test the connection again.<br>
<!-- bilingual-en:end -->
7. 附加题：
   - 安装并测试 `mosh`
   - 研究 `ssh -N` 与 `ssh -f`，找出后台端口转发命令
<!-- bilingual-en:start -->

&nbsp;
**7.** Extensions: install and test `mosh`; then study `ssh -N` and `ssh -f` and determine the command for running port forwarding in the background.<br>
<!-- bilingual-en:end -->
