---
student_os: knowledge-atom
atom_id: CS-CLI-018
atom_type: definition
aliases:
  - SSH远程执行通过加密连接在远端运行命令但本地shell仍先解析命令行
  - SSH remote command execution
status: source-checked
requires:
  - "[[Shell引用]]"
  - "[[Shell管道]]"
  - "[[SSH主机认证]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# SSH远程执行通过加密连接在远端运行命令但本地shell仍先解析命令行
<!-- bilingual-en:start -->
*SSH executes commands remotely over an encrypted connection, but the local shell still parses the command line first.*
<!-- bilingual-en:end -->

`ssh user@host command` 请求登录远端账户并在远端执行命令；不提供命令时通常请求交互 shell。连接时先明确[[SSH主机认证|对端身份]]，用户是否获准登录则由[[SSH公钥认证|用户认证]]等机制判断。
<!-- bilingual-en:start -->
`ssh user@host command` requests account access and command execution on the remote host; without a command it normally requests an interactive shell. Establish [[SSH主机认证|server identity]] and separately satisfy an account's [[SSH公钥认证|user authentication]] requirements.
<!-- bilingual-en:end -->

```bash
ssh host 'printf "%s\n" beta alpha' | sort
ssh host 'printf "%s\n" beta alpha | sort'
```

第一行在远端输出、在本地排序；第二行把管道包含在远端命令文本中，因此两步都在远端。外层引用控制哪些字符先被本地 shell 解释；同理，未包含在远端命令中的 `>` 通常是本地重定向。
<!-- bilingual-en:start -->
The first line prints remotely and sorts locally. The second includes the pipeline in the remote command text, so both stages run remotely. Outer quoting determines what the local shell interprets first; similarly, `>` outside the remote command is normally local redirection.
<!-- bilingual-en:end -->

OpenSSH 会将命令及附加参数以空格拼成远端命令文本，而不是原样传送本地 argv 边界。把本地变量双引号包住，并不自动安全地引用了远端解析；不要把不可信值直接拼进远端命令。需要传数据时，优先把数据与固定命令分离，例如通过标准输入。
<!-- bilingual-en:start -->
OpenSSH joins the command and additional arguments with spaces into remote command text rather than preserving local argv boundaries. Double-quoting a local variable does not automatically quote it safely for remote parsing. Avoid interpolating untrusted values; separate data from a fixed command, for example through standard input.
<!-- bilingual-en:end -->

## 来源与核验

[OpenSSH ssh(1), Description](https://man.openbsd.org/ssh.1#DESCRIPTION)：核对远端执行及参数以空格拼接；[MIT Remote Machines](https://missing.csail.mit.edu/2020/command-line/#remote-machines) 的 Executing commands 段支持本地与远端管道区别。上例为最小演示，不是连接已执行的记录。
<!-- bilingual-en:start -->
[OpenSSH ssh(1): Description](https://man.openbsd.org/ssh.1#DESCRIPTION) supports execution and space-joined command arguments. [MIT Remote Machines](https://missing.csail.mit.edu/2020/command-line/#remote-machines) supports pipeline placement. The example is illustrative, not a record of an executed connection.
<!-- bilingual-en:end -->
