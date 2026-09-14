---
student_os: knowledge-atom
atom_id: CS-CLI-020
atom_type: definition
aliases:
  - SSH本地端口转发把客户端监听端口的连接经SSH送到远端视角的目标
  - SSH local port forwarding
status: source-checked
requires:
  - "[[SSH远程执行]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# SSH本地端口转发把客户端监听端口的连接经SSH送到远端视角的目标
<!-- bilingual-en:start -->
*SSH local forwarding sends connections arriving at a client-side listener through SSH to a destination reached from the remote side.*
<!-- bilingual-en:end -->

`ssh -L [bind_address:]local_port:target_host:target_port user@server` 在客户端建立监听入口。连接该入口后，数据经 SSH 通道到服务器，再由服务器连接 `target_host:target_port`。因此监听位置与最终目标是两个端点，目标主机名从远端解析和访问。
<!-- bilingual-en:start -->
`ssh -L [bind_address:]local_port:target_host:target_port user@server` creates a client-side listener. Incoming connections travel through SSH and then from the server to `target_host:target_port`. The listener and final destination are distinct, and the destination hostname is resolved and reached from the remote side.
<!-- bilingual-en:end -->

```bash
ssh -N -L 127.0.0.1:9999:localhost:8888 user@server
```

此例把本地 `127.0.0.1:9999` 连到服务器视角的 `localhost:8888`；`-N` 不运行远端命令，只建立连接与转发。它不会自动启动远端服务，服务必须另外运行且能够从服务器访问。
<!-- bilingual-en:start -->
This example connects local `127.0.0.1:9999` to `localhost:8888` as seen by the server. `-N` requests no remote command. Forwarding does not start the destination service; that service must already be running and reachable from the server.
<!-- bilingual-en:end -->

显式 loopback 监听限制入口供本机访问；绑定所有接口会扩大可访问者范围。SSH 保护的是通道这一段，不自动赋予目标应用登录鉴权，也不保证服务器到另一目标主机这一段受 SSH 加密。端口已监听不等于目标服务正常，须分别验证。
<!-- bilingual-en:start -->
Explicit loopback binding limits the entry point to the local host; binding all interfaces broadens access. SSH protects its channel, not automatic application authorization or necessarily the server-to-another-host leg. A listening port alone does not establish destination-service health.
<!-- bilingual-en:end -->

## 来源与核验

[OpenSSH ssh(1), -L 与 -N](https://man.openbsd.org/ssh.1#L)：核对监听位置、远端发起目标连接和不执行远端命令；[ssh_config(5), LocalForward](https://man.openbsd.org/ssh_config.5#LocalForward) 支持配置等价形式与端点含义。
<!-- bilingual-en:start -->
[OpenSSH ssh(1): -L and -N](https://man.openbsd.org/ssh.1#L) supports the listener, remote-side destination connection, and absence of a remote command. [ssh_config(5): LocalForward](https://man.openbsd.org/ssh_config.5#LocalForward) supports the configuration form.
<!-- bilingual-en:end -->
