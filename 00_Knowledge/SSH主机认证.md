---
student_os: knowledge-atom
atom_id: CS-CLI-021
atom_type: definition
aliases:
  - SSH主机认证验证服务器使用的主机密钥是否属于预期主机
  - SSH server host authentication
status: source-checked
requires:
  - "[[数字签名与身份绑定]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# SSH主机认证验证服务器使用的主机密钥是否属于预期主机
<!-- bilingual-en:start -->
*SSH host authentication verifies that the server's host key belongs to the intended host.*
<!-- bilingual-en:end -->

SSH 客户端不仅需要建立加密通道，还要确认对端主机身份。OpenSSH 将服务器主机密钥与已信任的 `known_hosts` 记录或受信任主机证书等依据核对，避免只因某台机器能够响应连接就把它视为目标服务器。
<!-- bilingual-en:start -->
An SSH client needs server identity as well as an encrypted channel. OpenSSH checks the server's host key against trusted `known_hosts` records or other trust sources such as trusted host certificates, rather than accepting any machine that responds as the intended server.
<!-- bilingual-en:end -->

首次连接尚无记录时，屏幕显示的指纹只是待验证的主机密钥摘要，不能自行证明真实身份。应通过独立可信渠道核对指纹或使用已配置的信任来源；若只是接受首次见到的密钥，就是信任首次使用的假设，必须知道其首次连接风险。
<!-- bilingual-en:start -->
On first contact without a record, a displayed fingerprint identifies the offered key but does not prove the server's identity. Verify it through an independently trusted channel or an established trust source. Simply accepting the first observed key uses a trust-on-first-use assumption with a first-contact risk.
<!-- bilingual-en:end -->

已知主机密钥改变可能来自合法重装或轮换，也可能是冒充；先核实变化原因，不应为了消除提示就删除记录或关闭检查。`known_hosts` 验证服务器，而远端 `authorized_keys` 支持[[SSH公钥认证|用户登录授权]]，两份文件职责不同。
<!-- bilingual-en:start -->
A changed known key may reflect legitimate reinstallation or rotation, or impersonation. Verify the cause rather than deleting the record or disabling checks merely to suppress a warning. `known_hosts` verifies servers, while remote `authorized_keys` supports [[SSH公钥认证|user login authorization]]; their responsibilities differ.
<!-- bilingual-en:end -->

## 来源与核验

[OpenSSH ssh(1), Host Key Verification](https://man.openbsd.org/ssh.1#VERIFYING_HOST_KEYS)：核对指纹验证、known-host 数据库与可信验证渠道；[ssh_config(5), StrictHostKeyChecking](https://man.openbsd.org/ssh_config.5#StrictHostKeyChecking) 核对未知和改变密钥的接受策略。
<!-- bilingual-en:start -->
[OpenSSH ssh(1): Host Key Verification](https://man.openbsd.org/ssh.1#VERIFYING_HOST_KEYS) supports fingerprint checking, the known-host database, and trusted verification sources. [ssh_config(5): StrictHostKeyChecking](https://man.openbsd.org/ssh_config.5#StrictHostKeyChecking) supports acceptance policies for unknown and changed keys.
<!-- bilingual-en:end -->
