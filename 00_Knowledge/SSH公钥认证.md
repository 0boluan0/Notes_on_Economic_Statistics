---
student_os: knowledge-atom
atom_id: CS-CLI-019
atom_type: definition
aliases:
  - SSH公钥认证以会话相关签名证明用户能使用获授权公钥对应的私钥
  - SSH public key user authentication
status: source-checked
requires:
  - "[[数字签名]]"
part_of:
  - "[[Shell、数据整理与命令行环境.canvas]]"
---

# SSH公钥认证以会话相关签名证明用户能使用获授权公钥对应的私钥
<!-- bilingual-en:start -->
*SSH public-key user authentication uses a session-bound signature to prove access to the private key corresponding to an authorized public key.*
<!-- bilingual-en:end -->

客户端使用用户私钥签署包含会话标识与认证请求的数据，服务器检查签名，并判断该公钥是否被授权用于所请求账户。有效签名和账户授权缺一不可；服务器策略还可能要求额外认证步骤。
<!-- bilingual-en:start -->
The client signs data containing the session identifier and authentication request with the user's private key. The server verifies the signature and whether the public key is authorized for the requested account. Both checks are necessary, and policy can require additional authentication steps.
<!-- bilingual-en:end -->

OpenSSH 常用远端账户的 `authorized_keys` 保存允许的公钥。复制的是公钥，私钥留在客户端并应受访问控制和适当口令保护；可由 agent 代为签名，但“没有再次输入账户密码”不等于没有认证。
<!-- bilingual-en:start -->
OpenSSH commonly stores authorized public keys in the remote account's `authorized_keys`. Copy the public key, while protecting the private key on the client with access controls and an appropriate passphrase. An agent may perform signing; not retyping an account password does not mean authentication is absent.
<!-- bilingual-en:end -->

这项机制回答“客户端能否作为该用户登录”，不回答“连接的服务器是否为预期主机”。后者属于[[SSH主机认证]]；不要把用户私钥当作上传给服务器的口令，也不要把[[RSA公钥加密]]的加密推导直接当作 SSH 签名认证流程。
<!-- bilingual-en:start -->
This mechanism establishes user access, not whether the server is the intended host; that is [[SSH主机认证|SSH host authentication]]. Do not upload the user's private key as a server password or substitute an [[RSA公钥加密|RSA encryption]] derivation for SSH signature authentication.
<!-- bilingual-en:end -->

## 来源与核验

[RFC 4252 §7, Public Key Authentication Method](https://www.rfc-editor.org/rfc/rfc4252.html#section-7)：核对签名、账户授权、会话标识和可要求附加认证的边界。
<!-- bilingual-en:start -->
[RFC 4252 §7: Public Key Authentication Method](https://www.rfc-editor.org/rfc/rfc4252.html#section-7) supports signatures, account authorization, the session identifier, and possible additional authentication.
<!-- bilingual-en:end -->

[OpenSSH ssh(1), Authentication](https://man.openbsd.org/ssh.1#AUTHENTICATION)：支持 `authorized_keys`、私钥和 agent 的实际角色。
<!-- bilingual-en:start -->
[OpenSSH ssh(1): Authentication](https://man.openbsd.org/ssh.1#AUTHENTICATION) supports the roles of `authorized_keys`, private keys, and agents.
<!-- bilingual-en:end -->
