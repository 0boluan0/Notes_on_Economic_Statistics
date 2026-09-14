---
aliases:
  - "若采用 pepper，秘密密钥必须与口令 verifier 分离保存"
  - If a pepper is used its secret key must be stored separately from password verifiers
  - pepper 与验证器分离
  - verifier-only secret
student_os: knowledge-atom
atom_id: CS-SEC-016
atom_set: cryptographic-primitives-security-models
atom_type: implementation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[口令哈希存储]]"
part_of:
  - "[[密码学原语与安全模型.canvas]]"
related:
  - "[[原语安全不等于系统安全]]"
---

# 若采用 pepper，秘密密钥必须与口令 verifier 分离保存
<!-- bilingual-en:start -->
*If a pepper is used, its secret key must be stored separately from password verifiers*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 工程上常把 password hash 之外、由验证器掌握的额外秘密称为 pepper。现行 NIST 指南以 **SHOULD** 建议在 salted password hash 后增加 keyed hash 或加密操作；一旦采用，该秘密 key **必须** 与 password hashes 分开保存，并且**建议**在硬件保护区内使用。
> <!-- bilingual-en:start -->
> Engineering practice often calls an additional verifier-held secret outside the password hash a pepper. Current NIST guidance says a verifier **SHOULD** add a keyed hash or encryption operation after salted password hashing. If this option is used, the secret key **SHALL** be stored separately from password hashes and **SHOULD** be used in a hardware-protected area.
> <!-- bilingual-en:end -->

pepper 的价值来自故障域分离：只拿到 verifier 数据库的攻击者还缺少第二处秘密，不能仅凭数据库完成同样的离线验证。若 pepper 与 password hashes 同库存放、同一备份暴露或同一低权限进程可直接读取，它就没有形成独立防线。
<!-- bilingual-en:start -->
The value of a pepper comes from separating failure domains. An attacker who obtains only the verifier database still lacks a second secret and cannot perform the same offline verification with the database alone. If the pepper is stored in the same database, exposed by the same backup, or directly readable by the same low-privilege process, it has not created an independent barrier.
<!-- bilingual-en:end -->

这条建议不能代替独立 salt、适当成本和强口令策略。它还引入轮换、可用性与恢复责任：秘密丢失会让现有口令无法验证，秘密泄露则要有迁移方案。因此 pepper 是一项可选的系统控制，不是给 password hash “再加一个固定字符串”。
<!-- bilingual-en:start -->
This guidance does not replace per-record salts, an appropriate cost, or password-strength controls. It also introduces rotation, availability, and recovery responsibilities: losing the secret can make existing passwords unverifiable, while compromise requires a migration plan. A pepper is therefore an optional system control, not a fixed string casually appended to a password hash.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么把同一个 pepper 字符串和所有 password hashes 一起放在数据库的一列里，没有形成额外防线？
>
> **答案：** 数据库泄露会同时交出 verifier 和 pepper；攻击者仍可离线验证猜测。pepper 的安全作用依赖独立密钥与独立保护边界。

## 来源与核验

- [NIST SP 800-63B-4, “Password Verifiers”](https://pages.nist.gov/800-63-4/sp800-63b.html#password-verifiers)：核对额外 keyed hash／加密步骤的 **SHOULD** 层级，以及采用后密钥与 password hashes 分离保存的 **SHALL** 和硬件保护的 **SHOULD**。本卡用常见工程名 pepper 描述该 verifier-only secret，没有把建议误写成所有系统的强制要求。
