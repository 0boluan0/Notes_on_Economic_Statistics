---
aliases:
  - "GCM 的 IV 必须在同一密钥下满足唯一性要求"
  - GCM IVs must satisfy the uniqueness requirement under one key
  - GCM nonce 唯一性
student_os: knowledge-atom
atom_id: CS-SEC-006
atom_set: cryptographic-primitives-security-models
atom_type: implementation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[认证加密]]"
part_of:
  - "[[密码学原语与安全模型.canvas]]"
related:
  - "[[认证不等于防重放]]"
  - "[[原语安全不等于系统安全]]"
---

# GCM 的 IV 必须在同一密钥下满足唯一性要求
<!-- bilingual-en:start -->
*GCM IVs must satisfy the uniqueness requirement under one key*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> GCM 要求同一密钥下的不同加密输入不得重复使用同一 IV。NIST 对同时使用确定性与随机构造的实现给出更精确的终身概率边界：跨该密钥的所有加密实例，在两组不同输入上重复调用相同 IV 的概率不得超过 $2^{-32}$。IV 通常不需要保密，但必须按选定构造正确管理。
> <!-- bilingual-en:start -->
> GCM requires distinct encryption inputs under one key not to reuse the same IV. NIST gives a more precise lifetime probability bound for implementations that use deterministic and random constructions: across all encryption instances under that key, the probability of invoking the same IV on two distinct input sets must be no greater than $2^{-32}$. An IV normally need not be secret, but it must be managed according to the chosen construction.
> <!-- bilingual-en:end -->

确定性构造通过不重复来满足要求；基于随机数生成器（RBG）的构造通过限制重复概率来满足。IV 重复会复用计数器模式的密钥流，并可严重破坏认证边界，后果不只是“随机性稍差”。
<!-- bilingual-en:start -->
The deterministic construction meets the requirement by preventing reuse; the random-bit-generator construction meets it by bounding the probability of reuse. Repeating an IV reuses the counter-mode keystream and can severely undermine authentication. The consequence is not merely “slightly worse randomness.”
<!-- bilingual-en:end -->

标准的 RBG 构造要求随机字段至少 96 bit。除非实现只使用确定性构造的 96-bit IV，否则同一密钥下、跨所有 IV 长度和加密实例的认证加密调用总数不得超过 $2^{32}$。确定性计数器也要在多设备、并发、崩溃与回滚后保持不重复；若不能可靠保存状态，就要缩短密钥使用期或采用能满足标准要求的分配设计。
<!-- bilingual-en:start -->
The standard's RBG construction requires a random field of at least 96 bits. Unless an implementation exclusively uses 96-bit IVs from the deterministic construction, the total number of authenticated-encryption invocations under one key, across all IV lengths and instances, must not exceed $2^{32}$. Deterministic counters must remain non-repeating across devices, concurrency, crashes, and rollback. If state cannot be preserved reliably, the design must shorten key lifetime or allocate IVs in another standards-conforming way.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“每次从 0 开始计数”在单次运行中看似没问题，却可能在程序重启后破坏 GCM？
>
> **答案：** 重启会让同一密钥再次使用以前的 IV。唯一性约束跨该密钥的整个生命周期与所有实例，不只看一次进程运行。

## 来源与核验

- [NIST SP 800-38D, Sections 5.2 and 8](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-38d.pdf)：核对 $2^{-32}$ 重复概率上限、两种 IV 构造、RBG 随机字段下限、条件性的 $2^{32}$ 全局调用上限，以及 invocation/fixed field 的容量约束。
- [RFC 5116, Sections 3.1 and 5.1.1](https://datatracker.ietf.org/doc/html/rfc5116)：核对长寿命密钥下 nonce 状态要跨重启保持，以及 GCM nonce 重用会同时破坏机密性和认证性。
- 截至 2026-09-01，NIST CSRC 仍把 2007 版 SP 800-38D 列为 final，并已宣布修订计划；本卡没有把尚未定稿的修订意向写成现行要求。
