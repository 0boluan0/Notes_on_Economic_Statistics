---
aliases:
  - "MAC 只能认证某个共享密钥持有者，不能在持钥者之间归责"
  - A MAC authenticates some shared-key holder but cannot attribute a message among key holders
  - MAC 的来源边界
  - HMAC 的归属边界
student_os: knowledge-atom
atom_id: CS-SEC-003
atom_set: cryptographic-primitives-security-models
atom_type: concept-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[消息认证码]]"
part_of:
  - "[[密码学原语与安全模型.canvas]]"
related:
  - "[[数字签名]]"
  - "[[认证不等于防重放]]"
---

# MAC 只能认证某个共享密钥持有者，不能在持钥者之间归责
<!-- bilingual-en:start -->
*A MAC authenticates some shared-key holder but cannot attribute a message among key holders*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> MAC 验证成功可以在密钥未泄露等前提下排除不持钥者的伪造，但发送方与验证方共享同一秘密，双方原则上都能生成有效 tag。因此，MAC 的来源结论只到“某个持钥者”为止，不能仅凭 tag 向第三方证明是哪一位持钥者生成。
> <!-- bilingual-en:start -->
> Successful MAC verification can exclude forgery by a party without the key under assumptions such as an uncompromised key. The sender and verifier share the same secret, however, so either can in principle generate a valid tag. A MAC therefore establishes origin only from some key holder; the tag alone cannot prove to a third party which holder generated it.
> <!-- bilingual-en:end -->

例如 Alice 与 Bob 共用一把 MAC 密钥。Bob 收到一条有效消息后，能相信它不是由不持钥的网络攻击者随意改写；但 Bob 自己也能算出相同形式的 tag，所以不能把这条 tag 交给仲裁者并仅凭它证明“一定是 Alice 发的”。
<!-- bilingual-en:start -->
Suppose Alice and Bob share one MAC key. After receiving a valid message, Bob can treat it as not arbitrarily modified by a network attacker without the key. But Bob can also compute tags of the same form, so he cannot present the tag to an arbitrator and prove from it alone that Alice must have sent the message.
<!-- bilingual-en:end -->

若需要不持有签名私钥的第三方独立验证来源，应考虑数字签名以及相应的公钥身份绑定；这并不意味着签名自动解决所有身份与时间问题。
<!-- bilingual-en:start -->
If a third party that does not hold the signing secret must independently verify origin, a digital signature and an appropriate public-key identity binding may be needed. This does not mean that a signature automatically settles every identity or timing question.
<!-- bilingual-en:end -->

> [!question]- 自检
> Alice 与 Bob 共用同一 MAC 密钥。Bob 能否把一条有效 tag 给仲裁者，并仅凭它证明消息一定由 Alice 生成？
>
> **答案：** 不能。Bob 自己也能生成有效 tag；它只能排除没有密钥的伪造者，不能在持钥者之间归责。

## 来源与核验

- [RFC 9052, Section 8.2, “Message Authentication Code Algorithms”](https://www.rfc-editor.org/rfc/rfc9052.html#section-8.2)：直接核对共享密钥 MAC 不能向第三方证明发送者身份，因为所有持钥者都能生成 tag。
- [NIST Message Authentication Codes project](https://csrc.nist.gov/Projects/message-authentication-codes)：核对 MAC 要求发送方与接收方建立秘密密钥，并提供消息完整性与来源认证；本卡只收窄其“来源”结论的适用范围。
