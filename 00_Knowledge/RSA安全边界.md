---
aliases:
  - "RSA安全性依赖计算假设与编码方案而不能由解密正确性推出"
  - RSA security assumptions and encoding boundaries
student_os: knowledge-atom
atom_id: MCS-NT-023
atom_type: distinction
status: source-checked
part_of:
  - "[[数论与RSA.canvas]]"
requires:
  - "[[RSA公钥加密]]"
  - "[[RSA全消息正确性]]"
related:
  - "[[唯一素因数分解]]"
---

# RSA安全性依赖计算假设与编码方案而不能由解密正确性推出
<!-- bilingual-en:start -->
*RSA security depends on computational assumptions and encoding, not merely on correct decryption*
<!-- bilingual-en:end -->

[[RSA全消息正确性]] 证明有私钥时能还原消息；安全性还要回答攻击者仅有公开信息时能否恢复或区分消息。正确性是一条代数定理，抵抗攻击则依赖计算困难假设与具体编码方案，两者不能互相替代。
<!-- bilingual-en:start -->
[[RSA全消息正确性|Correctness]] proves recovery with the private key. Security also asks whether an attacker with public information can recover or distinguish messages. The algebraic theorem does not establish the computational and scheme-level guarantees.
<!-- bilingual-en:end -->

若能把 $n=pq$ 分解，就能算出 $\varphi(n)$，再求 $e$ 的逆元得到可用的 $d$，因此分解模数足以破坏 RSA。这个方向是已知推导；“任何恢复明文的方法都必然给出分解算法”不能由它倒推。也不要把教材关于分解困难的假设写成已经证明的复杂度下界。
<!-- bilingual-en:start -->
Factoring $n$ gives its totient and hence a usable inverse exponent, so factorization is sufficient to break RSA. That implication does not establish the converse that every way to recover plaintext yields a factoring algorithm. Assumed hardness is not a proved complexity lower bound.
<!-- bilingual-en:end -->

直接使用 $E(m)=m^e\bmod n$ 还存在不依赖分解的结构问题：同一个消息总产生同一个密文，任何人都能加密候选消息进行比较；并且
$$
E(m_1)E(m_2)\equiv E(m_1m_2)\pmod n.
$$
这种确定性与乘法可塑性说明，裸模幂公式本身没有提供完整的加密安全保证。
<!-- bilingual-en:start -->
The raw map is deterministic, allowing public comparison against candidate messages. It also preserves multiplication as displayed. These structural properties show why the modular-power formula alone does not provide a complete encryption-security guarantee.
<!-- bilingual-en:end -->

RFC 8017 区分两类方案：RSAES-OAEP 把随机化编码与 RSA 原语结合用于加密；RSASSA-PSS 把签名编码与 RSA 原语结合用于数字签名。PSS 不是加密填充。对于具体方案，还必须说明安全论证采用的假设与模型，不能把“使用 RSA”本身当作充分依据。
<!-- bilingual-en:start -->
RFC 8017 distinguishes RSAES-OAEP encryption from RSASSA-PSS digital signatures. OAEP combines randomized message encoding with RSA primitives; PSS is a signature encoding, not encryption padding. A scheme's guarantees depend on the assumptions and model of its security argument.
<!-- bilingual-en:end -->

课程把分解编码成 SAT，说明一个高效通用 SAT 算法会带来高效分解算法；这项归约本身没有证明现实中的分解或 SAT 已经容易。
<!-- bilingual-en:start -->
The course reduction to SAT shows that an efficient general SAT algorithm would give efficient factoring. The reduction itself does not establish that either problem is easy in practice.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=290|MIT 6.042J 教材，印刷第 281–282 页／PDF 第 290–291 页]]：核对分解可恢复私钥、绕过私钥解密与分解的关系未获等价结论，以及 SAT 归约方向；[[MIT6_042JS15_FactoringSAT.pdf#page=1|FactoringSAT，PDF 第 1–3 页]] 支持乘法电路与逐位 SAT 检验。
- [IETF RFC 8017 §5.1、§7.1、§8.1](https://www.rfc-editor.org/rfc/rfc8017.html#section-7.1)：核对 RSA 原语、OAEP 加密与 PSS 签名的职责及有条件的安全论证。确定性与乘法性质由原语公式直接核算；本卡不沿用教材带年代的攻击历史概括。
<!-- bilingual-en:start -->
- MIT printed pp. 281–282 / PDF pp. 290–291 and FactoringSAT verify the factoring and SAT implications without assuming their converses. RFC 8017 verifies the primitives and the distinct roles of OAEP and PSS. Determinism and multiplicativity follow directly from the primitive formula.
<!-- bilingual-en:end -->
