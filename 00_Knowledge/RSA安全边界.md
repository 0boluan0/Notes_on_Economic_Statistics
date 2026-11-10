---
aliases:
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

<!-- bilingual-en:start -->
# RSA security boundaries
<!-- bilingual-en:end -->

*RSA安全边界*

<!-- bilingual-en:start -->
Correct decryption and secrecy answer different questions. A lock may open reliably with its key and still be easy to open without it. For RSA, separate the integer operation, the computational problem faced by an attacker, and the complete message-encoding scheme.
<!-- bilingual-en:end -->

解密正确与保密回答不同的问题：一把锁用钥匙总能打开，并不说明没有钥匙就难以打开。对 RSA，需要分清整数运算本身、攻击者面对的计算问题，以及完整的消息编码方案。

<!-- bilingual-en:start -->
[[RSA全消息正确性|Correctness]] proves recovery with the private key. Security also asks whether an attacker with public information can recover or distinguish messages. The algebraic theorem does not establish the computational and scheme-level guarantees.
<!-- bilingual-en:end -->

[[RSA全消息正确性]] 证明有私钥时能还原消息；安全性还要回答攻击者仅有公开信息时能否恢复或区分消息。正确性是一条代数定理，抵抗攻击则依赖计算困难假设与具体编码方案，两者不能互相替代。

<!-- bilingual-en:start -->
The [[RSA公钥加密|RSA construction]] publishes $n,e$ and retains an inverse exponent $d$. Its raw integer map is $E(m)=m^e\bmod n$, where $0\le m<n$; $c=E(m)$ is the ciphertext representative. An attacker sees the public key and ciphertext. **Recovering** a message asks for its representative; **distinguishing** messages may ask only which of two known candidates was sent. The second task can be easier and is already relevant to secrecy.
<!-- bilingual-en:end -->

[[RSA公钥加密]]公开 $n,e$，保有逆指数 $d$；其裸整数映射为 $E(m)=m^e\bmod n$，其中 $0\le m<n$，$c=E(m)$ 是密文代表。攻击者看得到公钥和密文。“恢复”消息要求找出消息代表；“区分”消息可能只要求判断两个已知候选中发送了哪一个。后一项可以更容易，却已经关系到保密性。

<!-- bilingual-en:start -->
## Factoring gives an attack; the converse is another claim
<!-- bilingual-en:end -->

*分解给出攻击方法，反向结论是另一项主张*

<!-- bilingual-en:start -->
Factoring $n$ gives its totient and hence a usable inverse exponent, so factorization is sufficient to break RSA. That implication does not establish the converse that every way to recover plaintext yields a factoring algorithm. Assumed hardness is not a proved complexity lower bound.
<!-- bilingual-en:end -->

若能把 $n=pq$ 分解，就能算出 $\varphi(n)$，再求 $e$ 的逆元得到可用的 $d$，因此分解模数足以破坏 RSA。这个方向是已知推导；“任何恢复明文的方法都必然给出分解算法”不能由它倒推。也不要把教材关于分解困难的假设写成已经证明的复杂度下界。

<!-- bilingual-en:start -->
Here is the forward chain. [[欧拉函数|Euler’s totient]] $\varphi(n)$ counts the residue classes coprime to $n$, meaning their [[最大公因数|gcd]] with $n$ is one. Knowing the distinct primes $p,q$ gives $\lambda(n)=\operatorname{lcm}(p-1,q-1)$, where [[最小公倍数|lcm]] is the least common multiple. The valid-key condition makes $e$ invertible modulo $\lambda(n)$; the [[扩展欧几里得算法|extended Euclidean algorithm]] finds a positive integer representative $d$ with $ed\equiv1\pmod{\lambda(n)}$. The all-message correctness theorem then supplies decryption. One can also use $\varphi(n)=(p-1)(q-1)$: it has the same prime divisors as this lcm, so $e$ is coprime to it as well.
<!-- bilingual-en:end -->

正向链条如下：[[欧拉函数]] $\varphi(n)$ 统计与 $n$ 互素的余数类，也就是与 $n$ 的[[最大公因数]]为一的类。知道互异素数 $p,q$ 后，就能求 $\lambda(n)=\operatorname{lcm}(p-1,q-1)$，其中 lcm 是[[最小公倍数]]。合法密钥条件保证 $e$ 模 $\lambda(n)$ 可逆；[[扩展欧几里得算法]]求出满足 $ed\equiv1\pmod{\lambda(n)}$ 的正整数代表 $d$，全消息正确性定理随即给出解密。也可以使用 $\varphi(n)=(p-1)(q-1)$：它与该最小公倍数具有相同的素因子，所以 $e$ 与它也互素。

<!-- bilingual-en:start -->
RSA inversion, by contrast, asks for an input that produces a supplied value under the public power map, without assuming that the method first finds $p,q$ or $d$. Producing an answer by one route does not prove that all routes solve the same computational problem. No general equivalence between factoring and RSA inversion is asserted here. [[唯一素因数分解|Unique prime factorization]] guarantees that the prime factors are mathematically determined, not that finding them is fast or slow.
<!-- bilingual-en:end -->

RSA 求逆则问：对于公开幂映射给出的一个值，如何找出产生这个值的输入；不能预设每种方法都先找到 $p,q$ 或 $d$。一条路线能得到答案，不等于所有路线都解决同一个计算问题。这里不主张分解与 RSA 求逆已知一般等价。[[唯一素因数分解]]只保证素因子在数学上确定，并不保证求出它们是快还是慢。

<!-- bilingual-en:start -->
## The raw power map reveals structure
<!-- bilingual-en:end -->

*裸幂映射会暴露结构*

<!-- bilingual-en:start -->
The raw map is deterministic, allowing public comparison against candidate messages. It also preserves multiplication as displayed. These structural properties show why the modular-power formula alone does not provide a complete encryption-security guarantee.
<!-- bilingual-en:end -->

直接使用 $E(m)=m^e\bmod n$ 还存在不依赖分解的结构问题：同一个消息总产生同一个密文，任何人都能加密候选消息进行比较；并且
$$
E(m_1)E(m_2)\equiv E(m_1m_2)\pmod n.
$$
这种确定性与乘法可塑性说明，裸模幂公式本身没有提供完整的加密安全保证。

> [!example]
> <!-- bilingual-en:start -->
> Take the small public key $(n,e)=(55,3)$. If an attacker knows the message is either two or three, they compute $E(2)=8$ and $E(3)=27$ and compare the observed ciphertext with these two values. They need neither the factorization nor $d$. The numbers make the example easy to check; the candidate-comparison method comes from determinism and does not depend on their small size.
> <!-- bilingual-en:end -->
>
> 取小公钥 $(n,e)=(55,3)$。若攻击者知道消息只可能是二或三，就能公开计算 $E(2)=8$、$E(3)=27$，再与截获密文比较，无需分解或知道 $d$。这里的小数便于验算；候选比对来自确定性，并不依赖数值很小。

<!-- bilingual-en:start -->
The multiplication identity follows from $(m_1m_2)^e=m_1^e m_2^e$ and [[模运算|compatibility of remainders with multiplication]]. If $m_1m_2$ is outside the representative range, first take its remainder; this gives the same residue on the right. For the example, $8\cdot27\equiv51\equiv6^3\pmod{55}$. More generally, for an integer $k$, multiplying a ciphertext $c$ by $k^e$ modulo $n$ creates the encryption of the class $km$. This is a predictable relation between messages and ciphertexts; it does not by itself say that an arbitrary unknown $m$ has been recovered.
<!-- bilingual-en:end -->

乘法恒等式来自 $(m_1m_2)^e=m_1^e m_2^e$ 以及[[模运算]]与乘法的兼容性。若 $m_1m_2$ 超出代表范围，先取余，右侧仍给出同一个类。上例中 $8\cdot27\equiv51\equiv6^3\pmod{55}$。一般地，对整数 $k$，把密文 $c$ 乘 $k^e$ 后取模，会得到消息类 $km$ 的密文。这是消息与密文之间可预测的关系，本身并不表示已经恢复任意未知的 $m$。

<!-- bilingual-en:start -->
## Encoding belongs to a specified scheme
<!-- bilingual-en:end -->

*编码属于明确的方案*

<!-- bilingual-en:start -->
RFC 8017 distinguishes RSAES-OAEP encryption from RSASSA-PSS digital signatures. OAEP combines randomized message encoding with RSA primitives; PSS is a signature encoding, not encryption padding. A scheme's guarantees depend on the assumptions and model of its security argument.
<!-- bilingual-en:end -->

RFC 8017 区分两类方案：RSAES-OAEP 把随机化编码与 RSA 原语结合用于加密；RSASSA-PSS 把签名编码与 RSA 原语结合用于数字签名。PSS 不是加密填充。对于具体方案，还必须说明安全论证采用的假设与模型，不能把“使用 RSA”本身当作充分依据。

<!-- bilingual-en:start -->
In OAEP, a byte message is combined with a fresh random seed through the specified encoding procedure before the resulting integer is fed to the RSA power operation. Consequently, the raw candidate-to-ciphertext comparison above does not transfer unchanged to OAEP. Randomness alone is not a proof of security: the encoding and decoding rules, hash and mask-generation functions, and attack model all form part of the statement. RFC 8017’s discussion relates OAEP security to RSA inversion under stated assumptions and a random-oracle model, an idealization in which designated functions return random outputs while answering repeated inputs consistently. It does not prove that factoring is hard.
<!-- bilingual-en:end -->

OAEP 在把字节消息送入 RSA 幂运算之前，按指定过程把消息与新生成的随机种子结合成编码，再转换为整数。因此，上述裸映射的候选—密文直接比对，不能原样搬到 OAEP。仅仅“加了随机性”也不是安全证明：编码与解码规则、哈希及掩码生成函数、攻击模型都是论断的一部分。RFC 8017 在所述假设和随机预言机模型下，把 OAEP 的安全性联系到 RSA 求逆；该模型把指定函数理想化为随机应答、但同一输入始终得到相同输出。它没有证明分解困难。

<!-- bilingual-en:start -->
PSS serves a different goal: the private-key operation produces a signature and a public-key check verifies it against the message. This supports authentication and integrity under its assumptions; it is not a method for hiding the message from someone with the public key. Treating “OAEP or PSS” as interchangeable encryption padding loses this distinction.
<!-- bilingual-en:end -->

PSS 的目标不同：私钥运算产生签名，公钥检查它是否与消息相符；在相应假设下，它服务于认证与完整性，不是向持有公钥的人隐藏消息的方法。把“OAEP 或 PSS”当成可互换的加密填充，会丢掉这一区别。

<!-- bilingual-en:start -->
## A reduction is a conditional implication
<!-- bilingual-en:end -->

*归约表达条件蕴含*

<!-- bilingual-en:start -->
The course reduction to SAT shows that an efficient general SAT algorithm would give efficient factoring. The reduction itself does not establish that either problem is easy in practice.
<!-- bilingual-en:end -->

课程把分解编码成 SAT，说明一个高效通用 SAT 算法会带来高效分解算法；这项归约本身没有证明现实中的分解或 SAT 已经容易。

<!-- bilingual-en:start -->
SAT asks whether some truth-value assignment makes a Boolean formula true. The course builds a multiplication-checking circuit, encodes it by a formula, and uses satisfiability queries while fixing successive factor bits. Thus a sufficiently efficient general SAT procedure would provide an efficient factoring method. This explains the direction of the implication. It neither gives such a SAT procedure nor turns the converse into a theorem about every RSA attack.
<!-- bilingual-en:end -->

SAT 问一个布尔公式是否存在使它为真的真值赋值。课程先构造检查乘法的电路，再把电路编码为公式，并在逐位确定因子时询问可满足性。因此，足够高效的通用 SAT 过程会给出高效分解方法。这解释了蕴含的方向，并没有给出这样的 SAT 过程，也没有把逆命题变成关于所有 RSA 攻击的定理。

> [!question]
> <!-- bilingual-en:start -->
> A successful numerical encryption/decryption test checks an instance of correctness. It supplies no evidence that an attacker lacks a different route. When reading a security claim, identify the attacker’s task, the available information, the assumed hard problem, and the actual scheme. Keep those separate from the algebraic recovery proof.
> <!-- bilingual-en:end -->
>
> 一次数值加密—解密成功，只检查了正确性的一个实例，并不提供“攻击者没有其他路线”的证据。阅读安全主张时，先辨明攻击者的任务、可见信息、假设困难的问题和实际方案，再与代数还原证明区分。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT printed pp. 281–282 / PDF pp. 290–291 and FactoringSAT verify the factoring and SAT implications without assuming their converses. RFC 8017 verifies the primitives and the distinct roles of OAEP and PSS. Determinism and multiplicativity follow directly from the primitive formula.
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=290|MIT 6.042J 教材，印刷第 281–282 页／PDF 第 290–291 页]]：核对分解可恢复私钥、绕过私钥解密与分解的关系未获等价结论，以及 SAT 归约方向；[[MIT6_042JS15_FactoringSAT.pdf#page=1|FactoringSAT，PDF 第 1–3 页]] 支持乘法电路与逐位 SAT 检验。
- [IETF RFC 8017 §5.1、§7.1、§8.1](https://www.rfc-editor.org/rfc/rfc8017.html#section-7.1)：核对 RSA 原语、OAEP 加密与 PSS 签名的职责及有条件的安全论证。确定性与乘法性质由原语公式直接核算；本卡不沿用教材带年代的攻击历史概括。

<!-- bilingual-en:start -->
The RFC is cited for its exact primitive/scheme definitions and qualified security arguments, not for a proof of the hardness assumption. MIT textbook PDF290–291 and the three FactoringSAT pages are read for the direction of the factoring and SAT reductions. The textbook’s dated claims about attack history are not used as present-day security evidence. The two-candidate and multiplication examples are derived directly above.
<!-- bilingual-en:end -->

RFC 用于核对原语／方案的精确定义及附条件的安全论证，不作为计算困难假设已被证明的依据。MIT 教材 PDF290–291 与 FactoringSAT 三页用于核对分解和 SAT 归约方向。教材带年代的攻击历史说法不作为当今安全证据；两个候选及乘法例子均在正文直接推导。
