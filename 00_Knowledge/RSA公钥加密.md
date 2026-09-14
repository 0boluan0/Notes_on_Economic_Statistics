---
aliases:
  - "RSA公钥加密以公开的模幂运算编码并以保密的逆指数解码"
  - RSA public-key encryption
  - Textbook RSA
student_os: knowledge-atom
atom_id: MCS-NT-021
atom_type: definition
status: source-checked
part_of:
  - "[[数论与RSA.canvas]]"
requires:
  - "[[素数]]"
  - "[[欧拉函数]]"
  - "[[模逆元存在条件]]"
  - "[[扩展欧几里得算法]]"
  - "[[欧拉函数乘积公式]]"
related:
  - "[[模重复平方法]]"
leads_to:
  - "[[RSA全消息正确性]]"
  - "[[RSA安全边界]]"
---

# RSA公钥加密以公开的模幂运算编码并以保密的逆指数解码
<!-- bilingual-en:start -->
*RSA public-key encryption encodes by a public modular power and decodes using a private inverse exponent*
<!-- bilingual-en:end -->

RSA 公钥加密把两种模幂运算配成一对：发送方知道公钥 $(n,e)$，把整数消息代表 $m$ 编码为 $c=m^e\bmod n$；接收方保有私钥指数 $d$，计算 $c^d\bmod n$ 恢复 $m$。这里的 $m,c$ 都选在 $\{0,1,\ldots,n-1\}$；文字或字节消息需要先由具体编码方案变成这样的整数。
<!-- bilingual-en:start -->
RSA pairs two modular powers. The sender uses public key $(n,e)$ to transform the integer representative $m$ into $c=m^e\bmod n$; the recipient uses the private exponent $d$ to recover $m$. Representatives lie between zero and $n-1$. A concrete encoding scheme maps a byte message to such an integer.
<!-- bilingual-en:end -->

两素数的教学构造如下：取不同奇素数 $p,q$，令 $n=pq$、$\varphi(n)=(p-1)(q-1)$；选 $1<e<\varphi(n)$ 且 $\gcd(e,\varphi(n))=1$，再用 [[扩展欧几里得算法]] 求正整数 $d$ 满足
$$
ed\equiv1\pmod{\varphi(n)}.
$$
公开 $n,e$，保密 $d,p,q$。这里的 $\varphi(pq)=(p-1)(q-1)$ 是 [[欧拉函数乘积公式]] 的两素数情形；接收方已知自己选择的两个素因子，因此能够直接计算它。
<!-- bilingual-en:start -->
In the two-prime teaching construction, choose distinct odd primes, form $n$ and its totient, and choose $e$ coprime to the totient. The [[扩展欧几里得算法|extended Euclidean algorithm]] gives the inverse exponent. Publish $n,e$ and retain $d,p,q$ privately. The [[欧拉函数乘积公式|totient product formula]] gives $(p-1)(q-1)$ directly because the recipient already knows the two chosen prime factors.
<!-- bilingual-en:end -->

例如取 $p=5,q=11$，则 $n=55$、$\varphi(n)=40$。选 $e=3,d=27$，因为 $3\cdot27=81\equiv1\pmod{40}$。消息代表 $m=12$ 给出 $c=12^3\bmod55=23$，解码 $23^{27}\bmod55=12$；大指数用 [[模重复平方法]] 计算。
<!-- bilingual-en:start -->
For the numerical example $p=5,q=11$, choose $e=3,d=27$, since their product is one modulo forty. Representative twelve encrypts to twenty-three and decrypts to twelve. [[模重复平方法|Repeated squaring]] evaluates the powers efficiently.
<!-- bilingual-en:end -->

这是模幂原语的数学构造。[[RSA全消息正确性]] 证明它为何能还原全部消息代表；随机化编码和安全假设由 [[RSA安全边界]] 区分。标准条件也可写为 $ed\equiv1\pmod{\lambda(n)}$，其中两素数情形的 $\lambda(n)$ 是 $p-1$ 与 $q-1$ 的 [[最小公倍数]]；上述按 $\varphi(n)$ 求逆的构造满足这个条件。
<!-- bilingual-en:start -->
This constructs the modular-exponentiation primitive. [[RSA全消息正确性|All-message correctness]] proves inversion, while [[RSA安全边界|the security boundary]] separates that fact from randomized encoding and security assumptions. The standard condition uses the [[最小公倍数|least common multiple]] of $p-1$ and $q-1$; inversion modulo $\varphi(n)$ implies inversion modulo $\lambda(n)$.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=287|MIT 6.042J 教材，Lemma 8.10.9，印刷第 278 页／PDF 第 287 页]]：核对不同素数的欧拉函数计数；[[MIT6_042JS15_textbook.pdf#page=289|§8.11，印刷第 280 页／PDF 第 289 页]] 与 [[MIT6_042JS15_RSA_Encytion.pdf#page=2|RSA Encryption，PDF 第 2–3 页]] 支持密钥、整数范围及两次模幂。
- [IETF RFC 8017 §3.1–3.2、§5.1](https://www.rfc-editor.org/rfc/rfc8017.html#section-3)：核对标准密钥采用 $\lambda(n)$ 的条件以及 RSAEP/RSADP 原语；小整数算例逐次取模核算。
<!-- bilingual-en:start -->
- MIT Lemma 8.10.9 verifies the totient count; §8.11 and the RSA slides verify the key construction and powers. RFC 8017 §§3 and 5.1 verify the lambda condition and RSA primitives. The small numerical example was checked directly.
<!-- bilingual-en:end -->
