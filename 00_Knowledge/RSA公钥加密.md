---
aliases:
  - RSA public-key encryption
  - Textbook RSA
  - 教科书RSA
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

<!-- bilingual-en:start -->
# RSA public-key encryption
<!-- bilingual-en:end -->

*RSA公钥加密*

<!-- bilingual-en:start -->
RSA pairs two modular powers. The sender uses public key $(n,e)$ to transform the integer representative $m$ into $c=m^e\bmod n$; the recipient uses the private exponent $d$ to recover $m$. Representatives lie between zero and $n-1$. A concrete encoding scheme maps a byte message to such an integer.
<!-- bilingual-en:end -->

RSA 公钥加密把两种模幂运算配成一对：发送方知道公钥 $(n,e)$，把整数消息代表 $m$ 编码为 $c=m^e\bmod n$；接收方保有私钥指数 $d$，计算 $c^d\bmod n$ 恢复 $m$。这里的 $m,c$ 都选在 $\{0,1,\ldots,n-1\}$；文字或字节消息需要先由具体编码方案变成这样的整数。

<!-- bilingual-en:start -->
The roles explain the word public: the receiver can publish the information needed to transform a message while keeping the information used to recover it. This card describes the integer arithmetic underlying that arrangement. The mathematical round trip and the security of a complete encryption scheme are separate questions.
<!-- bilingual-en:end -->

“公钥”描述的是两种角色：接收方可以公开变换消息所需的信息，同时保留恢复消息所用的信息。本卡解释这套安排背后的整数运算；能否正确往返，以及完整加密方案是否安全，是两个不同的问题。

<!-- bilingual-en:start -->
A [[素数|prime]] is an integer greater than one with exactly two positive divisors. The [[欧拉函数|totient]] $\varphi(n)$ counts remainder classes coprime to $n$. Here [[模运算|modular exponentiation]] takes an integer power and keeps its standard remainder. The [[模逆元存在条件|inverse criterion]] says that an integer has a multiplicative inverse modulo a modulus greater than one exactly when their [[最大公因数|gcd]] is one.
<!-- bilingual-en:end -->

[[素数]]是大于一且正因子恰有两个的整数；[[欧拉函数]] $\varphi(n)$ 统计与 $n$ 互素的余数类个数。这里的[[模运算|模幂]]是计算整数幂后取标准余数。[[模逆元存在条件]]说明：模数大于一时，一个整数可逆当且仅当它与模数的[[最大公因数]]为一。

<!-- bilingual-en:start -->
## Construct the paired exponents
<!-- bilingual-en:end -->

*构造配对指数*

<!-- bilingual-en:start -->
Write $p,q$ for the chosen distinct odd primes, $n=pq$ for their product, and $\varphi(n)=(p-1)(q-1)$. Choose an integer $e$ with $1<e<\varphi(n)$ and $\gcd(e,\varphi(n))=1$. The displayed congruence below asks for a positive integer $d$ whose product with $e$ has remainder one modulo $\varphi(n)$.
<!-- bilingual-en:end -->

记选出的两个不同奇素数为 $p,q$，乘积为 $n=pq$，欧拉函数值为 $\varphi(n)=(p-1)(q-1)$。选整数 $e$，要求 $1<e<\varphi(n)$ 且 $\gcd(e,\varphi(n))=1$。下面的同余式要找正整数 $d$，使它与 $e$ 的乘积除以 $\varphi(n)$ 余一。

<!-- bilingual-en:start -->
In the two-prime teaching construction, choose distinct odd primes, form $n$ and its totient, and choose $e$ coprime to the totient. The [[扩展欧几里得算法|extended Euclidean algorithm]] gives the inverse exponent. Publish $n,e$ and retain $d,p,q$ privately. The [[欧拉函数乘积公式|totient product formula]] gives $(p-1)(q-1)$ directly because the recipient already knows the two chosen prime factors.
<!-- bilingual-en:end -->

两素数的教学构造如下：取不同奇素数 $p,q$，令 $n=pq$、$\varphi(n)=(p-1)(q-1)$；选 $1<e<\varphi(n)$ 且 $\gcd(e,\varphi(n))=1$，再用 [[扩展欧几里得算法]] 求正整数 $d$ 满足
$$
ed\equiv1\pmod{\varphi(n)}.
$$
公开 $n,e$，保密 $d,p,q$。这里的 $\varphi(pq)=(p-1)(q-1)$ 是 [[欧拉函数乘积公式]] 的两素数情形；接收方已知自己选择的两个素因子，因此能够直接计算它。

<!-- bilingual-en:start -->
The inverse is taken modulo $\varphi(n)$, not modulo $n$: $ed=1+h\varphi(n)$ for an integer $h$. Applying the two powers multiplies exponents, so $(m^e)^d=m^{ed}$ before taking remainders. The role of $d$ is to undo the exponentiation on the allowed message representatives; it is not the ordinary fraction $1/e$. The [[扩展欧几里得算法|extended Euclidean algorithm]] supplies coefficients $s,t$ with $se+t\varphi(n)=1$. The standard remainder of $s$ modulo $\varphi(n)$ gives the positive inverse $d$.
<!-- bilingual-en:end -->

这里是模 $\varphi(n)$ 求逆，不是模 $n$ 求逆：存在整数 $h$ 使 $ed=1+h\varphi(n)$。先后做两次幂运算会把指数相乘，取余前有 $(m^e)^d=m^{ed}$。$d$ 的作用是对允许的消息代表撤销幂变换，不是普通分数 $1/e$。[[扩展欧几里得算法]]给出 $se+t\varphi(n)=1$ 的整数系数 $s,t$，把 $s$ 对 $\varphi(n)$ 取标准余数，就得到正逆元 $d$。

<!-- bilingual-en:start -->
## Follow a complete small example
<!-- bilingual-en:end -->

*走完一个小例子*

<!-- bilingual-en:start -->
For the numerical example $p=5,q=11$, choose $e=3,d=27$, since their product is one modulo forty. Representative twelve encrypts to twenty-three and decrypts to twelve. [[模重复平方法|Repeated squaring]] evaluates the powers efficiently.
<!-- bilingual-en:end -->

例如取 $p=5,q=11$，则 $n=55$、$\varphi(n)=40$。选 $e=3,d=27$，因为 $3\cdot27=81\equiv1\pmod{40}$。消息代表 $m=12$ 给出 $c=12^3\bmod55=23$，解码 $23^{27}\bmod55=12$；大指数用 [[模重复平方法]] 计算。

> [!example] Where 27 comes from · 27 从哪里来
> <!-- bilingual-en:start -->
> With $\varphi(55)=40$ and $e=3$, division gives $40=13\cdot3+1$. Rearranging yields $1=40-13\cdot3$, so the coefficient of three is $-13$. Its remainder modulo forty is $27$: $3\cdot27-2\cdot40=1$. Thus $(55,3)$ is the public pair and $27$ is the chosen private exponent. These tiny numbers illustrate the arithmetic only.
> <!-- bilingual-en:end -->
>
> 当 $\varphi(55)=40$、$e=3$ 时，带余除法给出 $40=13\cdot3+1$，即 $1=40-13\cdot3$。三的系数是 $-13$，模四十的标准余数为 $27$：$3\cdot27-2\cdot40=1$。因此公钥对是 $(55,3)$，选取的私钥指数是 $27$。这些小数只用于展示运算。

<!-- bilingual-en:start -->
Encryption is $12^3=1728=31\cdot55+23$. To verify decryption by [[模重复平方法|repeated squaring]], the remainders of $23^1,23^2,23^4,23^8,23^{16}$ modulo fifty-five are $23,34,1,1,1$. Since $27=16+8+2+1$, the selected product reduces to $1\cdot1\cdot34\cdot23=782=14\cdot55+12$. This checks the numerical round trip without expanding $23^{27}$.
<!-- bilingual-en:end -->

加密时 $12^3=1728=31\cdot55+23$。用[[模重复平方法]]核对解密，$23^1,23^2,23^4,23^8,23^{16}$ 模五十五的余数依次为 $23,34,1,1,1$。因为 $27=16+8+2+1$，所选乘积约成 $1\cdot1\cdot34\cdot23=782=14\cdot55+12$。这样无需展开 $23^{27}$，就核对了这次往返。

<!-- bilingual-en:start -->
## Representatives, correctness and complete schemes
<!-- bilingual-en:end -->

*代表元、正确性与完整方案*

<!-- bilingual-en:start -->
This constructs the modular-exponentiation primitive. [[RSA全消息正确性|All-message correctness]] proves inversion, while [[RSA安全边界|the security boundary]] separates that fact from randomized encoding and security assumptions. The standard condition uses the [[最小公倍数|least common multiple]] of $p-1$ and $q-1$; inversion modulo $\varphi(n)$ implies inversion modulo $\lambda(n)$.
<!-- bilingual-en:end -->

这是模幂原语的数学构造。[[RSA全消息正确性]] 证明它为何能还原全部消息代表；随机化编码和安全假设由 [[RSA安全边界]] 区分。标准条件也可写为 $ed\equiv1\pmod{\lambda(n)}$，其中两素数情形的 $\lambda(n)$ 是 $p-1$ 与 $q-1$ 的 [[最小公倍数]]；上述按 $\varphi(n)$ 求逆的构造满足这个条件。

<!-- bilingual-en:start -->
Let $L=\operatorname{lcm}(p-1,q-1)$. Both $p-1$ and $q-1$ divide $\varphi(n)$, and their [[最小公倍数|least common multiple]] therefore divides $\varphi(n)$. Consequently $\varphi(n)\mid(ed-1)$ implies $L\mid(ed-1)$. This explains the implication to the standard lambda condition without claiming that $\varphi(n)$ and $L$ are always equal. In the example, $L=\operatorname{lcm}(4,10)=20$, whereas $\varphi(55)=40$.
<!-- bilingual-en:end -->

令 $L=\operatorname{lcm}(p-1,q-1)$。$p-1$ 与 $q-1$ 都整除 $\varphi(n)$，所以它们的[[最小公倍数]]也整除 $\varphi(n)$。因此 $\varphi(n)\mid(ed-1)$ 推出 $L\mid(ed-1)$。这解释了前述构造为什么满足标准的 lambda 条件，并没有声称 $\varphi(n)$ 与 $L$ 总是相等；例子里 $L=\operatorname{lcm}(4,10)=20$，而 $\varphi(55)=40$。

> [!warning] Know what the arithmetic accepts · 明确运算接收什么
> <!-- bilingual-en:start -->
> The integer primitive accepts $0\le m<n$, not an arbitrary-length sentence or an out-of-range integer. Reducing an out-of-range input first would identify distinct inputs, such as $12$ and $67$ modulo fifty-five. A complete encryption scheme must specify how a byte message becomes an admissible representative and is recovered. The zero representative and representatives sharing a prime factor with $n$ are allowed; the proof for those cases is in [[RSA全消息正确性|all-message correctness]].
> <!-- bilingual-en:end -->
>
> 整数原语接收 $0\le m<n$ 的代表，不直接接收任意长句子或超出范围的整数。如果先把越界输入随意取余，就会把不同输入合并，例如模五十五的 $12$ 与 $67$。完整加密方案必须规定字节消息如何成为合法代表、又如何还原。零代表以及与 $n$ 共享素因子的代表都允许；这些情况的证明见[[RSA全消息正确性]]。

<!-- bilingual-en:start -->
The public-key construction here is a teaching version using two primes and a totient inverse. RFC 8017 distinguishes the RSAEP/RSADP integer primitives from encryption schemes built around them; its valid-key conditions use the least common multiple of the prime factors minus one. Correct arithmetic by itself does not establish confidentiality; the exact distinction is developed in [[RSA安全边界|the RSA security boundary]].
<!-- bilingual-en:end -->

本卡的公钥构造采用两个素数和欧拉函数模数下的逆元，是教学版本。RFC 8017 区分 RSAEP/RSADP 整数原语与围绕它们构建的加密方案，合法密钥条件使用各素因子减一的最小公倍数。运算正确本身还不能证明保密性，具体区别见[[RSA安全边界]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT Lemma 8.10.9 verifies the totient count; §8.11 and the RSA slides verify the key construction and powers. RFC 8017 §§3 and 5.1 verify the lambda condition and RSA primitives. The small numerical example was checked directly.
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=287|MIT 6.042J 教材，Lemma 8.10.9，印刷第 278 页／PDF 第 287 页]]：核对不同素数的欧拉函数计数；[[MIT6_042JS15_textbook.pdf#page=289|§8.11，印刷第 280 页／PDF 第 289 页]] 与 [[MIT6_042JS15_RSA_Encytion.pdf#page=2|RSA Encryption，PDF 第 2–3 页]] 支持密钥、整数范围及两次模幂。
- [IETF RFC 8017 §3.1–3.2、§5.1](https://www.rfc-editor.org/rfc/rfc8017.html#section-3)：核对标准密钥采用 $\lambda(n)$ 的条件以及 RSAEP/RSADP 原语；小整数算例逐次取模核算。

<!-- bilingual-en:start -->
- The source scope is deliberate: the slides illustrate the construction and restrict their displayed message range to $[1,n)$; the textbook includes zero. RFC 8017 §§3, 5.1 and 6 support the standard key condition, integer ranges, and primitive/scheme distinction. The coefficient calculation and both modular powers above are derived explicitly here.
<!-- bilingual-en:end -->

- 来源范围已分别核对：课件展示构造，其显示的消息范围为 $[1,n)$；教材包含零。RFC 8017 的 §3、§5.1 与 §6 支持标准密钥条件、整数范围和原语／方案区分。上面的系数求法与两次模幂均在正文逐步推导。

