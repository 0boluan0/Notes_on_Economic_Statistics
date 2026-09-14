---
aliases:
  - "RSA逆指数条件通过逐素数分类与中国剩余定理保证所有消息代表都能还原"
  - RSA correctness for all message representatives
student_os: knowledge-atom
atom_id: MCS-NT-022
atom_type: theorem
status: source-checked
part_of:
  - "[[数论与RSA.canvas]]"
requires:
  - "[[RSA公钥加密]]"
  - "[[费马小定理]]"
  - "[[中国剩余定理]]"
  - "[[最小公倍数]]"
related:
  - "[[欧拉定理]]"
  - "[[RSA安全边界]]"
---

# RSA逆指数条件通过逐素数分类与中国剩余定理保证所有消息代表都能还原
<!-- bilingual-en:start -->
*RSA's inverse-exponent condition recovers every message representative by primewise cases and the Chinese remainder theorem*
<!-- bilingual-en:end -->

设 $p,q$ 为不同素数，$n=pq$，$e,d$ 为正整数，且
$$
ed\equiv1\pmod{\lambda(n)},
\qquad\lambda(n)=\operatorname{lcm}(p-1,q-1).
$$
其中 $\lambda(n)$ 是 $p-1$ 与 $q-1$ 的 [[最小公倍数]]。则对任意整数 $m$，$m^{ed}\equiv m\pmod n$。因此当 $m\in\{0,\ldots,n-1\}$ 时，先计算 $c=m^e\bmod n$ 再计算 $c^d\bmod n$，得到的整数代表恰好是 $m$，包括与 $n$ 不互素的消息。
<!-- bilingual-en:start -->
Here $\lambda(n)$ is the [[最小公倍数|least common multiple]] of $p-1$ and $q-1$. For distinct primes and positive exponents satisfying the displayed condition, $m^{ed}\equiv m\pmod n$ holds for every integer $m$. Encrypting and decrypting returns the exact representative between zero and $n-1$, including noncoprime representatives.
<!-- bilingual-en:end -->

先固定 $r=p$ 或 $r=q$。由于 $r-1\mid\lambda(n)$，可写 $ed=1+t_r(r-1)$，其中 $t_r\ge0$。若 $r\mid m$，则 $m^{ed}\equiv0\equiv m\pmod r$，因为 $ed\ge1$。若 $r\nmid m$，用 [[费马小定理]] 得
$$
m^{ed}=m\bigl(m^{r-1}\bigr)^{t_r}\equiv m\pmod r.
$$
两种情形覆盖全部 $m$。
<!-- bilingual-en:start -->
Fix either prime $r$. Its predecessor divides $\lambda(n)$, so $ed=1+t_r(r-1)$ with $t_r\ge0$. If $r$ divides $m$, the positive exponent gives zero on both sides. Otherwise, [[费马小定理|Fermat's little theorem]] yields the displayed congruence. These cases exhaust all messages.
<!-- bilingual-en:end -->

于是同时有 $m^{ed}\equiv m\pmod p$ 和 $m^{ed}\equiv m\pmod q$。由于 $\gcd(p,q)=1$，[[中国剩余定理]] 保证两者合并成模 $pq$ 的同余。最后，$c\equiv m^e\pmod n$ 蕴含 $c^d\equiv m^{ed}\pmod n$，而区间 $[0,n)$ 中的余数代表唯一，所以解密返回原整数。
<!-- bilingual-en:start -->
The [[中国剩余定理|Chinese remainder theorem]] combines the congruences modulo the two coprime primes into one modulo their product. Since $c\equiv m^e$, exponentiation gives $c^d\equiv m^{ed}$. Uniqueness of the representative in $[0,n)$ establishes exact recovery.
<!-- bilingual-en:end -->

教学构造使用 $ed\equiv1\pmod{(p-1)(q-1)}$，这是充分条件，因为 $\lambda(n)$ 整除 $(p-1)(q-1)$。证明不能直接对所有 $m$ 套 [[欧拉定理]]：例如 $n=55,e=3,d=27,m=5$ 不满足互素前提，但加密得到 $15$、解密仍得到 $5$。另外，不同素数这一条件很关键；把模数换成 $p^2$ 后，零余数分类与 CRT 合并不再是同一个证明。
<!-- bilingual-en:start -->
Inverting modulo $(p-1)(q-1)$ is sufficient because its lcm divides that product. [[欧拉定理|Euler's theorem]] alone misses noncoprime representatives: with $n=55,e=3,d=27$, message five encrypts to fifteen and still decrypts correctly. Distinct primes matter; replacing the modulus by a prime square does not preserve this proof.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=288|MIT 6.042J 教材，§8.11，印刷第 279 页／PDF 第 288 页]]：明确解密对不互素消息也成立；[[MIT6_042JS15_cp15.pdf#page=2|MIT 6.042J CP15，Problem 3(b)–(d)，PDF 第 2 页]] 给出逐素数证明再合并为不同素数乘积的路线。本卡以 CRT 说明其中的合并步骤，正指数条件明确排除零指数。
- [IETF RFC 8017 §3.1–3.2](https://www.rfc-editor.org/rfc/rfc8017.html#section-3)：核对不同素因子与 $ed\equiv1\pmod{\lambda(n)}$ 的标准条件。本卡逐项核对两个素数下的零／非零情形、正指数和唯一代表；$m=5$ 算例独立验算。
<!-- bilingual-en:start -->
- MIT §8.11 explicitly includes noncoprime messages. CP15 Problem 3(b)–(d), PDF page 2, supplies the primewise proof route and the distinct-prime product condition; this note uses CRT for the combination step. RFC 8017 §3 verifies the distinct-prime and lambda conditions. Both residue cases, positive exponents, and uniqueness of the recovered representative were checked directly.
<!-- bilingual-en:end -->
