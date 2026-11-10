---
aliases:
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

<!-- bilingual-en:start -->
# RSA correctness for every message
<!-- bilingual-en:end -->

*RSA全消息正确性*

<!-- bilingual-en:start -->
RSA correctness is a round-trip statement: encrypt an allowed integer representative with the public exponent, then decrypt with the paired private exponent, and the original representative returns. The claim includes zero and messages sharing a prime factor with the modulus. Those cases need their own step in the proof.
<!-- bilingual-en:end -->

RSA 正确性说明一次往返：把允许的整数消息代表用公开指数加密，再用配对的私钥指数解密，就回到原代表。这个结论包含零，也包含与模数共享素因子的消息；证明必须处理这些情况。

<!-- bilingual-en:start -->
In [[RSA公钥加密|RSA public-key encryption]], $n$ is the modulus, $e$ the public exponent, $d$ the private exponent, $m$ a message representative and $c$ its ciphertext representative. [[同余|Congruence]] $x\equiv y\pmod n$ means $n$ divides $x-y$; [[模运算|modular arithmetic]] lets us multiply such congruences and take positive integer powers. The notation $x\bmod n$ selects the unique remainder in $0,\ldots,n-1$.
<!-- bilingual-en:end -->

在[[RSA公钥加密]]中，$n$ 是模数，$e$ 是公开指数，$d$ 是私钥指数，$m$ 是消息代表，$c$ 是密文代表。[[同余]] $x\equiv y\pmod n$ 表示 $n$ 整除 $x-y$；[[模运算]]允许把这些同余式相乘并取正整数幂。记号 $x\bmod n$ 选出 $0,\ldots,n-1$ 中唯一的余数。

<!-- bilingual-en:start -->
## The condition on the exponents
<!-- bilingual-en:end -->

*指数要满足的条件*

<!-- bilingual-en:start -->
Choose two distinct [[素数|primes]] $p,q$ and put $n=pq$. Define $\lambda(n)$ here as the least common multiple of their predecessors. Its role is to provide an exponent increment divisible by both $p-1$ and $q-1$. Assume $e,d$ are positive integers and satisfy the following condition.
<!-- bilingual-en:end -->

取两个不同的[[素数]] $p,q$，令 $n=pq$。这里把 $\lambda(n)$ 定义为两素数各减一后的最小公倍数；它给出一个同时被 $p-1$ 和 $q-1$ 整除的指数步长。假定 $e,d$ 是正整数，并满足下面的条件。

<!-- bilingual-en:start -->
Here $\lambda(n)$ is the [[最小公倍数|least common multiple]] of $p-1$ and $q-1$. For distinct primes and positive exponents satisfying the displayed condition, $m^{ed}\equiv m\pmod n$ holds for every integer $m$. Encrypting and decrypting returns the exact representative between zero and $n-1$, including noncoprime representatives.
<!-- bilingual-en:end -->

设 $p,q$ 为不同素数，$n=pq$，$e,d$ 为正整数，且
$$
ed\equiv1\pmod{\lambda(n)},
\qquad\lambda(n)=\operatorname{lcm}(p-1,q-1).
$$
其中 $\lambda(n)$ 是 $p-1$ 与 $q-1$ 的 [[最小公倍数]]。则对任意整数 $m$，$m^{ed}\equiv m\pmod n$。因此当 $m\in\{0,\ldots,n-1\}$ 时，先计算 $c=m^e\bmod n$ 再计算 $c^d\bmod n$，得到的整数代表恰好是 $m$，包括与 $n$ 不互素的消息。

<!-- bilingual-en:start -->
The inverse condition is imposed on the exponents modulo $\lambda(n)$, while the message calculation is modulo $n$. These are different moduli. A [[模逆元|modular inverse]] means a product congruent to one, not the ordinary fraction $1/e$. The condition ensures $ed-1=k\lambda(n)$ for a nonnegative integer $k$, since $ed\ge1$.
<!-- bilingual-en:end -->

求逆条件作用在指数上，模数是 $\lambda(n)$；消息运算的模数则是 $n$，二者不同。[[模逆元]]表示乘积同余于一，不是普通分数 $1/e$。由于 $ed\ge1$，该条件使 $ed-1=k\lambda(n)$，其中 $k$ 是非负整数。

<!-- bilingual-en:start -->
## Recover the message modulo each prime
<!-- bilingual-en:end -->

*先在每个素数模数下恢复*

<!-- bilingual-en:start -->
Fix either prime $r$. Its predecessor divides $\lambda(n)$, so $ed=1+t_r(r-1)$ with $t_r\ge0$. If $r$ divides $m$, the positive exponent gives zero on both sides. Otherwise, [[费马小定理|Fermat's little theorem]] yields the displayed congruence. These cases exhaust all messages.
<!-- bilingual-en:end -->

先固定 $r=p$ 或 $r=q$。由于 $r-1\mid\lambda(n)$，可写 $ed=1+t_r(r-1)$，其中 $t_r\ge0$。若 $r\mid m$，则 $m^{ed}\equiv0\equiv m\pmod r$，因为 $ed\ge1$。若 $r\nmid m$，用 [[费马小定理]] 得
$$
m^{ed}=m\bigl(m^{r-1}\bigr)^{t_r}\equiv m\pmod r.
$$
两种情形覆盖全部 $m$。

<!-- bilingual-en:start -->
Why does the nonzero case meet Fermat’s hypothesis? A prime has no positive divisors other than one and itself. If $r\nmid m$, then their [[最大公因数|greatest common divisor]] is $\gcd(m,r)=1$, so $m^{r-1}\equiv1\pmod r$. Raising this to the nonnegative power $t_r$ leaves one; multiplying by $m$ restores $m$. If $r\mid m$, do not replace $m^{r-1}$ by one: every positive power of $m$ is instead divisible by $r$. This also handles $m=0$ without ever using $0^0$.
<!-- bilingual-en:end -->

非零情形为什么满足费马定理的前提？素数的正因子只有一和自身；若 $r\nmid m$，它们的[[最大公因数]]便是 $\gcd(m,r)=1$，所以 $m^{r-1}\equiv1\pmod r$。将其取非负整数次幂 $t_r$ 仍余一，再乘 $m$ 就恢复 $m$。若 $r\mid m$，则不能把 $m^{r-1}$ 换成一；此时 $m$ 的每个正幂都被 $r$ 整除。这也涵盖 $m=0$，从未使用 $0^0$。

<!-- bilingual-en:start -->
## Combine the two recoveries
<!-- bilingual-en:end -->

*把两个恢复结论合并*

<!-- bilingual-en:start -->
The [[中国剩余定理|Chinese remainder theorem]] combines the congruences modulo the two coprime primes into one modulo their product. Since $c\equiv m^e$, exponentiation gives $c^d\equiv m^{ed}$. Uniqueness of the representative in $[0,n)$ establishes exact recovery.
<!-- bilingual-en:end -->

于是同时有 $m^{ed}\equiv m\pmod p$ 和 $m^{ed}\equiv m\pmod q$。由于 $\gcd(p,q)=1$，[[中国剩余定理]] 保证两者合并成模 $pq$ 的同余。最后，$c\equiv m^e\pmod n$ 蕴含 $c^d\equiv m^{ed}\pmod n$，而区间 $[0,n)$ 中的余数代表唯一，所以解密返回原整数。

<!-- bilingual-en:start -->
More explicitly, $m^{ed}$ and $m$ have the same two remainders modulo $p$ and $q$. The uniqueness part of CRT says they therefore belong to the same class modulo $pq$. Replacing $m^e$ by its standard remainder $c$ does not change any positive power modulo $n$. Finally, two integers in $0,\ldots,n-1$ whose difference is a multiple of $n$ must be equal: their difference has absolute value less than $n$. This last step turns a congruence into exact recovery of the chosen integer.
<!-- bilingual-en:end -->

具体而言，$m^{ed}$ 与 $m$ 模 $p,q$ 的两个余数都相同；CRT 的唯一性说明它们属于同一个模 $pq$ 的类。把 $m^e$ 换成标准余数 $c$，不会改变其正整数次幂模 $n$ 的结果。最后，$0,\ldots,n-1$ 中两个整数若相差 $n$ 的倍数，就必定相等，因为差的绝对值小于 $n$。这一步把同余结论转成对所选整数的精确恢复。

<!-- bilingual-en:start -->
## A noncoprime message and the totient condition
<!-- bilingual-en:end -->

*非互素消息与欧拉函数条件*

<!-- bilingual-en:start -->
Inverting modulo $(p-1)(q-1)$ is sufficient because its lcm divides that product. [[欧拉定理|Euler's theorem]] alone misses noncoprime representatives: with $n=55,e=3,d=27$, message five encrypts to fifteen and still decrypts correctly. Distinct primes matter; replacing the modulus by a prime square does not preserve this proof.
<!-- bilingual-en:end -->

教学构造使用 $ed\equiv1\pmod{(p-1)(q-1)}$，这是充分条件，因为 $\lambda(n)$ 整除 $(p-1)(q-1)$。证明不能直接对所有 $m$ 套 [[欧拉定理]]：例如 $n=55,e=3,d=27,m=5$ 不满足互素前提，但加密得到 $15$、解密仍得到 $5$。另外，不同素数这一条件很关键；把模数换成 $p^2$ 后，零余数分类与 CRT 合并不再是同一个证明。

> [!example]
> <!-- bilingual-en:start -->
> For that example, [[欧拉函数|Euler’s totient]] counts the classes coprime to the modulus, while the lcm gives the common exponent step. The primes $p=5,q=11$ give $\lambda(55)=20$ and $\varphi(55)=40$. Encryption is $5^3=125=2\cdot55+15$. For decryption, the remainders of $15^1,15^2,15^4,15^8,15^{16}$ modulo fifty-five are $15,5,25,20,15$. Since $27=16+8+2+1$, their selected product is $15\cdot20\cdot5\cdot15=22500=409\cdot55+5$. The primewise proof explains the result: modulo five the message is zero; modulo eleven it is nonzero.
> <!-- bilingual-en:end -->
>
> 在这个例子里，[[欧拉函数]]统计与模数互素的类，lcm 给出共同的指数步长。素数 $p=5,q=11$ 给出 $\lambda(55)=20$、$\varphi(55)=40$。加密为 $5^3=125=2\cdot55+15$。解密时，$15^1,15^2,15^4,15^8,15^{16}$ 模五十五的余数依次是 $15,5,25,20,15$。由于 $27=16+8+2+1$，选中余数的乘积为 $15\cdot20\cdot5\cdot15=22500=409\cdot55+5$。逐素数证明解释了结果：消息模五为零，模十一不为零。

<!-- bilingual-en:start -->
The condition modulo the totient is stronger than needed for this theorem. Keeping $n=55,e=3$ but choosing $d=7$ gives $ed=21\equiv1\pmod{20}$, although $21\not\equiv1\pmod{40}$. The theorem still recovers every representative. For message five, $15^7\equiv25\cdot5\cdot15\equiv5\pmod{55}$. Thus the two valid private exponents illustrate sufficiency of the lambda condition; the proof does not require a totient inverse.
<!-- bilingual-en:end -->

模欧拉函数的条件比本定理所需条件更强。仍取 $n=55,e=3$，改选 $d=7$，则 $ed=21\equiv1\pmod{20}$，但 $21\not\equiv1\pmod{40}$。本定理仍保证全部代表都能恢复。对消息五，$15^7\equiv25\cdot5\cdot15\equiv5\pmod{55}$。这两个可用私钥指数展示了 lambda 条件的充分性；证明不要求一定模欧拉函数求逆。

<!-- bilingual-en:start -->
## Keep the theorem’s scope visible
<!-- bilingual-en:end -->

*明确这条定理的范围*

<!-- bilingual-en:start -->
The theorem concerns residue representatives. For an arbitrary integer such as $m=60$ with $n=55$, it proves recovery of the class, whose standard representative is five, not recovery of the original integer sixty. The RSA integer primitive therefore specifies its input range. Positive exponents matter, and the mathematics also works when one prime is two. RFC 8017 additionally requires distinct odd primes and further valid-key conditions; this more general algebraic statement is not a key-generation recommendation.
<!-- bilingual-en:end -->

定理处理的是余数代表。若输入任意整数，例如 $n=55$ 时的 $m=60$，结论保证恢复它所在的类，标准代表为五，而不是恢复原整数六十。因此 RSA 整数原语规定了输入范围。正指数条件不可省略；数学论证也允许其中一个素数为二。RFC 8017 另外要求不同奇素数及其他合法密钥条件；这里较一般的代数陈述不是密钥生成建议。

> [!warning]
> <!-- bilingual-en:start -->
> Repeated prime factors cannot be substituted into this two-prime argument. If one incorrectly takes $p=q=3$, computes $\operatorname{lcm}(2,2)=2$ and chooses $e=3,d=1$, the integer three modulo nine is sent to zero and does not return. The value two was obtained by using the distinct-prime formula outside its domain; no exponent theorem for modulus nine was established. Also, even perfect recovery does not establish confidentiality: $e=d=1$ satisfies this algebraic condition but leaves every message unchanged. See [[RSA安全边界|RSA security boundaries]].
> <!-- bilingual-en:end -->
>
> 不能把重复素因子代入这套两素数论证。如果错误地取 $p=q=3$，算 $\operatorname{lcm}(2,2)=2$ 并选 $e=3,d=1$，模九的消息三会变成零，无法恢复。这里的二来自超出适用范围使用互异素数公式，并没有建立模九的指数定理。此外，即使全部还原也不保证保密：$e=d=1$ 满足本代数条件，却让每个消息原样不变。见[[RSA安全边界]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- MIT §8.11 explicitly includes noncoprime messages. CP15 Problem 3(b)–(d), PDF page 2, supplies the primewise proof route and the distinct-prime product condition; this note uses CRT for the combination step. RFC 8017 §3 verifies the distinct-prime and lambda conditions. Both residue cases, positive exponents, and uniqueness of the recovered representative were checked directly.
<!-- bilingual-en:end -->

- [[MIT6_042JS15_textbook.pdf#page=288|MIT 6.042J 教材，§8.11，印刷第 279 页／PDF 第 288 页]]：明确解密对不互素消息也成立；[[MIT6_042JS15_cp15.pdf#page=2|MIT 6.042J CP15，Problem 3(b)–(d)，PDF 第 2 页]] 给出逐素数证明再合并为不同素数乘积的路线。本卡以 CRT 说明其中的合并步骤，正指数条件明确排除零指数。
- [IETF RFC 8017 §3.1–3.2](https://www.rfc-editor.org/rfc/rfc8017.html#section-3)：核对不同素因子与 $ed\equiv1\pmod{\lambda(n)}$ 的标准条件。本卡逐项核对两个素数下的零／非零情形、正指数和唯一代表；$m=5$ 算例独立验算。

<!-- bilingual-en:start -->
The textbook PDF288–289 establishes the message range and explicitly includes noncoprime inputs. CP15 PDF1–2 supplies the primewise route; its exercise omits an explicit positivity qualification on the displayed exponent, which is made explicit here. RFC 8017 §§3.1–3.2 and 5.1 distinguish valid odd-prime keys from the more general two-prime algebra proved here. The examples and the extension from the totient condition to the lcm condition are verified directly.
<!-- bilingual-en:end -->

教材 PDF288–289 给出消息范围并明确包含非互素输入；CP15 PDF1–2 支持逐素数路线，其中练习显示的指数未显式限定正值，本文已明确补齐。RFC 8017 的 §3.1–3.2 与 §5.1 区分合法奇素数密钥条件与本文较一般的两素数代数论证。例子及从欧拉函数条件到最小公倍数条件的推广均直接核验。
