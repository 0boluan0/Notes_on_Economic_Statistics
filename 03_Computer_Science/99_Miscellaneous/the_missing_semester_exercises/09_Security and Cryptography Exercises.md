---
aliases:
  - The Missing Semester Lecture 9 Exercises
  - Missing Semester Security Exercises
tags:
  - computer-science
  - tools
  - exercises
  - security
  - cryptography
  - the-missing-semester
---

# 第 9 讲 安全和密码学 练习
<!-- bilingual-en:start -->
*Lecture 9: Security and Cryptography Exercises*
<!-- bilingual-en:end -->

>[!note]
> 对应主笔记：[[the_missing_semester#第 9 讲 安全和密码学]]
> 
> 官方来源：https://missing-semester-cn.github.io/2020/security/
> 
> 官方解答：https://missing-semester-cn.github.io/missing-notes-and-solutions/2020/solutions//security-solution

## 练习清单
<!-- bilingual-en:start -->
*Exercise Checklist*
<!-- bilingual-en:end -->

1. 熵：
   - 假设密码由 4 个随机小写单词拼接而成，每个单词从 10 万词词典中等概率选取，求其熵
   - 假设另一个密码由 8 个随机大小写字母或数字组成，求其熵
   - 比较两者谁更强
   - 若攻击者每秒尝试 1 万个密码，估算破解时间
<!-- bilingual-en:start -->

&nbsp;
**1.** **Entropy:** Suppose a password is formed by concatenating four random lowercase words, each selected uniformly from a dictionary of 100,000 words. Calculate its entropy. Then calculate the entropy of an eight-character password whose characters are selected at random from uppercase letters, lowercase letters, and digits. Compare their strength and estimate the cracking time if an attacker can try 10,000 passwords per second.<br>
<!-- bilingual-en:end -->
2. 密码散列函数：
   - 从 Debian 镜像站下载一个光盘映像
   - 用 `sha256sum` 计算本地文件哈希
   - 与 Debian 官方公布的哈希值对比
<!-- bilingual-en:start -->

&nbsp;
**2.** **Cryptographic hash functions:** Download a disc image from a Debian mirror, compute the local file's hash with `sha256sum`, and compare it with the hash published by Debian.<br>
<!-- bilingual-en:end -->
3. 对称加密：
   - 使用 OpenSSL 的 AES 模式加密一个文件
   - 用 `cat` 或 `hexdump` 比较源文件与密文
   - 再解密回来
   - 用 `cmp` 验证解密后文件与原文件一致
<!-- bilingual-en:start -->

&nbsp;
**3.** **Symmetric encryption:** Encrypt a file using an OpenSSL AES mode, compare the plaintext and ciphertext with `cat` or `hexdump`, decrypt the file, and use `cmp` to verify that the decrypted file is identical to the original.<br>
<!-- bilingual-en:end -->
4. 非对称加密：
   - 在本机生成更安全的 ED25519 SSH 密钥对，并给私钥加密码
   - 配置 GPG
   - 给 Anish 发送一封加密邮件
   - 用 `git commit -S` 或 `git tag -s` 进行签名，并验证签名
<!-- bilingual-en:start -->

&nbsp;
**4.** **Asymmetric cryptography:** Generate a stronger ED25519 SSH key pair locally and protect the private key with a passphrase. Configure GPG, send Anish an encrypted email, and create and verify a signed Git commit or tag with `git commit -S` or `git tag -s`.<br>
<!-- bilingual-en:end -->
