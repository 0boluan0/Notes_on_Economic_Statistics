from __future__ import annotations

import difflib
from pathlib import Path


PATH = Path("01_Math/01_calculus/01_Differentiation.md")

REPLACEMENTS = [
    (r"""**Question:**How to calculate the abstract difference quotient?  Where do the unified rules for power functions come from?""", r"""**Question:** How do we evaluate an abstract difference quotient, and where does the general power rule come from?"""),
    (r"""** Prerequisite knowledge: ** Derivative definition, Fractional General Fraction, Binomial Expansion, Straight Intercept and Triangle Area.""", r"""**Prerequisites:** the definition of the derivative, combining fractions over a common denominator, the binomial theorem, line intercepts, and triangle area."""),
    (r"""*02b: Triangle Enclosed by Hyperbolic Tangent*""", r"""*02b: Triangle Enclosed by a Tangent to the Hyperbola*"""),
    (r"""**Problem.** $y=1/x$ What is the area of the triangle bounded by the two axes of the $P=(x_0,1/x_0)$ tangent at any point in the first quadrant?""", r"""**Problem.** For a point $P=(x_0,1/x_0)$ on $y=1/x$ in the first quadrant, what is the area of the triangle enclosed by the tangent line and the coordinate axes?"""),
    (r"""*02d: A Complete Derivation of the Law of Positive Integer Powers*""", r"""*02d: A Complete Derivation of the Positive-Integer Power Rule*"""),
    (r"""*02e: Linear Approximate Product (Lead Material)*""", r"""*02e: Products of Linear Approximations (Preview Material)*"""),
    (r"""** Question: Why is the ** tangent slope also representative of velocity, current, temperature gradient, and measurement sensitivity?""", r"""**Question:** Why can a tangent slope also represent velocity, electric current, a temperature gradient, or measurement sensitivity?"""),
    (r"""** Prerequisites: ** Difference Quotient, Derivative Definition, Variables and Units, Pythagorean Theorem.""", r"""**Prerequisites:** difference quotients, the definition of the derivative, variables and units, and the Pythagorean theorem."""),
    (r"""*03c:80 rice pumpkin down*""", r"""*03c: An 80-Metre Pumpkin Drop*"""),
    (r"""*04a: Limits, Ease Limits and Difficult Limits*""", r"""*04a: Limits, Direct-Substitution Cases, and Indeterminate Cases*"""),
    (r"""*04b: Three consecutive conditions*""", r"""*04b: The Three Conditions for Continuity*"""),
    (r"""** Question: How does ** continuity fail?  Why is it possible to derive a certain continuity, but not a certain continuity?""", r"""**Question:** In what ways can continuity fail? Why does differentiability imply continuity, while continuity does not imply differentiability?"""),
    (r"""** Prerequisite knowledge: ** Left-right limit, point continuous definition, derivative difference quotient.""", r"""**Prerequisites:** one-sided limits, continuity at a point, and the derivative as a difference-quotient limit."""),
    (r"""** Objective.  ** Proof $\lim_{x\to x_0}[f(x)-f(x_0)]=0$.""", r"""**Goal.** Prove that $\lim_{x\to x_0}[f(x)-f(x_0)]=0$."""),
    (r"""** Construction.  **For $x\ne x_0$, the function increment is split into "difference quotient×input increment":""", r"""**Construction.** For $x\ne x_0$, factor the function increment into “difference quotient × input increment”:"""),
    (r"""The ** problem: How do ** combine the known simple derivatives into the derivatives of new functions such as polynomials?""", r"""**Question:** How can known elementary derivatives be combined to differentiate new functions such as polynomials?"""),
    (r"""**Problem: **How to derive the derivatives of $\sin x$, $\cos x$ using only the derivative definition and trigonometric addition formula?""", r"""**Question:** How can the derivatives of $\sin x$ and $\cos x$ be derived using only the definition of the derivative and the angle-addition identities?"""),
    (r"""** Prerequisites: ** Difference Quotient, Limit Linear,""", r"""**Prerequisites:** difference quotients and linearity of limits."""),
    (r"""*07a: Algebraic Derivation of Sine Derivatives*""", r"""*07a: Algebraic Derivation of the Derivative of Sine*"""),
    (r"""*07b: Algebraic Derivation of Cosine Derivatives*""", r"""*07b: Algebraic Derivation of the Derivative of Cosine*"""),
    (r"""** Question: Why do the two trigonometric limits used in the previous section ** hold?  What is its geometric nature?""", r"""**Question:** Why do the two trigonometric limits used in the previous section hold, and what is their geometric basis?"""),
    (r"""**Prerequisite Knowledge:**Unit Circle, Radian, Triangle Area, Fixation Theorem, Conjugate.""", r"""**Prerequisites:** the unit circle, radians, triangle area, the squeeze theorem, and conjugate expressions."""),
    (r"""*08d: Geometric Image of Sinusoidal Derivative*""", r"""*08d: Geometric Interpretation of the Derivative of Sine*"""),
    (r"""**Question: **Multiply two variable quantities, and why is the instantaneous change of the product the sum of the two quantities?""", r"""**Question:** When two varying quantities are multiplied, why does the product's instantaneous rate of change contain two terms?"""),
    (r"""**Question: How do the derivatives of quotient consist of their respective derivatives when both the numerator and the denominator of **vary?""", r"""**Question:** When both numerator and denominator vary, how do their derivatives combine in the quotient rule?"""),
    (r"""*10a: Commercial law derivation*""", r"""*10a: Derivation of the Quotient Rule*"""),
    (r"""**Question: When **inputs are converted into multi-level functions, why is the total rate of change the product of the local rates of change at each level?""", r"""**Question:** When an input passes through several nested functions, why is the overall rate of change the product of the local rates at each level?"""),
    (r"""**Prerequisite knowledge: **Function composition, derivative, continuation, product rule.""", r"""**Prerequisites:** function composition, derivatives, continuity, and the product rule."""),
    (r"""** Question: How do we express and explain this "rate of change" when the ** derivative itself continues to change?""", r"""**Question:** When a derivative itself changes, how do higher derivatives express and interpret that change?"""),
    (r"""*Product Higher Derivatives: Leibniz Formula*""", r"""*Higher Derivatives of a Product: Leibniz's Formula*"""),
    (r"""**Problem Set 1 Summary: **These questions link the three levels of Part A's capabilities: read the diagram and domain first, then select the rules, and finally back to check with limits, parity, units, or function values.""", r"""**Problem Set 1 summary:** These questions connect three levels of skill in Part A: first read the graph and domain, then choose the appropriate rules, and finally check the result using limits, parity, units, or function values."""),
    (r"""**Question:**How do I ask for $dy/dx$ when $y$ is not easily solved separately?  How does the positive integer power rule extend to rational exponents?""", r"""**Question:** How can we find $dy/dx$ when solving explicitly for $y$ is inconvenient, and how does the positive-integer power rule extend to rational exponents?"""),
    (r"""**Prerequisite knowledge:**Chain rule, integer power rule, exponential operation, local $y$ as $y(x)$.""", r"""**Prerequisites:** the chain rule, the integer power rule, exponentiation, and treating $y$ locally as a function $y(x)$."""),
    (r"""*13a: Implicit Function Derivation*""", r"""*13a: Implicit Differentiation*"""),
    (r"""*13b: A Complete Derivation of the Rational Exponential Power Rule*""", r"""*13b: A Complete Derivation of the Rational-Exponent Power Rule*"""),
    (r"""** Question: When is the ** Implicit method shorter than solving $y$ first?  How do I handle products that also contain $x,y$?""", r"""**Question:** When is implicit differentiation shorter than first solving for $y$, and how should products involving both $x$ and $y$ be handled?"""),
    (r"""** Prerequisites: ** Chains, Product Rules, Branches and Domains.""", r"""**Prerequisites:** the chain rule, product rule, branches, and domains."""),
    (r"""** Question: Why are the slopes of the ** and inverse functions reciprocal?  How to derive the derivatives of arctangent and arcsine?""", r"""**Question:** Why are the slopes of a function and its inverse reciprocal, and how does this yield the derivatives of arctangent and arcsine?"""),
    (r"""**Prerequisite Knowledge: **One-to-One Correspondence, Function Composition, Implicit Function and Chain Rule, Trigonometric Identity.""", r"""**Prerequisites:** one-to-one functions, composition, implicit differentiation, the chain rule, and trigonometric identities."""),
    (r"""** Question: Why must the derivative of **$a^x$ be its own multiplier by a base-only constant?""", r"""**Question:** Why must the derivative of $a^x$ equal $a^x$ times a constant that depends only on the base?"""),
    (r"""** Prerequisite knowledge: ** Exponential law, continuity, derivative definitions.  Natural base $e$ has not been selected in this section.""", r"""**Prerequisites:** exponent laws, continuity, and the definition of the derivative. The natural base $e$ has not yet been selected."""),
    (r"""*16a-16b: Exponential function defined to real*""", r"""*16a–16b: Extending the Exponential Function to Real Exponents*"""),
    (r"""** Question: Can ** choose a base so that the derivative of the exponential function is exactly equal to itself?  Why is its inverse function derivative $1/x$?""", r"""**Question:** Can we choose a base whose exponential function is its own derivative, and why does the inverse function then have derivative $1/x$?"""),
    (r"""** Prerequisite knowledge: **$M(a)$, inverse derivative, exponential law.""", r"""**Prerequisites:** $M(a)$, the inverse-function derivative, and exponent laws."""),
    (r"""** Prerequisite knowledge: **$e^x,\ln x$'s derivative, product and chain rule, logarithm property.""", r"""**Prerequisites:** the derivatives of $e^x$ and $\ln x$, the product and chain rules, and logarithm laws."""),
    (r"""**Question:** Why does the limit of "very little, unlimited growth at a time" produce $e$?""", r"""**Question:** Why does the limit of increasingly frequent compounding produce $e$?"""),
    (r"""**Prerequisite knowledge: **log-exponential reciprocal, derivative-type limit, infinite limit variable substitution.""", r"""**Prerequisites:** the inverse relationship between logarithms and exponentials, derivative-form limits, and substitutions in limits at infinity."""),
    (r"""** Question: What trigonometric-like structures are produced by the symmetric combination of **$e^x,e^{-x}$?""", r"""**Question:** What trigonometric-like structure emerges from symmetric combinations of $e^x$ and $e^{-x}$?"""),
    (r"""** Prerequisite knowledge: ** Exponential derivative, product and chain rule, hyperbolic equation.""", r"""**Prerequisites:** exponential derivatives, the product and chain rules, and the hyperbola equation."""),
    (r"""**Problem Set 2 Summary: The core of **Part B is not to write a few more formulas, but to use the chain rule for "dependencies not explicitly written out": $y(x)$, inverse functions, powers of variables after logarithms, and hyperbolic functions defined by exponents.""", r"""**Problem Set 2 summary:** The core of Part B is not memorising more formulas, but applying the chain rule to dependencies that are not written explicitly: $y(x)$ in an implicit relation, inverse functions, variable powers after taking logarithms, and hyperbolic functions defined through exponentials."""),
    (r"""**Question:** How to quickly identify structures and check answers to synthetic derivation, proof of definition, tangent and piecewise function questions?""", r"""**Question:** How can we quickly identify the relevant structure and check answers in mixed differentiation, definition-proof, tangent-line, and piecewise-function problems?"""),
    (r"""** Prerequisites: **Session 1-20.  The goal is not to add a new rule, but to establish a call sequence.""", r"""**Prerequisites:** Sessions 1–20. The goal is not to add another rule, but to establish a reliable order of attack."""),
    (r"""** Answer Criteria: ** Write "Known/Target → Selection Rules → Step-by-Step Calculations → Domain or Geometry Checks → Final Answers" on each question.  There is no `Ses22` handout for local Session 22; this section is officially structured to consist of the Exam 1 original and the official answers.""", r"""**Answer standard:** For each question, write “known information and target → selected rule → step-by-step calculation → domain or geometric check → final answer.” There is no local `Ses22` handout; in the official structure, Session 22 consists of the original Exam 1 paper and its official solutions."""),
    (r"""*Problem 2: Tangent to a astroid*""", r"""*Problem 2: Tangent to an Astroid*"""),
    (r"""**Check.**Alternate tangent point: $(-\sqrt{27})/\sqrt3+4=-3+4=1$; the tangent does pass through the specified point.  The fractional power is understood as a real cube root at a negative number, and $x^{1/3}$ cannot be mistakenly taken as a positive root.""", r"""**Check.** At the alternative tangency point, $(-\sqrt{27})/\sqrt3+4=-3+4=1$, so the tangent does pass through the specified point. For negative inputs, the fractional power is interpreted using the real cube root; $x^{1/3}$ must not be treated as a nonnegative principal square root."""),
    (r"""** Construction.  **Add or subtract intermediate $f(x)g(x+h)$:""", r"""**Construction.** Add and subtract the intermediate term $f(x)g(x+h)$:"""),
    (r"""*Problem 5: Whether a piecewise function can be derived everywhere*""", r"""*Problem 5: Whether a Piecewise Function Is Differentiable Everywhere*"""),
]


def make_patch(original: str, revised: str) -> str:
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            revised.splitlines(keepends=True),
            n=3,
        )
    )[2:]
    hunks = ["@@\n" if line.startswith("@@") else line for line in diff]
    return "".join(
        [
            "*** Begin Patch\n",
            f"*** Update File: {PATH.resolve()}\n",
            *hunks,
            "*** End Patch\n",
        ]
    ) if hunks else ""


text = PATH.read_text()
original = text
missing = []
for old, new in REPLACEMENTS:
    if old not in text:
        missing.append(old)
    else:
        text = text.replace(old, new)
if missing:
    print("MISSING", len(missing))
    for item in missing:
        print(item)
else:
    print(make_patch(original, text), end="")
