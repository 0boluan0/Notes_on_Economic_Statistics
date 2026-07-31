from __future__ import annotations

import argparse
import difflib
from pathlib import Path


REPLACEMENTS: dict[str, dict[str, str]] = {
    "01_Differentiation.md": {
        "*three-way self-examination*": "*Three Quick Self-Checks*",
        "*Boundary Condition and False Points*": "*Boundary Cases and Common Pitfalls*",
        "** Problem: **": "**Problem:**",
        "** Question: **": "**Question:**",
        "** topic.  **": "**Problem.**",
        "** Check.  **": "**Check.**",
        "*Why the Contrary Proposition is Wrong: $|x|$*": "*Why the Converse Fails: $|x|$*",
        "Token": "Notation",
        "1. $h\ne0$ only;\n2. Algebraic deformations are used to eliminate the common factors that cause $0/0$.\n3. Then study what the simplified expression tends to when $h\to0$.": "1. Work with $h\ne0$.\n2. Use algebra to cancel the common factor responsible for the indeterminate form $0/0$.\n3. Then determine the limit of the simplified expression as $h\to0$.",
        "- The two-sided difference quotient limits must be the same.  The right and left slopes are different at the sharp corners, so they are not derivative.\n- When the limit approaches $\pm\infty$, it can be said that there is a vertical tangent, but it is still not differentiable according to the \"finite derivative\" convention in this course.\n- $f'(x_0)$ is a number; $f'(x)$ is a new function that changes with $x$, and should not be confused.\n- Tangents are local approximations that do not guarantee that they are close to curves on the entire image.": "- The two-sided limits of the difference quotient must agree. At a corner, the left- and right-hand slopes differ, so the function is not differentiable there.\n- If the difference quotient tends to $\pm\infty$, the curve may have a vertical tangent, but under this course's finite-derivative convention the function is still not differentiable.\n- $f'(x_0)$ is a number, whereas $f'(x)$ is a function of $x$; do not confuse them.\n- A tangent line is a local approximation and need not remain close to the curve globally.",
        "2. Why can't a line only intersect a curve define a tangent?": "2. Why can a tangent line not be defined merely as a line that intersects the curve at one point?",
        "3. For $f(x)=\frac12x^3-x$, when $x=0$ and $h=0.25$, how much difference does the slope of the secant and the slope of the tangent differ?": "3. For $f(x)=\frac12x^3-x$ at $x=0$ with $h=0.25$, what is the difference between the secant slope and the tangent slope?",
        "> 1. $[(3+h)^2-9]/h=6+h\to6$, tangent $y-9=6(x-3)$.": "> 1. $[(3+h)^2-9]/h=6+h\to6$, so the tangent line is $y-9=6(x-3)$.",
        "> 3. The slope of secant is $\frac12h^2-1=-0.96875$, the slope of tangent is $f'(0)=-1$, and the absolute error is $0.03125$.": "> 3. The secant slope is $\frac12h^2-1=-0.96875$, the tangent slope is $f'(0)=-1$, and the absolute difference is $0.03125$.",
        "Slope and rate of change are two languages with the same ratio: rise/run in the coordinate chart; physical or economic when the horizontal axis is interpreted as time, distance, or yield.": "Slope and rate of change are two interpretations of the same ratio. On a graph it is rise over run; when the horizontal axis represents time, distance, or output, the ratio acquires a physical or economic meaning.",
        "The courseware is designed to allow pumpkins to fall quietly from a height of about $80$ meters, ignoring the air resistance:": "The lecture models a pumpkin dropped from rest at a height of about $80$ metres, neglecting air resistance:",
        "The average speed of the whole process is": "The average velocity over the entire fall is",
        "So before the crash": "Therefore, immediately before impact,",
        "A negative sign indicates downward; the speed is the speed size $|v|=40\text{ m/s}$.  The average velocity is different from the terminal instantaneous velocity because the velocity changes during the descent.": "The negative sign indicates downward motion; the speed is the magnitude $|v|=40\text{ m/s}$. The average velocity differs from the final instantaneous velocity because the velocity changes throughout the fall.",
        "If $h$ takes m and $t$ takes seconds, then the unit of $dh/dt$ is m/s; then the acceleration is obtained by derivation": "If $h$ is measured in metres and $t$ in seconds, then $dh/dt$ has units of m/s; differentiating once more gives the acceleration",
        "- charge $q$ (Coulomb) derivative with respect to time $dq/dt$ is current (amperes);\n- temperature $T$ derivative of position $x$ $dT/dx$ is temperature gradient (degrees/meters);\n- Cost $C$ Derivative of Yield $q$ $dC/dq$ is the marginal cost (currency/item).": "- If charge $q$ is measured in coulombs, then $dq/dt$ is current in amperes.\n- The spatial derivative $dT/dx$ is a temperature gradient, measured in degrees per metre.\n- The derivative $dC/dq$ is marginal cost, measured in currency units per unit of output.",
        "*03d:GPS sensitivity*": "*03d: GPS Sensitivity*",
        "In the simplified plane model, the altitude of the satellite is known at $s$, the receivers measured the slant distance is $h$, and the horizontal distance is $L$:": "In the simplified planar model, the satellite altitude $s$ is known, the receiver measures the slant range $h$, and the horizontal distance is $L$:",
        "Derive for $h$:\n#confused": "Differentiate with respect to $h$:",
        "When the distance measurement error is small, $\Delta h$, the horizontal error is approximate": "For a small ranging error $\Delta h$, the horizontal error is approximately",
        "When the receiver is almost under the satellite, the $L$ is very small, and the amplification factor $h/L$ is very large: a very small slant error will also cause significant horizontal position error.  That's what derivatives mean as **sensitivity**.": "When the receiver is almost directly below the satellite, $L$ is small and the amplification factor $h/L$ is large: even a small error in slant range can produce a substantial horizontal-position error. This is the meaning of a derivative as **sensitivity**.",
        "- Describe the arguments before deriving them.  $dT/dx$ and $dT/dt$ answer different questions.\nThe sensitivity is close to infinity, which means that the actual error is infinity. It only shows that the linear amplification factor is very large and the measurement geometry is very unfavorable.": "- State the independent variable before differentiating: $dT/dx$ and $dT/dt$ answer different questions.\n- A sensitivity that becomes arbitrarily large does not mean the actual error must be infinite; it means the linear amplification factor is large and the measurement geometry is poorly conditioned.",
        "*self-test*": "*Three Quick Self-Checks*",
        "1. For $s(t)=3t^2-2t$, the average velocity of $[1,3]$ and the instantaneous velocity of $t=3$ are calculated.\n2. If $s=3,h=5$ in the GPS model, the ranging error is about $0.01$, and the estimation error is $L$.": "1. For $s(t)=3t^2-2t$, find the average velocity on $[1,3]$ and the instantaneous velocity at $t=3$.\n2. Why is the pumpkin's velocity at impact $-40$ rather than $40$? When should the answer be $40$?\n3. In the GPS model, if $s=3$, $h=5$, and the ranging error is approximately $0.01$, estimate the error in $L$.",
        "** Problem: ** Does \"approaching\" depend on the value of the function at the target point?  What guarantees can be brought in directly?": "**Problem:** Does a limit depend on the function's value at the target point? Under what conditions is direct substitution valid?",
        "** Extreme.  **The derivability hypothesis guarantees that the first factor tends to a finite number $f'(x_0)$, and the second factor tends to a finite number $0$:": "**Take the limit.** Differentiability guarantees that the first factor tends to the finite value $f'(x_0)$, while the second factor tends to $0$:",
        "Thus, $\lim_{x\to x_0}f(x)=f(x_0)$, or continuous.": "Therefore, $\lim_{x\to x_0}f(x)=f(x_0)$, so $f$ is continuous at $x_0$.",
        "** Boundary conditions.  ** It is proved that the dependent derivative is finite, and if the difference quotient tends to infinity, it cannot be written as \"finite number times zero\".  And because the limit process is always $x\ne x_0$, the middle divided by $x-x_0$ is legal.": "**Boundary condition.** The proof relies on the derivative being finite. If the difference quotient diverges, it cannot be treated as a finite number multiplied by zero. Because the limiting process always has $x\ne x_0$, division by $x-x_0$ in the intermediate step is valid.",
        "$f(x)=|x|$ is continuous in $0$, but": "$f(x)=|x|$ is continuous at $0$, but",
        "The left-right quotient limit is $1,-1$, so it is not derivative.  This shows that:": "The right- and left-hand limits of the difference quotient are $1$ and $-1$, respectively, so the function is not differentiable at $0$. Thus:",
        "3. It is proved that if the function is discontinuous at a point, it must be non-derivative at that point.": "3. Prove that if a function is discontinuous at a point, then it cannot be differentiable there.",
        "- $\infty$ is not a normal real number and cannot be arbitrarily $\infty-\infty$ in an algebraic expression.": "- $\infty$ is not an ordinary real number, so expressions such as $\infty-\infty$ cannot be manipulated algebraically without further analysis.",
    },
    "04_Techniques_of_Integration.md": {
        "Upon completion of this chapter, it shall be able to:": "After completing this chapter, you should be able to:",
        "1. Deal with $\int\sin^n x\cos^m x\,dx$ according to parity and deal with the power of $\tan,\sec,\cot,\csc$.\n2. Choose the right triangulation from the root shape, control the angle range, absolute value and generation.\n3. Divide any rational function long and then factorize and write the complete partial fractional template.\n4. Derive the partial integral from the product rule and determine which part should be differentiated.\n5. Arc length is deduced from the broken line limit, and surface area of revolution is deduced from the narrow truncated cone.\n6. Transform between Cartesian coordinates, parametric equations and polar coordinates, and correctly determine the trajectory, direction, repetition and integration range.\n7. Complete all assignment questions for Problem Set 9-11 and Exam 4 independently.": "1. Evaluate $\int\sin^n x\cos^m x\,dx$ by using parity, and handle powers of $\tan,\sec,\cot,\csc$.\n2. Choose the appropriate trigonometric substitution from the form of the radical, controlling the angle range, absolute values, and back-substitution.\n3. Apply polynomial long division to an improper rational function, factor the denominator, and write the complete partial-fraction decomposition.\n4. Derive integration by parts from the product rule and decide which factor to differentiate.\n5. Derive arc length from polygonal approximation and surface area of revolution from narrow frusta.\n6. Move between Cartesian, parametric, and polar descriptions while checking the path, orientation, repeated tracing, and integration interval.\n7. Complete all assigned problems from Problem Sets 9–11 and Exam 4 independently.",
        "**Summary:**The odd power provides a factor that can act as a $du$, and the square identity turns the rest into the same new variable.": "**Summary:** An odd power supplies a factor that can serve as $du$; the Pythagorean identity rewrites every remaining factor in terms of the same substitution variable.",
        "**Summary:**For all even orders, the power-down formula is replaced by \"angle doubling\" for \"halving the number of times.\"": "**Summary:** When both powers are even, the power-reduction identities double the angle while reducing the exponents, and repeated application eventually produces directly integrable terms.",
        "**Summary:**The triangle substitution is not magic; it embeds the Pythagorean identity in the root form, making the root form an edge of the right-angled triangle.": "**Summary:** Trigonometric substitution is not a trick: it builds a Pythagorean identity into the radical, turning the square root into a side of a right triangle.",
        "**Summary:**The power of the treatment $\tan,\sec$ is still \"leave a derivative factor and then unify the variables with identities\".": "**Summary:** Integrals involving powers of $\tan$ and $\sec$ follow the same strategy: reserve a factor that matches a derivative, then use an identity to express the rest in one variable.",
        "1. First of all, the quadratic is standardized into one of the three templates.\n2. Write $dx$ and declare the scope of the $\theta$.\n3. Erase the root form and complete the trigonometric integral by the identity.\n4. Return to $x$ with triangles or algebraic identities.\n5. Derive the result, if it is definite integral, change the upper and lower limits directly in order to avoid retroactive.": "1. Rewrite the quadratic expression in one of the three standard forms.\n2. Compute $dx$ and state the range of $\theta$.\n3. Use the relevant identity to eliminate the radical and evaluate the trigonometric integral.\n4. Return to $x$ using a reference triangle or an algebraic identity.\n5. Differentiate to check the result. For a definite integral, preferably transform the limits during substitution so that no back-substitution is needed.",
        "**Summary:**Formula translates the general quadratic into the Translated Standard Radical; then selects sine, tangent, or secant substitution by symbol.": "**Summary:** Completing the square converts a general quadratic into a shifted standard radical; the resulting sign pattern determines whether sine, tangent, or secant substitution is appropriate.",
        "> 3. Most of the symbol errors can be found by randomly substituting a non-pole after decomposition, but strictly guarantee the identity from the multiplied denominator.": "> 3. Substituting a convenient non-pole after decomposition catches many sign errors, but a rigorous check comes from recombining the fractions over the common denominator and verifying the identity.",
        "**Summarization:**Partial fractions complete an algebraic problem first, then complete multiple simple integrals; the coefficients occur before the integrals.": "**Summary:** Partial fractions first solves an algebraic decomposition problem and then reduces the task to several elementary integrals; determine the coefficients before integrating.",
        "**Summary:**\"Divide long, then complete column\" makes partial fractions a deterministic algorithm that covers all rational functions.": "**Summary:** “Long-divide first, then include every required term” turns partial fractions into a systematic procedure for any rational function.",
        "*Session 76:[[积分方法#三角积分、分部积分与部分分式|Segment Credits]]*": "*Session 76: [[积分方法#三角积分、分部积分与部分分式|Integration by Parts]]*",
        "**Summary:**The partial integral is a reverse-engineering of the product rule; a good choice causes a certain \"complexity index\" to decline at each step.": "**Summary:** Integration by parts reverses the product rule; a good choice of $u$ and $dv$ makes a clear measure of complexity decrease at each step.",
        "> 1. Why is the shell height $e-e^x$?  Since that riser extend from the lower boundary $y=e^x$ to the upper cover $y=e$.": "> 1. Why is the shell height $e-e^x$? Because each vertical strip extends from the lower boundary $y=e^x$ to the upper boundary $y=e$.",
        "**Summary: The**geometry slice determines the integral form; the integral technique then serves the calculation and you cannot choose the technique before you hard-wrap the geometry.": "**Summary:** The geometry of the slice determines the integral. Only then should an integration technique be chosen to carry out the calculation; the geometry cannot be forced to fit a preselected method.",
        "**Summary:**Arc length is the accumulation of Local Pythagorean length; $ds$ is a geometric object, and choosing $x,y$ or the parameter is just a different way of calculating it.": "**Summary:** Arc length accumulates local Pythagorean increments. The element $ds$ is geometric; using $x$, $y$, or a parameter merely gives different ways to compute it.",
        "*Question in this section: Why does a narrow cone give a formula?*": "*Question: Why Does a Narrow Frustum Lead to the Formula?*",
        "> 1. The $y=c>0$ has a side area of $2\pi cL$ around the $x$ axis in a length of $L$.\n> 2. The unit upper semicircle is given to $4\pi$ around the $x$ axis, not $2\pi$, because $x$ is from $-1$ to $1$.\n> 3. The radius is $|f(x)-k|$ when the line is $y=k$.": "> 1. Rotating the horizontal segment $y=c>0$ of length $L$ about the $x$-axis gives lateral area $2\pi cL$.\n> 2. Rotating the upper unit semicircle about the $x$-axis gives area $4\pi$, not $2\pi$, because $x$ runs from $-1$ to $1$.\n> 3. When the axis of rotation is $y=k$, the radius is $|f(x)-k|$.",
        "**Summary:**[[参数曲线与极坐标#弧长、曲面与检查|area of revolved surface]] is the summation of \"circumferential × arc-length infinitesimal elements\"; the real error-prone ones are geometric radii and $ds$, not the final integration technique.": "**Summary:** A [[参数曲线与极坐标#弧长、曲面与检查|surface of revolution]] accumulates “circumference × arc-length element.” The main sources of error are the geometric radius and $ds$, not the final integration technique.",
        "**Problem Set 10 Summary: The difficulty with**partial fractions is the complete algebraic decomposition; the difficulty with partial integrals is to reduce the complexity.  Finding the wrong coefficient is quicker than doing it all over again.": "**Problem Set 10 summary:** The challenge in partial fractions is completing the algebraic decomposition; the challenge in integration by parts is reducing complexity. Differentiating the finished result term by term usually reveals a coefficient error faster than redoing the calculation.",
        "> 1. $x=\cos2t,y=\sin2t,0\le t\le2\pi$ strokes the unit circle counterclockwise twice.\n> 2. $x=t^2,y=t^4$ satisfies $y=x^2$, but $t\in\mathbb R$ only covers $x\ge0$ and is stroked twice per point except for the origin.": "> 1. $x=\cos2t,y=\sin2t,0\le t\le2\pi$ traces the unit circle counterclockwise twice.\n> 2. $x=t^2,y=t^4$ satisfies $y=x^2$, but for $t\in\mathbb R$ it covers only $x\ge0$ and traces every nonzero point twice.",
        "**Summary:**Parameterization turns the curve into motion; $x',y'$ is the velocity component, $ds/dt$ is the rate, and the path is obtained by integrating the rate.": "**Summary:** A parametrization turns a curve into motion: $x'$ and $y'$ are velocity components, $ds/dt$ is speed, and integrating speed gives distance travelled.",
        "How to identify the track and judge the velocity from the parameter equation, and calculate the surface area of the parameter curve?": "How can we identify the path and motion from parametric equations, and how do we compute the surface area generated by a parametric curve?",
        "> 1. The ellipse is vertical in the tangent to $t=\pi/2$ because $x'=0,y'=-1$.\n> 2. The surface is computed twice for the same ellipse if $t$ is from $0$ to $2\pi$ and then around the $y$ axis.": "> 1. The ellipse has a vertical tangent at $t=\pi/2$ because $x'=0$ and $y'=-1$.\n> 2. If the same ellipse is traced over $0\le t\le2\pi$ and revolved about the $y$-axis, the resulting surface is counted twice.",
        "**Summarization:**Trajectory is identified by parameter elimination and motion is identified by parameter interval and derivative;Both types of information are indispensable when calculating arc length or area.": "**Summary:** Eliminating the parameter identifies the geometric path, while the parameter interval and derivatives identify the motion. Both are essential when computing arc length or surface area.",
        "**Summary:**The polar area is the Riemann sum of \"narrow sectors\"; the most important task before integration is to determine the angle interval that covers the target area only once.": "**Summary:** Polar area is the limit of a Riemann sum of narrow sectors. Before integrating, determine an angular interval that covers the target region exactly once.",
        "*Objective of this section: To avoid blinding dots*": "*Objective: Sketch the Curve Without Blindly Plotting Points*",
        "**Summary:**The core of polar mapping is not multiple points, but periods, symmetries, zeros, symbols, and repetition.": "**Summary:** Effective polar graphing relies on period, symmetry, zeros, the sign of $r$, and repeated tracing—not on plotting more points.",
        "given": "This gives",
        "This problem concatenates partial integrals, rational simplifications and inverse trigonometric primitive functions.": "This problem combines integration by parts, rational simplification, and inverse-trigonometric antiderivatives.",
        "> 1. $\int x^3/(x^2+1)dx$ should divide by $x-x/(x^2+1)$ and answer $x^2/2-\tfrac12\ln(x^2+1)+C$.\n> 2. Parametric surface formulas must use a rate of $\sqrt{x'^2+y'^2}$, not a signed $dx/dt$.\n> 3. The last five minutes are checked first: chain constant, logarithmic absolute value, upper and lower limit of definite integral, radius of surface area and repeated drawing in polar coordinates.": "> 1. For $\int x^3/(x^2+1)dx$, first divide to obtain $x-x/(x^2+1)$; the result is $x^2/2-\tfrac12\ln(x^2+1)+C$.\n> 2. A parametric surface-area formula must use the speed $\sqrt{x'^2+y'^2}$, not the signed quantity $dx/dt$.\n> 3. In the final five minutes, check chain-rule constants, absolute values in logarithms, limits of definite integrals, radii in surface-area formulas, and repeated tracing in polar coordinates.",
        "**Summary:**The Exam 4 really examines identifying structures and concatenation methods; writing \"why it\" first often avoids a dead end.": "**Summary:** Exam 4 primarily tests structural recognition and the ability to combine methods. Writing down why a method applies before calculating often prevents a dead end.",
        "There is no `Ses86` handout locally; this section is composed of [[Exam4_Problems.pdf#page=1|Exam 4 Original]] and [[Exam4_Solutions.pdf#page=1|Official Answer]] in the official structure.  Here are the complete answers on a question-by-question basis.": "There is no local `Ses86` handout. In the official course structure, this session consists of [[Exam4_Problems.pdf#page=1|the original Exam 4 paper]] and [[Exam4_Solutions.pdf#page=1|the official solutions]]. Complete solutions are given below, question by question.",
        "> - Question 1: Partial integral boundary terms and power exponents.\n> - Question 2: Set aside the $\sec^2\theta d\theta$ and replace the upper and lower limits.\n> - Question 3: The quadratic factor must be above a linear molecule.\n> - Question 4: Formula First, Declaration Domain, Control Root Symbol.\n> - Question 5: Arc length is $ds$, and the surface is multiplied by $2\pi\times$ to the distance from the axis of revolution.": "> - Question 1: Check the boundary term in integration by parts and the power exponent.\n> - Question 2: Reserve the factor $\sec^2\theta\,d\theta$ and transform the limits.\n> - Question 3: An irreducible quadratic factor requires a linear numerator.\n> - Question 4: Complete the square first, state the domain, and control the sign of the square root.\n> - Question 5: Use $ds$ for arc length; for surface area, multiply by $2\pi$ times the distance to the axis of rotation.",
    },
}


def make_patch(path: Path, original: str, revised: str) -> str:
    diff = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            revised.splitlines(keepends=True),
            n=3,
        )
    )[2:]
    if not diff:
        return ""
    hunks = ["@@\n" if line.startswith("@@") else line for line in diff]
    return "".join(
        [
            "*** Begin Patch\n",
            f"*** Update File: {path.resolve()}\n",
            *hunks,
            "*** End Patch\n",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    original = args.path.read_text()
    revised = original
    applied = []
    missing = []
    for old, new in REPLACEMENTS[args.path.name].items():
        count = revised.count(old)
        if count:
            revised = revised.replace(old, new)
            applied.append((count, old[:80]))
        else:
            missing.append(old[:80])
    if args.summary:
        print(f"applied_patterns={len(applied)} replacements={sum(c for c, _ in applied)}")
        print(f"missing_patterns={len(missing)}")
        for item in missing:
            print("MISSING", item)
        return
    print(make_patch(args.path, original, revised), end="")


if __name__ == "__main__":
    main()
