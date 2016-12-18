\newcommand{\floor}[1]{\left\lfloor{#1}\right\rfloor}

# Nontrivial Mathematical Deductions

## Challenge 46
In this challenge we are essentially given a parity function $e_x(k) = kx\mod N
\mod 2$ and the value of odd integer $N$, and we need to recover $x$ by querying
$e_x(k)$ for arbitrary $k$.

This can be solved by a bisection method. We define $l=\floor{l'N}$ and
$h=\floor{h'N}$, where $l'=\frac{a}{2^s}$ and $h'=\frac{a+1}{2^s}$. Initially we
have $a = 0,\,s=0$. By recursively bisecting on $(l', h')$ while ensuring $x$
within the range $(l, h]$, we can deduce $x=h$ when $l=h-1$.

Denote the set of even integers by $\mathbb{E}=\{2k\mid k\in\mathbb{Z}\}$.
The key is to find a bisection point $m$ and a scaling factor $f\in\mathbb{E}$,
such that
$$
\begin{eqnarray}
    \floor{\frac{(l+1)f}{N}} &=& \floor{\frac{mf}{N}} \\
        &=& \floor{\frac{(m+1)f}{N}}-1 \\
        &=& \floor{\frac{hf}{N}} - 1 \\
        &\in& \mathbb{E}
\end{eqnarray}
$$
So for we can have
$$
    e_x(f)=\left\{
    \begin{array}{cc}
        0 & \text{if } x\in(l, m] \\
        1 & \text{if } x\in(m, h]
    \end{array}
    \right.
$$

It can be proved that $f=2^{s+1}$ and $m=\floor{\frac{2a+1}{f}N}$ satisfies the
above equation. We only give the proof for the first term here, and it applies
to others similarly.

Given $l=\floor{\frac{2a}{f}N}$, we have
$$
    l \le \frac{2a}{f}N \lt l+1
$$
which is equivalent to
$$
    2a \lt \frac{(l+1)f}{N} \le 2a+\frac{f}{N}
$$
Since $f \lt N$ during the bisection, it follows
$$
    \floor{\frac{(l+1)f}{N}} = 2a
$$

Therefore, we can find the answer using above strategy.

[//]: # " vim: set textwidth=80 spell: "
