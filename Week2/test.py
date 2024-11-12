#Solution to Activity 1
from sympy import * 
import sympy as sp

t = sp.symbols('t')
x,y = sp.symbols('x y')
s_t = t ** 3
p_t = sp.exp(t)*sp.cos(t**2)

s_prime_t = s_t.diff(t)
s_prime_t1 = sp.diff(s_t,t)
f1 = lambdify(t,s_t)
p_prime_t = sp.diff(p_t,t)
print(p_prime_t)
print(f1(2))


print("Derivative of s(t) =", s_prime_t)
print("Derivative of s(t) =", s_prime_t1)

# partial derivative wrt y
f = x**4+ x*y**4
print(f.diff(y))

#chain rule
f2 = sp.cos(x**2)
print(f2.diff(x))

# quotient rule
t, d = sp.symbols('t d')
S_t = d / t

S_prime_t = sp.diff(S_t, t)
S_prime_t1 = sp.diff(S_t, d)

print("Derivative of S(t) =", S_prime_t)
print("Derivative of S(t) =", S_prime_t1)

