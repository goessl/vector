from HermiteFunction import HermiteFunction
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, +4, 1000)
for n in range(5):
    f = HermiteFunction(n)
    plt.plot(x, f(x), label=f'$h_{n}$')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('hermite_functions.png', dpi=300)



plt.clf()
np.random.seed(4)



f = HermiteFunction((1, 2, 3))
g = HermiteFunction.random(15)
h = HermiteFunction.fit(x, g(x), 10)
plt.plot(x, f(x), label='$f$')
plt.plot(x, g(x), '--', label='$g$')
plt.plot(x, h(x), ':', label='$h$')
plt.legend()
#plt.show()
plt.savefig('initialization.png', dpi=300)



plt.clf()



f_p = f.der()
f_pp = f.der(2)
plt.plot(x, f(x), label=rf"$f \ (\deg f={f.deg})$")
plt.plot(x, f_p(x), '--', label=rf"$f' \ (\deg f'={f_p.deg})$")
plt.plot(x, f_pp(x), ':', label=rf"$f'' \ (\deg f''={f_pp.deg})$")
plt.legend()
#plt.show()
plt.savefig('differentiation.png', dpi=300)



plt.clf()



g = HermiteFunction(4)
h = f + g
i = 0.5 * f
plt.plot(x, f(x), label='$f$')
plt.plot(x, g(x), '--', label='$g$')
plt.plot(x, h(x), ':', label='$h$')
plt.plot(x, i(x), '-.', label='$i$')
plt.legend()
#plt.show()
plt.savefig('arithmetic.png', dpi=300)
