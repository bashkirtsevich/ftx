from numba import jit


# Ideas for approximating tanh/atanh:
# * https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/
# * http://functions.wolfram.com/ElementaryFunctions/ArcTanh/10/0001/
# * https://mathr.co.uk/blog/2017-09-06_approximating_hyperbolic_tangent.html
# * https://math.stackexchange.com/a/446411


@jit(nopython=True)
def fast_tanh(x: float) -> float:
    if x < -4.97:
        return -1.0

    if x > 4.97:
        return 1.0

    x2 = x ** 2
    a = x * (945.0 + x2 * (105.0 + x2))
    b = 945.0 + x2 * (420.0 + x2 * 15.0)
    return a / b


@jit(nopython=True)
def fast_atanh(x: float) -> float:
    x2 = x ** 2
    a = x * (945.0 + x2 * (-735.0 + x2 * 64.0))
    b = (945.0 + x2 * (-1050.0 + x2 * 225.0))
    return a / b
