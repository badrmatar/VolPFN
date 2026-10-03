"""High-precision reference for the Gaussian log-density diagnostic."""
import decimal

def oracle(x, v):
    with decimal.localcontext() as c:
        c.prec = 100
        c.Emax = 999999999
        # High precision pi; binary64 x/v are the exact inputs being evaluated.
        pi = decimal.Decimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679")
        xx, vv = decimal.Decimal.from_float(float(x)), decimal.Decimal.from_float(float(v))
        return -((2 * pi).ln() + vv + xx * xx * (-vv).exp()) / 2
