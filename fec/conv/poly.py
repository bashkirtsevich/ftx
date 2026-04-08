# Convolutional coding polynomials. All are rate 1/2, K=32
# "NASA standard" code by Massey & Costello
# Nonsystematic, quick look-in, dmin=11, dfree=23
# used on Pioneer 10-12, Helios A,B
NASA_POLY1 = 0xbbef6bb7
NASA_POLY2 = 0xbbef6bb5

# Massey-Johannesson code
# Nonsystematic, quick look-in, dmin=13, dfree>=23
# Purported to be more computationally efficient than Massey-Costello
MJ_POLY1 = 0xb840a20f
MJ_POLY2 = 0xb840a20d

# Layland-Lushbaugh code
# Nonsystematic, non-quick look-in, dmin=?, dfree=?
LL_POLY1 = 0xf2d05351
LL_POLY2 = 0xe4613c47
