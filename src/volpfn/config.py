"""Architecture sizes and feature constants."""


SIZES = dict(
    tiny=dict(d=48, layers=2, heads=4, K=3),
    small=dict(d=128, layers=4, heads=4, K=4),
    base=dict(d=256, layers=6, heads=8, K=4),
    large=dict(d=384, layers=8, heads=8, K=5),
)


N_FEAT = 8


HALF_LIVES = (5.0, 20.0, 100.0)
