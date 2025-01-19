def test_fn(x, *args, y=1, **kw):
    print(x, args, y, kw)

test_fn(2, 3, 4,  3)