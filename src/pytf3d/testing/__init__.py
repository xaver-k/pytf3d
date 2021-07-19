try:
    import hypothesis
except ModuleNotFoundError:
    pass
else:
    from ._strategies import QuaternionStrategy, RotationStrategy, UnitQuaternionStrategy
finally:
    del hypothesis
