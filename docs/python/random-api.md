# Random API

FoamNordic uses immutable, backend-neutral keys for stochastic closure and
field-transform execution:

```python
key = fno.Random.key(42, scope="global")

def perturb(velocity, *, key):
    scale_key, noise_key = fno.Random.split(key, 2)
    scale = fno.Random.uniform(scale_key, low=0.995, high=1.005)
    noise = fno.Random.normal(noise_key, shape=velocity.shape, std=1.0e-6)
    return {"velocity": velocity * scale + noise}
```

Keys are never consumed. Reusing one key repeats the same draw; use `split()`
when a function needs several independent draws. The initial distribution
surface is `uniform`, `normal`, `integers`, and `bernoulli`.

At execution time FoamNordic folds the stable program identity and exchange
index into the declared root key. A `global` key is identical on every solver
rank for that invocation. A `rank` key additionally folds in the negotiated
solver rank. Global scope is suitable for one physical perturbation shared by
a decomposed field; rank scope is suitable for independent rank-local work.

`fno.Random.to_jax(key)` materializes a JAX typed key. Equinox residents do
this automatically. NumPy and JAX do not promise bit-identical samples, but
each backend is reproducible for the same FoamNordic key and execution context.

The legacy `seed=` declarations and `rng`/`seed` function injections remain
temporarily available for source compatibility. New code should use `key=`.
