# JAX connector

JAX execution belongs in a managed resident worker with a compiled callable.
It will consume the common packed tensor contract and keep Python outside the
OpenFOAM process. JAX itself is optional and is never a core dependency.
