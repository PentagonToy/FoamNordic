<p align="center">
  <img src="others/icon.png" alt="FoamNordic" width="220">
</p>

# FoamNordic

FoamNordic is a native C++ foundation for blocking, atomic machine-learning
closure exchange with OpenFOAM. Its hot path keeps fields outside Python and
uses UDS or shared memory for same-node coupling.

The project is under active research development. The native runtime and
OpenFOAM adapter are implemented; the public Python orchestration API is still
being designed.

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture/README.md)
- [Native C++ internals](docs/native/README.md)
- [Python API status and design](docs/python/README.md)

Linux x86_64 with OpenFOAM v2512 is the primary target. macOS ARM64 with
OpenFOAM v2606 is used as the second development platform.

## License

FoamNordic is distributed under the
[GNU General Public License v3.0](LICENSE).
