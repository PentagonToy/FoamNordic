# Python API documentation

FoamNordic's installable Python control plane compiles immutable case, closure,
observation, placement, and scheduler declarations into a deterministic native
plan. `Longship.launch()` prepares an isolated case and returns the
cancellation-safe `Run`; `Run.stop()` produces one durable `Result`. Native C++
remains authoritative for placement arithmetic, field exchange, inference,
and solver integration.

- [API design](design.md)
- [Provisional run-control API](run-control-api.md)
- [Postprocess API](postprocess-api.md)
- [Wheel publishing](publishing.md)
- [Native C++ documentation](../native/README.md)

The earlier SmartSim- and Redis-based FoamNordic APIs are historical reference
material, not compatibility requirements for this repository.
