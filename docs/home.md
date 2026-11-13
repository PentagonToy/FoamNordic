<div class="nordic-banner">
  <img src="../others/assets/nordic_banner_light.svg#gh-light-mode-only" alt="Nordic documentation" width="960">
  <img src="../others/assets/nordic_banner_dark.svg#gh-dark-mode-only" alt="Nordic documentation" width="960">
</div>

# Nordic documentation

Run OpenFOAM workloads with FoamNordic and prepare stored results with StackNordic. Each project has its own API and can be used independently.

<div class="nordic-projects" markdown="1">
<div class="nordic-project-card" markdown="1">

## FoamNordic

Run, monitor and couple OpenFOAM fields to mathematical or learned models.

[Install](foamnordic/installation.md) → [First OpenFOAM run](foamnordic/getting-started.md#quick-start)

[API reference](foamnordic/api/README.md)

</div>
<div class="nordic-project-card" markdown="1">

## StackNordic

Reconstruct results, export datasets and apply declared field reductions.

[Install](stacknordic/installation.md) → [First reconstruction](stacknordic/getting-started.md#reconstruct-a-case)

[API reference](stacknordic/api/README.md)

</div>
</div>

## Choose the next step

| Goal | Guide |
| --- | --- |
| Run an unchanged case locally | [Basic OpenFOAM run](foamnordic/examples/lid-driven-cavity.md) |
| Exchange fields or add a closure on Slurm | [FoamNordic tutorials](foamnordic/tutorials.md) |
| Turn stored results into a dataset | [StackNordic case-to-dataset example](stacknordic/tutorials/case-to-dataset.md) |
| Use both projects in one workflow | [Run to dataset](workflows/foamnordic-stacknordic.md) |

## Sources and versions

Documents remain in their project repositories. The site imports a fixed StackNordic revision and records both [source commits](versions.md). See [Site maintenance](foamnordic/developer/documentation.md) for builds and updates.
