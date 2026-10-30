# Documentation site

The [Nordic documentation site](https://pentagontoy.github.io/FoamNordic/) separates FoamNordic, StackNordic and shared workflows. The website is public; repository visibility remains separate. Every HTML page includes `noindex` to request exclusion from search results, without restricting visitors who know the URL.

## Sources and build

FoamNordic pages come from this repository. StackNordic pages come from its own `docs/` directory and README at the commit recorded in `tools/docs/stacknordic.ref`. Edit each project's original documents; the site creates no second source copy. `docs/home.md` and `docs/workflows/` own the shared overview and hand-off guides.

For a local build, set the StackNordic checkout explicitly or place it beside FoamNordic. Use the documentation environment with `requirements-docs.txt` installed. Zensical renders the site; MkDocs core remains pinned only for the existing source/file adapter:

```console
STACKNORDIC_DOCS_ROOT=/path/to/StackNordic python tools/docs/build.py --workspace /path/to/output/work --site-dir /path/to/output/site
```

The build script assembles temporary Markdown, rewrites project links and copies SVG assets before invoking Zensical. The Zensical renderer uses no Python hooks. Repeated builds replace only marked generated output; source documents remain untouched. Source-code links remain on GitHub and follow that repository's permissions. Builds execute neither notebooks nor scientific workloads.

## Publication

The Documentation workflow builds the pinned sources and deploys relevant pushes to `main`. For StackNordic updates:

1. Edit and publish the original documents in StackNordic.
2. Set `tools/docs/stacknordic.ref` in FoamNordic to that published commit's full SHA.
3. Build and check the shared site, then publish the FoamNordic update.
4. Check the [Sources page](https://pentagontoy.github.io/FoamNordic/versions/) for the deployed pair of revisions.

Keep the project URLs stable: [FoamNordic](https://pentagontoy.github.io/FoamNordic/foamnordic/), [StackNordic](https://pentagontoy.github.io/FoamNordic/stacknordic/) and [shared workflows](https://pentagontoy.github.io/FoamNordic/workflows/foamnordic-stacknordic/). A StackNordic push alone does not update the shared site.

`mkdocs.yml` retains the compatible navigation and Markdown settings. Zensical uses its modern theme with shared light/dark styling. The HTML override supplies the indexing policy. Check both project sections and shared links after changing site configuration. Cite matching package versions, document revisions and validation conditions in a paper.

## Notebook examples

Selected notebook cells are imported as copyable code during the documentation build without execution. Edit the original notebook to update its page. The basic-run, field-exchange and closure pages publish code only, without historical machine logs. Saved outputs are previews, not new validation results. The generated pages use the current notebook source, including its concise workload declarations and Hedgehogs figure helpers. Keep common options in examples and describe advanced execution settings in the API reference.

## Portable Markdown

Repository pages use standard Markdown and HTML details. Installation option groups use hidden `tabs` comments; source preparation converts them to rendered content tabs. Check both ordinary Markdown and the built site after editing shared pages.

## Artwork

`others/assets/icon.svg` is the canonical symbol used by the documentation home, header and favicon. The README uses `banner_light.svg` and `banner_dark.svg` from the same directory; GitHub and the site select the matching theme. Keep the symbol background and internal cut-outs transparent; check both themes after artwork changes. Source preparation adds a content-hash query to SVG references, including the header and favicon, to invalidate cached artwork. Browser-owned tab backgrounds remain outside site styling.

Run `python tools/docs/check_site.py /path/to/output/site` to check local links, anchors, indexing metadata and key generated features. The publication workflow runs the same check.

## Local preview

Serve the generated output with `python -m http.server 8000 --directory /path/to/output/site`. Check both themes, installation tabs, search, mathematics, diagrams, source links and legacy redirects. `tools/docs/build.py` stages each build independently, so no generated document copy needs maintenance.
