"""Build a shared site from independent repository documents."""
import os
import hashlib
import posixpath
import re
import subprocess
import sys
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/docs'))
from notebooks import render_notebook
from portable_markdown import enhance

STACK = Path(os.environ.get('STACKNORDIC_DOCS_ROOT', ROOT.parent / 'StackNordic')).resolve()
SOURCES = {}
ROUTES = {}
REPOSITORIES = {ROOT: 'https://github.com/PentagonToy/FoamNordic', STACK: 'https://github.com/PentagonToy/StackNordic'}


def assemble(files, config):
    if not (STACK / 'docs/README.md').is_file():
        raise ValueError('Set STACKNORDIC_DOCS_ROOT to a StackNordic checkout containing docs/.')
    SOURCES.clear()
    ROUTES.clear()

    def add(name, source, owner):
        source = source.resolve()
        SOURCES[name] = (source, owner)
        ROUTES[source] = name
        files.append(File.generated(config, name, abs_src_path=str(source)))

    for file in list(files):
        if file.is_documentation_page() and file.src_uri == 'home.md':
            files.remove(file)
            add('index.md', ROOT / 'docs/home.md', ROOT)
        elif file.is_documentation_page() and not file.src_uri.startswith('workflows/'):
            files.remove(file)
            add('foamnordic/' + file.src_uri, ROOT / 'docs' / file.src_uri, ROOT)
    for source in sorted((STACK / 'docs').rglob('*.md')):
        add('stacknordic/' + source.relative_to(STACK / 'docs').as_posix(), source, STACK)
    add('foamnordic/getting-started.md', ROOT / 'README.md', ROOT)
    add('foamnordic/tutorials.md', ROOT / 'tutorials/README.md', ROOT)
    add('stacknordic/getting-started.md', STACK / 'README.md', STACK)
    for owner, prefix in [(ROOT, 'foamnordic'), (STACK, 'stacknordic')]:
        for source in sorted((owner / 'others').rglob('*.svg')):
            add('assets/' + prefix + '/' + source.relative_to(owner / 'others').as_posix(), source, owner)
    examples = [
        ('openfoam_lid_driven_cavity', 'lid-driven-cavity', 'Basic run: local cavity',
         'Run an unchanged OpenFOAM case through Longship with empty closures, transforms and observations.',
         'This example runs locally with `scheduler=None` and OpenFOAM.app (`of_cmd="openfoam"`, `shell="zsh"`). On Linux, select the local OpenFOAM activation and shell in the case declaration.',
         'A successful run reports success and exposes U and p through postprocessing. Field sizes and values depend on the case.'),
        ('field_exchange_lid_driven_cavity', 'field-exchange', 'Field exchange: Slurm cavity',
         'Apply a small seeded velocity perturbation at the start of each time step, observe U and p, and inspect the completed case.',
         'This example uses Slurm and `module load openfoam/2512` with bash. Replace the allocation account, partition and resource settings with those supported by the cluster before submitting. Run the notebook client where Slurm commands and the selected OpenFOAM module are available. The random perturbation illustrates a field transform; it does not define an equation-level closure.',
         'Inspect the run result and observation records for U and p, then load the final fields. The structured visualisation assumes the tutorial cavity grid; saved shapes and mesh dimensions should be checked against the actual case.'),
        ('smagorinsky_pitz_daily', 'smagorinsky-closure', 'Closure: Smagorinsky on Slurm',
         'Declare a native symbolic Smagorinsky operator and couple it through the nutFjord LES adapter in a Pitz Daily case.',
         'This example uses Slurm and `module load openfoam/2512` with bash. Replace the allocation account, partition and resources with local values. The supplied tutorial case and the built nutFjord adapter are required. The mathematical closure uses FoamNordic algebra and requires no trained-model download.',
         'Check successful completion and postprocessed U and p. The example demonstrates the closure integration path; it does not establish a new turbulence-model validation result.'),
    ]
    for stem, slug, title, purpose, environment, outcome in examples:
        source = ROOT / 'tutorials/foamnordic_tutorials/incompressible' / f'{stem}.ipynb'
        name = f'foamnordic/examples/{slug}.md'
        introduction = purpose + '\n\n## Before running\n\n[Install FoamNordic](../installation.md), build its integration and check the OpenFOAM environment. Start inside a FoamNordic checkout containing the tutorial cases. Runs are copied below the notebook output directory.\n\n' + environment
        introduction += f'\n\nRead or download [the original notebook](https://github.com/PentagonToy/FoamNordic/blob/main/tutorials/foamnordic_tutorials/incompressible/{stem}.ipynb). Run the original cells below in order.'
        content = render_notebook(source, title, introduction, config, files, outputs=False)
        content += '\n## Check the result\n\n' + outcome + '\n'
        files.append(File.generated(config, name, content=content))
        SOURCES[name] = (source, ROOT)
    revisions = [("FoamNordic", ROOT, subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()),
                 ("StackNordic", STACK, (ROOT / 'tools/docs/stacknordic.ref').read_text().strip())]
    content = '# Documentation sources\n\n| Project | Document revision |\n| --- | --- |\n'
    for name, owner, ref in revisions:
        content += f'| {name} | [{ref}]({REPOSITORIES[owner]}/commit/{ref}) |\n'
    content += '\nThe build combines these source revisions. Package compatibility and numerical validation remain documented within each project.\n'
    files.append(File.generated(config, 'versions.md', content=content))
    return files


def rewrite_links(markdown, page, config, files):
    # The site toggle can differ from the operating-system theme used by picture.
    markdown = re.sub(
        r'<picture>\s*<source media="\(prefers-color-scheme: dark\)" srcset="([^"]+)">\s*<img src="([^"]+)"([^>]+)>\s*</picture>',
        lambda m: f'<img src="{m[1]}#gh-dark-mode-only"{m[3]}>\n<img src="{m[2]}#gh-light-mode-only"{m[3]}>',
        markdown,
    )
    markdown = enhance(markdown)
    source, owner = SOURCES.get(page.file.src_uri, (ROOT / 'docs' / page.file.src_uri, ROOT))

    def resolve(target, html=False):
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            for project, repository in REPOSITORIES.items():
                if target.rstrip('/') == repository:
                    local = ROUTES[(project / 'docs/README.md').resolve()]
                    return posixpath.relpath(local, posixpath.dirname(page.file.src_uri) or '.')
            return target
        local = posixpath.normpath(posixpath.join(posixpath.dirname(page.file.src_uri), unquote(url.path)))
        if files.get_file_from_path(local) is None:
            destination = (source.parent / unquote(url.path)).resolve()
            local = ROUTES.get(destination)
            if local is None:
                if destination.is_relative_to(owner) and destination.exists():
                    kind = 'tree' if destination.is_dir() else 'blob'
                    ref = (ROOT / 'tools/docs/stacknordic.ref').read_text().strip() if owner == STACK else 'main'
                    return f'{REPOSITORIES[owner]}/{kind}/{ref}/{quote(destination.relative_to(owner).as_posix())}' + (f'#{url.fragment}' if url.fragment else '')
                raise ValueError(f'Unresolved repository link in {source}: {target}')
        result = posixpath.relpath(local, posixpath.dirname(page.file.src_uri) or '.')
        query = url.query
        if local.endswith('.svg') and not query:
            asset = files.get_file_from_path(local)
            query = 'revision=' + hashlib.sha256(Path(asset.abs_src_path).read_bytes()).hexdigest()[:12]
        return result + (f'?{query}' if query else '') + (f'#{url.fragment}' if url.fragment else '')

    if page.file.src_uri == 'foamnordic/tutorials.md':
        markdown = markdown.replace('## Run a tutorial', 'Notebook sources are linked on GitHub; access follows repository permissions.\n\n## Run a tutorial')
    markdown = re.sub(r'(\]\()([^\s)]+)(\))', lambda m: m[1] + resolve(m[2]) + m[3], markdown)
    return re.sub(r'(\b(?:src|href)=["\'])([^"\']+)(["\'])', lambda m: m[1] + resolve(m[2], html=True) + m[3], markdown)


def write_redirects(config):
    """Keep bookmarks to the former FoamNordic-only paths working."""
    aliases = {'getting-started.md': 'foamnordic/getting-started.md', 'tutorials.md': 'foamnordic/tutorials.md'}
    for source, target in ROUTES.items():
        if source.is_relative_to(ROOT / 'docs') and target.startswith('foamnordic/'):
            original = source.relative_to(ROOT / 'docs').as_posix()
            if original != 'README.md':
                aliases[original] = target
    for original, target in aliases.items():
        old = File.generated(config, original, content='')
        new = File.generated(config, target, content='')
        destination = escape(new.url_relative_to(old), quote=True)
        path = Path(old.abs_dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex"><meta http-equiv="refresh" content="0; url={destination}"><title>FoamNordic documentation</title></head><body><a href="{destination}">Continue to FoamNordic documentation</a></body></html>\n')
