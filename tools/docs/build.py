"""Assemble repository sources, then render the shared site with Zensical."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import hashlib
import shutil
import os
from types import SimpleNamespace

import yaml
from mkdocs.config import load_config
from mkdocs.structure.files import get_files

import sources

ROOT = Path(__file__).resolve().parents[2]


def prepare(workspace: Path, output: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    # MkDocs supplies only the existing source/file adapter, not site rendering.
    config = load_config(str(ROOT / 'mkdocs.yml'), theme={'name': 'mkdocs'}, hooks=[], site_dir=str(output))
    config.plugins._current_plugin = 'source-preparation'
    files = sources.assemble(get_files(config), config)
    staged = workspace / 'docs'
    if staged.exists():
        raise ValueError(f'Use an empty workspace to avoid stale pages: {workspace}')
    for file in files:
        destination = staged / file.src_uri
        destination.parent.mkdir(parents=True, exist_ok=True)
        if file.is_documentation_page():
            content = sources.rewrite_links(file.content_string, SimpleNamespace(file=file), config, files)
            if file.src_uri.startswith('stacknordic/'):
                content = '---\nnordic_project: StackNordic\n---\n\n' + content
            destination.write_text(content)
        else:
            destination.write_bytes(file.content_bytes)
    settings = yaml.load((ROOT / 'mkdocs.yml').read_text(), Loader=yaml.Loader)
    settings.pop('hooks', None)
    settings['docs_dir'] = 'docs'
    settings['site_dir'] = 'site'
    shutil.copytree(ROOT / 'tools/docs/overrides', workspace / 'overrides')
    settings['theme']['custom_dir'] = 'overrides'
    revision = hashlib.sha256((ROOT / 'others/assets/icon.svg').read_bytes()).hexdigest()[:12]
    for key in ('logo', 'favicon'):
        settings['theme'][key] += '?revision=' + revision
    target = workspace / 'mkdocs.yml'
    target.write_text(yaml.dump(settings, sort_keys=False))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True, help='Directory for temporary source preparation.')
    parser.add_argument('--site-dir', type=Path, required=True)
    args = parser.parse_args()
    workspace, output = args.workspace.resolve(), args.site_dir.resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Output must not contain the repository.')
    if output.exists() and not (output / '.nordic-site').is_file():
        raise ValueError(f'Refusing to replace an unmarked directory: {output}')
    workspace.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sources-', dir=workspace) as temporary:
        target = prepare(Path(temporary), output)
        subprocess.run([sys.executable, '-m', 'zensical', 'build', '--strict', '-f', str(target)], check=True, cwd=target.parent)
        rendered = target.parent / 'site'
        config = load_config(str(ROOT / 'mkdocs.yml'), theme={'name': 'mkdocs'}, hooks=[], site_dir=str(rendered))
        config.plugins._current_plugin = 'source-preparation'
        sources.write_redirects(config)
        (rendered / '.nordic-site').write_text('Generated Nordic documentation\n')
        # Keep the previous generated site available until its replacement is ready.
        with tempfile.TemporaryDirectory(prefix='publish-', dir=output.parent) as publication:
            pending, previous = Path(publication) / 'site', Path(publication) / 'previous'
            shutil.copytree(rendered, pending)
            if output.exists():
                os.replace(output, previous)
            try:
                os.replace(pending, output)
            except OSError:
                if previous.exists():
                    os.replace(previous, output)
                raise


if __name__ == '__main__':
    main()
