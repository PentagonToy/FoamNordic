"""Render reviewed notebook cells and saved outputs without executing code."""
import base64
import hashlib
import json
import re
from html.parser import HTMLParser

from mkdocs.structure.files import File


class Table(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.rows = []
        self.cell = None
        self.row = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        if tag == 'tr':
            self.row = []
        elif tag in ('td', 'th'):
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ('td', 'th') and self.cell is not None:
            self.row.append(' '.join(''.join(self.cell).split()).replace('|', '\\|'))
            self.cell = None
        elif tag == 'tr' and self.row:
            self.rows.append(self.row)


def text(value):
    return value if isinstance(value, str) else ''.join(value)


def render_notebook(source, title, introduction, config, files, outputs=True):
    notebook = json.loads(source.read_text())
    result = f'# {title}\n\n{introduction}\n\n'
    image_count = 0
    for cell in notebook['cells']:
        content = text(cell.get('source', [])).strip()
        if not content:
            continue
        if cell['cell_type'] == 'markdown':
            if content.startswith('# ') and content != '# Preamble':
                continue
            result += re.sub(r'^# ', '## ', content, flags=re.M) + '\n\n'
        elif cell['cell_type'] == 'code':
            result += '````python\n' + content + '\n````\n\n'
            if not outputs:
                continue
            for output in cell.get('outputs', []):
                data = output.get('data', {})
                if 'image/png' in data:
                    image_count += 1
                    payload = base64.b64decode(text(data['image/png']))
                    revision = hashlib.sha256(payload).hexdigest()[:12]
                    name = f'assets/tutorials/{source.stem}-{image_count}-{revision}.png'
                    files.append(File.generated(config, name, content=payload))
                    result += f'![Saved notebook output](/' + name + '){ .notebook-output }\n\n'
                elif 'text/html' in data and '<table' in text(data['text/html']):
                    rows = Table(text(data['text/html'])).rows
                    if rows:
                        result += '| ' + ' | '.join(rows[0]) + ' |\n'
                        result += '| ' + ' | '.join('---' for _ in rows[0]) + ' |\n'
                        result += ''.join('| ' + ' | '.join(row) + ' |\n' for row in rows[1:]) + '\n'
                else:
                    value = output.get('text') if output.get('output_type') == 'stream' else data.get('text/plain')
                    if value:
                        clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text(value)).strip()
                        if clean:
                            result += '````text\n' + clean + '\n````\n\n'
    return result.replace('](/assets/tutorials/', '](../assets/tutorials/')
