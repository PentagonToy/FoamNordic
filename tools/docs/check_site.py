"""Check generated document links and Nordic site contracts before publication."""
import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids, self.links, self.noindex = set(), [], False
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])
        if tag == 'meta' and attrs.get('name') == 'robots' and 'noindex' in attrs.get('content', ''):
            self.noindex = True


def check(site):
    pages = {file: Page(file.read_text()) for file in site.rglob('*.html')}
    errors, count = [], 0
    if not pages:
        raise ValueError(f'No HTML output in {site}')
    for file, page in pages.items():
        if not page.noindex:
            errors.append(f'{file.relative_to(site)}: noindex missing')
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            path = unquote(url.path)
            if path.startswith('/FoamNordic/'):
                path = path[len('/FoamNordic'):]
            target = ((site / path.lstrip('/')) if path.startswith('/') else (file.parent / path)).resolve() if path else file
            if target.is_dir():
                target = target / 'index.html'
            if not target.exists():
                errors.append(f'{file.relative_to(site)}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append(f'{file.relative_to(site)}: missing anchor {link}')
            count += 1
    expected = {
        'foamnordic/installation/index.html': 'tabbed-set',
        'stacknordic/installation/index.html': 'tabbed-set',
        'stacknordic/api/index.html': 'href="https://github.com/PentagonToy/StackNordic"',
        'foamnordic/architecture/principles/index.html': 'arithmatex',
        'foamnordic/architecture/runtime/index.html': 'mermaid',
    }
    for name, marker in expected.items():
        file = site / name
        if not file.exists() or marker not in file.read_text():
            errors.append(f'{name}: expected {marker}')
    if not (site / 'search.json').is_file():
        errors.append('Search index missing')
    if errors:
        raise ValueError('\n'.join(errors))
    print(f'{len(pages)} HTML pages, {count} local links and site contracts passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', type=Path)
    check(parser.parse_args().site.resolve())
