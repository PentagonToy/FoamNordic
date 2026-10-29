"""Enhance portable repository Markdown only when building the site."""
import re


def enhance(markdown):
    def details(match):
        title, body = match.groups()
        content = '\n'.join('    ' + line if line else '' for line in body.strip().splitlines())
        return f'??? info "{title}"\n\n{content}\n'

    markdown = re.sub(r'<details>\n<summary>([^<]+)</summary>\n([\s\S]*?)</details>', details, markdown)

    def tabs(match):
        indent, body = match.groups()
        chunks = re.split(r'^' + indent + r'### (.+)\n', body, flags=re.M)
        result = []
        for index in range(1, len(chunks), 2):
            result.append(indent + '=== "' + chunks[index] + '"\n')
            result.extend('    ' + line if line else '' for line in chunks[index + 1].strip('\n').splitlines())
        return '\n'.join(result) + '\n'

    return re.sub(r'^( *)<!-- tabs -->\n([\s\S]*?)^\1<!-- /tabs -->', tabs, markdown, flags=re.M)
