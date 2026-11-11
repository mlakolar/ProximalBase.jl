#!/usr/bin/env python3
"""Check generated pages, navigation, local links/assets and anchor targets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'

class Page(HTMLParser):
    def __init__(self, path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids, self.links, self.assets = set(), [], []
        self.menu_links, self.toggles = [], []
        self.in_sidebar = False
        self.math = self.code = self.footer = 0
        self.feed(path.read_text())
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if 'id' in a:
            self.ids.add(a['id'])
        if tag == 'nav' and a.get('class') == 'docs-sidebar':
            self.in_sidebar = True
        if tag == 'nav' and a.get('class') == 'docs-footer':
            self.footer += 1
        if tag == 'a' and a.get('href'):
            self.links.append(a['href'])
            if self.in_sidebar:
                self.menu_links.append(a['href'])
        if tag in ('script', 'img') and a.get('src'):
            self.assets.append(a['src'])
        if tag == 'link' and a.get('rel') == 'stylesheet':
            self.assets.append(a.get('href', ''))
        if tag == 'input' and a.get('class') == 'collapse-toggle':
            self.toggles.append(a)
        if 'math-container' in a.get('class', ''):
            self.math += 1
        if tag == 'code':
            self.code += 1
    def handle_endtag(self, tag):
        if tag == 'nav':
            self.in_sidebar = False

def resolve(page, href):
    url = urlsplit(href)
    if url.scheme or url.netloc:
        return None
    dest = (page.path.parent / unquote(url.path)).resolve() if url.path else page.path
    if dest.is_dir():
        dest /= 'index.html'
    return dest, unquote(url.fragment)

def main():
    pages = {p.resolve(): Page(p.resolve()) for p in BUILD.rglob('*.html')}
    assert pages, 'No generated HTML; run docs/make.jl first'
    errors = []
    link_count = 0
    for page in pages.values():
        for href in page.links + page.assets:
            # Documenter adds these version-selector scripts during deployment,
            # not makedocs. Their absence in a local artifact is expected.
            if href in page.assets and urlsplit(href).path.split('/')[-1] in ('siteinfo.js', 'versions.js'):
                continue
            target = resolve(page, href)
            if target is None:
                continue
            link_count += 1
            dest, fragment = target
            if not dest.is_file():
                errors.append(f'{page.path.relative_to(BUILD)}: missing {href}')
            elif fragment and dest in pages and fragment not in pages[dest].ids:
                errors.append(f'{page.path.relative_to(BUILD)}: missing anchor {href}')
        assert page.footer, f'Missing previous/next footer: {page.path}'
        assert len(page.toggles) == 7, f'Expected seven shallow expandable categories: {page.path}'
    # Every authored/generated Markdown page must be rendered and reachable in
    # navigation. The historical storage landing page is intentionally hidden.
    home = pages[(BUILD / 'index.html').resolve()]
    nav_targets = {resolve(home, href)[0] for href in home.menu_links if resolve(home, href)}
    source_count = 0
    for source in (ROOT / 'src').rglob('*.md'):
        rel = source.relative_to(ROOT / 'src')
        flat = (BUILD / rel).with_suffix('.html').resolve()
        pretty = (BUILD / rel.with_suffix('') / 'index.html').resolve()
        # Documenter treats a directory index.md as directory/index.html.
        if rel.name == 'index.md':
            pretty = (BUILD / rel).with_suffix('.html').resolve()
        dest = flat if flat in pages else pretty
        if dest not in pages:
            errors.append(f'Unrendered source page: {rel}')
        elif rel.as_posix() != 'storage.md' and dest not in nav_targets:
            errors.append(f'Orphan source page: {rel}')
        source_count += 1
    assert all('checked' not in toggle for toggle in home.toggles), 'Home categories must start collapsed'
    assert home.math >= 2 and home.code, 'Home should contain mathematics and executable code'
    assert not errors, '\n'.join(errors)
    report = dict(html_pages=len(pages), source_pages=source_count,
                  checked_local_links_and_assets=link_count,
                  expandable_categories=len(home.toggles),
                  home_categories_initially_collapsed=True,
                  broken_links=0, orphan_pages=0,
                  note='Structural validation; actual browser inspection is recorded separately.')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
