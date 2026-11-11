#!/usr/bin/env python3
"""Check generated pages, navigation, local links/assets and anchor targets."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json
import subprocess

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
    tracked = set(subprocess.check_output(['git', 'ls-files'], cwd=ROOT.parent, text=True).splitlines())
    link_count = 0
    checked_source_links = 0
    for page in pages.values():
        for href in page.links + page.assets:
            url = urlsplit(href)
            for mode in ('blob', 'edit'):
                prefix = f'/mlakolar/ProximalBase.jl/{mode}/master/'
                if url.netloc == 'github.com' and url.path.startswith(prefix):
                    checked_source_links += 1
                    source = unquote(url.path[len(prefix):])
                    if source not in tracked:
                        errors.append(f'{page.path.relative_to(BUILD)}: untracked repository source/edit target {href}')
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
        assert len(page.toggles) == 8, f'Expected eight shallow expandable categories: {page.path}'
    # These pages are generated from docs/design/*.md. Their math and code
    # must render as formatted elements, not plain prose in the HTML.
    contracts = [
        'developer/contracts/block-prox-0.6',
        'developer/contracts/iterate-family-0.6',
        'developer/contracts/symmetric-sparse-iterate-0.6',
    ]
    for base in contracts:
        flat = (BUILD / (base + '.html')).resolve()
        pretty = (BUILD / base / 'index.html').resolve()
        target = flat if flat in pages else pretty
        if target not in pages:
            errors.append(f'Missing rendered contract: {base}')
            continue
        contract = pages[target]
        if contract.code < 10:
            errors.append(f'{base}: expected formatted Julia API names and code blocks')
        if contract.math < 1:
            errors.append(f'{base}: expected rendered mathematical expressions')
    # All authored/generated Markdown pages must render. Only the historical
    # storage landing page and old interface URL are hidden from navigation.
    hidden_pages = {'storage.md', 'interface.md'}
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
        elif rel.as_posix() not in hidden_pages and dest not in nav_targets:
            errors.append(f'Orphan source page: {rel}')
        source_count += 1
    assert all('checked' not in toggle for toggle in home.toggles), 'Home categories must start collapsed'
    assert home.math >= 2 and home.code, 'Home should contain mathematics and executable code'
    assert not errors, '\n'.join(errors)
    report = dict(html_pages=len(pages), source_pages=source_count,
                  checked_local_links_and_assets=link_count,
                  checked_repository_source_links=checked_source_links,
                  expandable_categories=len(home.toggles),
                  home_categories_initially_collapsed=True,
                  broken_links=0, orphan_pages=0,
                  note='Structural validation; actual browser inspection is recorded separately.')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
