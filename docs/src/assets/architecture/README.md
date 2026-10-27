# Architecture diagram sources

These six Mermaid sources and their SVG renders belong to the maintained
[package-structure chapter](../../architecture.md). The initial views were
recovered from `docs/audit/release-0.5.4/architecture`; that directory remains a
historical record. Update these maintained sources when the implementation
changes.

Render with `@mermaid-js/mermaid-cli` 12.0.0 and Chrome/Chromium. Install the
renderer outside the package environment. With a Puppeteer configuration
that selects your browser, run from this directory:

```sh
for source in 0*.mmd; do
    mmdc -p /path/to/puppeteer.json -c mermaid.json --no-font-embed \
        --size 1300 -b white -I "pb-${source%.mmd}" \
        -i "$source" -o "${source%.mmd}.svg"
done
```

The configuration makes diagram IDs deterministic. SVGs use system fonts;
font binaries are not embedded. The documentation build serves the committed
SVGs and does not require Node.js, a browser, or a Mermaid service. Inspect all
renders after changing the sources. Keep inheritance, composition, dispatch
and runtime-call edges consistent with the chapter's legend.
