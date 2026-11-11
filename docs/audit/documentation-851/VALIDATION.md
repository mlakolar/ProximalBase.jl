# Validation and rendered-site inspection

Baseline: post-#850 master `2b373c54`; historical reference
`eff21b6d7d33c622313122f46c6ee71693c101ee`. Implementation branch:
`docs/851-minimal-core-manual`. Linux x86_64, Julia 1.13.3, one BLAS thread.

## Completed local gates

| Check | Result | Complete evidence |
|:--|:--|:--|
| Live API coverage | All 49 exported bindings (module included) and 10 qualified public hooks have explicit grouped reference entries and matrix coverage | logs/documenter.log; logs/api-methods.log; docs/src/api/coverage.md |
| Strict Documenter / doctests / Literate | Pass; exports checked, doctest=true, warnonly=false; all eight scripts run alone and in page sandboxes | logs/documenter.log |
| Retained package inventory | All 18 test files pass, including FP32/FP64 mathematical/storage suites | logs/package-tests.log |
| Frozen-reference integrity | All 102 differential probes pass | logs/benchmark-check.log |
| Rendered internal links / navigation | 48 source/HTML pages; no broken internal links or orphan pages; seven initially collapsed categories | logs/html-structure.log |
| Source scope | All 17 source files preserve executable text exactly after removing docstrings | logs/source-check.log; check_source.py |
| Exceptional contract audit | Infinite-weight/non-finite-trial ambiguity reported separately as #853; documented current behavior | logs/api-methods.log |

The docs build logs include the independent residual/gradient/direct-solve,
orthogonal-lasso, KKT, block and symmetric invariants asserted by examples.
Package tests include independent BigFloat ridge calculations and exceptional
product checks. Complete logs are committed here; CI additionally publishes
HTML and full test/docs logs as separate downloadable artifacts.

## Actual browser inspection

Chrome 157 on Linux loaded generated CI-style directory URLs through a local
HTTP server. Inspected Home, Getting started, Smooth functions, symmetric owner,
Writing your own function, compact block extension example, grouped interface
API and developer overview at 1440×1050; Getting started at 390×844.

- All seven sidebar category labels were clicked open and closed successfully.
- All displayed mathematics rendered with KaTeX; no KaTeX error nodes.
- All inspected code blocks received syntax highlighting.
- Desktop and mobile document widths fit their viewports; long code scrolls
  inside code blocks, preserving standard Documenter behavior.
- Internal link/fragment targets and previous/next footers exist on all 48 pages.
  Actual next/previous links were followed successfully in Chrome. The mobile
  sidebar opened with its menu control and closed on a content click.
- The standard read-only local build omits Documenter's deployment-generated
  siteinfo.js/versions.js; these optional selector scripts are not missing
  authored content. Network abort events during page navigation were observed;
  there were no JavaScript runtime exceptions.

The exact-head browser report is [inspection/browser-report.json](inspection/browser-report.json).
Representative screenshots: [Home](inspection/home.png),
[mathematical function guide](inspection/smooth.png),
[extension code](inspection/extending.png), [mobile tutorial](inspection/getting-started-mobile.png)
and [mobile sidebar](inspection/mobile-sidebar-open.png).
The inspected content head is `6d9174b3ae280fff4a2fad8b78063054b21a08f8`;
this receipt adds evidence without modifying rendered content. Structural validation is reproducible with
`python3 docs/check_html.py`. Live browser checks complement that script.

## Independent review and corrections

[GPT-6.1 Sol high-effort review](https://github.com/mlakolar/ProximalBase.jl/pull/854#pullrequestreview-5861490159)
was posted directly on PR #854. The P2 centered-ridge restriction constant and
P3 ZeroFunction state-copy clarification are addressed in the guides/docstrings
and executable working-set example. Kernel code remains unchanged. The reviewer
also independently checked FP32/FP64 prox stationarity, fused values, aliases,
owner geometry, array steps, API coverage, source preservation and HTML links.
Final re-review and CI results are recorded in the PR conversation.

## Remaining merge gates

The single linked PR must receive independent GPT-6.1 Sol high-effort review
posted directly to GitHub, and all required GitHub checks must pass on the
reviewed head. The final receipt will record review, checks and merge links.
