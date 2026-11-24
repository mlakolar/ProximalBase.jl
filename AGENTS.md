# Working on ProximalBase.jl

## Release branches and PR targets

- `master` is the 0.7 development line. New families, new public APIs and architectural changes target master.
- `release/0.6` is feature-frozen. Only demonstrated correctness fixes, focused performance-regression fixes, tests, documentation and release-readiness work are eligible. GroupMCP and GroupSCAD belong to 0.7.
- Use `release:0.6` and `release:0.7+` to identify the intended scope. A shared fix normally lands on master and is backported by a separate reviewed PR if applicable to 0.6. An urgent release-first fix must be independently forward-ported. Never merge master wholesale into the release branch.
- Before editing, verify the intended base and current SHA. Review and test each branch-specific diff on its actual target; preserve the existing numerical, CI and performance gates. Do not merge, tag or publish without the maintainer's approval.
- Keep ProximalSolvers on tested 0.6 pins unless its maintainer explicitly approves a different dependency. Downstream adoption is tracked separately from feature freeze; see #1204 and #527.

## Find the right owner

- [API coverage](docs/src/api/coverage.md) maps public symbols to their retained operations and tests.
- [Numerical contracts](docs/src/spec/index.md) give the family-specific rules and status vocabulary. [Array and precision conventions](docs/src/numerical_usage.md) and [numerical range and endpoint rules](docs/src/numerics.md) cover shared behavior.
- [Function and oracle dispatch](docs/src/developer/oracle_dispatch.md) and [numerical routines](docs/src/developer/routines/index.md) describe the implementation. Consult the current source and tests when checking an implementation detail.
- [Writing documentation](docs/src/developer/writing_documentation.md) covers stable anchors, evidence and executable examples. [The change process](docs/src/developer/change_process.md) covers validation.

## Validate changes

From the repository root, use the commands in [CONTRIBUTING.md](CONTRIBUTING.md).
The strict documentation build checks docstrings, doctests and executable
examples. Check the rendered navigation and links with
`python3 docs/check_html.py`. Run the contract schema check with
`python3 docs/check_contract_schema.py` when changing a family page.

Author runnable Literate examples in `examples/`. Documenter generates
`docs/src/examples/` and the copies of the design contracts under
`docs/src/developer/contracts/`.
