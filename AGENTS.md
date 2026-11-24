# Working on ProximalBase.jl

## Release branches and PR targets

Follow the [release branches and PR targets](CONTRIBUTING.md#release-branches-and-pr-targets) policy in CONTRIBUTING.md before choosing a branch or making changes.

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
