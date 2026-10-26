# Documentation coverage

Every baseline Markdown document below was read; historical records retain their original claims and dates. Source docstrings are covered file by file in the three implementation ledgers. New audit outputs are the report rather than an additional input corpus.

| File | Status | Review and validation scope |
| --- | --- | --- |
| `CONTRIBUTING.md` | Maintained package entry or contributor guidance | Installation, normalization, naming, workflow, compatibility/removal policy |
| `README.md` | Maintained package entry or contributor guidance | Installation, normalization, naming, workflow, compatibility/removal policy |
| `benchmark/README.md` | Maintained performance instructions | Setup/selection/output commands, analysis controls and exploratory versus replicated evidence |
| `benchmark/gpu_solver/README.md` | Maintained performance instructions | Setup/selection/output commands, analysis controls and exploratory versus replicated evidence |
| `docs/src/api/calculus.md` | Maintained API assembly | Exports/qualified names and autodocs source lists; Documenter cross-reference and export checks |
| `docs/src/api/functions.md` | Maintained API assembly | Exports/qualified names and autodocs source lists; Documenter cross-reference and export checks |
| `docs/src/api/interface.md` | Maintained API assembly | Exports/qualified names and autodocs source lists; Documenter cross-reference and export checks |
| `docs/src/api/iterates.md` | Maintained API assembly | Exports/qualified names and autodocs source lists; Documenter cross-reference and export checks |
| `docs/src/calculus.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/coordinate_descent.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/extending.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/getting_started.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/index.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/interface.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/iterates.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/migration.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/penalties.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/sets.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `docs/src/smooth.md` | Maintained narrative and executable guide | Formulas, assumptions, type/trait/return conventions, device and allocation scope; executable blocks in documentation build |
| `examples/README.md` | Maintained tutorial index | All 18 scripts read; names, commands and check-strength claims; execution in suite and docs |
| `test/cuda/README.md` | Maintained optional validation instructions | Environment/setup/commands, acceptance criteria and physical/oracle scope; historical results separated |
| `test/cuda/validation/discovery-2026-10-25/README.md` | Historical result record | Edition and limitations read; never treated as validation of this audit head |
| `test/oracles/README.md` | Maintained optional validation instructions | Environment/setup/commands, acceptance criteria and physical/oracle scope; historical results separated |
| `test/oracles/RESULTS.md` | Historical result record | Edition and limitations read; never treated as validation of this audit head |

All 18 maintained `examples/*.jl` scripts were read end to end. The test workflow executes each in an independent module; the documentation build also executes their Literate pages. Formula checks include scalar and block updates, symmetric-coordinate factors, Moreau/Fenchel identities, affine Schur complements, spectral reductions, and the IRLS curvature floor.
