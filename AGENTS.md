<!-- Generated: 2026-06-25 -->

# hyperspy-ml

## Purpose
`hyperspy-ml` is a HyperSpy extension package providing multivariate analysis (MVA) tools: decomposition (PCA, SVD, NMF, RPCA, MLPCA), blind source separation (ICA), and clustering. It wraps `hyperspy-ml-algorithms` and scikit-learn to expose these capabilities through the HyperSpy signal API.

## Key Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Build system config, dependencies, optional extras |
| `README.md` | Project overview and quick-start |
| `CHANGES.rst` | Changelog |
| `.readthedocs.yaml` | Read the Docs build config |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `hyperspy_ml/` | Main Python package |
| `doc/` | Sphinx documentation source |
| `upcoming_changes/` | Towncrier news fragments for the next release |
| `scripts/` | Development tooling for pre-commit hooks and CI checks |

## For AI Agents

### Working in This Repository

- Run tests: `pytest hyperspy_ml/tests/`
- Lint: `ruff check hyperspy_ml/`
- Format: `ruff format hyperspy_ml/`
- Add changelog entries for every user-facing change in `upcoming_changes/` (see format in `upcoming_changes/README.rst`)
- Never edit `.rst` files in `doc/_build/` — those are generated

### Testing Requirements

```bash
pytest hyperspy_ml/tests/                  # full suite
pytest -x hyperspy_ml/tests/              # stop on first failure
```

## Dependencies

### Internal
- `hyperspy-ml-algorithms` — standalone ML algorithm implementations
- `hyperspy` — parent library providing signal framework

### External
- `scikit-learn` — decomposition and clustering algorithms
- `numpy` — numerical core (via hyperspy)
- `matplotlib` — plotting
- `array-api-compat` — array API compatibility layer
- `zarr` — array storage

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->

## AI Agent Setup

Before implementing any changes, verify that the development environment
is correctly configured.  Do **not** start editing files until these pass:

- ``python -c "import hyperspy_ml"`` — confirms the package is installed in
  editable mode.  If this fails: **stop**.  The current environment may not
  be set up.  Ask the user whether to proceed with
  ``pip install -e ".[dev]"`` — do not install into an unknown environment.

- ``pre-commit install`` — activates lint, format, and AI co-author checks
  on every commit.  If ``pre-commit`` is not installed, ask the user
  whether to install it (``pip install pre-commit``).

- ``pre-commit run --all-files`` — should pass cleanly.  Fix any reported
  issues before proceeding.

- **Disable ``Co-authored-by:`` injection.**  Many AI coding tools add a
  ``Co-authored-by:`` trailer to commits by default (Claude Code, Cursor,
  GitHub Copilot, etc.).  HyperSpy **blocks** these with a pre-commit
  hook — your commits will fail.  Before editing any files, check your
  tool's settings and disable any feature that automatically injects AI
  attribution.  HyperSpy uses ``Assisted-by: <tool>:<model>`` instead;
  add it manually to each commit.

  For common tools:
  * **Claude Code**: set ``"includeCoAuthoredBy": false`` in settings
  * **Cursor**: disable commit-message AI attribution in Cursor Settings
  * **GitHub Copilot**: disable ``github.copilot.chat.commitMessageGeneration``
  * **OpenCode / oh-my-openagent**: check skill and plugin configs
    that inject AI co-author trailers

  If unsure, run the pre-commit hook against a test commit message::

      echo "test" | pre-commit run check-ai-co-author --hook-stage commit-msg

Add setup steps to the first todo item of every session — do not skip it.

## Agent Completion Checklist

Before claiming any task is complete, verify ALL of the following:

### Code Quality
- [ ] `ruff check` passes on all changed files — zero new errors
- [ ] `ruff format` applied — no formatting inconsistencies

### Testing
- [ ] Affected tests pass: `pytest hyperspy_ml/tests/`
- [ ] New code has corresponding tests that mirror the source structure
- [ ] Floating-point comparisons use `np.testing.assert_allclose`, not raw `==`

### Changelog
- [ ] Every user-facing change has an `upcoming_changes/<issue>.<type>.rst` entry
- [ ] The `<type>` matches one of: `new`, `bugfix`, `doc`, `deprecation`, `enhancements`, `api`, `maintenance`

### Documentation
- [ ] New public API has updated docstrings
- [ ] Never edit `.rst` files in `doc/_build/` — those are generated

### Commits
- [ ] Commit following best practices (atomic units, repo-consistent messages, no secrets)
- [ ] MUST NOT use ``Co-authored-by:`` trailer for AI tools — use ``Assisted-by: <tool>:<model>`` instead
- [ ] Never push unless explicitly asked

### Repository Hygiene
- [ ] Never modify AGENTS.md generated sections — only add notes below `<!-- MANUAL -->` lines
- [ ] Never suppress type/lint errors with blanket ignores (`# type: ignore`, `# noqa` without justification)

### Pre-CI Validation (run BEFORE pushing)
- [ ] `python scripts/check-docs.py` passes — validates fragment filenames, `towncrier --draft`, and Sphinx build with warnings-as-errors (equivalent to CI doc build)
