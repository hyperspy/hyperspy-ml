# hyperspy-ml

Machine learning extension for [HyperSpy](https://hyperspy.org) — MVA decomposition, blind source separation, and clustering.

[![Tests status](https://github.com/hyperspy/hyperspy-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/hyperspy/hyperspy-ml/actions/workflows/tests.yml)
[![Documentation status](https://readthedocs.org/projects/hyperspy-ml/badge/?version=latest)](https://hyperspy-ml.readthedocs.io)
[![codecov](https://codecov.io/gh/hyperspy/hyperspy-ml/graph/badge.svg)](https://codecov.io/gh/hyperspy/hyperspy-ml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/hyperspy/hyperspy-ml/main.svg)](https://results.pre-commit.ci/latest/github/hyperspy/hyperspy-ml/main)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`hyperspy-ml` provides multivariate analysis capabilities for HyperSpy signals, built on top of `hyperspy-ml-algorithms` and scikit-learn.

## Installation

```bash
pip install hyperspy-ml
```

For GPU support:

```bash
pip install hyperspy-ml[gpu]
```

## Development

```bash
git clone https://github.com/hyperspy/hyperspy-ml.git
cd hyperspy-ml
pip install -e ".[dev]"
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```

HyperSpy enforces AI attribution standards — `Co-authored-by:` trailers referencing AI tools are blocked by pre-commit. Use `Assisted-by: <tool>:<model>` instead.

## License

GNU General Public License v3.0 or later.
