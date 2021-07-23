[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/xaver-k/pytf3d/master.svg)](https://results.pre-commit.ci/latest/github/xaver-k/pytf3d/master)

# `pytf3d` - Python Transformations in 3D

`pytf3d` is intended as a lightweight library for handling rigid body transformations in 3D.
In contrast to existing python libraries out there, `pytf3d` focuses on:
* lightweight dependencies
* object-oriented interface

Also see the [Why this library?](#why-this-library)-section for a comparison with similar libraries.

# Getting Started

## Installation

TODO
* incl. extras

## Usage

TODO

# For developers

TODO

## `poetry`
[`Poetry`](https://python-poetry.org/docs/basic-usage/) is used for package and dependency management.

Common use:

  * installation: [see here](https://python-poetry.org/docs/#installation)
  * To use the right interpreter in IDEs, set the interpreter to the output of `echo "$(poetry env info -p)/bin/python3"`. Or use a poetry plugin for your IDE from the start.

## `pre-commit`
Use `pre-commit` to format, check and lint the code in this repository.
After cloning, switch to the repository root and run `pre-commit install`.

## `bump2version`
Automatically bump version numbers and create tags by using `bump2version` (`pip3 install bump2version`):
```bash
bump2version major|minor|patch  # chose one
```

  You can test the bumping first by running:
```bash
bump2version patch --dry-run --allow-dirty --verbose
```

## testing
Run tests by running `poetry run pytest` in the root of your repository (or just `pytest` if you have the correct project interpreter set).

# Why this library?

TODO:
* heavy dependencies (scipy, )
* unclear returns and ordering of e.g. quaternions (wxyz vs. xyzw), easy to make mistakes
* integration with ROS?
* property-based testing
