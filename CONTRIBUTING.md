# Contributing to SciRAG-UQ

Thank you for your interest in contributing to SciRAG-UQ! This document outlines the process for contributing to the project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/kaushalrog/scirag-uq.git
   cd scirag-uq
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up pre-commit hooks (if applicable):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and write tests if applicable.

3. Run the test suite:
   ```bash
   make test
   ```

4. Commit your changes using conventional commit messages:
   ```bash
   git commit -m "feat(module): description of changes"
   ```

5. Push to your branch and submit a Pull Request.

## Code Style
- We follow PEP 8.
- Use `black` for formatting and `isort` for import sorting.
- Use type hints wherever possible.

## Submitting Issues
If you find a bug or have a feature request, please open an issue using the provided GitHub templates.
