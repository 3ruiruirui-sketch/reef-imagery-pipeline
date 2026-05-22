# Contributing to Reef Imagery Pipeline

## Code Style
- Use Python 3.10+
- Follow PEP 8 conventions
- Use type hints where possible
- Maximum line length: 120 characters

## Development Workflow

1. **Clone and setup**
   ```bash
   git clone <repo>
   cd reef_imagery_pipeline
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements_v3.txt
   ```

2. **Make changes on a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Run tests and linting**
   ```bash
   flake8 .
   black .
   mypy orchestrator*.py
   pytest
   ```

4. **Commit with clear messages**
   - Use conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`
   - Keep commits focused and atomic

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Project Structure

```
reef_imagery_pipeline/
├── orchestrator.py              # Main comparison orchestrator
├── orchestrator_run.py           # Physical analysis runner
├── reef_ml_predictor*.py         # ML prediction models
├── hybrid_stac_physical_orchestrator.py  # STAC streaming
├── dashboard/                   # Flask web dashboard
├── reef_Output_Master/          # Consolidated output directories (36 datasets)
├── .github/workflows/           # CI/CD automation
├── requirements*.txt            # Dependencies
└── README*.md                   # Documentation
```

## Key Files Edited

- `orchestrator.py` - Updated paths for reef_Output_Master consolidation
- `orchestrator_run.py` - Updated paths for reef_Output_Master consolidation
- `run_benthic_physics_comparison.py` - Updated paths for reef_Output_Master consolidation
- `.gitignore` - Enhanced with virtualenv, IDE, cache patterns

## Testing

The CI/CD pipeline runs:
- Python syntax checks (py_compile)
- Linting (flake8)
- Code formatting (black)
- Type checking (mypy)
- Security scanning (bandit, safety)
- Dependency verification

## Questions?

See README.md, README_v2.md, README_v3.md for documentation on specific pipelines.
