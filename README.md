## Functionality isolation for surya update

### Running the project:
> Create venv:
```bash
python -m venv .venv
source .venv/bin/activate
```

> Install dependencies:
```bash
pip install pipenv
pipenv install
```

> Running FastAPI endpoint:
```bash
uvicorn fastapi_app:app --reload
```