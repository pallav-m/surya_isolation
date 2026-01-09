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

> Running through infer.py:
```bash
# Single image OCR
python infer.py --images document.jpg --task extract_text --output results.json

# Multiple images for layout detection
python infer.py --images page1.png page2.png --task detect_layout --output layouts.json

# Process entire directory for table extraction
python infer.py --input-dir ./scanned_docs --task process_tables --output tables.json

# Text extraction with plain text output
python infer.py --images receipt.jpg --task extract_text --output text.txt --format txt

```
