# IBM Cloud Deployment Notes

Smart Lender is designed so it can be deployed on IBM Cloud with the Flask app entry point:

```text
Flask.app1:app
```

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
gunicorn Flask.app1:app
```

Required project files:

- `requirements.txt`
- `Procfile`
- `runtime.txt`
- `Flask/app1.py`
- `Flask/templates/`
- `Flask/static/`
- `Flask/rdf.pkl`
- `Flask/scale1.pkl`
