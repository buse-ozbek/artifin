FROM python:3.9-slim

WORKDIR /app

# install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the webservice code
COPY dockerization_and_deployment/webservices/ ./webservices/

EXPOSE 8001

# point uvicorn at the FastAPI app inside predict.py
CMD ["uvicorn", "webservices.predict:app", "--host", "0.0.0.0", "--port", "8001"]
