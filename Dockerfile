FROM python:3.12

# Install requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code
COPY smart_stethoscope smart_stethoscope
COPY models models
COPY setup.py setup.py

CMD ["sh", "-c", "uvicorn smart_stethoscope.api.fast:app --host 0.0.0.0 --port ${PORT}"]
