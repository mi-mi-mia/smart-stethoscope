#TO DO - does the FROM line match the version running? in terminaly python -- version
FROM python:3.12

# Install requirements
#TO DO - did we add anything else that needs copying over when updating?
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code
COPY smart_stethoscope smart_stethoscope
#TO DO delete COPY models models as we load from GCSE, it's adding weight
COPY models models
COPY setup.py setup.py

#TO DO check entry point name still references the file - if the fast.py name was changed. (not no py needed below)
CMD ["sh", "-c", "uvicorn smart_stethoscope.api.fast:app --host 0.0.0.0 --port ${PORT}"]
