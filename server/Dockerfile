FROM python:3.12-slim
WORKDIR /app
# Copy the requirements file
COPY ../requirements.txt requirements.txt
# Install the dependencies
RUN pip install -r requirements.txt --no-cache-dir
# Copy the rest of the code
COPY server.py /app
# Run the server
CMD ["python", "server.py"]