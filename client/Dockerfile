FROM python:3.12-slim
WORKDIR /app
# Copy the requirements file
COPY requirements.txt requirements.txt
# Install the dependencies
RUN pip install -r requirements.txt --no-cache-dir
# Copy the rest of the code
COPY client.py /app

# Add the argument
ENV PARTITION_ID=0
ENV SERVER_ADDRESS="server:8080"

# Run the server
CMD python client.py --partition-id $PARTITION_ID --server-address $SERVER_ADDRESS