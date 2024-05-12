#!/bin/bash
docker network create federated_learning

docker build -t federated_learning_server:latest server/.

# Trap Ctrl+C to stop the server and exit
trap "echo -e '\nShutting down...' && docker stop server && docker rm server && echo 'Server stopped.' && exit" SIGINT

# Start the server
echo "Starting server..."
docker run --name server \
    --env NUM_ROUNDS=8 \
    --network federated_learning \
    federated_learning_server
echo "Server started."