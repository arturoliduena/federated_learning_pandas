#!/bin/bash
docker build -t federated_learning_client:latest client/.

start_client() {
    client_id=$1
    partition_id=$2
    client_name="client-${client_id}"
    docker run -d --name "$client_name" \
        --env SERVER_ADDRESS=server:8080 \
        --env PARTITION_ID="$partition_id" \
        --env NUMBER_OF_CLIENTS=2 \
        --network federated_learning \
        --restart on-failure \
        federated_learning_client
    echo "Started $client_name"
}

stop_client() {
    client_id=$1
    client_name="client-${client_id}"
    docker stop "$client_name" > /dev/null
    docker rm "$client_name" > /dev/null
    echo "Stopped $client_name"
}

create_random_clients() {
    for ((i=0; i<10; i++)); do
        client_id=$((RANDOM % 1000))
        start_client "$client_id" "$i"
        sleep 1  # Adjust sleep duration if needed
    done
}

create_random_clients
