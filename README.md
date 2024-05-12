# Flower Tutorial using Pandas

Tutorial for a federated learning using flower and pandas.

## What is Federated Learning?
...


## Project Setup

### Installation

To begin, ensure you have the necessary libraries installed. You can install them using pip:

```bash 
pip install numpy pandas flwr flwr_datasets
```
## Flower client 
Next, let's set up the Flower client. Create a file called `client.py` and add the following code:


```python
import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import flwr as fl

from flwr_datasets import FederatedDataset


column_names = ["sepal_length", "sepal_width"]


def compute_hist(df: pd.DataFrame, col_name: str) -> np.ndarray:
    freqs, _ = np.histogram(df[col_name])
    return freqs


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, X: pd.DataFrame):
        self.X = X

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        hist_list = []
        # Execute query locally
        for c in self.X.columns:
            hist = compute_hist(self.X, c)
            hist_list.append(hist)
        return (
            hist_list,
            len(self.X),
            {},
        )


if __name__ == "__main__":
    N_CLIENTS = os.environ.get("N_CLIENTS", 10)

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--partition-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the partition id of artificially partitioned datasets.",
    )

    parser.add_argument(
        "--server-address",
        type=str,
        required=True,
        help="The address of the Flower server. server + port",
    )
    args = parser.parse_args()
    partition_id = args.partition_id
    server_address = args.server_address

    # Load the partition data
    fds = FederatedDataset(dataset="hitorilabs/iris", partitioners={"train": N_CLIENTS})

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # Use just the specified columns
    X = dataset[column_names]

    # Start Flower client
    fl.client.start_client(
        server_address=server_address,
        client=FlowerClient(X).to_client(),
    )
```

### Flower Server
Now, let's set up the Flower server. Create a file called `server.py` and add the following code:

```python
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class FedAnalytics(Strategy):
    def initialize_parameters(
        self, client_manager: Optional[ClientManager] = None
    ) -> Optional[Parameters]:
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(num_clients=2, min_num_clients=2)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Get results from fit
        # Convert results
        values_aggregated = [
            (parameters_to_ndarrays(fit_res.parameters)) for _, fit_res in results
        ]
        length_agg_hist = 0
        width_agg_hist = 0
        for val in values_aggregated:
            length_agg_hist += val[0]
            width_agg_hist += val[1]

        ndarr = np.concatenate(
            (["Length:"], length_agg_hist, ["Width:"], width_agg_hist)
        )
        return ndarrays_to_parameters(ndarr), {}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        agg_hist = [arr.item() for arr in parameters_to_ndarrays(parameters)]
        return 0, {"Aggregated histograms": agg_hist}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        pass

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        pass

num_rounds = int(os.getenv("NUM_ROUNDS", 3))

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=FedAnalytics(),
)
```


Explanation:

- client.py: This file sets up the Flower client. It imports necessary libraries, defines a custom client class, and implements the fit method to compute histograms locally. The client connects to the Flower server specified via command line arguments (--partition-id for the dataset partition ID and --server-address for the server address).

- server.py: This file sets up the Flower server. It defines a custom federated learning strategy (FedAnalytics) which configures the fit process, aggregates fit results, and evaluates aggregated parameters. The server starts on address 0.0.0.0:8080 with a specified number of rounds for federated learning.

These files provide a basic setup for running federated learning experiments using Flower.

## Run project
Start by cloning the example project. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
$ git clone git@github.com:arturoliduena/federated_learning_pandas.git
```

After cloning the repository, you can run the containers using the following command:
```shell
$ docker-compose up --build
```
