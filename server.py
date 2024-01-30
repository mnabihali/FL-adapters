import flwr as fl
from utils import STRATEGY


if __name__ == "__main__":
    # Define strategy
    strategy = STRATEGY(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
    )

    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )