"""
@author : Tien Nguyen
@date   : 2023-Oct-25
"""
import logging
import socket
import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    logging.info("IP Address: %s", ip_address)

