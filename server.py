import flwr as fl
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import argparse
import logging
import json
import os

from model import TinyBERTModel
from quantization import dequantize_model_update

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Strategy for federated learning that saves the best model."""
    
    def __init__(
        self,
        *args,
        model: TinyBERTModel = None,
        save_dir: str = "models",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.round_metrics = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Optional[fl.common.NDArrays]:
        """Aggregate model updates from clients and save progress."""
        
        # Log client participation
        logger.info(f"Round {server_round}: {len(results)} clients participated")
        client_ids = [str(client.cid) for client, _ in results]
        with open(f"{self.save_dir}/round_{server_round}_participants.json", "w") as f:
            json.dump(client_ids, f)
        
        # Aggregate updates
        aggregated_updates = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_updates is not None:
            # Convert to OrderedDict for the model
            aggregated_weights = fl.common.parameters_to_ndarrays(aggregated_updates)
            state_dict = OrderedDict(
                {
                    k: torch.tensor(v) 
                    for k, v in zip(self.model.state_dict().keys(), aggregated_weights)
                }
            )
            
            # Update the model
            self.model.load_state_dict(state_dict, strict=True)
            
            # Save the model
            torch.save(self.model.state_dict(), f"{self.save_dir}/model_round_{server_round}.pt")
            
            # Calculate and save metrics from this round
            metrics = {}
            if results:
                metrics["num_clients"] = len(results)
                metrics["round"] = server_round
                
                # Collect loss values
                loss_list = []
                accuracy_list = []
                for _, fit_res in results:
                    if fit_res.metrics:
                        loss = fit_res.metrics.get("loss", 0.0)
                        accuracy = fit_res.metrics.get("accuracy", 0.0)
                        loss_list.append(loss)
                        accuracy_list.append(accuracy)
                
                if loss_list:
                    metrics["mean_loss"] = sum(loss_list) / len(loss_list)
                    metrics["mean_accuracy"] = sum(accuracy_list) / len(accuracy_list)
            
                self.round_metrics.append(metrics)
                with open(f"{self.save_dir}/round_metrics.json", "w") as f:
                    json.dump(self.round_metrics, f)
            
        return aggregated_updates

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Optional[float]:
        """Aggregate evaluation results from clients."""
        
        if not results:
            return None
        
        # Weigh accuracy based on number of examples used
        accuracies = []
        examples = []
        
        for _, eval_res in results:
            if eval_res.metrics:
                accuracies.append(eval_res.metrics.get("accuracy", 0.0))
                examples.append(eval_res.num_examples)
        
        # Calculate weighted accuracy
        if examples and accuracies:
            weighted_accuracy = np.average(accuracies, weights=examples)
            
            # Save evaluation metrics
            eval_metrics = {
                "round": server_round,
                "weighted_accuracy": float(weighted_accuracy),
                "num_clients": len(results)
            }
            
            with open(f"{self.save_dir}/eval_round_{server_round}.json", "w") as f:
                json.dump(eval_metrics, f)
            
            return weighted_accuracy
        
        return None

def start_server(host: str = "0.0.0.0", port: int = 8080, num_rounds: int = 10):
    """Start the FL server."""
    # Initialize the model
    model = TinyBERTModel()
    
    # Define strategy
    strategy = SaveModelStrategy(
        model=model,
        save_dir="models",
        # Default FedAvg parameters
        fraction_fit=0.5,  # Sample 50% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2,  # At least 2 clients needed for training
        min_evaluate_clients=2,  # At least 2 clients needed for evaluation
        min_available_clients=2,  # Wait until at least 2 clients are available
    )

    # Specify client resources
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # Start server
    fl.server.start_server(
        server_address=f"{host}:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (default: 8080)"
    )
    
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds of federated learning (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Start the server
    start_server(host=args.host, port=args.port, num_rounds=args.rounds)
