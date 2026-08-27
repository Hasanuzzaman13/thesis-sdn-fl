import flwr as fl

def weighted_average(metrics):
    examples = [n for n, _ in metrics]
    total_examples = sum(examples)
    
    # সব মেট্রিক্সের ওয়েটেড অ্যাভারেজ বের করা
    agg_metrics = {}
    for key in ["accuracy", "precision", "recall", "f1"]:
        agg_metrics[key] = sum(n * m[key] for n, m in metrics) / total_examples
        
    return agg_metrics

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, fraction_evaluate=1.0,
        min_fit_clients=4, min_evaluate_clients=4, min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    print("🚀 Starting Flower Server (Pro Version with F1-Score)...")
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy
    )
    print("✅ Training completed!")
