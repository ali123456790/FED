import json
import os
from datetime import datetime
import numpy as np

class MetricsLogger:
    def __init__(self, save_dir="metrics"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.metrics_history = []
        
    def log_metrics(self, round_number, metrics):
        """Log metrics for a specific round."""
        metrics_entry = {
            "round": round_number,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # Save to file
        filename = f"round_{round_number}_metrics.json"
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_entry, f, indent=4)
            
        # Save complete history
        history_file = os.path.join(self.save_dir, "metrics_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
            
    def load_history(self):
        """Load complete metrics history."""
        history_file = os.path.join(self.save_dir, "metrics_history.json")
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.metrics_history = json.load(f)
        return self.metrics_history
        
    def get_metric_progression(self, metric_name):
        """Get progression of a specific metric across rounds."""
        rounds = []
        values = []
        for entry in self.metrics_history:
            if metric_name in entry["metrics"]:
                rounds.append(entry["round"])
                values.append(entry["metrics"][metric_name])
        return rounds, values
        
    def print_summary(self):
        """Print summary of metrics across all rounds."""
        if not self.metrics_history:
            print("No metrics history available")
            return
            
        print("\nMetrics Summary Across All Rounds:")
        latest_metrics = self.metrics_history[-1]["metrics"]
        
        for metric_name in ["precision", "recall", "f1", "accuracy"]:
            if metric_name in latest_metrics:
                rounds, values = self.get_metric_progression(metric_name)
                print(f"\n{metric_name.capitalize()}:")
                print(f"  Latest: {values[-1]:.4f}")
                print(f"  Best: {max(values):.4f}")
                print(f"  Average: {np.mean(values):.4f}")
                print(f"  Std Dev: {np.std(values):.4f}")
                
        if "confusion_matrix" in latest_metrics:
            print("\nLatest Confusion Matrix:")
            print(np.array(latest_metrics["confusion_matrix"])) 