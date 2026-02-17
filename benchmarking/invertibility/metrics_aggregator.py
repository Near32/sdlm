"""
Utilities for aggregating and processing metrics across multiple samples.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from metrics_registry import METRIC_RANGES

logger = logging.getLogger("metrics_aggregator")

class MetricsAggregator:
    """
    Class for aggregating and processing metrics across multiple samples.
    """

    def __init__(self):
        """Initialize the metrics aggregator."""
        self.metrics_by_k = defaultdict(lambda: defaultdict(list))
        self.all_metrics = defaultdict(list)
        # For per-epoch metrics: epoch -> k_value -> list of lcs_ratio values
        self.lcs_ratio_by_epoch_by_k = defaultdict(lambda: defaultdict(list))
        # For per-position prompt metrics: position -> list of match indicators (0 or 1)
        self.per_position_matches = defaultdict(list)
        # For per-epoch prompt metrics: metric_name -> epoch -> k_value -> list of values
        self.prompt_metrics_by_epoch_by_k = {
            "prompt_token_accuracy": defaultdict(lambda: defaultdict(list)),
            "prompt_exact_match": defaultdict(lambda: defaultdict(list)),
            "prompt_lcs_ratio": defaultdict(lambda: defaultdict(list)),
            "prompt_cosine_similarity": defaultdict(lambda: defaultdict(list)),
        }
        # For per-epoch semantic metrics (BERT/SentenceBERT): metric_name -> epoch -> k_value -> list of values
        self.semantic_metrics_by_epoch_by_k = {
            "output_bertscore_f1": defaultdict(lambda: defaultdict(list)),
            "output_bertscore_precision": defaultdict(lambda: defaultdict(list)),
            "output_bertscore_recall": defaultdict(lambda: defaultdict(list)),
            "output_sentencebert_similarity": defaultdict(lambda: defaultdict(list)),
            "prompt_bertscore_f1": defaultdict(lambda: defaultdict(list)),
            "prompt_bertscore_precision": defaultdict(lambda: defaultdict(list)),
            "prompt_bertscore_recall": defaultdict(lambda: defaultdict(list)),
            "prompt_sentencebert_similarity": defaultdict(lambda: defaultdict(list)),
        }
        
    def add_sample(self, metrics: Dict[str, Any], k_value: Optional[int] = None):
        """
        Add a sample's metrics to the aggregator.

        Args:
            metrics: Dictionary of metric values
            k_value: Optional k value to categorize the metrics
        """
        # Add to overall metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.all_metrics[name].append(value)

        # Add to k-specific metrics if provided
        if k_value is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.metrics_by_k[k_value][name].append(value)

        # Handle per_position_match for prompt reconstruction metrics
        if "per_position_match" in metrics:
            per_pos = metrics["per_position_match"]
            if isinstance(per_pos, list):
                for pos, match in enumerate(per_pos):
                    self.per_position_matches[pos].append(match)
    
    def add_batch(self, batch_metrics: List[Dict[str, Any]], k_values: Optional[List[int]] = None):
        """
        Add a batch of metrics to the aggregator.
        
        Args:
            batch_metrics: List of metric dictionaries
            k_values: Optional list of k values for each sample
        """
        if k_values is None:
            for metrics in batch_metrics:
                self.add_sample(metrics)
        else:
            assert len(batch_metrics) == len(k_values), "Length of metrics and k_values must match"
            for metrics, k in zip(batch_metrics, k_values):
                self.add_sample(metrics, k)

    def add_epoch_metrics(self, lcs_ratio_history: List[float], k_value: int,
                          prompt_metrics_history: Optional[Dict[str, List[float]]] = None,
                          semantic_metrics_history: Optional[Dict[str, List[float]]] = None):
        """
        Add per-epoch lcs_ratio values, prompt metrics, and semantic metrics for a sample.

        Args:
            lcs_ratio_history: List of lcs_ratio values, one per epoch
            k_value: The k value for this sample
            prompt_metrics_history: Dict mapping metric_name -> list of values per epoch
            semantic_metrics_history: Dict mapping metric_name -> list of values per epoch (BERT/SentenceBERT)
        """
        for epoch, lcs_ratio in enumerate(lcs_ratio_history):
            self.lcs_ratio_by_epoch_by_k[epoch][k_value].append(lcs_ratio)

        # Add prompt metrics history if provided
        if prompt_metrics_history:
            for metric_name, history in prompt_metrics_history.items():
                if metric_name in self.prompt_metrics_by_epoch_by_k:
                    for epoch, value in enumerate(history):
                        self.prompt_metrics_by_epoch_by_k[metric_name][epoch][k_value].append(value)

        # Add semantic metrics history if provided
        if semantic_metrics_history:
            for metric_name, history in semantic_metrics_history.items():
                if metric_name in self.semantic_metrics_by_epoch_by_k:
                    for epoch, value in enumerate(history):
                        self.semantic_metrics_by_epoch_by_k[metric_name][epoch][k_value].append(value)

    def compute_avg_lcs_ratio_by_epoch_by_k(self) -> Dict[int, Dict[int, float]]:
        """
        Compute avg_lcs_ratio for each (epoch, k_value) pair.

        Returns:
            Dictionary mapping epoch -> k_value -> avg_lcs_ratio
        """
        result = {}
        for epoch, k_dict in self.lcs_ratio_by_epoch_by_k.items():
            result[epoch] = {}
            for k_value, values in k_dict.items():
                result[epoch][k_value] = sum(values) / len(values) if values else 0.0
        return result

    def compute_avg_prompt_metrics_by_epoch_by_k(self) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute average prompt metrics for each (metric, epoch, k_value) combination.

        Returns:
            Dictionary mapping metric_name -> epoch -> k_value -> avg_value
        """
        result = {}
        for metric_name, epoch_dict in self.prompt_metrics_by_epoch_by_k.items():
            result[metric_name] = {}
            for epoch, k_dict in epoch_dict.items():
                result[metric_name][epoch] = {}
                for k_value, values in k_dict.items():
                    result[metric_name][epoch][k_value] = sum(values) / len(values) if values else 0.0
        return result

    def has_prompt_metrics_history(self) -> bool:
        """Check if per-epoch prompt metrics have been collected."""
        for metric_dict in self.prompt_metrics_by_epoch_by_k.values():
            if metric_dict:
                return True
        return False

    def compute_avg_semantic_metrics_by_epoch_by_k(self) -> Dict[str, Dict[int, Dict[int, float]]]:
        """
        Compute average semantic metrics (BERT/SentenceBERT) for each (metric, epoch, k_value) combination.

        Returns:
            Dictionary mapping metric_name -> epoch -> k_value -> avg_value
        """
        result = {}
        for metric_name, epoch_dict in self.semantic_metrics_by_epoch_by_k.items():
            result[metric_name] = {}
            for epoch, k_dict in epoch_dict.items():
                result[metric_name][epoch] = {}
                for k_value, values in k_dict.items():
                    result[metric_name][epoch][k_value] = sum(values) / len(values) if values else 0.0
        return result

    def has_semantic_metrics_history(self) -> bool:
        """Check if per-epoch semantic metrics (BERT/SentenceBERT) have been collected."""
        for metric_dict in self.semantic_metrics_by_epoch_by_k.values():
            if metric_dict:
                return True
        return False

    def compute_per_position_success_rate(self) -> List[float]:
        """
        Compute per-position success rate for prompt reconstruction.

        Returns:
            List of success rates, one per position
        """
        if not self.per_position_matches:
            return []

        max_position = max(self.per_position_matches.keys())
        success_rates = []
        for pos in range(max_position + 1):
            matches = self.per_position_matches.get(pos, [])
            if matches:
                success_rates.append(sum(matches) / len(matches))
            else:
                success_rates.append(0.0)
        return success_rates

    def has_prompt_reconstruction_metrics(self) -> bool:
        """Check if prompt reconstruction metrics have been collected."""
        return bool(self.per_position_matches)

    def compute_average(self, metric_name: str) -> float:
        """
        Compute the average value for a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Average value or 0 if no samples exist
        """
        values = self.all_metrics.get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
    def compute_average_by_k(self, metric_name: str, k_value: int) -> float:
        """
        Compute the average value for a specific metric and k value.
        
        Args:
            metric_name: Name of the metric
            k_value: The k value
            
        Returns:
            Average value or 0 if no samples exist
        """
        values = self.metrics_by_k.get(k_value, {}).get(metric_name, [])
        return sum(values) / len(values) if values else 0.0
    
    def get_k_values(self) -> List[int]:
        """Get all k values with data."""
        return sorted(self.metrics_by_k.keys())
    
    def get_metric_names(self) -> List[str]:
        """Get all metric names with data."""
        return list(self.all_metrics.keys())
    
    def get_sample_count(self, k_value: Optional[int] = None) -> int:
        """
        Get the number of samples for a specific k value or overall.
        
        Args:
            k_value: Optional k value to count samples for
            
        Returns:
            Number of samples
        """
        if k_value is None:
            # Use the first metric's count as representative
            metrics = self.all_metrics
            first_key = next(iter(metrics), None)
            return len(metrics[first_key]) if first_key else 0
        else:
            # Use the first metric's count for this k value
            metrics = self.metrics_by_k.get(k_value, {})
            first_key = next(iter(metrics), None)
            return len(metrics[first_key]) if first_key else 0
    
    def compute_aggregated_metrics(self) -> Dict[str, float]:
        """
        Compute aggregated metrics across all samples.
        
        Returns:
            Dictionary of aggregated metrics following the expected naming convention
        """
        aggregated = {}
        
        for name, values in self.all_metrics.items():
            if values:
                # Use original naming convention for compatibility with existing visualizations
                if name == "exact_match":
                    aggregated["success_rate"] = sum(values) / len(values)
                else:
                    aggregated[f"avg_{name}"] = sum(values) / len(values)
                
                aggregated[f"min_{name}"] = min(values)
                aggregated[f"max_{name}"] = max(values)
                aggregated[f"std_{name}"] = np.std(values).item() if len(values) > 1 else 0.0
        
        aggregated["sample_count"] = self.get_sample_count()
        
        return aggregated
    
    def compute_aggregated_metrics_by_k(self) -> Dict[int, Dict[str, float]]:
        """
        Compute aggregated metrics for each k value.
        
        Returns:
            Dictionary mapping k values to aggregated metrics
        """
        aggregated = {}
        
        for k, metrics in self.metrics_by_k.items():
            k_aggregated = {}
            
            for name, values in metrics.items():
                if values:
                    # Match original naming conventions for backward compatibility
                    if name == "exact_match":
                        k_aggregated["success_rate"] = sum(values) / len(values)
                    else:
                        k_aggregated[f"avg_{name}"] = sum(values) / len(values)
                    
                    k_aggregated[f"min_{name}"] = min(values)
                    k_aggregated[f"max_{name}"] = max(values)
                    k_aggregated[f"std_{name}"] = np.std(values).item() if len(values) > 1 else 0.0
            
            k_aggregated["sample_count"] = self.get_sample_count(k)
            aggregated[k] = k_aggregated
        
        return aggregated
    
    def compute_auc(self, metric_name: str) -> Tuple[float, float]:
        """
        Compute the Area Under the Curve for a metric across k values.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (raw AUC, normalized AUC)
        """
        k_values = self.get_k_values()
        
        if not k_values:
            return 0.0, 0.0
        
        # Collect values for each k
        values = []
        for k in k_values:
            avg_value = self.compute_average_by_k(metric_name, k)
            values.append(avg_value)
        
        # Calculate AUC using trapezoidal rule
        auc = 0.0
        for i in range(1, len(k_values)):
            x1, x2 = k_values[i-1], k_values[i]
            y1, y2 = values[i-1], values[i]
            auc += 0.5 * (x2 - x1) * (y1 + y2)
        
        # Get normalization range
        metric_info = METRIC_RANGES.get(metric_name, {"min": 0, "max": 1})
        min_y = metric_info.get("min", 0)
        max_y = metric_info.get("max", 1)
        
        # Compute max possible AUC
        max_k = max(k_values)
        min_k = min(k_values)
        max_possible_auc = abs(max_y - min_y) * abs(max_k - min_k)
        
        # Normalize AUC
        normalized_auc = auc / max_possible_auc if max_possible_auc != 0 else 0
        
        return auc, normalized_auc
    
    def compute_all_auc_metrics(self) -> Dict[str, float]:
        """
        Compute AUC for all metrics.
        
        Returns:
            Dictionary mapping metric names to AUC values using the expected naming convention
        """
        auc_results = {}
        
        for metric_name in self.get_metric_names():
            raw_auc, normalized_auc = self.compute_auc(metric_name)
            
            # Use original naming convention for compatibility with existing visualizations
            if metric_name == "exact_match":
                auc_results["AuC/Raw/success_rate"] = raw_auc
                auc_results["AuC/Normalized/success_rate"] = normalized_auc
            else:
                auc_results[f"AuC/Raw/{metric_name}"] = raw_auc
                auc_results[f"AuC/Normalized/{metric_name}"] = normalized_auc
        
        return auc_results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a complete summary of all computed metrics.

        Returns:
            Dictionary with all aggregated metrics and AUC results
        """
        summary = {
            "overall": self.compute_aggregated_metrics(),
            "by_k": self.compute_aggregated_metrics_by_k(),
            "auc": self.compute_all_auc_metrics(),
            "k_values": self.get_k_values(),
            "metric_names": self.get_metric_names(),
            "total_samples": self.get_sample_count(),
            "avg_lcs_ratio_by_epoch_by_k": self.compute_avg_lcs_ratio_by_epoch_by_k(),
        }

        # Add prompt reconstruction specific metrics if available
        if self.has_prompt_reconstruction_metrics():
            per_pos_success = self.compute_per_position_success_rate()
            summary["per_position_success_rate"] = per_pos_success
            summary["has_prompt_reconstruction_metrics"] = True

            # Add summary prompt metrics to overall
            if "prompt_exact_match" in self.all_metrics:
                values = self.all_metrics["prompt_exact_match"]
                summary["overall"]["prompt_success_rate"] = sum(values) / len(values) if values else 0.0

        # Add per-epoch prompt metrics if available
        if self.has_prompt_metrics_history():
            summary["avg_prompt_metrics_by_epoch_by_k"] = self.compute_avg_prompt_metrics_by_epoch_by_k()
            summary["has_prompt_metrics_history"] = True

        # Add per-epoch semantic metrics (BERT/SentenceBERT) if available
        if self.has_semantic_metrics_history():
            summary["avg_semantic_metrics_by_epoch_by_k"] = self.compute_avg_semantic_metrics_by_epoch_by_k()
            summary["has_semantic_metrics_history"] = True

        return summary
