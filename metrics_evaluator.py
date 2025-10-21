"""
Metrics evaluation system for prompt performance assessment.
Implements various metrics to measure prompt effectiveness.
"""

import json
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    # High-value detection metrics (>250 GBP)
    high_value_precision: float
    high_value_recall: float
    high_value_f1: float
    high_value_accuracy: float

    # Anomaly detection metrics
    anomaly_precision: float
    anomaly_recall: float
    anomaly_f1: float
    anomaly_accuracy: float

    # Overall metrics
    overall_accuracy: float
    overall_f1: float

    # Response quality metrics
    response_parseable: bool
    response_format_score: float  # 0-1 score for format correctness
    response_time_seconds: float

    # Error metrics
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int

    # Combined score (weighted)
    composite_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def __str__(self) -> str:
        """String representation."""
        return f"""Performance Metrics:
  High-Value Detection: P={self.high_value_precision:.3f}, R={self.high_value_recall:.3f}, F1={self.high_value_f1:.3f}
  Anomaly Detection: P={self.anomaly_precision:.3f}, R={self.anomaly_recall:.3f}, F1={self.anomaly_f1:.3f}
  Overall: Accuracy={self.overall_accuracy:.3f}, F1={self.overall_f1:.3f}
  Composite Score: {self.composite_score:.3f}
  Response Time: {self.response_time_seconds:.2f}s
"""


class MetricsEvaluator:
    """Evaluate prompt performance using multiple metrics."""

    def __init__(self, weight_high_value: float = 0.3, weight_anomaly: float = 0.5,
                 weight_response_quality: float = 0.2):
        """
        Initialize evaluator with weights for different aspects.

        Args:
            weight_high_value: Weight for high-value detection performance
            weight_anomaly: Weight for anomaly detection performance
            weight_response_quality: Weight for response quality
        """
        self.weight_high_value = weight_high_value
        self.weight_anomaly = weight_anomaly
        self.weight_response_quality = weight_response_quality

        # Normalize weights
        total = weight_high_value + weight_anomaly + weight_response_quality
        self.weight_high_value /= total
        self.weight_anomaly /= total
        self.weight_response_quality /= total

    @staticmethod
    def _calculate_metrics(true_positive: int, false_positive: int,
                          true_negative: int, false_negative: int) -> Tuple[float, float, float, float]:
        """Calculate precision, recall, F1, and accuracy."""
        # Precision: TP / (TP + FP)
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

        # F1: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = true_positive + true_negative + false_positive + false_negative
        accuracy = (true_positive + true_negative) / total if total > 0 else 0.0

        return precision, recall, f1, accuracy

    def evaluate_response_format(self, response: str) -> Tuple[bool, float]:
        """
        Evaluate the quality of the response format.

        Returns:
            (is_parseable, format_score)
        """
        try:
            parsed = json.loads(response)

            # Check required keys
            has_above_250 = 'above_250' in parsed
            has_anomalies = 'anomalies' in parsed

            if not (has_above_250 and has_anomalies):
                return True, 0.5  # Parseable but incomplete

            # Check format quality
            score = 1.0

            # Validate above_250 is a list
            if not isinstance(parsed['above_250'], list):
                score -= 0.2

            # Validate anomalies format
            if isinstance(parsed['anomalies'], list):
                # Check if anomalies have proper structure
                for anomaly in parsed['anomalies']:
                    if isinstance(anomaly, dict):
                        if 'transaction_id' not in anomaly:
                            score -= 0.1
                        if 'reason' not in anomaly:
                            score -= 0.1
            else:
                score -= 0.3

            return True, max(0.0, score)

        except json.JSONDecodeError:
            # Try to extract JSON from text
            try:
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    parsed = json.loads(json_str)
                    return True, 0.7  # Parseable but with extra text
            except:
                pass

            return False, 0.0

    def parse_response(self, response: str) -> Tuple[Set[str], Set[str]]:
        """
        Parse LLM response to extract detected high-value transactions and anomalies.

        Returns:
            (set of high_value transaction IDs, set of anomaly transaction IDs)
        """
        try:
            # Try direct JSON parse
            parsed = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
            else:
                return set(), set()

        # Extract high-value transactions
        high_value = set()
        if 'above_250' in parsed and isinstance(parsed['above_250'], list):
            high_value = set(str(x) for x in parsed['above_250'])

        # Extract anomalies
        anomalies = set()
        if 'anomalies' in parsed:
            if isinstance(parsed['anomalies'], list):
                for item in parsed['anomalies']:
                    if isinstance(item, dict) and 'transaction_id' in item:
                        anomalies.add(str(item['transaction_id']))
                    elif isinstance(item, str):
                        anomalies.add(item)

        return high_value, anomalies

    def evaluate(self, ground_truth_df: pd.DataFrame, llm_response: str,
                response_time: float) -> PerformanceMetrics:
        """
        Evaluate LLM response against ground truth.

        Args:
            ground_truth_df: DataFrame with ground truth data
            llm_response: Raw LLM response
            response_time: Time taken to get response (seconds)

        Returns:
            PerformanceMetrics object
        """
        # Parse response
        parseable, format_score = self.evaluate_response_format(llm_response)
        predicted_high_value, predicted_anomalies = self.parse_response(llm_response)

        # Ground truth sets
        true_high_value = set(
            ground_truth_df[ground_truth_df['above_250'] == True]['transaction_id'].astype(str)
        )
        true_anomalies = set(
            ground_truth_df[ground_truth_df['is_anomaly'] == True]['transaction_id'].astype(str)
        )

        all_transactions = set(ground_truth_df['transaction_id'].astype(str))

        # High-value detection metrics
        hv_tp = len(predicted_high_value & true_high_value)
        hv_fp = len(predicted_high_value - true_high_value)
        hv_fn = len(true_high_value - predicted_high_value)
        hv_tn = len(all_transactions - true_high_value - predicted_high_value)

        hv_precision, hv_recall, hv_f1, hv_accuracy = self._calculate_metrics(
            hv_tp, hv_fp, hv_tn, hv_fn
        )

        # Anomaly detection metrics
        an_tp = len(predicted_anomalies & true_anomalies)
        an_fp = len(predicted_anomalies - true_anomalies)
        an_fn = len(true_anomalies - predicted_anomalies)
        an_tn = len(all_transactions - true_anomalies - predicted_anomalies)

        an_precision, an_recall, an_f1, an_accuracy = self._calculate_metrics(
            an_tp, an_fp, an_tn, an_fn
        )

        # Overall metrics
        total_tp = hv_tp + an_tp
        total_fp = hv_fp + an_fp
        total_fn = hv_fn + an_fn
        total_tn = hv_tn + an_tn

        overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0.0
        overall_f1 = (hv_f1 + an_f1) / 2

        # Calculate composite score
        composite_score = (
            self.weight_high_value * hv_f1 +
            self.weight_anomaly * an_f1 +
            self.weight_response_quality * format_score
        )

        # Penalty for slow responses (>10 seconds)
        if response_time > 10:
            composite_score *= (10 / response_time)

        return PerformanceMetrics(
            high_value_precision=hv_precision,
            high_value_recall=hv_recall,
            high_value_f1=hv_f1,
            high_value_accuracy=hv_accuracy,
            anomaly_precision=an_precision,
            anomaly_recall=an_recall,
            anomaly_f1=an_f1,
            anomaly_accuracy=an_accuracy,
            overall_accuracy=overall_accuracy,
            overall_f1=overall_f1,
            response_parseable=parseable,
            response_format_score=format_score,
            response_time_seconds=response_time,
            false_positives=total_fp,
            false_negatives=total_fn,
            true_positives=total_tp,
            true_negatives=total_tn,
            composite_score=composite_score
        )


class MetricsTracker:
    """Track and compare metrics across multiple prompt tests."""

    def __init__(self):
        self.results: Dict[str, List[PerformanceMetrics]] = {}

    def add_result(self, prompt_name: str, metrics: PerformanceMetrics):
        """Add a result for a prompt."""
        if prompt_name not in self.results:
            self.results[prompt_name] = []
        self.results[prompt_name].append(metrics)

    def get_average_metrics(self, prompt_name: str) -> Dict:
        """Get average metrics for a prompt across all tests."""
        if prompt_name not in self.results or len(self.results[prompt_name]) == 0:
            return {}

        metrics_list = self.results[prompt_name]
        num_results = len(metrics_list)

        avg_metrics = {
            'num_tests': num_results,
            'avg_composite_score': np.mean([m.composite_score for m in metrics_list]),
            'avg_high_value_f1': np.mean([m.high_value_f1 for m in metrics_list]),
            'avg_anomaly_f1': np.mean([m.anomaly_f1 for m in metrics_list]),
            'avg_overall_f1': np.mean([m.overall_f1 for m in metrics_list]),
            'avg_response_time': np.mean([m.response_time_seconds for m in metrics_list]),
            'std_composite_score': np.std([m.composite_score for m in metrics_list]),
            'parseable_rate': sum([m.response_parseable for m in metrics_list]) / num_results
        }

        return avg_metrics

    def get_best_prompt(self) -> Tuple[str, Dict]:
        """Get the best performing prompt based on composite score."""
        best_prompt = None
        best_score = -1

        for prompt_name in self.results:
            avg_metrics = self.get_average_metrics(prompt_name)
            if avg_metrics['avg_composite_score'] > best_score:
                best_score = avg_metrics['avg_composite_score']
                best_prompt = prompt_name

        return best_prompt, self.get_average_metrics(best_prompt) if best_prompt else {}

    def get_leaderboard(self) -> pd.DataFrame:
        """Get a leaderboard of all prompts sorted by performance."""
        leaderboard_data = []

        for prompt_name in self.results:
            avg_metrics = self.get_average_metrics(prompt_name)
            leaderboard_data.append({
                'prompt_name': prompt_name,
                'composite_score': avg_metrics['avg_composite_score'],
                'high_value_f1': avg_metrics['avg_high_value_f1'],
                'anomaly_f1': avg_metrics['avg_anomaly_f1'],
                'overall_f1': avg_metrics['avg_overall_f1'],
                'response_time': avg_metrics['avg_response_time'],
                'parseable_rate': avg_metrics['parseable_rate'],
                'num_tests': avg_metrics['num_tests']
            })

        df = pd.DataFrame(leaderboard_data)
        if len(df) > 0:
            df = df.sort_values('composite_score', ascending=False)

        return df


if __name__ == "__main__":
    # Example usage
    evaluator = MetricsEvaluator()

    # Mock data for testing
    import pandas as pd

    ground_truth = pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
        'above_250': [True, False, True, False],
        'is_anomaly': [False, True, True, False]
    })

    llm_response = json.dumps({
        'above_250': ['TXN001', 'TXN003'],
        'anomalies': [
            {'transaction_id': 'TXN002', 'reason': 'Suspicious pattern'},
            {'transaction_id': 'TXN003', 'reason': 'High value + unusual'}
        ]
    })

    metrics = evaluator.evaluate(ground_truth, llm_response, response_time=1.5)
    print(metrics)
