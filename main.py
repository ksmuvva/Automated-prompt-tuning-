"""
Main orchestrator for automated prompt tuning system.
Coordinates data loading, prompt testing, evaluation, and optimization.
"""

import os
import glob
import json
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import argparse

from data_generator import BankDataGenerator
from prompt_templates import PromptTemplateLibrary
from llm_interface import LLMFactory, LLMTester, format_transaction_data
from metrics_evaluator import MetricsEvaluator, MetricsTracker
from prompt_optimizer import PromptOptimizer, AdaptiveOptimizer, PromptCandidate


class PromptTuningOrchestrator:
    """Main orchestrator for the prompt tuning system."""

    def __init__(self, data_dir: str = "bank_data",
                 llm_provider: str = "mock",
                 llm_config: Optional[Dict] = None,
                 max_generations: int = 5,
                 population_size: int = 15,
                 num_test_files: int = 5):
        """
        Initialize the orchestrator.

        Args:
            data_dir: Directory containing CSV files
            llm_provider: Type of LLM provider to use
            llm_config: Configuration for LLM provider
            max_generations: Maximum optimization generations
            population_size: Number of prompts per generation
            num_test_files: Number of CSV files to test each prompt on
        """
        self.data_dir = data_dir
        self.num_test_files = num_test_files
        self.max_generations = max_generations

        # Initialize components
        print(f"Initializing LLM provider: {llm_provider}")
        llm_config = llm_config or {}
        llm = LLMFactory.create(llm_provider, **llm_config)
        self.llm_tester = LLMTester(llm)

        self.evaluator = MetricsEvaluator()
        self.tracker = MetricsTracker()
        self.optimizer = PromptOptimizer(population_size=population_size, mutation_rate=0.3)
        self.adaptive = AdaptiveOptimizer()

        # Results storage
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.all_results = []

    def load_data(self) -> List[pd.DataFrame]:
        """Load CSV files from data directory."""
        csv_files = sorted(glob.glob(os.path.join(self.data_dir, "*.csv")))

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        print(f"Found {len(csv_files)} CSV files")

        # Load subset for testing
        selected_files = csv_files[:self.num_test_files]
        dataframes = []

        for file_path in selected_files:
            df = pd.read_csv(file_path)
            dataframes.append(df)

        print(f"Loaded {len(dataframes)} files for testing")
        return dataframes

    def test_prompt_on_data(self, prompt_template, test_data: List[pd.DataFrame]) -> List:
        """
        Test a prompt template on multiple data files.

        Returns:
            List of PerformanceMetrics for each file
        """
        metrics_list = []

        for idx, df in enumerate(test_data):
            # Format data for prompt
            data_str = format_transaction_data(df)
            full_prompt = prompt_template.format(data=data_str)

            # Query LLM
            try:
                response, response_time = self.llm_tester.test_prompt(full_prompt, use_cache=False)

                # Evaluate response
                metrics = self.evaluator.evaluate(df, response, response_time)
                metrics_list.append(metrics)

            except Exception as e:
                print(f"Error testing prompt on file {idx}: {e}")
                # Create zero metrics for failed tests
                from metrics_evaluator import PerformanceMetrics
                metrics = PerformanceMetrics(
                    high_value_precision=0, high_value_recall=0, high_value_f1=0, high_value_accuracy=0,
                    anomaly_precision=0, anomaly_recall=0, anomaly_f1=0, anomaly_accuracy=0,
                    overall_accuracy=0, overall_f1=0,
                    response_parseable=False, response_format_score=0, response_time_seconds=0,
                    false_positives=0, false_negatives=0, true_positives=0, true_negatives=0,
                    composite_score=0
                )
                metrics_list.append(metrics)

        return metrics_list

    def evaluate_population(self, population: List[PromptCandidate],
                           test_data: List[pd.DataFrame]) -> List[PromptCandidate]:
        """Evaluate all prompts in population."""
        print(f"\nEvaluating {len(population)} prompts...")

        for idx, candidate in enumerate(population):
            print(f"  Testing prompt {idx+1}/{len(population)}: {candidate.template.name}")

            # Test on all data files
            metrics_list = self.test_prompt_on_data(candidate.template, test_data)

            # Calculate average metrics
            if metrics_list:
                avg_composite = sum(m.composite_score for m in metrics_list) / len(metrics_list)
                avg_high_value_f1 = sum(m.high_value_f1 for m in metrics_list) / len(metrics_list)
                avg_anomaly_f1 = sum(m.anomaly_f1 for m in metrics_list) / len(metrics_list)
                avg_response_time = sum(m.response_time_seconds for m in metrics_list) / len(metrics_list)

                # Use first metrics as representative, but update composite score
                candidate.metrics = metrics_list[0]
                candidate.metrics.composite_score = avg_composite

                # Track results
                self.tracker.add_result(candidate.template.name, candidate.metrics)

                print(f"    Score: {avg_composite:.3f} | HV-F1: {avg_high_value_f1:.3f} | "
                      f"Anom-F1: {avg_anomaly_f1:.3f} | Time: {avg_response_time:.2f}s")

                # Store detailed results
                self.all_results.append({
                    'prompt_name': candidate.template.name,
                    'generation': candidate.generation,
                    'composite_score': avg_composite,
                    'high_value_f1': avg_high_value_f1,
                    'anomaly_f1': avg_anomaly_f1,
                    'response_time': avg_response_time
                })

        return population

    def run_optimization(self) -> PromptCandidate:
        """
        Run the full optimization process.

        Returns:
            Best performing prompt candidate
        """
        print("="*70)
        print("AUTOMATED PROMPT TUNING SYSTEM")
        print("="*70)

        # Load test data
        test_data = self.load_data()

        # Initialize population
        print("\nInitializing prompt population...")
        population = self.optimizer.initialize_population()

        # Evaluate initial population
        print(f"\n{'='*70}")
        print(f"GENERATION 0 - Initial Population")
        print(f"{'='*70}")
        population = self.evaluate_population(population, test_data)

        best_candidate = self.optimizer.get_best_candidate(population)
        print(f"\nBest prompt: {best_candidate.template.name}")
        print(f"Best score: {best_candidate.fitness:.3f}")

        # Evolution loop
        for gen in range(1, self.max_generations):
            print(f"\n{'='*70}")
            print(f"GENERATION {gen}")
            print(f"{'='*70}")

            # Check if should continue
            avg_score = sum(c.fitness for c in population) / len(population)
            self.adaptive.record_generation(gen - 1, best_candidate.fitness, avg_score)

            if not self.adaptive.should_continue(gen, self.max_generations):
                print("Optimization plateau reached. Stopping early.")
                break

            # Evolve population
            print("Evolving population...")
            population = self.optimizer.evolve_generation(population)

            # Add targeted improvements
            improved_template = self.optimizer.create_targeted_improvement(best_candidate)
            population.append(PromptCandidate(improved_template, generation=gen))

            # Evaluate new generation
            population = self.evaluate_population(population, test_data)

            # Update best
            current_best = self.optimizer.get_best_candidate(population)
            if current_best.fitness > best_candidate.fitness:
                print(f"\n🎉 New best prompt found!")
                print(f"   Previous: {best_candidate.template.name} ({best_candidate.fitness:.3f})")
                print(f"   New: {current_best.template.name} ({current_best.fitness:.3f})")
                print(f"   Improvement: +{(current_best.fitness - best_candidate.fitness):.3f}")
                best_candidate = current_best
            else:
                print(f"\nNo improvement in this generation")
                print(f"Best remains: {best_candidate.template.name} ({best_candidate.fitness:.3f})")

        # Final results
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nBest Prompt: {best_candidate.template.name}")
        print(f"Final Score: {best_candidate.fitness:.3f}")
        if best_candidate.metrics:
            print(best_candidate.metrics)

        # Save results
        self.save_results(best_candidate)

        return best_candidate

    def save_results(self, best_candidate: PromptCandidate):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best prompt
        prompt_file = os.path.join(self.results_dir, f"best_prompt_{timestamp}.txt")
        with open(prompt_file, 'w') as f:
            f.write(f"Prompt Name: {best_candidate.template.name}\n")
            f.write(f"Score: {best_candidate.fitness:.3f}\n")
            f.write(f"Generation: {best_candidate.generation}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("PROMPT TEMPLATE:\n")
            f.write("="*70 + "\n\n")
            f.write(best_candidate.template.template)

        print(f"\nBest prompt saved to: {prompt_file}")

        # Save metrics
        metrics_file = os.path.join(self.results_dir, f"metrics_{timestamp}.json")
        with open(metrics_file, 'w') as f:
            json.dump({
                'best_prompt': best_candidate.template.name,
                'best_score': best_candidate.fitness,
                'metrics': best_candidate.metrics.to_dict() if best_candidate.metrics else {},
                'all_results': self.all_results
            }, f, indent=2)

        print(f"Metrics saved to: {metrics_file}")

        # Save leaderboard
        leaderboard = self.tracker.get_leaderboard()
        leaderboard_file = os.path.join(self.results_dir, f"leaderboard_{timestamp}.csv")
        leaderboard.to_csv(leaderboard_file, index=False)
        print(f"Leaderboard saved to: {leaderboard_file}")

        # Print leaderboard
        print(f"\n{'='*70}")
        print("TOP 10 PROMPTS")
        print(f"{'='*70}")
        print(leaderboard.head(10).to_string(index=False))

    def quick_test(self, prompt_name: Optional[str] = None):
        """
        Quick test of a specific prompt or all base prompts.

        Args:
            prompt_name: Name of prompt to test, or None for all
        """
        print("Quick Test Mode")
        print("="*70)

        # Load data
        test_data = self.load_data()

        # Get prompts
        if prompt_name:
            templates = [t for t in PromptTemplateLibrary.get_all_templates()
                        if t.name == prompt_name]
            if not templates:
                print(f"Prompt '{prompt_name}' not found")
                return
        else:
            templates = PromptTemplateLibrary.get_all_templates()

        # Test each prompt
        for template in templates:
            print(f"\nTesting: {template.name}")
            candidate = PromptCandidate(template)
            metrics_list = self.test_prompt_on_data(template, test_data[:2])  # Test on 2 files

            if metrics_list:
                avg_score = sum(m.composite_score for m in metrics_list) / len(metrics_list)
                print(f"Average Score: {avg_score:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Automated Prompt Tuning System")
    parser.add_argument("--mode", choices=["generate", "optimize", "quick-test"],
                       default="optimize", help="Mode to run")
    parser.add_argument("--provider", choices=["openai", "anthropic", "ollama", "mock"],
                       default="mock", help="LLM provider")
    parser.add_argument("--model", type=str, help="Model name (provider-specific)")
    parser.add_argument("--api-key", type=str, help="API key for provider")
    parser.add_argument("--generations", type=int, default=5, help="Max generations")
    parser.add_argument("--population", type=int, default=15, help="Population size")
    parser.add_argument("--test-files", type=int, default=5, help="Number of test files")
    parser.add_argument("--prompt-name", type=str, help="Specific prompt to test (quick-test mode)")

    args = parser.parse_args()

    if args.mode == "generate":
        # Generate sample data
        print("Generating sample data...")
        generator = BankDataGenerator(num_files=30, transactions_per_file=100)
        generator.generate_all_files()
        stats = generator.get_ground_truth_stats()
        print("\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    elif args.mode == "optimize":
        # Run optimization
        llm_config = {}
        if args.model:
            llm_config['model'] = args.model
        if args.api_key:
            llm_config['api_key'] = args.api_key

        orchestrator = PromptTuningOrchestrator(
            llm_provider=args.provider,
            llm_config=llm_config,
            max_generations=args.generations,
            population_size=args.population,
            num_test_files=args.test_files
        )

        best = orchestrator.run_optimization()

    elif args.mode == "quick-test":
        # Quick test
        llm_config = {}
        if args.model:
            llm_config['model'] = args.model
        if args.api_key:
            llm_config['api_key'] = args.api_key

        orchestrator = PromptTuningOrchestrator(
            llm_provider=args.provider,
            llm_config=llm_config
        )

        orchestrator.quick_test(args.prompt_name)


if __name__ == "__main__":
    main()
