"""
Example usage of the Automated Prompt Tuning System.
Demonstrates various ways to use the system.
"""

from data_generator import BankDataGenerator
from prompt_templates import PromptTemplateLibrary, PromptTemplate
from llm_interface import LLMFactory, LLMTester, format_transaction_data
from metrics_evaluator import MetricsEvaluator
from prompt_optimizer import PromptOptimizer
from main import PromptTuningOrchestrator
import pandas as pd


def example_1_generate_data():
    """Example 1: Generate sample bank transaction data."""
    print("="*70)
    print("EXAMPLE 1: Generate Sample Data")
    print("="*70)

    generator = BankDataGenerator(num_files=5, transactions_per_file=50)
    files = generator.generate_all_files()

    print(f"\nGenerated {len(files)} files")

    # Show statistics
    stats = generator.get_ground_truth_stats()
    print("\nGround Truth Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def example_2_test_single_prompt():
    """Example 2: Test a single prompt template."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Test Single Prompt")
    print("="*70)

    # Get a prompt template
    templates = PromptTemplateLibrary.get_all_templates()
    template = templates[0]  # Use first template

    print(f"\nTesting prompt: {template.name}")
    print(f"Style: {template.style}")

    # Create mock LLM
    llm = LLMFactory.create("mock", accuracy=0.85)
    tester = LLMTester(llm)

    # Load a test file
    df = pd.read_csv("bank_data/bank_account_00.csv")

    # Format data and create prompt
    data_str = format_transaction_data(df)
    full_prompt = template.format(data=data_str)

    # Test the prompt
    response, response_time = tester.test_prompt(full_prompt)

    print(f"\nResponse received in {response_time:.2f} seconds")
    print(f"Response: {response[:200]}...")

    # Evaluate
    evaluator = MetricsEvaluator()
    metrics = evaluator.evaluate(df, response, response_time)

    print(f"\nMetrics:")
    print(metrics)


def example_3_compare_prompts():
    """Example 3: Compare multiple prompt styles."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare Prompt Styles")
    print("="*70)

    # Get prompts by style
    templates = PromptTemplateLibrary.get_all_templates()
    styles = set(t.style for t in templates)

    print(f"\nAvailable styles: {', '.join(styles)}")

    # Test one prompt from each style
    llm = LLMFactory.create("mock", accuracy=0.8)
    tester = LLMTester(llm)
    evaluator = MetricsEvaluator()

    # Load test data
    df = pd.read_csv("bank_data/bank_account_00.csv")
    data_str = format_transaction_data(df)

    results = []

    for style in list(styles)[:3]:  # Test first 3 styles
        # Get first template of this style
        template = next(t for t in templates if t.style == style)

        print(f"\nTesting {style}: {template.name}")

        full_prompt = template.format(data=data_str)
        response, response_time = tester.test_prompt(full_prompt)
        metrics = evaluator.evaluate(df, response, response_time)

        results.append({
            'style': style,
            'name': template.name,
            'score': metrics.composite_score,
            'time': response_time
        })

        print(f"  Score: {metrics.composite_score:.3f}")
        print(f"  Time: {response_time:.2f}s")

    # Show comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    for r in sorted(results, key=lambda x: x['score'], reverse=True):
        print(f"{r['style']:20s} | Score: {r['score']:.3f} | Time: {r['time']:.2f}s")


def example_4_optimization():
    """Example 4: Run full optimization."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Full Optimization (Mock LLM)")
    print("="*70)

    orchestrator = PromptTuningOrchestrator(
        llm_provider="mock",
        llm_config={"accuracy": 0.85},
        max_generations=3,  # Small number for demo
        population_size=8,
        num_test_files=3
    )

    best = orchestrator.run_optimization()

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Best prompt: {best.template.name}")
    print(f"Score: {best.fitness:.3f}")


def example_5_custom_prompt():
    """Example 5: Create and test custom prompt."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Custom Prompt Template")
    print("="*70)

    # Create custom template
    custom_template = PromptTemplate(
        name="my_custom_prompt",
        style="custom",
        template="""BANK TRANSACTION SECURITY ANALYSIS

Your role: Senior fraud detection specialist

Review the transactions below and flag:
1. Amounts exceeding 250 GBP
2. Suspicious or fraudulent patterns

Data:
{data}

Provide JSON response:
{{"above_250": ["TXN..."], "anomalies": [{{"transaction_id": "...", "reason": "..."}}]}}

Be thorough and accurate."""
    )

    print(f"Created custom template: {custom_template.name}")

    # Test it
    llm = LLMFactory.create("mock", accuracy=0.9)
    tester = LLMTester(llm)
    evaluator = MetricsEvaluator()

    df = pd.read_csv("bank_data/bank_account_00.csv")
    data_str = format_transaction_data(df)

    full_prompt = custom_template.format(data=data_str)
    response, response_time = tester.test_prompt(full_prompt)
    metrics = evaluator.evaluate(df, response, response_time)

    print(f"\nPerformance:")
    print(f"  Composite Score: {metrics.composite_score:.3f}")
    print(f"  High-Value F1: {metrics.high_value_f1:.3f}")
    print(f"  Anomaly F1: {metrics.anomaly_f1:.3f}")


def example_6_prompt_mutations():
    """Example 6: Demonstrate prompt mutations."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Prompt Mutations")
    print("="*70)

    optimizer = PromptOptimizer()
    base_template = PromptTemplateLibrary.get_all_templates()[0]

    print(f"Base template: {base_template.name}")
    print(f"Original length: {len(base_template.template)} characters")

    # Apply different mutations
    mutations = [
        ("Emphasis", optimizer._add_emphasis(base_template)),
        ("Structure", optimizer._add_structure(base_template)),
        ("Examples", optimizer._add_examples(base_template)),
        ("Constraints", optimizer._add_constraints(base_template)),
        ("Simplified", optimizer._simplify(base_template))
    ]

    for mutation_type, mutated in mutations:
        print(f"\n{mutation_type}:")
        print(f"  Name: {mutated.name}")
        print(f"  Length: {len(mutated.template)} characters")
        print(f"  Preview: {mutated.template[:100]}...")


def main():
    """Run all examples."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     AUTOMATED PROMPT TUNING SYSTEM - EXAMPLE USAGE                   ║
╚══════════════════════════════════════════════════════════════════════╝
""")

    # Check if data exists, generate if not
    import os
    if not os.path.exists("bank_data"):
        print("Generating sample data first...")
        example_1_generate_data()
    else:
        print("Using existing data in bank_data/")

    # Run examples
    try:
        example_2_test_single_prompt()
    except Exception as e:
        print(f"Example 2 error: {e}")

    try:
        example_3_compare_prompts()
    except Exception as e:
        print(f"Example 3 error: {e}")

    try:
        example_5_custom_prompt()
    except Exception as e:
        print(f"Example 5 error: {e}")

    try:
        example_6_prompt_mutations()
    except Exception as e:
        print(f"Example 6 error: {e}")

    # Optional: Run full optimization
    print("\n" + "="*70)
    print("Run full optimization? (example_4)")
    print("This will take a few minutes...")
    print("Uncomment the line below in the code to run it.")
    print("="*70)
    # example_4_optimization()  # Uncomment to run

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
