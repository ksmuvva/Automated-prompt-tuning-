"""
Example usage of the Automated Prompt Tuning System.
Demonstrates various ways to use the system, including the USPM Agent.
"""

from data_generator import BankDataGenerator
from prompt_templates import PromptTemplateLibrary, PromptTemplate
from llm_interface import LLMFactory, LLMTester, format_transaction_data
from metrics_evaluator import MetricsEvaluator
from prompt_optimizer import PromptOptimizer
from main import PromptTuningOrchestrator
from uspm_agent import USPMAgent
import pandas as pd
import os


def example_0_uspm_agent():
    """Example 0: Using USPM Agent with different LLM providers."""
    print("="*70)
    print("EXAMPLE 0: USPM Agent - Unified Smart Prompt Management")
    print("="*70)

    print("\n--- USPM Agent with Different Providers ---\n")

    # Example 1: OpenAI
    print("# OpenAI (GPT-4)")
    print("agent = USPMAgent(mode='guided')  # or mode='quick'")
    print("# Set environment: export OPENAI_API_KEY='your-key'")
    print("agent.run_interactive()\n")

    # Example 2: Anthropic
    print("# Anthropic (Claude)")
    print("agent = USPMAgent(mode='guided')")
    print("# Set environment: export ANTHROPIC_API_KEY='your-key'")
    print("agent.run_interactive()\n")

    # Example 3: Ollama (Local)
    print("# Local models (Ollama)")
    print("agent = USPMAgent(mode='quick')")
    print("# Ensure Ollama is running: ollama serve")
    print("agent.run_interactive()\n")

    print("--- Two Modes Available ---\n")
    print("1. Guided Mode (mode='guided'):")
    print("   - Full 10-step interactive workflow")
    print("   - AI reasoning & explainability")
    print("   - Bias detection & human validation")
    print("   - Perfect for first-time users\n")

    print("2. Quick Mode (mode='quick'):")
    print("   - Fast natural language commands")
    print("   - Direct task execution")
    print("   - Expert-friendly interface")
    print("   - Example: 'optimize prompts for 10 generations'\n")

    print("--- Programmatic Usage ---\n")
    print("# Guided mode - Full workflow")
    print("agent_guided = USPMAgent(mode='guided')")
    print("agent_guided.run_interactive()\n")

    print("# Quick mode - Execute specific tasks")
    print("agent_quick = USPMAgent(mode='quick')")
    print("task = agent_quick.parse_intent('generate 50 files with 200 transactions')")
    print("result = agent_quick.execute_task(task)")
    print("print(result)\n")

    print("# Switch modes anytime")
    print("agent = USPMAgent(mode='quick')")
    print("agent.switch_mode('guided')  # Switch from quick to guided")
    print("\nNote: Set your LLM API keys as environment variables before running")


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

    # Note: Configure LLM provider (openai, anthropic, or ollama)
    # Example: llm = LLMFactory.create("openai", model="gpt-4", api_key=os.getenv("OPENAI_API_KEY"))
    print("\nSkipping test - Please configure an LLM provider first")
    return

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

    # Note: Configure LLM provider (openai, anthropic, or ollama)
    print("\nSkipping test - Please configure an LLM provider first")
    return

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
    print("EXAMPLE 4: Full Optimization")
    print("="*70)

    # Note: Configure LLM provider (openai, anthropic, or ollama)
    # Example:
    # orchestrator = PromptTuningOrchestrator(
    #     llm_provider="openai",
    #     llm_config={"model": "gpt-4", "api_key": os.getenv("OPENAI_API_KEY")},
    #     max_generations=3,
    #     population_size=8,
    #     num_test_files=3
    # )
    # best = orchestrator.run_optimization()

    print("\nSkipping optimization - Please configure an LLM provider first")
    print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable and uncomment code")


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

    # Note: Configure LLM provider (openai, anthropic, or ollama)
    print("\nSkipping test - Please configure an LLM provider first")
    return

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

    # Show USPM Agent examples first
    try:
        example_0_uspm_agent()
    except Exception as e:
        print(f"Example 0 error: {e}")

    # Check if data exists, generate if not
    if not os.path.exists("bank_data"):
        print("\nGenerating sample data first...")
        example_1_generate_data()
    else:
        print("\nUsing existing data in bank_data/")

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
