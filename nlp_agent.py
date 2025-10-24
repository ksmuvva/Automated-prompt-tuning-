"""
NLP Agent for handling natural language tasks in the prompt tuning system.
This agent can understand and execute various commands through natural language.
"""

import re
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from data_generator import BankDataGenerator
from prompt_templates import PromptTemplateLibrary
from llm_interface import LLMFactory, LLMTester
from metrics_evaluator import MetricsEvaluator, MetricsTracker
from prompt_optimizer import PromptOptimizer, AdaptiveOptimizer


class TaskType(Enum):
    """Types of tasks the agent can handle."""
    GENERATE_DATA = "generate_data"
    OPTIMIZE_PROMPTS = "optimize_prompts"
    TEST_PROMPT = "test_prompt"
    COMPARE_PROMPTS = "compare_prompts"
    LIST_PROMPTS = "list_prompts"
    SHOW_RESULTS = "show_results"
    CONFIGURE = "configure"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class AgentTask:
    """Represents a task for the agent to execute."""
    task_type: TaskType
    parameters: Dict[str, Any]
    confidence: float


class NLPAgent:
    """
    Intelligent NLP agent for prompt tuning system.
    Can understand natural language commands and execute appropriate actions.
    """

    def __init__(self, data_dir: str = "bank_data", results_dir: str = "results"):
        """
        Initialize the NLP agent.

        Args:
            data_dir: Directory for data files
            results_dir: Directory for results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.config = self._load_config()

        # Agent state
        self.current_provider = None
        self.llm_tester = None
        self.evaluator = MetricsEvaluator()
        self.tracker = MetricsTracker()
        self.optimizer = None
        self.adaptive = AdaptiveOptimizer()

        # Conversation context
        self.context = {
            'last_task': None,
            'last_results': None,
            'provider_configured': False
        }

        print("NLP Agent initialized. Type 'help' for available commands.")

    def _load_config(self) -> Dict:
        """Load configuration from config.json if available."""
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def parse_intent(self, user_input: str) -> AgentTask:
        """
        Parse user input to understand intent and extract parameters.

        Args:
            user_input: Natural language input from user

        Returns:
            AgentTask with parsed intent and parameters
        """
        user_input = user_input.lower().strip()

        # Pattern matching for different intents
        patterns = {
            TaskType.GENERATE_DATA: [
                r"generate.*data",
                r"create.*data",
                r"make.*transactions",
                r"generate.*files"
            ],
            TaskType.OPTIMIZE_PROMPTS: [
                r"optimize.*prompts?",
                r"run.*optimization",
                r"improve.*prompts?",
                r"tune.*prompts?",
                r"evolve.*prompts?"
            ],
            TaskType.TEST_PROMPT: [
                r"test.*prompt",
                r"try.*prompt",
                r"evaluate.*prompt",
                r"run.*prompt"
            ],
            TaskType.COMPARE_PROMPTS: [
                r"compare.*prompts?",
                r"compare.*templates?",
                r"which.*better"
            ],
            TaskType.LIST_PROMPTS: [
                r"list.*prompts?",
                r"show.*prompts?",
                r"available.*prompts?",
                r"what.*prompts?"
            ],
            TaskType.SHOW_RESULTS: [
                r"show.*results?",
                r"display.*results?",
                r"view.*results?",
                r"see.*results?"
            ],
            TaskType.CONFIGURE: [
                r"configure.*provider",
                r"set.*provider",
                r"use.*provider",
                r"setup.*llm",
                r"configure.*api"
            ],
            TaskType.HELP: [
                r"help",
                r"what.*can.*you.*do",
                r"commands?",
                r"instructions?"
            ]
        }

        # Try to match patterns
        for task_type, task_patterns in patterns.items():
            for pattern in task_patterns:
                if re.search(pattern, user_input):
                    params = self._extract_parameters(user_input, task_type)
                    return AgentTask(task_type, params, confidence=0.9)

        # If no match found
        return AgentTask(TaskType.UNKNOWN, {}, confidence=0.0)

    def _extract_parameters(self, user_input: str, task_type: TaskType) -> Dict[str, Any]:
        """
        Extract parameters from user input based on task type.

        Args:
            user_input: User's natural language input
            task_type: Identified task type

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Extract numbers
        numbers = re.findall(r'\d+', user_input)

        if task_type == TaskType.GENERATE_DATA:
            if numbers:
                params['num_files'] = int(numbers[0]) if len(numbers) > 0 else 30
                params['transactions_per_file'] = int(numbers[1]) if len(numbers) > 1 else 100
            else:
                params['num_files'] = 30
                params['transactions_per_file'] = 100

        elif task_type == TaskType.OPTIMIZE_PROMPTS:
            params['generations'] = int(numbers[0]) if numbers else 5
            params['population_size'] = int(numbers[1]) if len(numbers) > 1 else 15

        elif task_type == TaskType.TEST_PROMPT:
            # Extract prompt name if mentioned
            templates = PromptTemplateLibrary.get_all_templates()
            for template in templates:
                if template.name.lower() in user_input:
                    params['prompt_name'] = template.name
                    break

        elif task_type == TaskType.CONFIGURE:
            # Extract provider type
            if 'openai' in user_input or 'gpt' in user_input:
                params['provider'] = 'openai'
                # Extract model if mentioned
                if 'gpt-4' in user_input:
                    params['model'] = 'gpt-4'
                elif 'gpt-3.5' in user_input:
                    params['model'] = 'gpt-3.5-turbo'
            elif 'anthropic' in user_input or 'claude' in user_input:
                params['provider'] = 'anthropic'
                if 'claude-3-5-sonnet' in user_input or 'sonnet' in user_input:
                    params['model'] = 'claude-3-5-sonnet-20241022'
            elif 'ollama' in user_input or 'local' in user_input:
                params['provider'] = 'ollama'
                if 'llama' in user_input:
                    params['model'] = 'llama2'

        return params

    def execute_task(self, task: AgentTask) -> str:
        """
        Execute the identified task.

        Args:
            task: AgentTask to execute

        Returns:
            Result message
        """
        if task.task_type == TaskType.GENERATE_DATA:
            return self._generate_data(task.parameters)
        elif task.task_type == TaskType.OPTIMIZE_PROMPTS:
            return self._optimize_prompts(task.parameters)
        elif task.task_type == TaskType.TEST_PROMPT:
            return self._test_prompt(task.parameters)
        elif task.task_type == TaskType.COMPARE_PROMPTS:
            return self._compare_prompts(task.parameters)
        elif task.task_type == TaskType.LIST_PROMPTS:
            return self._list_prompts()
        elif task.task_type == TaskType.SHOW_RESULTS:
            return self._show_results()
        elif task.task_type == TaskType.CONFIGURE:
            return self._configure_provider(task.parameters)
        elif task.task_type == TaskType.HELP:
            return self._show_help()
        else:
            return self._handle_unknown(task)

    def _check_provider_configured(self) -> bool:
        """Check if LLM provider is configured."""
        if not self.context['provider_configured']:
            return False
        return True

    def _generate_data(self, params: Dict) -> str:
        """Generate sample bank transaction data."""
        num_files = params.get('num_files', 30)
        transactions_per_file = params.get('transactions_per_file', 100)

        print(f"\nGenerating {num_files} files with {transactions_per_file} transactions each...")
        generator = BankDataGenerator(num_files=num_files, transactions_per_file=transactions_per_file)
        generator.generate_all_files()
        stats = generator.get_ground_truth_stats()

        result = f"\nData generation complete!\n"
        result += f"Files created: {num_files}\n"
        result += f"Transactions per file: {transactions_per_file}\n"
        result += f"\nStatistics:\n"
        for key, value in stats.items():
            result += f"  {key}: {value}\n"

        self.context['last_task'] = 'generate_data'
        return result

    def _configure_provider(self, params: Dict) -> str:
        """Configure LLM provider."""
        provider = params.get('provider')
        if not provider:
            return ("Please specify a provider. Available: openai, anthropic, ollama\n"
                   "Example: 'configure openai provider' or 'use anthropic claude'")

        llm_config = {}
        if 'model' in params:
            llm_config['model'] = params['model']

        # Get API key from environment or prompt
        if provider in ['openai', 'anthropic']:
            api_key_env = 'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'
            api_key = os.getenv(api_key_env)
            if api_key:
                llm_config['api_key'] = api_key
            else:
                return (f"\n{api_key_env} environment variable not set.\n"
                       f"Please set it: export {api_key_env}='your-api-key'")

        try:
            self.current_provider = LLMFactory.create(provider, **llm_config)
            self.llm_tester = LLMTester(self.current_provider)
            self.context['provider_configured'] = True

            model_info = llm_config.get('model', 'default')
            return f"\nProvider configured successfully!\nProvider: {provider}\nModel: {model_info}\n"
        except Exception as e:
            return f"\nError configuring provider: {str(e)}\n"

    def _optimize_prompts(self, params: Dict) -> str:
        """Run prompt optimization."""
        if not self._check_provider_configured():
            return "\nPlease configure an LLM provider first. Example: 'configure openai provider'\n"

        generations = params.get('generations', 5)
        population_size = params.get('population_size', 15)

        print(f"\nRunning optimization for {generations} generations...")

        from main import PromptTuningOrchestrator

        # Create orchestrator with configured provider
        orchestrator = PromptTuningOrchestrator(
            data_dir=self.data_dir,
            llm_provider='openai',  # This will be overridden
            max_generations=generations,
            population_size=population_size
        )

        # Override with our configured provider
        orchestrator.llm_tester = self.llm_tester

        best = orchestrator.run_optimization()
        self.context['last_task'] = 'optimize'
        self.context['last_results'] = best

        return f"\nOptimization complete! Best prompt: {best.template.name} (Score: {best.fitness:.3f})\n"

    def _test_prompt(self, params: Dict) -> str:
        """Test a specific prompt."""
        if not self._check_provider_configured():
            return "\nPlease configure an LLM provider first. Example: 'configure openai provider'\n"

        prompt_name = params.get('prompt_name')
        if not prompt_name:
            return "\nPlease specify a prompt name. Use 'list prompts' to see available prompts.\n"

        from main import PromptTuningOrchestrator

        orchestrator = PromptTuningOrchestrator(data_dir=self.data_dir)
        orchestrator.llm_tester = self.llm_tester

        orchestrator.quick_test(prompt_name)
        return f"\nTest complete for prompt: {prompt_name}\n"

    def _compare_prompts(self, params: Dict) -> str:
        """Compare multiple prompts."""
        if not self._check_provider_configured():
            return "\nPlease configure an LLM provider first. Example: 'configure openai provider'\n"

        print("\nComparing all available prompts...")

        from main import PromptTuningOrchestrator

        orchestrator = PromptTuningOrchestrator(data_dir=self.data_dir)
        orchestrator.llm_tester = self.llm_tester

        orchestrator.quick_test(None)  # Test all prompts
        return "\nComparison complete! Check results above.\n"

    def _list_prompts(self) -> str:
        """List all available prompt templates."""
        templates = PromptTemplateLibrary.get_all_templates()

        result = "\nAvailable Prompt Templates:\n"
        result += "=" * 50 + "\n"
        for i, template in enumerate(templates, 1):
            result += f"{i}. {template.name}\n"
            result += f"   Style: {template.style}\n"
            if template.description:
                result += f"   Description: {template.description}\n"
            result += "\n"

        return result

    def _show_results(self) -> str:
        """Show last results."""
        if not self.context.get('last_results'):
            return "\nNo results available yet. Run an optimization or test first.\n"

        result = "\nLast Results:\n"
        result += "=" * 50 + "\n"
        last_results = self.context['last_results']
        result += f"Prompt: {last_results.template.name}\n"
        result += f"Score: {last_results.fitness:.3f}\n"
        if last_results.metrics:
            result += str(last_results.metrics)

        return result

    def _show_help(self) -> str:
        """Show help information."""
        help_text = """
NLP Agent - Natural Language Interface for Prompt Tuning
========================================================

Available Commands:
-------------------

1. Configure Provider:
   - "configure openai provider"
   - "use anthropic claude"
   - "setup ollama"

2. Generate Data:
   - "generate data"
   - "create 50 files with 200 transactions"
   - "generate sample data"

3. Optimize Prompts:
   - "optimize prompts"
   - "run optimization for 10 generations"
   - "improve prompts with 20 population size"

4. Test Prompts:
   - "test prompt concise_direct"
   - "evaluate prompt detailed_analytical"

5. Compare Prompts:
   - "compare prompts"
   - "which prompt is better"

6. List Prompts:
   - "list prompts"
   - "show available prompts"

7. Show Results:
   - "show results"
   - "display last results"

8. Help:
   - "help"
   - "what can you do"

Examples:
---------
> configure openai provider
> generate 30 files
> optimize prompts for 5 generations
> test prompt concise_direct
> show results

Note: You need to configure an LLM provider before running optimization or testing.
Set API keys via environment variables:
  - export OPENAI_API_KEY='your-key'
  - export ANTHROPIC_API_KEY='your-key'
"""
        return help_text

    def _handle_unknown(self, task: AgentTask) -> str:
        """Handle unknown commands."""
        return ("\nI didn't understand that command. Type 'help' to see available commands.\n"
                "You can use natural language like:\n"
                "  - 'generate data'\n"
                "  - 'optimize prompts'\n"
                "  - 'configure openai provider'\n")

    def run_interactive(self):
        """Run agent in interactive mode."""
        print("\n" + "=" * 60)
        print("NLP Agent - Interactive Mode")
        print("=" * 60)
        print("Type 'help' for available commands or 'exit' to quit.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Agent> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\nGoodbye!\n")
                    break

                # Parse and execute
                task = self.parse_intent(user_input)

                if task.confidence > 0.5:
                    print(f"\n[Understanding: {task.task_type.value}]")

                result = self.execute_task(task)
                print(result)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.\n")
            except Exception as e:
                print(f"\nError: {str(e)}\n")

    def run_command(self, command: str) -> str:
        """
        Run a single command and return result.

        Args:
            command: Natural language command

        Returns:
            Result string
        """
        task = self.parse_intent(command)
        return self.execute_task(task)


if __name__ == "__main__":
    agent = NLPAgent()
    agent.run_interactive()
