"""
USPM Agent - Unified Smart Prompt Management Agent
Merges EnhancedAgent (guided workflow) and NLPAgent (quick commands)
into a single, intelligent agent with both modes.

Created using Tree of Thought and Beam Reasoning:
- Beam 1 (9.5/10): Unified class with mode selection - SELECTED
- Beam 2 (8.0/10): Strategy pattern with delegation
- Beam 3 (7.0/10): Feature flags approach

Architecture: Single USPMAgent class with dual modes:
  - Guided Mode: Full 10-step workflow with reasoning & explainability
  - Quick Mode: Natural language commands for fast execution
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from data_generator import BankDataGenerator
from prompt_templates import PromptTemplateLibrary, PromptTemplate
from llm_interface import LLMFactory, LLMTester
from metrics_evaluator import MetricsEvaluator


class AgentMode(Enum):
    """Operating modes for USPM agent."""
    GUIDED = "guided"      # Full workflow with reasoning
    QUICK = "quick"        # Fast natural language commands


class WorkflowStage(Enum):
    """Stages in the guided workflow."""
    INITIALIZATION = "initialization"
    API_KEY_VALIDATION = "api_key_validation"
    USE_CASE_DETERMINATION = "use_case_determination"
    DATA_REQUIREMENTS = "data_requirements"
    DATA_GENERATION = "data_generation"
    GROUND_TRUTH_GENERATION = "ground_truth_generation"
    METRICS_PRESCRIPTION = "metrics_prescription"
    METRICS_VALIDATION = "metrics_validation"
    PROMPT_LOADING = "prompt_loading"
    OPTIMIZATION = "optimization"
    COMPLETED = "completed"


class TaskType(Enum):
    """Types of tasks in quick mode."""
    GENERATE_DATA = "generate_data"
    OPTIMIZE_PROMPTS = "optimize_prompts"
    TEST_PROMPT = "test_prompt"
    COMPARE_PROMPTS = "compare_prompts"
    LIST_PROMPTS = "list_prompts"
    SHOW_RESULTS = "show_results"
    CONFIGURE = "configure"
    SWITCH_MODE = "switch_mode"
    HELP = "help"
    UNKNOWN = "unknown"


@dataclass
class UseCaseContext:
    """Context information about the determined use case."""
    use_case_name: str
    use_case_type: str
    domain: str
    description: str
    key_objectives: List[str]
    data_characteristics: Dict[str, Any]
    reasoning: str
    confidence: float


@dataclass
class DataRequirements:
    """Requirements for synthetic data generation."""
    num_files: int
    records_per_file: int
    data_type: str
    features: List[str]
    anomaly_rate: float
    special_conditions: Dict[str, Any]
    reasoning: str


@dataclass
class MetricsPrescription:
    """LLM-prescribed metrics for evaluation."""
    primary_metrics: List[str]
    secondary_metrics: List[str]
    metric_definitions: Dict[str, str]
    thresholds: Dict[str, float]
    reasoning: str
    llm_explanation: str


@dataclass
class ReasoningLog:
    """Log entry for reasoning and explainability."""
    stage: str
    timestamp: str
    action: str
    reasoning: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    bias_check: Dict[str, Any]


@dataclass
class AgentTask:
    """Represents a task for quick mode execution."""
    task_type: TaskType
    parameters: Dict[str, Any]
    confidence: float


class USPMAgent:
    """
    Unified Smart Prompt Management Agent

    Combines guided workflow (with reasoning & explainability) and
    quick natural language commands into a single intelligent agent.

    Modes:
      - Guided: Step-by-step workflow with full transparency
      - Quick: Fast natural language command execution

    Features:
      - Reasoning & Explainability (guided mode)
      - Bias Detection (guided mode)
      - Human-in-the-Loop validation (guided mode)
      - Natural language understanding (both modes)
      - Workflow state management (guided mode)
      - Context tracking (both modes)
    """

    def __init__(self, mode: str = "guided", data_dir: str = "bank_data", results_dir: str = "results"):
        """
        Initialize USPM agent.

        Args:
            mode: "guided" for full workflow or "quick" for fast commands
            data_dir: Directory for data files
            results_dir: Directory for results
        """
        self.mode = AgentMode.GUIDED if mode.lower() == "guided" else AgentMode.QUICK
        self.data_dir = data_dir
        self.results_dir = results_dir

        # Shared state
        self.llm_provider = None
        self.llm_tester = None
        self.provider_type = None
        self.reasoning_logs: List[ReasoningLog] = []

        # Guided mode state
        self.current_stage = WorkflowStage.INITIALIZATION
        self.use_case_context: Optional[UseCaseContext] = None
        self.data_requirements: Optional[DataRequirements] = None
        self.metrics_prescription: Optional[MetricsPrescription] = None
        self.generated_data_path = None
        self.ground_truth = None
        self.validation_results = {}

        # Quick mode state
        self.context = {
            'last_task': None,
            'last_results': None,
            'provider_configured': False
        }
        self.config = self._load_config()

        self._print_header()

    def _print_header(self):
        """Print agent header based on mode."""
        if self.mode == AgentMode.GUIDED:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║         USPM Agent - Unified Smart Prompt Management        ║")
            print("║              GUIDED MODE - Full Workflow with Reasoning      ║")
            print("╚══════════════════════════════════════════════════════════════╝\n")
        else:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║         USPM Agent - Unified Smart Prompt Management        ║")
            print("║              QUICK MODE - Natural Language Commands          ║")
            print("╚══════════════════════════════════════════════════════════════╝\n")
            print("Type 'help' for available commands or 'guided mode' to switch.\n")

    def _load_config(self) -> Dict:
        """Load configuration from config.json if available."""
        config_path = "config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    # ========================================================================
    # SHARED METHODS - Used by both modes
    # ========================================================================

    def log_reasoning(self, stage: str, action: str, reasoning: str,
                     input_data: Dict, output_data: Dict,
                     confidence: float = 1.0, bias_check: Optional[Dict] = None):
        """Log reasoning for explainability (used in both modes)."""
        log_entry = ReasoningLog(
            stage=stage,
            timestamp=datetime.now().isoformat(),
            action=action,
            reasoning=reasoning,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            bias_check=bias_check or {}
        )
        self.reasoning_logs.append(log_entry)

    def check_bias(self, text: str, context: str) -> Dict[str, Any]:
        """Check for potential biases in text/decisions."""
        bias_indicators = {
            "language_bias": [],
            "assumption_bias": [],
            "selection_bias": [],
            "severity": "low"
        }

        # Check for biased language
        biased_terms = ["always", "never", "obviously", "clearly", "certainly"]
        for term in biased_terms:
            if term.lower() in text.lower():
                bias_indicators["language_bias"].append(f"Absolute term detected: '{term}'")

        # Check for assumptions
        assumption_patterns = [r"should be", r"must be", r"obviously", r"of course"]
        for pattern in assumption_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                bias_indicators["assumption_bias"].append(f"Assumption detected: '{pattern}'")

        # Determine severity
        total_issues = len(bias_indicators["language_bias"]) + len(bias_indicators["assumption_bias"])
        if total_issues > 3:
            bias_indicators["severity"] = "high"
        elif total_issues > 1:
            bias_indicators["severity"] = "medium"

        return bias_indicators

    def validate_api_keys(self) -> Tuple[bool, Dict[str, bool]]:
        """Validate available API keys for LLM providers."""
        print("\n" + "="*60)
        print("API KEY VALIDATION")
        print("="*60)

        key_status = {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "ollama": True  # Ollama is local, no key needed
        }

        print("\nChecking for available LLM providers:")
        for provider, available in key_status.items():
            status = "✓ Available" if available else "✗ Not configured"
            print(f"  {provider.capitalize():12} : {status}")
            if not available and provider != "ollama":
                env_var = f"{provider.upper()}_API_KEY"
                print(f"               (Set {env_var} environment variable)")

        has_any = any(key_status.values())

        if not has_any:
            print("\n⚠️  No LLM provider configured!")
            print("Please set at least one API key:")
            print("  export OPENAI_API_KEY='your-key'")
            print("  export ANTHROPIC_API_KEY='your-key'")
            print("Or use Ollama locally (requires Ollama installation)")

        self.log_reasoning(
            stage="api_validation",
            action="validate_api_keys",
            reasoning="Checked environment for LLM provider API keys",
            input_data={},
            output_data={"key_status": key_status, "has_any": has_any},
            confidence=1.0
        )

        return has_any, key_status

    def configure_llm_provider(self, key_status: Optional[Dict[str, bool]] = None,
                              provider: Optional[str] = None, model: Optional[str] = None) -> bool:
        """Configure LLM provider (used by both modes)."""
        if not key_status:
            _, key_status = self.validate_api_keys()

        available_providers = [k for k, v in key_status.items() if v]
        if not available_providers:
            return False

        # Determine provider
        if not provider:
            if len(available_providers) == 1:
                provider = available_providers[0]
                print(f"\nAuto-selecting: {provider}")
            else:
                provider = input(f"\nSelect provider ({'/'.join(available_providers)}): ").strip().lower()
                if provider not in available_providers:
                    print("Invalid provider.")
                    return False

        # Configure provider
        try:
            llm_config = {}

            if provider == "openai":
                if not model:
                    model = input("Enter model (default: gpt-4): ").strip() or "gpt-4"
                llm_config = {"model": model, "api_key": os.getenv("OPENAI_API_KEY")}
            elif provider == "anthropic":
                if not model:
                    model = input("Enter model (default: claude-3-5-sonnet-20241022): ").strip() or "claude-3-5-sonnet-20241022"
                llm_config = {"model": model, "api_key": os.getenv("ANTHROPIC_API_KEY")}
            elif provider == "ollama":
                if not model:
                    model = input("Enter model (default: llama2): ").strip() or "llama2"
                llm_config = {"model": model}

            print(f"\n🔧 Initializing {provider} with model {llm_config.get('model', 'default')}...")

            self.llm_provider = LLMFactory.create(provider, **llm_config)
            self.llm_tester = LLMTester(self.llm_provider)
            self.provider_type = provider
            self.context['provider_configured'] = True

            print(f"✓ {provider.capitalize()} configured successfully!\n")

            self.log_reasoning(
                stage="provider_configuration",
                action="configure_llm",
                reasoning=f"Selected {provider} based on available keys",
                input_data={"available_providers": available_providers},
                output_data={"provider": provider, "config": llm_config},
                confidence=1.0
            )

            return True

        except Exception as e:
            print(f"\n✗ Error configuring {provider}: {str(e)}")
            return False

    def switch_mode(self, new_mode: str):
        """Switch between guided and quick modes."""
        old_mode = self.mode.value
        self.mode = AgentMode.GUIDED if new_mode.lower() == "guided" else AgentMode.QUICK

        print(f"\n🔄 Switching from {old_mode} mode to {self.mode.value} mode...")
        self._print_header()

        self.log_reasoning(
            stage="mode_switch",
            action="switch_mode",
            reasoning=f"User requested mode switch from {old_mode} to {self.mode.value}",
            input_data={"old_mode": old_mode},
            output_data={"new_mode": self.mode.value},
            confidence=1.0
        )

    # ========================================================================
    # GUIDED MODE METHODS - Full workflow with reasoning
    # ========================================================================

    def determine_use_case(self) -> UseCaseContext:
        """Determine use case through user interaction and LLM reasoning."""
        print("\n" + "="*60)
        print("STEP 3: USE CASE DETERMINATION")
        print("="*60)

        print("\nDo you already know your use case?")
        knows_use_case = input("(yes/no): ").strip().lower() in ['yes', 'y']

        if knows_use_case:
            # User knows the use case
            print("\nPlease describe your use case:")
            use_case_desc = input("Description: ").strip()
            domain = input("Domain (e.g., finance, healthcare, retail): ").strip()
            objectives = input("Key objectives (comma-separated): ").strip().split(',')
            objectives = [obj.strip() for obj in objectives]

            use_case_context = UseCaseContext(
                use_case_name=input("Use case name: ").strip(),
                use_case_type=input("Type (e.g., fraud_detection, classification): ").strip(),
                domain=domain,
                description=use_case_desc,
                key_objectives=objectives,
                data_characteristics={},
                reasoning="User provided explicit use case description",
                confidence=0.95
            )

        else:
            # Use LLM to determine use case
            print("\nLet me help you determine the use case.")
            print("Please provide information about your needs:\n")

            user_input = {}
            user_input['problem'] = input("1. What problem are you trying to solve? ").strip()
            user_input['data_type'] = input("2. What type of data will you work with? ").strip()
            user_input['goals'] = input("3. What are your main goals? ").strip()
            user_input['constraints'] = input("4. Any specific constraints or requirements? ").strip()

            print("\n🤔 Analyzing your requirements with LLM reasoning...")

            # Use LLM to determine use case
            prompt = f"""Analyze the following user requirements and determine the best use case for a prompt tuning system.

User Requirements:
- Problem: {user_input['problem']}
- Data Type: {user_input['data_type']}
- Goals: {user_input['goals']}
- Constraints: {user_input['constraints']}

Provide a JSON response with:
{{
    "use_case_name": "descriptive name",
    "use_case_type": "type (e.g., fraud_detection, sentiment_analysis, classification)",
    "domain": "industry domain",
    "description": "detailed description",
    "key_objectives": ["objective1", "objective2"],
    "data_characteristics": {{"feature1": "description"}},
    "reasoning": "explain why this use case fits the requirements",
    "confidence": 0.0-1.0
}}

Be specific and provide clear reasoning."""

            try:
                response, _ = self.llm_tester.test_prompt(prompt, use_cache=False)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    use_case_data = json.loads(json_match.group())
                    use_case_context = UseCaseContext(**use_case_data)

                    print("\n✓ Use case determined!")
                    print(f"\n📋 Use Case: {use_case_context.use_case_name}")
                    print(f"   Type: {use_case_context.use_case_type}")
                    print(f"   Domain: {use_case_context.domain}")
                    print(f"\n💭 Reasoning: {use_case_context.reasoning}")
                    print(f"   Confidence: {use_case_context.confidence:.0%}")
                else:
                    print("\n⚠️  Could not parse LLM response. Using default.")
                    use_case_context = self._get_default_use_case()

            except Exception as e:
                print(f"\n⚠️  Error with LLM reasoning: {str(e)}")
                print("Using default bank transaction use case.")
                use_case_context = self._get_default_use_case()

        # Bias check
        bias_check = self.check_bias(use_case_context.reasoning, "use_case_determination")
        if bias_check["severity"] != "low":
            print(f"\n⚠️  Bias detected in use case reasoning (severity: {bias_check['severity']})")

        self.use_case_context = use_case_context
        self.log_reasoning(
            stage="use_case_determination",
            action="determine_use_case",
            reasoning=use_case_context.reasoning,
            input_data={"knows_use_case": knows_use_case},
            output_data=asdict(use_case_context),
            confidence=use_case_context.confidence,
            bias_check=bias_check
        )

        self.current_stage = WorkflowStage.DATA_REQUIREMENTS
        return use_case_context

    def _get_default_use_case(self) -> UseCaseContext:
        """Return default bank transaction use case."""
        return UseCaseContext(
            use_case_name="Bank Transaction Analysis",
            use_case_type="fraud_detection",
            domain="finance",
            description="Detect high-value transactions and anomalies in bank data",
            key_objectives=["Identify transactions above 250 GBP", "Detect anomalous patterns"],
            data_characteristics={"transaction_data": "financial transactions with amounts, dates, merchants"},
            reasoning="Default use case for financial transaction analysis",
            confidence=0.90
        )

    def gather_data_requirements(self) -> DataRequirements:
        """Interactively gather data generation requirements from user."""
        print("\n" + "="*60)
        print("STEP 4: DATA REQUIREMENTS GATHERING")
        print("="*60)

        print(f"\nBased on use case: {self.use_case_context.use_case_name}")
        print("Let's determine the data requirements for synthetic data generation.\n")

        print("Data Volume:")
        num_files = int(input("  Number of data files to generate (default: 30): ").strip() or "30")
        records_per_file = int(input("  Records per file (default: 100): ").strip() or "100")

        print("\nData Characteristics:")
        data_type = input("  Data type (e.g., transactions, events, records): ").strip() or "transactions"

        print("\nFeatures/Columns:")
        print("  Enter features one by one (press Enter on empty line to finish)")
        features = []
        while True:
            feature = input(f"  Feature {len(features)+1}: ").strip()
            if not feature:
                break
            features.append(feature)

        if not features:
            features = ["transaction_id", "date", "amount_gbp", "category", "merchant", "description"]
            print(f"  Using default features: {', '.join(features)}")

        print("\nAnomaly Configuration:")
        anomaly_rate = float(input("  Anomaly rate (0.0-1.0, default: 0.1): ").strip() or "0.1")

        print("\nSpecial Conditions:")
        special_input = input("  Conditions (comma-separated, or press Enter to skip): ").strip()
        special_conditions = {}
        if special_input:
            for condition in special_input.split(','):
                condition = condition.strip()
                if '>' in condition:
                    key, value = condition.split('>')
                    special_conditions[key.strip()] = {"operator": ">", "value": float(value.strip())}
                else:
                    special_conditions[condition] = True

        reasoning = f"Data requirements gathered for {data_type} with {num_files} files and {records_per_file} records each"

        data_reqs = DataRequirements(
            num_files=num_files,
            records_per_file=records_per_file,
            data_type=data_type,
            features=features,
            anomaly_rate=anomaly_rate,
            special_conditions=special_conditions,
            reasoning=reasoning
        )

        print("\n📊 Data Requirements Summary:")
        print(f"  Total records: {num_files * records_per_file}")
        print(f"  Expected anomalies: ~{int(num_files * records_per_file * anomaly_rate)}")
        print(f"  Features: {len(features)}")

        confirm = input("\nProceed with these requirements? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Let's try again...")
            return self.gather_data_requirements()

        self.data_requirements = data_reqs
        self.log_reasoning(
            stage="data_requirements",
            action="gather_requirements",
            reasoning=reasoning,
            input_data={},
            output_data=asdict(data_reqs),
            confidence=0.95
        )

        self.current_stage = WorkflowStage.DATA_GENERATION
        return data_reqs

    def generate_synthetic_data(self) -> str:
        """Generate synthetic data based on gathered requirements."""
        print("\n" + "="*60)
        print("STEP 5: SYNTHETIC DATA GENERATION")
        print("="*60)

        print(f"\n🔄 Generating {self.data_requirements.num_files} files with {self.data_requirements.records_per_file} records each...")

        generator = BankDataGenerator(
            num_files=self.data_requirements.num_files,
            transactions_per_file=self.data_requirements.records_per_file
        )

        files = generator.generate_all_files()
        print(f"\n✓ Generated {len(files)} files successfully!")

        self.generated_data_path = "bank_data"
        self.log_reasoning(
            stage="data_generation",
            action="generate_synthetic_data",
            reasoning=f"Generated {len(files)} CSV files using BankDataGenerator",
            input_data=asdict(self.data_requirements),
            output_data={"files_generated": len(files), "path": self.generated_data_path},
            confidence=1.0
        )

        self.current_stage = WorkflowStage.GROUND_TRUTH_GENERATION
        return self.generated_data_path

    def generate_ground_truth(self) -> Dict:
        """Generate ground truth labels and statistics."""
        print("\n" + "="*60)
        print("STEP 6: GROUND TRUTH GENERATION")
        print("="*60)

        print("\n📈 Calculating ground truth statistics...")

        generator = BankDataGenerator(
            num_files=self.data_requirements.num_files,
            transactions_per_file=self.data_requirements.records_per_file
        )

        stats = generator.get_ground_truth_stats()

        print("\n✓ Ground Truth Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        self.ground_truth = stats
        self.log_reasoning(
            stage="ground_truth",
            action="generate_ground_truth",
            reasoning="Calculated ground truth statistics from generated data",
            input_data={"data_path": self.generated_data_path},
            output_data=stats,
            confidence=1.0
        )

        self.current_stage = WorkflowStage.METRICS_PRESCRIPTION
        return stats

    def prescribe_metrics_with_llm(self) -> MetricsPrescription:
        """Use LLM to prescribe appropriate metrics for the use case."""
        print("\n" + "="*60)
        print("STEP 7: METRICS PRESCRIPTION (LLM-Assisted)")
        print("="*60)

        print("\n🤔 Consulting LLM for metrics recommendations...")

        prompt = f"""As an expert in ML evaluation metrics, prescribe appropriate metrics for this use case:

Use Case: {self.use_case_context.use_case_name}
Type: {self.use_case_context.use_case_type}
Domain: {self.use_case_context.domain}
Objectives: {', '.join(self.use_case_context.key_objectives)}

Ground Truth Stats:
{json.dumps(self.ground_truth, indent=2)}

Provide a JSON response with:
{{
    "primary_metrics": ["metric1", "metric2"],
    "secondary_metrics": ["metric3", "metric4"],
    "metric_definitions": {{
        "metric1": "clear definition of what this measures"
    }},
    "thresholds": {{
        "metric1": 0.85
    }},
    "reasoning": "explain why these metrics are appropriate for this use case",
    "llm_explanation": "detailed explanation of the metric selection rationale"
}}"""

        try:
            response, _ = self.llm_tester.test_prompt(prompt, use_cache=False)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                metrics_data = json.loads(json_match.group())
                prescription = MetricsPrescription(**metrics_data)

                print("\n✓ Metrics prescribed by LLM!")
                print(f"\n📊 Primary Metrics: {', '.join(prescription.primary_metrics)}")
                print(f"   Secondary Metrics: {', '.join(prescription.secondary_metrics)}")
                print(f"\n💭 LLM Reasoning:\n{prescription.llm_explanation}")
            else:
                prescription = self._get_default_metrics()
        except Exception as e:
            print(f"\n⚠️  Error: {str(e)}")
            prescription = self._get_default_metrics()

        # Bias check
        bias_check = self.check_bias(prescription.llm_explanation, "metrics_prescription")
        if bias_check["severity"] != "low":
            print(f"\n⚠️  Bias detected (severity: {bias_check['severity']})")

        self.metrics_prescription = prescription
        self.log_reasoning(
            stage="metrics_prescription",
            action="prescribe_metrics",
            reasoning=prescription.reasoning,
            input_data={"use_case": asdict(self.use_case_context)},
            output_data=asdict(prescription),
            confidence=0.85,
            bias_check=bias_check
        )

        self.current_stage = WorkflowStage.METRICS_VALIDATION
        return prescription

    def _get_default_metrics(self) -> MetricsPrescription:
        """Return default metrics."""
        return MetricsPrescription(
            primary_metrics=["high_value_f1", "anomaly_f1", "composite_score"],
            secondary_metrics=["precision", "recall", "response_time"],
            metric_definitions={
                "high_value_f1": "F1 score for detecting transactions above 250 GBP",
                "anomaly_f1": "F1 score for detecting anomalous transactions",
                "composite_score": "Weighted combination of all metrics"
            },
            thresholds={"high_value_f1": 0.85, "anomaly_f1": 0.80, "composite_score": 0.75},
            reasoning="Standard metrics for fraud detection tasks",
            llm_explanation="These metrics balance precision and recall for both high-value and anomaly detection"
        )

    def validate_metrics_with_human(self) -> bool:
        """Get human validation for prescribed metrics."""
        print("\n" + "="*60)
        print("STEP 8: METRICS VALIDATION (Human Review)")
        print("="*60)

        print("\n📋 Prescribed Metrics:")
        print(f"\nPrimary: {', '.join(self.metrics_prescription.primary_metrics)}")
        print(f"Secondary: {', '.join(self.metrics_prescription.secondary_metrics)}")

        print("\n📖 Definitions:")
        for metric, definition in self.metrics_prescription.metric_definitions.items():
            print(f"  • {metric}: {definition}")

        print("\n🎯 Suggested Thresholds:")
        for metric, threshold in self.metrics_prescription.thresholds.items():
            print(f"  • {metric}: {threshold:.2f}")

        print("\n" + "-"*60)
        sufficient = input("\nAre these metrics sufficient? (yes/no/modify): ").strip().lower()

        if sufficient in ['yes', 'y']:
            print("\n✓ Metrics validated and approved!")
            self.validation_results['metrics_approved'] = True
        elif sufficient == 'modify':
            additional = input("Additional metrics (comma-separated): ").strip()
            if additional:
                new_metrics = [m.strip() for m in additional.split(',')]
                self.metrics_prescription.secondary_metrics.extend(new_metrics)
                print(f"\n✓ Added: {', '.join(new_metrics)}")
                self.validation_results['metrics_approved'] = True
        else:
            self.validation_results['metrics_approved'] = False
            return False

        self.log_reasoning(
            stage="metrics_validation",
            action="human_validation",
            reasoning="Human reviewed and validated prescribed metrics",
            input_data=asdict(self.metrics_prescription),
            output_data=self.validation_results,
            confidence=1.0
        )

        self.current_stage = WorkflowStage.PROMPT_LOADING
        return True

    def load_or_generate_prompts(self) -> List[PromptTemplate]:
        """Load existing prompt templates or generate new ones dynamically."""
        print("\n" + "="*60)
        print("STEP 9: PROMPT TEMPLATE LOADING/GENERATION")
        print("="*60)

        print("\nWould you like to:")
        print("  1. Load existing prompt templates")
        print("  2. Generate new prompts dynamically")
        print("  3. Both (load existing + generate new)")

        choice = input("\nChoice (1/2/3): ").strip()
        templates = []

        if choice in ['1', '3']:
            templates = PromptTemplateLibrary.get_all_templates()
            print(f"✓ Loaded {len(templates)} existing templates")

        if choice in ['2', '3']:
            num_generate = int(input("How many new prompts to generate? (default: 5): ").strip() or "5")
            generated = self._generate_dynamic_prompts(num_generate)
            templates.extend(generated)
            print(f"✓ Generated {len(generated)} new templates")

        if not templates:
            templates = PromptTemplateLibrary.get_all_templates()

        print(f"\n✓ Total templates available: {len(templates)}")
        self.log_reasoning(
            stage="prompt_loading",
            action="load_or_generate_prompts",
            reasoning=f"Loaded/generated {len(templates)} prompt templates",
            input_data={"choice": choice},
            output_data={"num_templates": len(templates)},
            confidence=1.0
        )

        self.current_stage = WorkflowStage.OPTIMIZATION
        return templates

    def _generate_dynamic_prompts(self, num_prompts: int) -> List[PromptTemplate]:
        """Generate prompts dynamically using LLM."""
        generated_templates = []
        for i in range(num_prompts):
            prompt = f"""Generate a unique prompt template for: {self.use_case_context.use_case_name}

Objectives: {', '.join(self.use_case_context.key_objectives)}
Domain: {self.use_case_context.domain}

Provide just the prompt template text with {{data}} placeholder."""

            try:
                response, _ = self.llm_tester.test_prompt(prompt, use_cache=False)
                template = PromptTemplate(
                    name=f"dynamic_generated_{i+1}",
                    template=response.strip(),
                    style="dynamic"
                )
                generated_templates.append(template)
            except:
                pass
        return generated_templates

    def run_guided_workflow(self):
        """Run the complete guided workflow."""
        print("\n🚀 Starting USPM Agent Guided Workflow\n")

        try:
            # Steps 1-2
            has_keys, key_status = self.validate_api_keys()
            if not has_keys or not self.configure_llm_provider(key_status):
                print("\n❌ Cannot proceed without LLM provider.")
                return

            # Steps 3-10
            self.determine_use_case()
            self.gather_data_requirements()
            self.generate_synthetic_data()
            self.generate_ground_truth()
            self.prescribe_metrics_with_llm()

            if not self.validate_metrics_with_human():
                print("\nWorkflow paused. Please review metrics.")
                return

            templates = self.load_or_generate_prompts()

            print("\n" + "="*60)
            print("✓ WORKFLOW COMPLETE - READY FOR OPTIMIZATION")
            print("="*60)

            proceed = input("\nProceed with optimization? (yes/no): ").strip().lower()
            if proceed in ['yes', 'y']:
                self._run_optimization(templates)
            else:
                self._save_workflow_state()

            self.current_stage = WorkflowStage.COMPLETED

        except KeyboardInterrupt:
            print("\n\n⚠️  Workflow interrupted by user.")
            self._save_workflow_state()
        except Exception as e:
            print(f"\n❌ Error in workflow: {str(e)}")
            self._save_workflow_state()

    def _run_optimization(self, templates: List[PromptTemplate]):
        """Run the optimization with reasoning logging."""
        print("\n" + "="*60)
        print("STEP 10: OPTIMIZATION")
        print("="*60)

        from main import PromptTuningOrchestrator

        generations = int(input("  Max generations (default: 5): ").strip() or "5")
        population = int(input("  Population size (default: 15): ").strip() or "15")

        print(f"\n🔄 Running optimization with {generations} generations...")

        orchestrator = PromptTuningOrchestrator(max_generations=generations, population_size=population)
        orchestrator.llm_tester = self.llm_tester
        best = orchestrator.run_optimization()

        print(f"\n✓ Optimization complete!")
        print(f"  Best Prompt: {best.template.name}")
        print(f"  Score: {best.fitness:.3f}")

        self.log_reasoning(
            stage="optimization",
            action="run_optimization",
            reasoning=f"Ran {generations} generations with population {population}",
            input_data={"generations": generations, "population": population},
            output_data={"best_prompt": best.template.name, "score": best.fitness},
            confidence=best.fitness
        )

    def _save_workflow_state(self):
        """Save the current workflow state."""
        print("\n💾 Saving workflow state...")

        state = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode.value,
            "current_stage": self.current_stage.value if self.mode == AgentMode.GUIDED else None,
            "use_case": asdict(self.use_case_context) if self.use_case_context else None,
            "data_requirements": asdict(self.data_requirements) if self.data_requirements else None,
            "metrics": asdict(self.metrics_prescription) if self.metrics_prescription else None,
            "reasoning_logs": [asdict(log) for log in self.reasoning_logs]
        }

        os.makedirs("workflow_states", exist_ok=True)
        filename = f"workflow_states/uspm_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"✓ Workflow state saved to: {filename}")

    # ========================================================================
    # QUICK MODE METHODS - Natural language commands
    # ========================================================================

    def parse_intent(self, user_input: str) -> AgentTask:
        """Parse user input to understand intent and extract parameters."""
        user_input = user_input.lower().strip()

        patterns = {
            TaskType.GENERATE_DATA: [r"generate.*data", r"create.*data", r"make.*transactions"],
            TaskType.OPTIMIZE_PROMPTS: [r"optimize.*prompts?", r"run.*optimization", r"improve.*prompts?"],
            TaskType.TEST_PROMPT: [r"test.*prompt", r"try.*prompt", r"evaluate.*prompt"],
            TaskType.COMPARE_PROMPTS: [r"compare.*prompts?", r"which.*better"],
            TaskType.LIST_PROMPTS: [r"list.*prompts?", r"show.*prompts?", r"available.*prompts?"],
            TaskType.SHOW_RESULTS: [r"show.*results?", r"display.*results?"],
            TaskType.CONFIGURE: [r"configure.*provider", r"set.*provider", r"use.*provider"],
            TaskType.SWITCH_MODE: [r"guided mode", r"switch.*guided", r"full workflow"],
            TaskType.HELP: [r"help", r"what.*can.*you.*do", r"commands?"]
        }

        for task_type, task_patterns in patterns.items():
            for pattern in task_patterns:
                if re.search(pattern, user_input):
                    params = self._extract_parameters(user_input, task_type)
                    return AgentTask(task_type, params, confidence=0.9)

        return AgentTask(TaskType.UNKNOWN, {}, confidence=0.0)

    def _extract_parameters(self, user_input: str, task_type: TaskType) -> Dict[str, Any]:
        """Extract parameters from user input based on task type."""
        params = {}
        numbers = re.findall(r'\d+', user_input)

        if task_type == TaskType.GENERATE_DATA:
            params['num_files'] = int(numbers[0]) if numbers else 30
            params['transactions_per_file'] = int(numbers[1]) if len(numbers) > 1 else 100

        elif task_type == TaskType.OPTIMIZE_PROMPTS:
            params['generations'] = int(numbers[0]) if numbers else 5
            params['population_size'] = int(numbers[1]) if len(numbers) > 1 else 15

        elif task_type == TaskType.TEST_PROMPT:
            templates = PromptTemplateLibrary.get_all_templates()
            for template in templates:
                if template.name.lower() in user_input:
                    params['prompt_name'] = template.name
                    break

        elif task_type == TaskType.CONFIGURE:
            if 'openai' in user_input or 'gpt' in user_input:
                params['provider'] = 'openai'
            elif 'anthropic' in user_input or 'claude' in user_input:
                params['provider'] = 'anthropic'
            elif 'ollama' in user_input:
                params['provider'] = 'ollama'

        return params

    def execute_task(self, task: AgentTask) -> str:
        """Execute the identified task in quick mode."""
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
            return self._configure_provider_quick(task.parameters)
        elif task.task_type == TaskType.SWITCH_MODE:
            self.switch_mode("guided")
            return "\n✓ Switched to guided mode. Starting workflow..."
        elif task.task_type == TaskType.HELP:
            return self._show_help()
        else:
            return self._handle_unknown(task)

    def _generate_data(self, params: Dict) -> str:
        """Generate sample bank transaction data."""
        num_files = params.get('num_files', 30)
        transactions_per_file = params.get('transactions_per_file', 100)

        print(f"\n🔄 Generating {num_files} files with {transactions_per_file} transactions each...")
        generator = BankDataGenerator(num_files=num_files, transactions_per_file=transactions_per_file)
        generator.generate_all_files()
        stats = generator.get_ground_truth_stats()

        result = f"\n✓ Data generation complete!\n"
        result += f"Files created: {num_files}\n"
        result += f"Transactions per file: {transactions_per_file}\n"
        result += f"\nStatistics:\n"
        for key, value in stats.items():
            result += f"  {key}: {value}\n"

        self.context['last_task'] = 'generate_data'
        return result

    def _configure_provider_quick(self, params: Dict) -> str:
        """Configure LLM provider in quick mode."""
        provider = params.get('provider')
        if not provider:
            return "Please specify a provider: openai, anthropic, or ollama"

        _, key_status = self.validate_api_keys()
        if self.configure_llm_provider(key_status, provider=provider):
            return f"\n✓ {provider.capitalize()} configured successfully!\n"
        return f"\n✗ Failed to configure {provider}\n"

    def _optimize_prompts(self, params: Dict) -> str:
        """Run prompt optimization."""
        if not self.context['provider_configured']:
            return "\nPlease configure an LLM provider first. Example: 'configure openai provider'\n"

        from main import PromptTuningOrchestrator

        generations = params.get('generations', 5)
        population_size = params.get('population_size', 15)

        print(f"\n🔄 Running optimization for {generations} generations...")

        orchestrator = PromptTuningOrchestrator(max_generations=generations, population_size=population_size)
        orchestrator.llm_tester = self.llm_tester

        best = orchestrator.run_optimization()
        self.context['last_task'] = 'optimize'
        self.context['last_results'] = best

        return f"\n✓ Optimization complete! Best prompt: {best.template.name} (Score: {best.fitness:.3f})\n"

    def _test_prompt(self, params: Dict) -> str:
        """Test a specific prompt."""
        if not self.context['provider_configured']:
            return "\nPlease configure an LLM provider first.\n"

        prompt_name = params.get('prompt_name')
        if not prompt_name:
            return "\nPlease specify a prompt name. Use 'list prompts' to see available prompts.\n"

        from main import PromptTuningOrchestrator

        orchestrator = PromptTuningOrchestrator(data_dir=self.data_dir)
        orchestrator.llm_tester = self.llm_tester
        orchestrator.quick_test(prompt_name)

        return f"\n✓ Test complete for prompt: {prompt_name}\n"

    def _compare_prompts(self, params: Dict) -> str:
        """Compare multiple prompts."""
        if not self.context['provider_configured']:
            return "\nPlease configure an LLM provider first.\n"

        from main import PromptTuningOrchestrator

        orchestrator = PromptTuningOrchestrator(data_dir=self.data_dir)
        orchestrator.llm_tester = self.llm_tester
        orchestrator.quick_test(None)

        return "\n✓ Comparison complete!\n"

    def _list_prompts(self) -> str:
        """List all available prompt templates."""
        templates = PromptTemplateLibrary.get_all_templates()

        result = "\n📚 Available Prompt Templates:\n"
        result += "=" * 50 + "\n"
        for i, template in enumerate(templates, 1):
            result += f"{i}. {template.name} ({template.style})\n"

        return result

    def _show_results(self) -> str:
        """Show last results."""
        if not self.context.get('last_results'):
            return "\nNo results available yet. Run an optimization or test first.\n"

        result = "\n📊 Last Results:\n"
        result += "=" * 50 + "\n"
        last_results = self.context['last_results']
        result += f"Prompt: {last_results.template.name}\n"
        result += f"Score: {last_results.fitness:.3f}\n"

        return result

    def _show_help(self) -> str:
        """Show help information."""
        return """
╔══════════════════════════════════════════════════════════════╗
║              USPM Agent - Quick Mode Commands                ║
╚══════════════════════════════════════════════════════════════╝

Available Commands:
-------------------

1. Configure Provider:
   - "configure openai provider"
   - "use anthropic claude"
   - "setup ollama"

2. Generate Data:
   - "generate data"
   - "create 50 files with 200 transactions"

3. Optimize Prompts:
   - "optimize prompts"
   - "run optimization for 10 generations"

4. Test Prompts:
   - "test prompt concise_direct"
   - "compare prompts"

5. View Information:
   - "list prompts"
   - "show results"

6. Switch Mode:
   - "guided mode" (switch to full workflow)

7. Help:
   - "help"

8. Exit:
   - "exit" or "quit"

Note: Configure an LLM provider before running optimization or testing.
Set API keys: export OPENAI_API_KEY='your-key'
"""

    def _handle_unknown(self, task: AgentTask) -> str:
        """Handle unknown commands."""
        return ("\n❓ I didn't understand that command. Type 'help' to see available commands.\n"
                "Or type 'guided mode' to switch to the full guided workflow.\n")

    def run_interactive(self):
        """Run agent in interactive mode (quick mode)."""
        if self.mode == AgentMode.GUIDED:
            self.run_guided_workflow()
            return

        while True:
            try:
                user_input = input("USPM> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Goodbye!\n")
                    break

                # Parse and execute
                task = self.parse_intent(user_input)

                if task.confidence > 0.5:
                    print(f"[Understanding: {task.task_type.value}]")

                result = self.execute_task(task)
                print(result)

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted. Type 'exit' to quit.\n")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

    def show_reasoning_log(self):
        """Display the reasoning log for explainability."""
        print("\n" + "="*60)
        print("REASONING & EXPLAINABILITY LOG")
        print("="*60)

        for i, log in enumerate(self.reasoning_logs, 1):
            print(f"\n[{i}] {log.stage.upper()} - {log.action}")
            print(f"    Time: {log.timestamp}")
            print(f"    Reasoning: {log.reasoning}")
            print(f"    Confidence: {log.confidence:.0%}")

            if log.bias_check and log.bias_check.get('severity') != 'low':
                print(f"    ⚠️  Bias Detected: {log.bias_check['severity']}")

        print("\n" + "="*60)


if __name__ == "__main__":
    import sys

    mode = "guided"
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        mode = "quick"

    agent = USPMAgent(mode=mode)
    agent.run_interactive()
