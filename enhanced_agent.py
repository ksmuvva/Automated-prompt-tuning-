"""
Enhanced NLP Agent with reasoning, explainability, and bias detection.
Provides guided workflow for use case determination, data generation, metrics prescription,
and prompt optimization with full transparency.
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


class WorkflowStage(Enum):
    """Stages in the enhanced agent workflow."""
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


@dataclass
class UseCaseContext:
    """Context information about the determined use case."""
    use_case_name: str
    use_case_type: str  # e.g., "fraud_detection", "sentiment_analysis", etc.
    domain: str
    description: str
    key_objectives: List[str]
    data_characteristics: Dict[str, Any]
    reasoning: str  # How the use case was determined
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


class EnhancedAgent:
    """
    Enhanced NLP agent with reasoning, explainability, and bias detection.
    Guides users through the entire workflow from use case to optimization.
    """

    def __init__(self):
        """Initialize the enhanced agent."""
        self.current_stage = WorkflowStage.INITIALIZATION
        self.reasoning_logs: List[ReasoningLog] = []

        # Context tracking
        self.use_case_context: Optional[UseCaseContext] = None
        self.data_requirements: Optional[DataRequirements] = None
        self.metrics_prescription: Optional[MetricsPrescription] = None

        # LLM configuration
        self.llm_provider = None
        self.llm_tester = None
        self.provider_type = None

        # Results tracking
        self.generated_data_path = None
        self.ground_truth = None
        self.validation_results = {}

        print("╔══════════════════════════════════════════════════════════════╗")
        print("║      Enhanced Prompt Tuning Agent with Reasoning            ║")
        print("║      Version 2.0 - With Explainability & Bias Detection     ║")
        print("╚══════════════════════════════════════════════════════════════╝\n")

    def log_reasoning(self, stage: str, action: str, reasoning: str,
                     input_data: Dict, output_data: Dict,
                     confidence: float = 1.0, bias_check: Optional[Dict] = None):
        """Log reasoning for explainability."""
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
        """
        Check for potential biases in text/decisions.
        Returns bias analysis results.
        """
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
        """
        Validate available API keys for LLM providers.
        Returns (has_any_key, key_status_dict)
        """
        print("\n" + "="*60)
        print("STEP 1: API KEY VALIDATION")
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

    def configure_llm_provider(self, key_status: Dict[str, bool]) -> bool:
        """Configure LLM provider based on available keys."""
        print("\n" + "="*60)
        print("STEP 2: LLM PROVIDER CONFIGURATION")
        print("="*60)

        available_providers = [k for k, v in key_status.items() if v]

        if not available_providers:
            return False

        print(f"\nAvailable providers: {', '.join(available_providers)}")

        # Auto-select if only one available
        if len(available_providers) == 1:
            provider = available_providers[0]
            print(f"\nAuto-selecting: {provider}")
        else:
            while True:
                provider = input(f"\nSelect provider ({'/'.join(available_providers)}): ").strip().lower()
                if provider in available_providers:
                    break
                print("Invalid provider. Please choose from available providers.")

        # Configure provider
        try:
            llm_config = {}

            if provider == "openai":
                model = input("Enter model (default: gpt-4): ").strip() or "gpt-4"
                llm_config = {"model": model, "api_key": os.getenv("OPENAI_API_KEY")}
            elif provider == "anthropic":
                model = input("Enter model (default: claude-3-5-sonnet-20241022): ").strip() or "claude-3-5-sonnet-20241022"
                llm_config = {"model": model, "api_key": os.getenv("ANTHROPIC_API_KEY")}
            elif provider == "ollama":
                model = input("Enter model (default: llama2): ").strip() or "llama2"
                llm_config = {"model": model}

            print(f"\n🔧 Initializing {provider} with model {llm_config.get('model', 'default')}...")

            self.llm_provider = LLMFactory.create(provider, **llm_config)
            self.llm_tester = LLMTester(self.llm_provider)
            self.provider_type = provider

            print(f"✓ {provider.capitalize()} configured successfully!\n")

            self.log_reasoning(
                stage="provider_configuration",
                action="configure_llm",
                reasoning=f"Selected {provider} based on available keys and user preference",
                input_data={"available_providers": available_providers},
                output_data={"provider": provider, "config": llm_config},
                confidence=1.0
            )

            self.current_stage = WorkflowStage.USE_CASE_DETERMINATION
            return True

        except Exception as e:
            print(f"\n✗ Error configuring {provider}: {str(e)}")
            return False

    def determine_use_case(self) -> UseCaseContext:
        """
        Determine use case through user interaction and LLM reasoning.
        """
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

                # Parse LLM response
                # Extract JSON from response
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
                    # Fallback to default
                    print("\n⚠️  Could not parse LLM response. Using default bank transaction use case.")
                    use_case_context = self._get_default_use_case()

            except Exception as e:
                print(f"\n⚠️  Error with LLM reasoning: {str(e)}")
                print("Using default bank transaction use case.")
                use_case_context = self._get_default_use_case()

        # Bias check on use case determination
        bias_check = self.check_bias(
            use_case_context.reasoning,
            "use_case_determination"
        )

        if bias_check["severity"] != "low":
            print(f"\n⚠️  Bias detected in use case reasoning (severity: {bias_check['severity']})")
            for bias_type, issues in bias_check.items():
                if issues and bias_type != "severity":
                    print(f"   {bias_type}: {issues}")

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
        """
        Interactively gather data generation requirements from user.
        """
        print("\n" + "="*60)
        print("STEP 4: DATA REQUIREMENTS GATHERING")
        print("="*60)

        print(f"\nBased on use case: {self.use_case_context.use_case_name}")
        print("Let's determine the data requirements for synthetic data generation.\n")

        # Ask for data specifications
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
        print("  Any special conditions? (e.g., 'high_value>250', 'seasonal_patterns')")
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
        """
        Generate synthetic data based on gathered requirements.
        """
        print("\n" + "="*60)
        print("STEP 5: SYNTHETIC DATA GENERATION")
        print("="*60)

        print(f"\n🔄 Generating {self.data_requirements.num_files} files with {self.data_requirements.records_per_file} records each...")

        # Use existing BankDataGenerator
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
        """
        Generate ground truth labels and statistics.
        """
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
        """
        Use LLM to prescribe appropriate metrics for the use case.
        """
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
}}

Consider:
1. The use case objectives
2. Data characteristics
3. Industry best practices
4. Balance between different aspects (precision, recall, etc.)"""

        try:
            response, _ = self.llm_tester.test_prompt(prompt, use_cache=False)

            # Parse response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                metrics_data = json.loads(json_match.group())
                prescription = MetricsPrescription(**metrics_data)

                print("\n✓ Metrics prescribed by LLM!")
                print(f"\n📊 Primary Metrics: {', '.join(prescription.primary_metrics)}")
                print(f"   Secondary Metrics: {', '.join(prescription.secondary_metrics)}")
                print(f"\n💭 LLM Reasoning:\n{prescription.llm_explanation}")

            else:
                print("\n⚠️  Could not parse LLM response. Using default metrics.")
                prescription = self._get_default_metrics()

        except Exception as e:
            print(f"\n⚠️  Error with LLM: {str(e)}")
            print("Using default metrics.")
            prescription = self._get_default_metrics()

        # Bias check
        bias_check = self.check_bias(
            prescription.llm_explanation,
            "metrics_prescription"
        )

        if bias_check["severity"] != "low":
            print(f"\n⚠️  Bias detected in metrics reasoning (severity: {bias_check['severity']})")

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
        """Return default metrics for bank transaction analysis."""
        return MetricsPrescription(
            primary_metrics=["high_value_f1", "anomaly_f1", "composite_score"],
            secondary_metrics=["precision", "recall", "response_time"],
            metric_definitions={
                "high_value_f1": "F1 score for detecting transactions above 250 GBP",
                "anomaly_f1": "F1 score for detecting anomalous transactions",
                "composite_score": "Weighted combination of all metrics"
            },
            thresholds={
                "high_value_f1": 0.85,
                "anomaly_f1": 0.80,
                "composite_score": 0.75
            },
            reasoning="Standard metrics for fraud detection tasks",
            llm_explanation="These metrics balance precision and recall for both high-value and anomaly detection"
        )

    def validate_metrics_with_human(self) -> bool:
        """
        Get human validation for prescribed metrics.
        """
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
            print("\nWhich metrics would you like to add? (comma-separated)")
            additional = input("Additional metrics: ").strip()
            if additional:
                new_metrics = [m.strip() for m in additional.split(',')]
                self.metrics_prescription.secondary_metrics.extend(new_metrics)
                print(f"\n✓ Added: {', '.join(new_metrics)}")
                self.validation_results['metrics_modified'] = True
                self.validation_results['metrics_approved'] = True
        else:
            print("\n❌ Metrics not approved. Please work with the LLM to adjust.")
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
        """
        Load existing prompt templates or generate new ones dynamically.
        """
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
            print("\n📚 Loading existing prompt templates...")
            templates = PromptTemplateLibrary.get_all_templates()
            print(f"✓ Loaded {len(templates)} existing templates")

        if choice in ['2', '3']:
            print("\n🎨 Generating new prompts dynamically...")
            num_generate = int(input("How many new prompts to generate? (default: 5): ").strip() or "5")

            generated = self._generate_dynamic_prompts(num_generate)
            templates.extend(generated)
            print(f"✓ Generated {len(generated)} new templates")

        if not templates:
            print("\n⚠️  No templates loaded. Using defaults.")
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
        print("\n🤔 Using LLM to generate custom prompts...")

        generated_templates = []

        for i in range(num_prompts):
            prompt = f"""Generate a unique prompt template for: {self.use_case_context.use_case_name}

Objectives: {', '.join(self.use_case_context.key_objectives)}
Domain: {self.use_case_context.domain}

The prompt should:
1. Be clear and specific
2. Request JSON output with {', '.join(self.metrics_prescription.primary_metrics)}
3. Include relevant instructions for {self.use_case_context.use_case_type}
4. Variation {i+1}: Use a different style/approach

Provide just the prompt template text with {{data}} placeholder."""

            try:
                response, _ = self.llm_tester.test_prompt(prompt, use_cache=False)

                template = PromptTemplate(
                    name=f"dynamic_generated_{i+1}",
                    template=response.strip(),
                    style="dynamic"
                )
                generated_templates.append(template)
                print(f"  Generated template {i+1}/{num_prompts}")

            except Exception as e:
                print(f"  ⚠️  Error generating template {i+1}: {str(e)}")

        return generated_templates

    def run_guided_workflow(self):
        """
        Run the complete guided workflow.
        """
        print("\n🚀 Starting Enhanced Agent Guided Workflow\n")

        try:
            # Step 1-2: Validate API keys and configure LLM
            has_keys, key_status = self.validate_api_keys()
            if not has_keys:
                print("\n❌ Cannot proceed without LLM provider. Exiting.")
                return

            if not self.configure_llm_provider(key_status):
                print("\n❌ Failed to configure LLM provider. Exiting.")
                return

            # Step 3: Determine use case
            use_case = self.determine_use_case()

            # Step 4: Gather data requirements
            data_reqs = self.gather_data_requirements()

            # Step 5: Generate synthetic data
            data_path = self.generate_synthetic_data()

            # Step 6: Generate ground truth
            ground_truth = self.generate_ground_truth()

            # Step 7: Prescribe metrics with LLM
            metrics = self.prescribe_metrics_with_llm()

            # Step 8: Human validation
            if not self.validate_metrics_with_human():
                print("\nWorkflow paused. Please review metrics.")
                return

            # Step 9: Load/generate prompts
            templates = self.load_or_generate_prompts()

            # Step 10: Ready for optimization
            print("\n" + "="*60)
            print("✓ WORKFLOW COMPLETE - READY FOR OPTIMIZATION")
            print("="*60)

            print(f"\n📊 Summary:")
            print(f"  Use Case: {use_case.use_case_name}")
            print(f"  Data: {data_reqs.num_files} files x {data_reqs.records_per_file} records")
            print(f"  Metrics: {len(metrics.primary_metrics)} primary + {len(metrics.secondary_metrics)} secondary")
            print(f"  Prompts: {len(templates)} templates")

            proceed = input("\nProceed with optimization? (yes/no): ").strip().lower()

            if proceed in ['yes', 'y']:
                self._run_optimization(templates)
            else:
                print("\n✓ Workflow data saved. You can run optimization later.")
                self._save_workflow_state()

            self.current_stage = WorkflowStage.COMPLETED

        except KeyboardInterrupt:
            print("\n\n⚠️  Workflow interrupted by user.")
            self._save_workflow_state()
        except Exception as e:
            print(f"\n❌ Error in workflow: {str(e)}")
            import traceback
            traceback.print_exc()
            self._save_workflow_state()

    def _run_optimization(self, templates: List[PromptTemplate]):
        """Run the optimization with reasoning logging."""
        print("\n" + "="*60)
        print("STEP 10: OPTIMIZATION")
        print("="*60)

        from main import PromptTuningOrchestrator

        print("\nConfiguration:")
        generations = int(input("  Max generations (default: 5): ").strip() or "5")
        population = int(input("  Population size (default: 15): ").strip() or "15")

        print(f"\n🔄 Running optimization with {generations} generations...")

        orchestrator = PromptTuningOrchestrator(
            max_generations=generations,
            population_size=population
        )
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
            "current_stage": self.current_stage.value,
            "use_case": asdict(self.use_case_context) if self.use_case_context else None,
            "data_requirements": asdict(self.data_requirements) if self.data_requirements else None,
            "metrics": asdict(self.metrics_prescription) if self.metrics_prescription else None,
            "reasoning_logs": [asdict(log) for log in self.reasoning_logs]
        }

        os.makedirs("workflow_states", exist_ok=True)
        filename = f"workflow_states/workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"✓ Workflow state saved to: {filename}")

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
    agent = EnhancedAgent()
    agent.run_guided_workflow()
