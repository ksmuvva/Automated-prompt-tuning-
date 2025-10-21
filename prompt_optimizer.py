"""
Automated prompt optimization system using metrics-driven refinement.
Implements genetic algorithm and feedback-based optimization strategies.
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy
from prompt_templates import PromptTemplate, PromptTemplateLibrary
from metrics_evaluator import PerformanceMetrics


@dataclass
class PromptCandidate:
    """Represents a prompt candidate with its performance."""
    template: PromptTemplate
    metrics: Optional[PerformanceMetrics] = None
    generation: int = 0

    @property
    def fitness(self) -> float:
        """Get fitness score (composite score from metrics)."""
        return self.metrics.composite_score if self.metrics else 0.0


class PromptOptimizer:
    """Optimize prompts using evolutionary and feedback-based strategies."""

    def __init__(self, population_size: int = 20, mutation_rate: float = 0.3,
                 elite_size: int = 5):
        """
        Initialize prompt optimizer.

        Args:
            population_size: Number of prompts in each generation
            mutation_rate: Probability of mutation (0-1)
            elite_size: Number of top performers to keep
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generation = 0
        self.best_prompts_history: List[PromptCandidate] = []

    def initialize_population(self) -> List[PromptCandidate]:
        """Create initial population from template library."""
        templates = PromptTemplateLibrary.get_all_templates()

        # Start with all base templates
        population = [PromptCandidate(t, generation=0) for t in templates]

        # Generate variations to reach population size
        while len(population) < self.population_size:
            base_template = random.choice(templates)
            mutated = self._mutate_template(base_template)
            population.append(PromptCandidate(mutated, generation=0))

        return population[:self.population_size]

    def _mutate_template(self, template: PromptTemplate) -> PromptTemplate:
        """Apply mutations to a template."""
        mutations = [
            self._add_emphasis,
            self._add_structure,
            self._add_examples,
            self._simplify,
            self._add_constraints,
            self._modify_tone,
            self._add_thinking_steps
        ]

        mutation = random.choice(mutations)
        return mutation(template)

    def _add_emphasis(self, template: PromptTemplate) -> PromptTemplate:
        """Add emphasis to key parts."""
        new_text = template.template
        emphasis_words = {
            "identify": "CAREFULLY IDENTIFY",
            "analyze": "THOROUGHLY ANALYZE",
            "detect": "ACCURATELY DETECT",
            "find": "PRECISELY FIND",
            "all": "ALL",
            "any": "ANY"
        }

        for old, new in emphasis_words.items():
            new_text = new_text.replace(old, new)

        return PromptTemplate(
            f"{template.name}_emphasized_{self.generation}",
            new_text,
            template.style
        )

    def _add_structure(self, template: PromptTemplate) -> PromptTemplate:
        """Add more structure to the prompt."""
        structured_prefix = """TASK STRUCTURE:
Step 1: Parse all transaction data
Step 2: Identify high-value transactions (>250 GBP)
Step 3: Analyze for anomalies
Step 4: Format output as JSON

DETAILED INSTRUCTIONS:
"""
        new_text = structured_prefix + template.template
        return PromptTemplate(
            f"{template.name}_structured_{self.generation}",
            new_text,
            "structured"
        )

    def _add_examples(self, template: PromptTemplate) -> PromptTemplate:
        """Add few-shot examples."""
        examples = """
EXAMPLES:

Example 1 - High value, not anomaly:
Input: TXN001, 2024-01-01 10:00:00, 500.00, Electronics, BestBuy, Large purchase
Output: {{"above_250": ["TXN001"], "anomalies": []}}

Example 2 - Low value, anomaly:
Input: TXN002, 2024-01-01 03:00:00, 50.00, Suspicious Wire Transfer, UnknownMerchant, Late night
Output: {{"above_250": [], "anomalies": [{{"transaction_id": "TXN002", "reason": "Suspicious category and timing"}}]}}

Example 3 - High value anomaly:
Input: TXN003, 2024-01-01 14:00:00, 2000.00, Foreign Transaction, HighRiskVendor, Unusual location
Output: {{"above_250": ["TXN003"], "anomalies": [{{"transaction_id": "TXN003", "reason": "High value + suspicious merchant + foreign"}}]}}

NOW ANALYZE THE FOLLOWING:
"""
        new_text = examples + template.template
        return PromptTemplate(
            f"{template.name}_examples_{self.generation}",
            new_text,
            "few-shot"
        )

    def _simplify(self, template: PromptTemplate) -> PromptTemplate:
        """Create simpler version."""
        simple = f"""Analyze bank transactions below.

Task:
1. Find all transactions > 250 GBP
2. Find all anomalies

Data:
{{data}}

Output JSON: {{"above_250": [...], "anomalies": [...]}}"""

        return PromptTemplate(
            f"{template.name}_simple_{self.generation}",
            simple,
            "simplified"
        )

    def _add_constraints(self, template: PromptTemplate) -> PromptTemplate:
        """Add output constraints."""
        constraints = """\n\nOUTPUT REQUIREMENTS:
- MUST be valid JSON only
- NO additional commentary
- Include transaction_id as string
- For anomalies, always include "reason" field
- Be comprehensive but avoid false positives"""

        new_text = template.template + constraints
        return PromptTemplate(
            f"{template.name}_constrained_{self.generation}",
            new_text,
            template.style
        )

    def _modify_tone(self, template: PromptTemplate) -> PromptTemplate:
        """Modify the tone of the prompt."""
        tones = [
            ("You are an expert fraud analyst.", "professional"),
            ("Act as a senior bank security officer.", "authoritative"),
            ("Please help analyze the following transactions.", "friendly"),
            ("Your mission is to detect fraud and protect customers.", "mission-driven")
        ]

        tone_prefix, tone_type = random.choice(tones)
        new_text = f"{tone_prefix}\n\n{template.template}"

        return PromptTemplate(
            f"{template.name}_{tone_type}_{self.generation}",
            new_text,
            template.style
        )

    def _add_thinking_steps(self, template: PromptTemplate) -> PromptTemplate:
        """Add chain-of-thought prompting."""
        thinking = """Think through this step-by-step:
1. What patterns do you notice in the data?
2. Which transactions exceed 250 GBP?
3. What makes a transaction anomalous?
4. Apply your reasoning to identify issues.

"""
        new_text = thinking + template.template
        return PromptTemplate(
            f"{template.name}_cot_{self.generation}",
            new_text,
            "chain-of-thought"
        )

    def _crossover(self, parent1: PromptTemplate, parent2: PromptTemplate) -> PromptTemplate:
        """Combine two templates."""
        # Simple crossover: take prefix from one, suffix from another
        text1_parts = parent1.template.split('\n\n')
        text2_parts = parent2.template.split('\n\n')

        # Randomly combine parts
        if len(text1_parts) > 1 and len(text2_parts) > 1:
            split_point = random.randint(1, min(len(text1_parts), len(text2_parts)) - 1)
            new_text = '\n\n'.join(text1_parts[:split_point] + text2_parts[split_point:])
        else:
            new_text = parent1.template if random.random() < 0.5 else parent2.template

        return PromptTemplate(
            f"crossover_{parent1.name}_{parent2.name}_{self.generation}",
            new_text,
            "hybrid"
        )

    def select_parents(self, population: List[PromptCandidate], k: int = 3) -> PromptCandidate:
        """Tournament selection."""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def evolve_generation(self, population: List[PromptCandidate]) -> List[PromptCandidate]:
        """Create next generation of prompts."""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Keep elite
        new_population = population[:self.elite_size]

        # Generate offspring
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # 70% crossover
                parent1 = self.select_parents(population)
                parent2 = self.select_parents(population)
                child_template = self._crossover(parent1.template, parent2.template)
            else:  # 30% mutation only
                parent = self.select_parents(population)
                child_template = copy.deepcopy(parent.template)

            # Apply mutation
            if random.random() < self.mutation_rate:
                child_template = self._mutate_template(child_template)

            new_population.append(
                PromptCandidate(child_template, generation=self.generation + 1)
            )

        self.generation += 1
        return new_population[:self.population_size]

    def get_best_candidate(self, population: List[PromptCandidate]) -> PromptCandidate:
        """Get the best performing candidate."""
        return max(population, key=lambda x: x.fitness)

    def analyze_performance_gaps(self, candidate: PromptCandidate) -> Dict[str, str]:
        """
        Analyze what aspects of performance need improvement.

        Returns:
            Dictionary of improvement suggestions
        """
        if not candidate.metrics:
            return {}

        suggestions = {}
        metrics = candidate.metrics

        # Check high-value detection
        if metrics.high_value_recall < 0.7:
            suggestions['high_value_recall'] = "Low recall on high-value transactions. Consider emphasizing amount thresholds."

        if metrics.high_value_precision < 0.7:
            suggestions['high_value_precision'] = "Too many false positives on high-value. Add stricter criteria."

        # Check anomaly detection
        if metrics.anomaly_recall < 0.6:
            suggestions['anomaly_recall'] = "Missing anomalies. Add more anomaly indicators and examples."

        if metrics.anomaly_precision < 0.6:
            suggestions['anomaly_precision'] = "Too many false anomaly flags. Be more specific about what constitutes an anomaly."

        # Check format
        if metrics.response_format_score < 0.9:
            suggestions['format'] = "Response format issues. Add stricter JSON output requirements."

        if not metrics.response_parseable:
            suggestions['parseable'] = "Response not parseable. Emphasize JSON-only output."

        # Check speed
        if metrics.response_time_seconds > 5:
            suggestions['speed'] = "Slow response. Consider simplifying the prompt."

        return suggestions

    def create_targeted_improvement(self, candidate: PromptCandidate) -> PromptTemplate:
        """Create an improved version based on performance gaps."""
        gaps = self.analyze_performance_gaps(candidate)
        improved_template = copy.deepcopy(candidate.template)

        # Apply targeted improvements
        if 'high_value_recall' in gaps:
            improved_template.template = improved_template.template.replace(
                "above 250 GBP",
                "above 250 GBP (IMPORTANT: Be thorough, check every transaction)"
            )

        if 'anomaly_recall' in gaps:
            anomaly_indicators = """\n\nANOMALY INDICATORS TO CHECK:
- Suspicious categories (wire transfers, unknown merchants)
- Unusual timing (late night, weekends for business accounts)
- Duplicate patterns
- Round numbers suggesting manual entry
- Geographic anomalies
- Velocity issues (too frequent)"""
            improved_template.template += anomaly_indicators

        if 'format' in gaps or 'parseable' in gaps:
            format_emphasis = """\n\nCRITICAL: Output ONLY valid JSON. No explanations, no markdown, just JSON."""
            improved_template.template += format_emphasis

        improved_template.name = f"{candidate.template.name}_improved_{self.generation}"

        return improved_template


class AdaptiveOptimizer:
    """Adaptive optimization using feedback loops."""

    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.performance_history: List[Dict] = []

    def suggest_next_templates(self, current_best: PromptCandidate,
                               population: List[PromptCandidate]) -> List[PromptTemplate]:
        """
        Suggest next templates to try based on current performance.

        Returns:
            List of suggested templates
        """
        suggestions = []
        optimizer = PromptOptimizer()

        # Always try targeted improvement
        improved = optimizer.create_targeted_improvement(current_best)
        suggestions.append(improved)

        # Try variations of top performers
        top_5 = sorted(population, key=lambda x: x.fitness, reverse=True)[:5]
        for candidate in top_5[:3]:
            mutated = optimizer._mutate_template(candidate.template)
            suggestions.append(mutated)

        return suggestions

    def should_continue(self, generation: int, max_generations: int,
                       improvement_threshold: float = 0.01) -> bool:
        """
        Determine if optimization should continue.

        Args:
            generation: Current generation number
            max_generations: Maximum generations allowed
            improvement_threshold: Minimum improvement required to continue

        Returns:
            True if should continue, False otherwise
        """
        if generation >= max_generations:
            return False

        # Check if we're still improving
        if len(self.performance_history) >= 3:
            recent = self.performance_history[-3:]
            scores = [h['best_score'] for h in recent]
            improvement = max(scores) - min(scores)

            if improvement < improvement_threshold:
                return False  # Plateaued

        return True

    def record_generation(self, generation: int, best_score: float, avg_score: float):
        """Record performance of a generation."""
        self.performance_history.append({
            'generation': generation,
            'best_score': best_score,
            'avg_score': avg_score
        })


if __name__ == "__main__":
    # Example usage
    optimizer = PromptOptimizer(population_size=10, mutation_rate=0.3)

    # Initialize population
    population = optimizer.initialize_population()
    print(f"Initialized population with {len(population)} prompts")

    # Show some examples
    for i, candidate in enumerate(population[:3]):
        print(f"\nCandidate {i+1}: {candidate.template.name}")
        print(f"Style: {candidate.template.style}")
