# Automated Prompt Tuning System for Bank Transaction Analysis

An intelligent system that automatically tests, evaluates, and optimizes LLM prompts for analyzing bank transactions, detecting high-value transactions (>250 GBP), and identifying anomalies.

## Features

- **Automated Prompt Testing**: Tests multiple prompt formats and styles automatically
- **Metrics-Driven Optimization**: Uses precision, recall, F1-score, and composite metrics
- **Genetic Algorithm**: Evolves prompts across generations for continuous improvement
- **Multiple LLM Support**: Works with OpenAI, Anthropic, Ollama, and mock providers
- **Comprehensive Evaluation**: Tracks 10+ performance metrics per prompt
- **Smart Refinement**: Identifies performance gaps and creates targeted improvements

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Prompt Tuning System                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   Data     │───▶│   Prompt     │───▶│     LLM       │  │
│  │ Generator  │    │  Templates   │    │  Interface    │  │
│  └────────────┘    └──────────────┘    └───────────────┘  │
│                            │                     │          │
│                            ▼                     ▼          │
│                    ┌──────────────┐    ┌───────────────┐  │
│                    │   Prompt     │◀───│   Metrics     │  │
│                    │  Optimizer   │    │  Evaluator    │  │
│                    └──────────────┘    └───────────────┘  │
│                            │                               │
│                            ▼                               │
│                    ┌──────────────┐                        │
│                    │   Results    │                        │
│                    │   & Report   │                        │
│                    └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Automated-prompt-tuning-

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# Generate 30 CSV files with bank transaction data
python main.py --mode generate
```

This creates sample data in the `bank_data/` directory with:
- Normal transactions
- High-value transactions (>250 GBP)
- Anomalous transactions
- Ground truth labels for evaluation

### 3. Run Optimization (Mock Mode)

```bash
# Test with mock LLM (no API key needed)
python main.py --mode optimize --provider mock --generations 5
```

### 4. Run with Real LLM

#### OpenAI:
```bash
export OPENAI_API_KEY="your-api-key"
python main.py --mode optimize --provider openai --model gpt-4 --generations 5
```

#### Anthropic Claude:
```bash
export ANTHROPIC_API_KEY="your-api-key"
python main.py --mode optimize --provider anthropic --model claude-3-5-sonnet-20241022 --generations 5
```

#### Ollama (Local):
```bash
# Make sure Ollama is running locally
python main.py --mode optimize --provider ollama --model llama2 --generations 5
```

## Usage Examples

### Quick Test a Specific Prompt

```bash
python main.py --mode quick-test --prompt-name concise_direct
```

### Full Optimization with Custom Parameters

```bash
python main.py --mode optimize \
  --provider openai \
  --model gpt-4 \
  --generations 10 \
  --population 20 \
  --test-files 10
```

### Using the API Programmatically

```python
from main import PromptTuningOrchestrator

# Initialize orchestrator
orchestrator = PromptTuningOrchestrator(
    llm_provider="openai",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_generations=5,
    population_size=15
)

# Run optimization
best_prompt = orchestrator.run_optimization()

print(f"Best prompt: {best_prompt.template.name}")
print(f"Score: {best_prompt.fitness:.3f}")
```

## How It Works

### 1. Initial Prompt Population

The system starts with 11 base prompt templates in different styles:
- **Concise**: Direct, minimal instructions
- **Detailed**: Comprehensive, step-by-step guidance
- **Structured**: XML/table formatted
- **Conversational**: Friendly, natural language
- **Few-shot**: With examples
- **Role-based**: Assigns specific roles
- **Chain-of-thought**: Encourages reasoning

### 2. Prompt Mutations

Each generation creates variations through:
- **Emphasis**: Adding emphasis words (CAREFULLY, THOROUGHLY)
- **Structure**: Adding step-by-step breakdowns
- **Examples**: Including few-shot examples
- **Constraints**: Stricter output requirements
- **Tone**: Modifying communication style
- **Chain-of-thought**: Adding reasoning steps

### 3. Evaluation Metrics

Each prompt is evaluated on:

**High-Value Detection (>250 GBP)**
- Precision: Accuracy of high-value identifications
- Recall: Completeness of high-value detection
- F1 Score: Harmonic mean of precision and recall
- Accuracy: Overall correctness

**Anomaly Detection**
- Precision: Accuracy of anomaly flags
- Recall: Coverage of actual anomalies
- F1 Score: Balance of precision and recall
- Accuracy: Overall correctness

**Response Quality**
- Parseability: Can the response be parsed as JSON?
- Format Score: How well does it match expected format?
- Response Time: Speed of LLM response

**Composite Score** (Weighted combination):
- 30% High-value F1 score
- 50% Anomaly F1 score
- 20% Response quality

### 4. Genetic Evolution

- **Selection**: Tournament selection favors high performers
- **Crossover**: Combines successful prompt patterns
- **Mutation**: Introduces variations for exploration
- **Elitism**: Preserves top performers each generation

### 5. Adaptive Refinement

The system identifies specific weaknesses:
- Low recall → Add emphasis on thoroughness
- Low precision → Add stricter criteria
- Format issues → Add JSON constraints
- Slow responses → Simplify prompts

## Project Structure

```
Automated-prompt-tuning-/
├── main.py                    # Main orchestrator
├── data_generator.py          # Generate sample CSV data
├── prompt_templates.py        # Prompt template library
├── llm_interface.py           # LLM provider integrations
├── metrics_evaluator.py       # Performance metrics system
├── prompt_optimizer.py        # Genetic algorithm optimizer
├── config.json               # Configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── bank_data/               # Generated CSV files (30 files)
│   ├── bank_account_00.csv
│   ├── bank_account_01.csv
│   └── ...
│
└── results/                 # Output results
    ├── best_prompt_YYYYMMDD_HHMMSS.txt
    ├── metrics_YYYYMMDD_HHMMSS.json
    └── leaderboard_YYYYMMDD_HHMMSS.csv
```

## Output Files

### Best Prompt File
Contains the winning prompt template with its score and generation number.

### Metrics File (JSON)
Detailed metrics for all prompts including:
- Composite scores
- F1 scores for high-value and anomaly detection
- Response times
- All evaluation metrics

### Leaderboard File (CSV)
Ranked list of all tested prompts with their performance metrics.

## Configuration

Edit `config.json` to customize:

```json
{
  "optimization": {
    "max_generations": 5,      // Number of evolution cycles
    "population_size": 15,     // Prompts per generation
    "mutation_rate": 0.3,      // Probability of mutation
    "num_test_files": 5        // CSV files to test on
  },
  "metrics": {
    "weight_high_value": 0.3,    // Weight for high-value detection
    "weight_anomaly": 0.5,       // Weight for anomaly detection
    "weight_response_quality": 0.2  // Weight for response quality
  }
}
```

## Performance Tips

1. **Start with mock mode** to test the system without API costs
2. **Use fewer test files** initially (2-3) for faster iterations
3. **Increase population size** (20-30) for better exploration
4. **Run more generations** (8-10) for optimal results
5. **Adjust weights** based on your priorities (high-value vs anomaly detection)

## Advanced Usage

### Creating Custom Prompts

```python
from prompt_templates import PromptTemplate, PromptTemplateLibrary

# Create custom template
custom_template = PromptTemplate(
    name="my_custom_prompt",
    template="""Your custom prompt here with {data} placeholder""",
    style="custom"
)

# Add to library
PromptTemplateLibrary.create_custom_template(
    "my_custom_prompt",
    custom_template.template,
    "custom"
)
```

### Custom Metrics Weights

```python
from metrics_evaluator import MetricsEvaluator

# Create evaluator with custom weights
evaluator = MetricsEvaluator(
    weight_high_value=0.4,     # 40% weight on high-value detection
    weight_anomaly=0.4,        # 40% weight on anomaly detection
    weight_response_quality=0.2 # 20% weight on response quality
)
```

## Troubleshooting

**Issue**: No CSV files found
- **Solution**: Run `python main.py --mode generate` first

**Issue**: API key errors
- **Solution**: Set environment variables: `export OPENAI_API_KEY="your-key"`

**Issue**: Ollama connection errors
- **Solution**: Ensure Ollama is running: `ollama serve`

**Issue**: Slow performance
- **Solution**: Reduce `--test-files` or `--population` parameters

## Metrics Interpretation

- **Composite Score > 0.8**: Excellent performance
- **Composite Score 0.6-0.8**: Good performance
- **Composite Score < 0.6**: Needs improvement

- **F1 Score > 0.9**: High accuracy and coverage
- **F1 Score 0.7-0.9**: Acceptable performance
- **F1 Score < 0.7**: Poor performance

## Contributing

Contributions are welcome! Areas for improvement:
- Additional prompt mutation strategies
- New LLM provider integrations
- Enhanced visualization of results
- Multi-objective optimization
- Real-time monitoring dashboard

## License

MIT License - See LICENSE file for details

## Citation

If you use this system in your research or work, please cite:

```bibtex
@software{automated_prompt_tuning,
  title={Automated Prompt Tuning System for Bank Transaction Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/Automated-prompt-tuning-}
}
```

## Contact

For questions or support, please open an issue on GitHub.
