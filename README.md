# Automated Prompt Tuning System for Bank Transaction Analysis

An intelligent system that automatically tests, evaluates, and optimizes LLM prompts for analyzing bank transactions, detecting high-value transactions (>250 GBP), and identifying anomalies.

## Features

- **🌟 USPM Agent - Unified Smart Prompt Management**:
  - **Guided Mode**: Full 10-step workflow with reasoning & explainability
    - Automated API key validation
    - Use case determination with LLM reasoning
    - Interactive data requirements gathering
    - LLM-based metrics prescription with human validation
    - Dynamic prompt generation
    - Bias detection across all stages
    - Complete reasoning logs for transparency
  - **Quick Mode**: Fast natural language command execution
    - Instant command recognition
    - Context-aware operation
    - Seamless mode switching
- **Automated Prompt Testing**: Tests multiple prompt formats and styles automatically
- **Metrics-Driven Optimization**: Uses precision, recall, F1-score, and composite metrics
- **Genetic Algorithm**: Evolves prompts across generations for continuous improvement
- **Multiple LLM Support**: Works with OpenAI, Anthropic, and Ollama providers
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

### 2. Run USPM Agent - Guided Mode (🌟 RECOMMENDED)

```bash
# Start USPM agent in guided mode with full workflow
python main.py --mode guided

# Or simply (guided is now the default)
python main.py
```

The USPM Agent in **Guided Mode** provides a complete workflow with reasoning and explainability:

**10-Step Guided Process:**

1. **API Key Validation** - Automatically checks for available LLM providers
2. **LLM Provider Configuration** - Guides you through provider setup
3. **Use Case Determination** - Uses LLM reasoning to understand your needs OR accepts your defined use case
4. **Data Requirements Gathering** - Interactive questions about your data needs
5. **Synthetic Data Generation** - Creates data based on your specifications
6. **Ground Truth Generation** - Generates labels and statistics
7. **Metrics Prescription** - LLM recommends appropriate evaluation metrics
8. **Human Validation** - You review and approve/modify suggested metrics
9. **Prompt Loading/Generation** - Load existing or generate new prompts dynamically
10. **Optimization** - Run the full optimization with reasoning logs

**Key Features:**
- ✅ **Reasoning & Explainability**: Every decision is logged with reasoning
- ✅ **Bias Detection**: Automatic bias checking at each stage
- ✅ **Human-in-the-Loop**: You validate critical decisions
- ✅ **Workflow State Saving**: Resume work from any step
- ✅ **Complete Transparency**: View reasoning logs anytime

**Example Session:**
```bash
$ python main.py

╔══════════════════════════════════════════════════════════════╗
║         USPM Agent - Unified Smart Prompt Management        ║
║              GUIDED MODE - Full Workflow with Reasoning      ║
╚══════════════════════════════════════════════════════════════╝

============================================================
STEP 1: API KEY VALIDATION
============================================================

Checking for available LLM providers:
  OpenAI       : ✓ Available
  Anthropic    : ✓ Available
  Ollama       : ✓ Available

============================================================
STEP 2: LLM PROVIDER CONFIGURATION
============================================================

Available providers: openai, anthropic, ollama

Select provider (openai/anthropic/ollama): openai
Enter model (default: gpt-4):

🔧 Initializing openai with model gpt-4...
✓ Openai configured successfully!

============================================================
STEP 3: USE CASE DETERMINATION
============================================================

Do you already know your use case?
(yes/no): no

Let me help you determine the use case.
Please provide information about your needs:

1. What problem are you trying to solve? Detect fraudulent transactions
2. What type of data will you work with? Bank transaction data
3. What are your main goals? Identify high-value and suspicious transactions
4. Any specific constraints or requirements? Need high precision

🤔 Analyzing your requirements with LLM reasoning...

✓ Use case determined!

📋 Use Case: Fraud Detection in Bank Transactions
   Type: fraud_detection
   Domain: finance

💭 Reasoning: Based on the user's goal to detect fraudulent and high-value
transactions in banking data, this is a fraud detection use case requiring
anomaly detection capabilities with emphasis on precision.
   Confidence: 92%

... [Continues through all 10 steps with reasoning]
```

### 3. Run USPM Agent - Quick Mode (For Fast Operations)

```bash
# Start USPM agent in quick mode for natural language commands
python main.py --mode quick
```

The USPM Agent in **Quick Mode** provides fast natural language command execution:
```
USPM> configure openai provider
USPM> generate 30 files
USPM> optimize prompts for 5 generations
USPM> show results
USPM> guided mode  (switch to guided mode anytime)
```

### 4. Generate Sample Data (CLI Mode)

```bash
# Generate 30 CSV files with bank transaction data
python main.py --mode generate
```

This creates sample data in the `bank_data/` directory with:
- Normal transactions
- High-value transactions (>250 GBP)
- Anomalous transactions
- Ground truth labels for evaluation

### 5. Run with LLM Provider (CLI Mode)

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

## CLI Usage Guide

The system provides **one unified USPM Agent** with two operating modes:

### USPM Agent - Unified Smart Prompt Management (🌟 RECOMMENDED)

The USPM Agent merges the best of both guided workflows and quick commands into a single, intelligent agent.

**Operating Modes:**

#### 1. Guided Mode - Full Workflow (Best for Beginners)

**Start Command:**
```bash
python main.py --mode guided
# or just
python main.py
```

**Features:**
- ✅ Guided 10-step workflow
- ✅ LLM-powered reasoning at each step
- ✅ Automatic use case determination
- ✅ Interactive data requirements gathering
- ✅ Metrics prescription with human validation
- ✅ Dynamic prompt generation
- ✅ Bias detection and explainability
- ✅ Workflow state saving/resuming

**When to Use:**
- First time using the system
- Complex or unfamiliar use cases
- Need transparency and explainability
- Want LLM assistance in decision-making

---

#### 2. Quick Mode - Fast Commands (Best for Experts)

**Start Command:**
```bash
python main.py --mode quick
```

The Quick Mode provides a conversational interface with natural language commands.

**Available Commands:**

| Command | Description | Example |
|---------|-------------|---------|
| **Configure Provider** | Set up LLM provider | `configure openai provider`<br>`use anthropic claude`<br>`setup ollama` |
| **Generate Data** | Create sample datasets | `generate data`<br>`create 50 files with 200 transactions`<br>`generate sample data` |
| **Optimize Prompts** | Run optimization | `optimize prompts`<br>`run optimization for 10 generations`<br>`improve prompts with 20 population size` |
| **Test Prompt** | Test specific prompt | `test prompt concise_direct`<br>`evaluate prompt detailed_analytical` |
| **Compare Prompts** | Compare all prompts | `compare prompts`<br>`which prompt is better` |
| **List Prompts** | Show available prompts | `list prompts`<br>`show available prompts` |
| **Show Results** | Display last results | `show results`<br>`display last results` |
| **Help** | Show help | `help`<br>`what can you do` |
| **Switch Mode** | Switch to guided mode | `guided mode`<br>`full workflow` |
| **Exit** | Quit the agent | `exit`<br>`quit`<br>`q` |

**Example Session:**
```bash
$ python main.py --mode quick

╔══════════════════════════════════════════════════════════════╗
║         USPM Agent - Unified Smart Prompt Management        ║
║              QUICK MODE - Natural Language Commands          ║
╚══════════════════════════════════════════════════════════════╝

Type 'help' for available commands or 'guided mode' to switch.

USPM> configure openai provider

Provider configured successfully!
Provider: openai
Model: gpt-4

USPM> generate 30 files

Generating 30 files with 100 transactions each...

✓ Data generation complete!
Files created: 30
Transactions per file: 100

USPM> optimize prompts for 5 generations

[Understanding: optimize_prompts]

🔄 Running optimization for 5 generations...

✓ Optimization complete! Best prompt: detailed_analytical (Score: 0.847)

USPM> show results

📊 Last Results:
Prompt: detailed_analytical
Score: 0.847

USPM> guided mode

🔄 Switching from quick mode to guided mode...

╔══════════════════════════════════════════════════════════════╗
║         USPM Agent - Unified Smart Prompt Management        ║
║              GUIDED MODE - Full Workflow with Reasoning      ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting USPM Agent Guided Workflow

USPM> exit

👋 Goodbye!
```

---

### Traditional CLI (For Scripting/Automation)

The traditional CLI uses command-line arguments for non-interactive use.

**Available Modes:**

| Mode | Description | Command |
|------|-------------|---------|
| `guided` | Start USPM agent in guided mode (default) | `python main.py --mode guided` |
| `quick` | Start USPM agent in quick mode | `python main.py --mode quick` |
| `generate` | Generate sample data | `python main.py --mode generate` |
| `optimize` | Run prompt optimization | `python main.py --mode optimize --provider openai` |
| `quick-test` | Quick test prompts | `python main.py --mode quick-test --provider openai` |

**Available Arguments:**

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--mode` | choice | Mode to run (agent, generate, optimize, quick-test) | `--mode optimize` |
| `--provider` | choice | LLM provider (openai, anthropic, ollama) | `--provider openai` |
| `--model` | string | Model name (provider-specific) | `--model gpt-4` |
| `--api-key` | string | API key for provider | `--api-key sk-xxx` |
| `--generations` | int | Max generations for optimization | `--generations 10` |
| `--population` | int | Population size per generation | `--population 20` |
| `--test-files` | int | Number of test files to use | `--test-files 5` |
| `--prompt-name` | string | Specific prompt to test | `--prompt-name concise_direct` |

**Example Commands:**

```bash
# 1. Generate sample data
python main.py --mode generate

# 2. Quick test a specific prompt (OpenAI)
export OPENAI_API_KEY="your-api-key"
python main.py --mode quick-test --provider openai --prompt-name concise_direct

# 3. Run optimization with OpenAI (default parameters)
python main.py --mode optimize --provider openai --model gpt-4

# 4. Run optimization with custom parameters
python main.py --mode optimize \
  --provider openai \
  --model gpt-4 \
  --generations 10 \
  --population 20 \
  --test-files 10

# 5. Run optimization with Anthropic Claude
export ANTHROPIC_API_KEY="your-api-key"
python main.py --mode optimize \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --generations 5

# 6. Run optimization with Ollama (local)
python main.py --mode optimize \
  --provider ollama \
  --model llama2 \
  --generations 5

# 7. Pass API key via argument (not recommended for security)
python main.py --mode optimize \
  --provider openai \
  --api-key sk-your-key-here \
  --generations 5
```

**Help Command:**
```bash
python main.py --help
```

---

### Using the API Programmatically

For advanced use cases, you can import and use the classes directly in Python:

```python
from main import PromptTuningOrchestrator
from uspm_agent import USPMAgent

# Method 1: Use the orchestrator directly
orchestrator = PromptTuningOrchestrator(
    llm_provider="openai",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_generations=5,
    population_size=15
)

best_prompt = orchestrator.run_optimization()
print(f"Best prompt: {best_prompt.template.name}")
print(f"Score: {best_prompt.fitness:.3f}")

# Method 2: Use USPM agent in guided mode
agent = USPMAgent(mode="guided")
agent.run_interactive()

# Method 3: Use USPM agent in quick mode
agent = USPMAgent(mode="quick")
result = agent.parse_intent("configure openai provider")
agent.execute_task(result)
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
├── main.py                    # Main orchestrator and CLI entry point
├── uspm_agent.py              # 🌟 USPM Agent - Unified Smart Prompt Management
│                              #    (Guided Mode + Quick Mode)
├── data_generator.py          # Synthetic data generation
├── prompt_templates.py        # Prompt template library
├── llm_interface.py           # LLM provider integrations (OpenAI, Anthropic, Ollama)
├── metrics_evaluator.py       # Performance metrics evaluation system
├── prompt_optimizer.py        # Genetic algorithm optimizer
├── config.json                # Configuration file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── AGENTS_SUMMARY.md          # Complete agent documentation
│
├── bank_data/                 # Generated CSV files (30 files)
│   ├── bank_account_00.csv
│   ├── bank_account_01.csv
│   └── ...
│
├── results/                   # Optimization results
│   ├── best_prompt_YYYYMMDD_HHMMSS.txt
│   ├── metrics_YYYYMMDD_HHMMSS.json
│   └── leaderboard_YYYYMMDD_HHMMSS.csv
│
└── workflow_states/           # USPM agent workflow states (resumable)
    └── uspm_workflow_YYYYMMDD_HHMMSS.json
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

1. **Use USPM Agent in Guided Mode** for first-time users or complex use cases - it provides guidance at every step
2. **Use USPM Agent in Quick Mode** for fast operations when you know what you need
3. **Switch modes anytime** - type "guided mode" in quick mode to get full workflow assistance
4. **Use fewer test files** initially (2-3) for faster iterations
5. **Increase population size** (20-30) for better exploration
6. **Run more generations** (8-10) for optimal results
7. **Adjust weights** based on your priorities (high-value vs anomaly detection)
8. **Review reasoning logs** via `agent.show_reasoning_log()` for transparency and debugging

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
