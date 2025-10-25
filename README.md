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

#### USPM Agent - Complete API Examples

##### Example 1: Initialize with OpenAI
```python
from uspm_agent import USPMAgent
import os

# Set API key
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# Initialize USPM Agent in guided mode
agent = USPMAgent(
    mode='guided',              # 'guided' or 'quick'
    data_dir='bank_data',       # Directory for data files
    results_dir='results'       # Directory for results
)

# Run the complete guided workflow
agent.run_interactive()
```

##### Example 2: Initialize with Anthropic Claude
```python
from uspm_agent import USPMAgent
import os

# Set API key
os.environ['ANTHROPIC_API_KEY'] = 'your-api-key'

# Initialize and run
agent = USPMAgent(mode='guided')
agent.run_interactive()
```

##### Example 3: Initialize with Ollama (Local Models)
```python
from uspm_agent import USPMAgent

# No API key needed for Ollama
# Make sure Ollama is running: ollama serve

agent = USPMAgent(mode='guided')
agent.run_interactive()
```

##### Example 4: Quick Mode - Programmatic Task Execution
```python
from uspm_agent import USPMAgent

# Initialize in quick mode
agent = USPMAgent(mode='quick')

# Configure provider
task = agent.parse_intent("configure openai provider")
result = agent.execute_task(task)
print(result)

# Generate data
task = agent.parse_intent("generate 30 files with 100 transactions")
result = agent.execute_task(task)
print(result)

# Run optimization
task = agent.parse_intent("optimize prompts for 5 generations")
result = agent.execute_task(task)
print(result)

# Show results
task = agent.parse_intent("show results")
result = agent.execute_task(task)
print(result)
```

##### Example 5: Access Reasoning Logs & Explainability
```python
from uspm_agent import USPMAgent

# Run guided workflow
agent = USPMAgent(mode='guided')
agent.run_interactive()

# After workflow completes, access reasoning logs
agent.show_reasoning_log()

# Access specific reasoning data
for log in agent.reasoning_logs:
    print(f"Stage: {log.stage}")
    print(f"Action: {log.action}")
    print(f"Reasoning: {log.reasoning}")
    print(f"Confidence: {log.confidence}")
    print(f"Bias Check: {log.bias_check}")
    print(f"Timestamp: {log.timestamp}")
    print("---")
```

##### Example 6: Mode Switching
```python
from uspm_agent import USPMAgent

# Start in quick mode
agent = USPMAgent(mode='quick')

# Do some quick tasks
agent.execute_task(agent.parse_intent("generate data"))

# Switch to guided mode for full workflow
agent.switch_mode('guided')
agent.run_interactive()
```

##### Example 7: Access Workflow State
```python
from uspm_agent import USPMAgent

agent = USPMAgent(mode='guided')
agent.run_interactive()

# Access determined use case
if agent.use_case_context:
    print(f"Use Case: {agent.use_case_context.use_case_name}")
    print(f"Type: {agent.use_case_context.use_case_type}")
    print(f"Domain: {agent.use_case_context.domain}")
    print(f"Reasoning: {agent.use_case_context.reasoning}")
    print(f"Confidence: {agent.use_case_context.confidence}")

# Access data requirements
if agent.data_requirements:
    print(f"Files: {agent.data_requirements.num_files}")
    print(f"Records per file: {agent.data_requirements.records_per_file}")
    print(f"Features: {agent.data_requirements.features}")

# Access prescribed metrics
if agent.metrics_prescription:
    print(f"Primary metrics: {agent.metrics_prescription.primary_metrics}")
    print(f"Secondary metrics: {agent.metrics_prescription.secondary_metrics}")
    print(f"Reasoning: {agent.metrics_prescription.reasoning}")
```

##### Example 8: Traditional Orchestrator (Low-Level API)
```python
from main import PromptTuningOrchestrator

# Use the orchestrator directly for custom workflows
orchestrator = PromptTuningOrchestrator(
    llm_provider="openai",
    llm_config={"model": "gpt-4", "api_key": "your-key"},
    max_generations=5,
    population_size=15,
    num_test_files=5,
    data_dir="bank_data",
    results_dir="results"
)

# Run optimization
best_prompt = orchestrator.run_optimization()
print(f"Best prompt: {best_prompt.template.name}")
print(f"Score: {best_prompt.fitness:.3f}")

# Or quick test a specific prompt
orchestrator.quick_test("concise_direct")
```

---

### USPMAgent API Reference

#### Class: `USPMAgent`

**Description:** Unified Smart Prompt Management Agent combining guided workflows and quick commands.

##### Constructor

```python
USPMAgent(mode='guided', data_dir='bank_data', results_dir='results')
```

**Parameters:**
- `mode` (str): Operating mode - `'guided'` for full workflow or `'quick'` for fast commands. Default: `'guided'`
- `data_dir` (str): Directory path for data files. Default: `'bank_data'`
- `results_dir` (str): Directory path for results. Default: `'results'`

**Returns:** USPMAgent instance

---

##### Instance Attributes

**Shared Attributes (Both Modes):**
- `mode` (AgentMode): Current operating mode (GUIDED or QUICK)
- `llm_provider` (LLMProvider): Configured LLM provider instance
- `llm_tester` (LLMTester): LLM testing interface
- `provider_type` (str): Provider type ('openai', 'anthropic', 'ollama')
- `reasoning_logs` (List[ReasoningLog]): Complete reasoning log for explainability

**Guided Mode Attributes:**
- `current_stage` (WorkflowStage): Current workflow stage
- `use_case_context` (UseCaseContext): Determined use case information
- `data_requirements` (DataRequirements): Data generation requirements
- `metrics_prescription` (MetricsPrescription): LLM-prescribed metrics
- `generated_data_path` (str): Path to generated data
- `ground_truth` (dict): Ground truth labels and statistics
- `validation_results` (dict): Human validation results

**Quick Mode Attributes:**
- `context` (dict): Task execution context including:
  - `last_task`: Last executed task type
  - `last_results`: Last execution results
  - `provider_configured`: Provider configuration status

---

##### Main Methods

###### `run_interactive()`
Run the agent in interactive mode.

```python
agent = USPMAgent(mode='guided')
agent.run_interactive()
```

**Behavior:**
- **Guided Mode**: Runs complete 10-step workflow
- **Quick Mode**: Starts interactive command prompt

**Returns:** None

---

###### `switch_mode(new_mode)`
Switch between guided and quick modes.

```python
agent.switch_mode('guided')  # Switch to guided mode
agent.switch_mode('quick')   # Switch to quick mode
```

**Parameters:**
- `new_mode` (str): Target mode ('guided' or 'quick')

**Returns:** None

---

###### `show_reasoning_log()`
Display the complete reasoning and explainability log.

```python
agent.show_reasoning_log()
```

**Returns:** None (prints to console)

---

##### Guided Mode Methods

###### `validate_api_keys()`
Check for available LLM provider API keys.

```python
has_keys, key_status = agent.validate_api_keys()
```

**Returns:**
- `has_keys` (bool): Whether any keys were found
- `key_status` (dict): Status for each provider

---

###### `configure_llm_provider(key_status, provider=None)`
Configure the LLM provider interactively or programmatically.

```python
agent.configure_llm_provider(key_status)
# or
agent.configure_llm_provider(key_status, provider='openai')
```

**Parameters:**
- `key_status` (dict): Available provider keys
- `provider` (str, optional): Specific provider to configure

**Returns:** bool (success status)

---

###### `determine_use_case()`
Use LLM reasoning to determine the use case or accept user input.

```python
agent.determine_use_case()
```

**Side Effects:** Sets `agent.use_case_context`

**Returns:** None

---

###### `gather_data_requirements()`
Interactively gather data generation requirements from user.

```python
agent.gather_data_requirements()
```

**Side Effects:** Sets `agent.data_requirements`

**Returns:** None

---

###### `generate_synthetic_data()`
Generate synthetic data based on gathered requirements.

```python
agent.generate_synthetic_data()
```

**Side Effects:**
- Creates data files in `data_dir`
- Sets `agent.generated_data_path`

**Returns:** None

---

###### `generate_ground_truth()`
Generate ground truth labels and statistics for evaluation.

```python
agent.generate_ground_truth()
```

**Side Effects:** Sets `agent.ground_truth`

**Returns:** None

---

###### `prescribe_metrics_with_llm()`
Use LLM to prescribe appropriate evaluation metrics.

```python
agent.prescribe_metrics_with_llm()
```

**Side Effects:** Sets `agent.metrics_prescription`

**Returns:** None

---

###### `validate_metrics_with_human()`
Present prescribed metrics to user for validation/modification.

```python
approved = agent.validate_metrics_with_human()
```

**Returns:** bool (whether user approved metrics)

---

###### `load_or_generate_prompts()`
Load existing prompt templates or generate new ones dynamically.

```python
templates = agent.load_or_generate_prompts()
```

**Returns:** List[PromptTemplate]

---

##### Quick Mode Methods

###### `parse_intent(user_input)`
Parse natural language input to understand user intent.

```python
task = agent.parse_intent("optimize prompts for 10 generations")
```

**Parameters:**
- `user_input` (str): Natural language command

**Returns:** AgentTask (with task_type, parameters, confidence)

---

###### `execute_task(task)`
Execute a parsed task.

```python
result = agent.execute_task(task)
```

**Parameters:**
- `task` (AgentTask): Parsed task object

**Returns:** str (result message)

---

##### Utility Methods

###### `log_reasoning(stage, action, reasoning, input_data, output_data, confidence, bias_check)`
Log reasoning for explainability.

```python
agent.log_reasoning(
    stage="use_case_determination",
    action="analyze_user_input",
    reasoning="User wants fraud detection based on keywords",
    input_data={"user_response": "detect fraud"},
    output_data={"use_case": "fraud_detection"},
    confidence=0.92,
    bias_check={"severity": "low"}
)
```

**Parameters:**
- `stage` (str): Workflow stage name
- `action` (str): Action being performed
- `reasoning` (str): Explanation of decision
- `input_data` (dict): Input data for this step
- `output_data` (dict): Output/result data
- `confidence` (float): Confidence level (0.0-1.0)
- `bias_check` (dict, optional): Bias detection results

**Returns:** None

---

###### `check_bias(text, context)`
Check for potential biases in text or decisions.

```python
bias_result = agent.check_bias(
    text="This will obviously work",
    context="use_case_determination"
)
```

**Parameters:**
- `text` (str): Text to check
- `context` (str): Context of the check

**Returns:** dict with bias indicators and severity

---

##### Data Classes

###### `UseCaseContext`
```python
@dataclass
class UseCaseContext:
    use_case_name: str
    use_case_type: str
    domain: str
    description: str
    key_objectives: List[str]
    data_characteristics: Dict[str, Any]
    reasoning: str
    confidence: float
```

###### `DataRequirements`
```python
@dataclass
class DataRequirements:
    num_files: int
    records_per_file: int
    data_type: str
    features: List[str]
    anomaly_rate: float
    special_conditions: Dict[str, Any]
    reasoning: str
```

###### `MetricsPrescription`
```python
@dataclass
class MetricsPrescription:
    primary_metrics: List[str]
    secondary_metrics: List[str]
    metric_definitions: Dict[str, str]
    thresholds: Dict[str, float]
    reasoning: str
    llm_explanation: str
```

###### `ReasoningLog`
```python
@dataclass
class ReasoningLog:
    stage: str
    timestamp: str
    action: str
    reasoning: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    bias_check: Dict[str, Any]
```

---

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
