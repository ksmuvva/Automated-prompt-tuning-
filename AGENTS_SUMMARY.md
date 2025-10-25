# Agents Summary - Complete Overview

## Total Agents in System: **2 Main Agents**

---

## 1. EnhancedAgent (enhanced_agent.py)

**Type:** Guided Workflow Agent with Reasoning & Explainability
**Lines of Code:** 945
**Default Mode:** Yes (runs with `python main.py`)

### Purpose
A comprehensive, step-by-step guided agent that walks users through the entire prompt tuning workflow with full reasoning, explainability, and bias detection at every stage.

### Main Class
```python
class EnhancedAgent:
    """
    Enhanced NLP agent with reasoning, explainability, and bias detection.
    Guides users through the entire workflow from use case to optimization.
    """
```

### Supporting Classes (6)
1. **WorkflowStage** (Enum) - Tracks current stage in 10-step workflow
2. **UseCaseContext** - Stores use case determination results
3. **DataRequirements** - Stores data generation specifications
4. **MetricsPrescription** - Stores LLM-prescribed metrics
5. **ReasoningLog** - Stores reasoning for each decision
6. **AgentTask** - N/A (not used in this file)

### Key Methods (24)
| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize enhanced agent |
| `log_reasoning()` | Log decisions with reasoning |
| `check_bias()` | Detect biases in text/decisions |
| `validate_api_keys()` | Check available LLM providers |
| `configure_llm_provider()` | Set up LLM provider |
| `determine_use_case()` | Determine use case with LLM reasoning |
| `_get_default_use_case()` | Return default use case |
| `gather_data_requirements()` | Interactive data requirements |
| `generate_synthetic_data()` | Generate data based on requirements |
| `generate_ground_truth()` | Generate ground truth labels |
| `prescribe_metrics_with_llm()` | LLM recommends metrics |
| `_get_default_metrics()` | Return default metrics |
| `validate_metrics_with_human()` | Human reviews metrics |
| `load_or_generate_prompts()` | Load or generate prompts |
| `_generate_dynamic_prompts()` | LLM generates custom prompts |
| `run_guided_workflow()` | Execute complete 10-step workflow |
| `_run_optimization()` | Run optimization with reasoning |
| `_save_workflow_state()` | Save workflow for resuming |
| `show_reasoning_log()` | Display reasoning log |

### 10-Step Workflow
1. **API Key Validation** - Check for OpenAI, Anthropic, Ollama keys
2. **LLM Provider Configuration** - Select and configure provider
3. **Use Case Determination** - User-defined OR LLM-assisted discovery
4. **Data Requirements Gathering** - Interactive specifications
5. **Synthetic Data Generation** - Generate based on requirements
6. **Ground Truth Generation** - Calculate statistics
7. **Metrics Prescription** - LLM recommends evaluation metrics
8. **Human Validation** - User reviews and approves metrics
9. **Prompt Loading/Generation** - Load existing or generate new
10. **Optimization** - Run full optimization with logging

### Features
- ✅ **Reasoning & Explainability** - Every decision logged
- ✅ **Bias Detection** - Automatic checking at each stage
- ✅ **Human-in-the-Loop** - Critical decisions validated
- ✅ **Workflow State Saving** - Resume from any step
- ✅ **Complete Transparency** - View reasoning logs
- ✅ **LLM-Powered** - Uses LLM for recommendations
- ✅ **Guided Process** - Step-by-step for beginners

### Use Cases
- **Best For:**
  - First-time users
  - Complex or unfamiliar use cases
  - Need for transparency and explainability
  - Want LLM assistance in decision-making
  - Require bias detection
  - Need workflow state management

### How to Run
```bash
# Default mode
python main.py

# Explicit mode
python main.py --mode enhanced
```

---

## 2. NLPAgent (nlp_agent.py)

**Type:** Natural Language Command Agent
**Lines of Code:** 508
**Default Mode:** No (requires `--mode agent`)

### Purpose
A quick, interactive natural language agent for users who know what they want and need fast execution without guided workflow.

### Main Class
```python
class NLPAgent:
    """
    Intelligent NLP agent for prompt tuning system.
    Can understand natural language commands and execute appropriate actions.
    """
```

### Supporting Classes (2)
1. **TaskType** (Enum) - Types of tasks the agent can handle
2. **AgentTask** - Represents a parsed task with parameters

### Task Types (9)
1. `GENERATE_DATA` - Generate sample data
2. `OPTIMIZE_PROMPTS` - Run optimization
3. `TEST_PROMPT` - Test specific prompt
4. `COMPARE_PROMPTS` - Compare prompts
5. `LIST_PROMPTS` - List available prompts
6. `SHOW_RESULTS` - Display results
7. `CONFIGURE` - Configure LLM provider
8. `HELP` - Show help
9. `UNKNOWN` - Unknown command

### Key Methods (17)
| Method | Purpose |
|--------|---------|
| `__init__()` | Initialize NLP agent |
| `_load_config()` | Load configuration |
| `parse_intent()` | Parse natural language to task |
| `_extract_parameters()` | Extract parameters from input |
| `execute_task()` | Execute identified task |
| `_check_provider_configured()` | Check if provider is set |
| `_generate_data()` | Generate synthetic data |
| `_configure_provider()` | Configure LLM provider |
| `_optimize_prompts()` | Run optimization |
| `_test_prompt()` | Test specific prompt |
| `_compare_prompts()` | Compare prompts |
| `_list_prompts()` | List templates |
| `_show_results()` | Show last results |
| `_show_help()` | Display help |
| `_handle_unknown()` | Handle unknown commands |
| `run_interactive()` | Run interactive mode |
| `run_command()` | Run single command |

### Natural Language Commands
| Command Type | Examples |
|--------------|----------|
| Configure | "configure openai provider", "use anthropic claude", "setup ollama" |
| Generate | "generate data", "create 50 files with 200 transactions" |
| Optimize | "optimize prompts", "run optimization for 10 generations" |
| Test | "test prompt concise_direct", "evaluate prompt detailed_analytical" |
| Compare | "compare prompts", "which prompt is better" |
| List | "list prompts", "show available prompts" |
| Results | "show results", "display last results" |
| Help | "help", "what can you do" |
| Exit | "exit", "quit", "q" |

### Features
- ✅ **Natural Language** - Understands conversational commands
- ✅ **Intent Parsing** - Pattern matching for commands
- ✅ **Parameter Extraction** - Automatically extracts numbers, names
- ✅ **Quick Execution** - Fast for experienced users
- ✅ **Interactive Mode** - Chat-like interface
- ✅ **Context Tracking** - Remembers last task and results

### Use Cases
- **Best For:**
  - Experienced users who know what they want
  - Quick operations
  - Batch testing
  - Scripting with natural language
  - When guided workflow is too slow

### How to Run
```bash
python main.py --mode agent
```

### Example Session
```
Agent> configure openai provider
Agent> generate 30 files
Agent> optimize prompts for 5 generations
Agent> show results
Agent> exit
```

---

## Comparison Table

| Feature | EnhancedAgent | NLPAgent |
|---------|---------------|----------|
| **Lines of Code** | 945 | 508 |
| **Complexity** | High | Medium |
| **Guided Workflow** | ✅ Yes (10 steps) | ❌ No |
| **Reasoning & Explainability** | ✅ Full logging | ❌ No |
| **Bias Detection** | ✅ Yes | ❌ No |
| **LLM-Assisted Decisions** | ✅ Yes | ❌ No |
| **Human Validation** | ✅ Yes | ❌ No |
| **Workflow State Saving** | ✅ Yes | ❌ No |
| **Natural Language** | ⚠️ Limited (prompts) | ✅ Full support |
| **Use Case Determination** | ✅ Automatic | ❌ Manual |
| **Metrics Prescription** | ✅ LLM-powered | ❌ Uses defaults |
| **Dynamic Prompt Gen** | ✅ Yes | ❌ No |
| **Speed** | ⭐⭐⭐ Slower (guided) | ⭐⭐⭐⭐⭐ Fast |
| **Ease for Beginners** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐ Good |
| **Transparency** | ⭐⭐⭐⭐⭐ Complete | ⭐⭐ Limited |
| **Default Mode** | ✅ Yes | ❌ No |

---

## When to Use Which Agent?

### Use **EnhancedAgent** When:
1. 🆕 You're using the system for the first time
2. 🤔 Your use case is complex or unfamiliar
3. 📊 You need to understand why decisions were made
4. ⚖️ Bias detection is important
5. 👥 You want human validation at critical steps
6. 💾 You need to save/resume workflows
7. 🎓 You're learning the system
8. 🔍 Transparency and explainability are required

### Use **NLPAgent** When:
1. ⚡ You need quick execution
2. ✅ You know exactly what you want
3. 🔄 Running repeated operations
4. 📝 Scripting natural language commands
5. 💪 You're an experienced user
6. 🚀 Speed is more important than guidance
7. 🎯 Simple, straightforward tasks

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   main.py (Entry Point)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Mode Selection:                                            │
│  ├─ --mode enhanced  ──────▶ EnhancedAgent                 │
│  ├─ --mode agent     ──────▶ NLPAgent                      │
│  ├─ --mode optimize  ──────▶ PromptTuningOrchestrator      │
│  ├─ --mode generate  ──────▶ BankDataGenerator             │
│  └─ --mode quick-test ─────▶ PromptTuningOrchestrator      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     EnhancedAgent                           │
│              (Guided Workflow Agent)                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  10-Step Workflow:                                          │
│  1. API Key Validation                                      │
│  2. LLM Configuration                                       │
│  3. Use Case Determination      ◄─── LLM Reasoning         │
│  4. Data Requirements                                       │
│  5. Data Generation                                         │
│  6. Ground Truth                                            │
│  7. Metrics Prescription        ◄─── LLM Reasoning         │
│  8. Human Validation            ◄─── Human Input           │
│  9. Prompt Loading/Generation   ◄─── LLM Generation        │
│  10. Optimization                                           │
│                                                             │
│  Cross-Cutting:                                             │
│  • Reasoning Logs (all stages)                              │
│  • Bias Detection (all stages)                              │
│  • Workflow State Saving                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       NLPAgent                              │
│           (Natural Language Command Agent)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Intent Parsing:                                            │
│  User Input ──▶ parse_intent() ──▶ AgentTask               │
│                                                             │
│  Task Execution:                                            │
│  AgentTask ──▶ execute_task() ──▶ _method_for_task()       │
│                                                             │
│  Supported Tasks:                                           │
│  • Configure Provider                                       │
│  • Generate Data                                            │
│  • Optimize Prompts                                         │
│  • Test Prompt                                              │
│  • Compare Prompts                                          │
│  • List Prompts                                             │
│  • Show Results                                             │
│  • Help                                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary Statistics

```
Total Agent Files:        2
Total Lines of Code:      1,453
Total Classes:            8 (3 main classes + 5 data classes)
Total Enums:              2
Total Methods:            41+

Agent Distribution:
├── EnhancedAgent:        945 lines (65%)
└── NLPAgent:            508 lines (35%)

Class Distribution:
├── Main Agent Classes:   3
│   ├── EnhancedAgent
│   ├── NLPAgent
│   └── AgentTask (data class)
├── Data Classes:         5
│   ├── UseCaseContext
│   ├── DataRequirements
│   ├── MetricsPrescription
│   ├── ReasoningLog
│   └── AgentTask
└── Enums:               2
    ├── WorkflowStage (10 stages)
    └── TaskType (9 types)
```

---

## File Locations

```
/home/user/Automated-prompt-tuning-/
├── enhanced_agent.py    # EnhancedAgent (default mode)
├── nlp_agent.py         # NLPAgent (quick mode)
└── main.py              # Entry point (integrates both)
```

---

## Quick Reference

### Start EnhancedAgent
```bash
python main.py
# or
python main.py --mode enhanced
```

### Start NLPAgent
```bash
python main.py --mode agent
```

### Choose Between Them
- **Beginners / Complex Tasks / Need Transparency** → Use EnhancedAgent
- **Experienced / Quick Tasks / Know What You Want** → Use NLPAgent

---

**Last Updated:** 2025-10-25
**Version:** 2.0
**Branch:** claude/fix-nlp-cli-agent-011CUSv3hBkJZw5aCJEK9dUV
