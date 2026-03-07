---
name: Benchmark Bug Report
about: Report issues with benchmarks, performance, or evaluation metrics
title: '[BENCHMARK BUG] '
labels: 'bug, benchmark'
assignees: ''

---

**📊 Benchmark Bug Description**:
<!-- Clearly describe what the benchmark issue is -->
- What benchmark or evaluation is affected?
- What unexpected behavior did you observe?
- What were you trying to measure or evaluate?

**🎯 Benchmark Bug Category**:
<!-- Check the primary category for this benchmark bug -->
- [ ] 📈 Performance Issue (slow execution, memory usage, timeouts)
- [ ] 📊 Metrics Issue (incorrect calculations, missing metrics)
- [ ] 🔄 Reproducibility Issue (inconsistent results across runs)
- [ ] 💾 Data Issue (incorrect datasets, missing data, data corruption)
- [ ] ⚖️ Evaluation Issue (scoring problems, comparison errors)
- [ ] 🏃 Execution Issue (benchmark fails to run, crashes during evaluation)

## 📋 Benchmark Details
**Benchmark Information**:
- Benchmark Name: [e.g., Gaia2, custom scenario set]
- Benchmark Version/Commit: [e.g., commit hash, tag, branch]
- Dataset Used: [e.g., meta-agents-research-environments/gaia2, local dataset path]
- Dataset Split: [e.g., validation, test]
- Dataset Config: [e.g., mini, execution, search, adaptability, time, ambiguity]

**Model Configuration**:
- Model: [e.g., meta-llama/llama3-70b-instruct, gpt-4, Llama-3.1-70B-Instruct]
- Provider: [e.g., huggingface, openai, llama-api, anthropic, local]
- Endpoint: [e.g., custom endpoint URL if using local provider]
- Agent: [e.g., default]

**Evaluation Parameters**:
- Number of runs: [e.g., 3 (default for gaia2-run), 1, 5]
- Scenario timeout: [e.g., 900 seconds (default)]
- Max concurrent scenarios: [e.g., auto-detected, 4]
- Limit: [e.g., no limit, 10, 50]
- Oracle mode: [e.g., enabled, disabled]
- Noise augmentation: [e.g., enabled, disabled]
- Agent2Agent proportion: [e.g., 0.0 (disabled), 1.0 (all scenarios)]

**Judge Configuration** (if applicable):
- Judge model: [e.g., meta-llama/Meta-Llama-3.3-70B-Instruct (default)]
- Judge provider: [e.g., same as main model, custom provider]
- Judge endpoint: [e.g., custom endpoint if different from main model]
```

**Command Used**:
```bash
# Paste the exact command that caused the issue
# Examples:
# are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation --hf-config mini --model meta-llama/llama3-70b-instruct --provider huggingface --agent default
# are-benchmark gaia2-run --hf-dataset meta-agents-research-environments/gaia2 --model gpt-4 --provider openai --agent default --output_dir ./gaia2_results --hf_upload my-org/gaia2-results
# are-benchmark judge --dataset /path/to/traces --judge_model meta-llama/Meta-Llama-3.3-70B-Instruct --judge_provider huggingface
# are-benchmark run --dataset /path/to/local/scenarios --agent default --limit 10 --max_concurrent_scenarios 4
```

## 🔄 Steps to Reproduce
**Environment Setup**:
1. Dataset preparation: ...
2. Model/Agent configuration: ...
3. Environment setup: ...
4. Command execution: ...

**Reproduction Steps**:
1. Step 1
2. Step 2
3. Step 3
4. Observe the issue

**Frequency**:
- [ ] Always happens
- [ ] Happens most of the time (>75%)
- [ ] Happens sometimes (25-75%)
- [ ] Rarely happens (<25%)
- [ ] Only under specific conditions (describe below)

**Specific Conditions** (if applicable):
<!-- When does this benchmark issue occur? What makes it more/less likely? -->

## 📊 Performance Data
**Expected Performance**:
<!-- What performance/results did you expect? -->
- Expected runtime: [e.g., 5 minutes, 2 hours]
- Expected memory usage: [e.g., 2GB, 8GB]
- Expected accuracy/score: [e.g., 85%, specific metric values]

**Actual Performance**:
<!-- What performance/results did you observe? -->
- Actual runtime: [e.g., 30 minutes, crashed after 1 hour]
- Actual memory usage: [e.g., 16GB, out of memory]
- Actual accuracy/score: [e.g., 45%, NaN, error]

**Performance Comparison** (if available):
<!-- How does this compare to previous runs or expected baselines? -->

## 📈 Metrics & Results
**Expected Metrics**:
```
# Paste expected benchmark results/metrics
```

**Actual Metrics**:
```
# Paste actual benchmark results/metrics
```

**Metric Calculations** (if relevant):
<!-- Are there issues with how metrics are calculated? -->

**Missing Metrics**:
<!-- Are there metrics that should be reported but aren't? -->

## 🔧 System & Environment Information
**Hardware**:
- CPU: [e.g., Intel i7-12700K, Apple M2, AMD Ryzen 9 5900X]
- RAM: [e.g., 32GB, 64GB]
- GPU: [e.g., NVIDIA RTX 4090, Apple M2, None]
- Storage: [e.g., NVMe SSD, HDD]

**Software Environment**:
- OS: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
- Architecture: [e.g., x64, arm64, Apple Silicon]

## 🐍 Python Environment Details
**Python Information**:
- Python version: [e.g., 3.9.7, 3.11.5]
- Python implementation: [e.g., CPython, PyPy]
- Virtual environment: [e.g., venv, conda, poetry, none]

**Meta Agents Research Environments Installation**:
- Package version: [e.g., 1.0.0, commit hash, branch name]
- Installation method: [e.g., pip install meta-agents-research-environments, pip install -e ., conda install]
- Installation location: [e.g., site-packages, local development]

**Key Dependencies**:
```
# Paste output of: pip list | grep -E "(torch|numpy|pandas|transformers|litellm|meta-agents-research-environments|huggingface)"
# Or run: pip show meta-agents-research-environments
```

**Full Environment** (if dependency-related):
```
# Paste output of: pip freeze
# Only include if the issue might be dependency-related
```

**Resource Constraints**:
- Available RAM: [e.g., 16GB available]
- Available GPU memory: [e.g., 24GB VRAM]
- Available disk space: [e.g., 100GB free]

## 📋 Error Information
**Error Messages**:
```
Paste any error messages here
```

**Stack Trace** (if applicable):
```
Paste the full stack trace here
```

**Log Output**:
```
Paste relevant log output, especially performance logs, timing information, etc.
```

**Resource Usage Logs** (if available):
<!-- Any memory usage, CPU usage, or GPU usage logs -->

## 🔍 Data & Configuration Context
**Dataset Information**:
- Dataset size: [e.g., 1000 samples, 50GB]
- Data format: [e.g., JSON, CSV, custom format]
- Data source: [e.g., local file, downloaded, generated]

**Model/Agent Configuration**:
```python
# Paste relevant model or agent configuration
```

**Scenario Configuration**:
```python
# Paste relevant scenario configuration
```

**Environment Variables** (if relevant):
```bash
# List any relevant environment variables
# Model provider authentication
LLAMA_API_KEY=***
OPENAI_API_KEY=***
ANTHROPIC_API_KEY=***
HUGGINGFACE_API_TOKEN=***
AZURE_API_KEY=***
AZURE_API_BASE=***

# System configuration
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=8
```

## 📊 Impact Assessment
**Severity**:
- [ ] 🔴 Critical (benchmark completely unusable)
- [ ] 🟠 High (significantly impacts evaluation reliability)
- [ ] 🟡 Medium (minor impact on benchmark results)
- [ ] 🟢 Low (cosmetic or edge case)

**Research Impact**:
- [ ] Blocks research progress
- [ ] Affects result reliability
- [ ] Impacts performance comparisons
- [ ] Minor inconvenience

**Reproducibility Impact**:
- [ ] Makes results non-reproducible
- [ ] Causes inconsistent results
- [ ] Minor variation in results
- [ ] No impact on reproducibility

## 🔧 Workaround
**Available Workaround**:
- [ ] Yes (describe below)
- [ ] No

**Workaround Description**:
<!-- If you found a way to work around this benchmark issue, please describe it -->

**Alternative Evaluation Methods**:
<!-- Are there alternative ways to evaluate or benchmark? -->

## 📎 Additional Context
<!-- Any other information that might be helpful -->
- Comparison with other benchmark tools
- Related issues or discussions
- When did this start happening?
- Did this benchmark work before? If so, when did it break?
- Any recent changes to your setup or data?
- Links to relevant papers, datasets, or external resources
