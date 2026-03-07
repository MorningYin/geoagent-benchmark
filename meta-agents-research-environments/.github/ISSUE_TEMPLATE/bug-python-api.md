---
name: Python API Bug Report
about: Report issues with Python library, modules, or API functionality
title: '[API BUG] '
labels: 'bug, python-api'
assignees: ''

---

**🐍 Python API Bug Description**:
<!-- Clearly describe what the Python API issue is -->
- What Python functionality is affected?
- What unexpected behavior did you observe?
- What were you trying to accomplish with the API?

**🏗️ Meta Agents Research Environments Components Affected**:
<!-- Check all components where the Python API bug occurs -->
- [ ] **Environment** (`are/simulation/environment.py`, `are/simulation/core/`, environment setup/configuration)
- [ ] **Scenarios** (`are/simulation/scenarios/`, scenario definitions, scenario runner)
- [ ] **Agents** (`are/simulation/agents/`, agent implementations)
- [ ] **Build/CI** (GitHub Actions, Docker, dependencies)

**🎯 Python API Bug Category**:
<!-- Check the primary category for this Python API bug -->
- [ ] 🔧 Function/Method Issue (incorrect behavior, wrong return values)
- [ ] 📦 Import/Module Issue (import errors, missing modules)
- [ ] 🏗️ Class/Object Issue (initialization, attribute access, inheritance)
- [ ] 📊 Data Processing Issue (incorrect data handling, serialization)
- [ ] 🔗 Integration Issue (compatibility with other libraries)
- [ ] 💥 Exception/Error Issue (unexpected crashes, unhandled exceptions)

## 💻 Code Example
**Minimal Reproducible Example**:
```python
# Provide a minimal code example that reproduces the issue
# Include imports, setup, and the specific code that fails

import are
# ... setup code ...

# The problematic code:
result = some_function()  # This should work but doesn't
```

**Expected Code Behavior**:
```python
# What you expected the code to do or return
expected_result = "some expected value"
```

**Actual Code Behavior**:
```python
# What actually happened (error, wrong result, etc.)
# actual_result = Exception or wrong value
```

## 🔄 Steps to Reproduce
**Environment Setup**:
1. Installation method: [e.g., pip install, conda, from source]
2. Virtual environment: [e.g., venv, conda env, system Python]
3. Dependencies installed: [e.g., pip install -r requirements.txt]

**Code Execution Steps**:
1. Import modules: `import are, ...`
2. Initialize objects: `env = are.simulation.Environment(...)`
3. Call problematic function: `result = env.some_method(...)`
4. Observe the issue

**Frequency**:
- [ ] Always happens with this code
- [ ] Happens most of the time
- [ ] Happens sometimes (intermittent)
- [ ] Only under specific conditions (describe below)

**Specific Conditions** (if applicable):
<!-- When does this API issue occur? What makes it more/less likely? -->

## 🐍 Python Environment Details
**Python Information**:
- Python version: [e.g., 3.9.7, 3.11.5]
- Python implementation: [e.g., CPython, PyPy]
- Virtual environment: [e.g., venv, conda, poetry, none]

**Meta Agents Research Environments Installation**:
- are version: [e.g., 1.0.0, commit hash, branch name]
- Installation method: [e.g., pip install meta-agents-research-environments, pip install -e ., conda install]
- Installation location: [e.g., site-packages, local development]

**Key Dependencies**:
```
# Paste output of: pip list | grep -E "(torch|numpy|pandas|transformers|meta-agents-research-environments)"
# Or relevant dependencies for your use case
```

**Full Environment** (if needed):
```
# Paste output of: pip freeze
# Only include if the issue might be dependency-related
```

## 📋 Error Information
**Error Message**:
```python
# Paste the complete error message here
Traceback (most recent call last):
  File "...", line ..., in ...
    ...
ErrorType: Error message
```

**Stack Trace Analysis**:
<!-- Which part of the are codebase is involved in the error? -->

**Related Warnings** (if any):
```python
# Any warnings that appear before the error
```

## 🔍 API Context & Usage
**Import Context**:
```python
# Show how you're importing are modules
from are import ...
import are.simulation.agents as ...
```

**Object Initialization**:
```python
# Show how you're creating are objects
env = are.Environment(...)
agent = are.simulation.agents.SomeAgent(...)
```

**Method/Function Usage**:
```python
# Show the specific API calls that are problematic
result = obj.method(param1=value1, param2=value2)
```

**Data Types & Values**:
<!-- What types of data are you passing to the API? -->
- Input data types: [e.g., dict, list, numpy array, pandas DataFrame]
- Input data shapes/sizes: [e.g., array shape (100, 50), dict with 10 keys]
- Parameter values: [e.g., specific configuration values]

## 🔧 System Information
**Operating System**:
- OS: [e.g., Windows 11, macOS 14, Ubuntu 22.04]
- Architecture: [e.g., x64, arm64, Apple Silicon]

**Hardware** (if relevant):
- RAM: [e.g., 16GB, 32GB]
- CPU: [e.g., Intel i7, Apple M2, AMD Ryzen]
- GPU: [e.g., NVIDIA RTX 4090, Apple M2, None]

## 📊 Impact Assessment
**Severity**:
- [ ] 🔴 Critical (API completely unusable)
- [ ] 🟠 High (major functionality blocked)
- [ ] 🟡 Medium (workaround available but inconvenient)
- [ ] 🟢 Low (minor issue or edge case)

**Development Impact**:
- [ ] Blocks core development workflow
- [ ] Prevents using key features
- [ ] Requires significant workarounds
- [ ] Minor inconvenience

**API Usage Context**:
- [ ] Production code
- [ ] Research/experimentation
- [ ] Learning/tutorial following
- [ ] Testing/debugging

## 🔧 Workaround
**Available Workaround**:
- [ ] Yes (describe below)
- [ ] No

**Workaround Code**:
```python
# If you found a way to work around this API issue, show the code
# Alternative approach that works
```

**Workaround Limitations**:
<!-- What are the downsides of the workaround? -->

## 🧪 Additional Testing
**Other Python Versions Tested**:
- [ ] Python 3.8
- [ ] Python 3.9
- [ ] Python 3.10
- [ ] Python 3.11
- [ ] Python 3.12

**Other Environments Tested**:
- [ ] Different OS (specify: _______)
- [ ] Different installation method
- [ ] Fresh virtual environment
- [ ] Different hardware

**Comparison with Documentation**:
<!-- Does the API behave as documented? Link to relevant docs if available -->

## 📎 Additional Context
<!-- Any other information that might be helpful -->
- Related GitHub issues or discussions
- Links to relevant documentation
- Comparison with similar APIs in other libraries
- When did this start happening?
- Did this API work before? If so, when did it break?
- Any recent changes to your code or environment?
- Links to external resources (Stack Overflow, etc.)
