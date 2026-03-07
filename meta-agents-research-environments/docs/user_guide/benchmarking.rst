..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`graph` Benchmarking with Meta Agents Research Environments
====================================================================

Meta Agents Research Environments (ARE) provides comprehensive benchmarking capabilities for evaluating AI agent performance across various scenarios.
This guide covers how to run benchmarks, analyze results, and integrate with evaluation pipelines.

:octicon:`info` Overview
------------------------

The Benchmark CLI is a powerful tool for systematic agent evaluation. It supports:

* **Local and Remote Datasets**: Work with local scenario files or Hugging Face datasets
* **Multiple Model Providers**: Connect to various AI models through `LiteLLM <https://www.litellm.ai/>`_
* **Parallel Execution**: Run multiple scenarios concurrently for efficiency
* **Result Management**: Automatic result collection and upload to Hugging Face

:octicon:`terminal` Main Commands
---------------------------------

The benchmark CLI provides three primary commands:

``run``
~~~~~~~
   Execute scenarios with AI agents and collect performance metrics.

``judge``
~~~~~~~~~
   Validate the scenarios runs against a ground truth of expected outcomes in the environment.

``gaia2-run``
~~~~~~~~~~~~~
   Complete Gaia2 leaderboard evaluation pipeline that automatically runs all required configurations and phases for submission.

:octicon:`gear` Command Line Reference
--------------------------------------

For complete parameter documentation with all options and examples, see the auto-generated reference:

.. click:: are.simulation.benchmark.cli:main
   :prog: are-benchmark
   :nested: full

Key Parameters Overview
~~~~~~~~~~~~~~~~~~~~~~~

**Command Selection**
    * ``run``: Execute scenarios with AI agents and collect performance metrics
    * ``judge``: Validate scenarios runs against ground truth
    * ``gaia2-run``: Complete Gaia2 leaderboard evaluation and submission pipeline

**Local Dataset Configuration**
    * ``--dataset``: Local directory containing JSON scenario files or JSONL file listing scenarios
    * ``--config``: Dataset config name (e.g., ``execution``)
    * ``--split``: Dataset split name (e.g., ``validation``)

**Hugging Face Dataset Configuration**
    * ``--hf-dataset``: Hugging Face dataset path (e.g., ``meta-agents-research-environments/gaia2``)
    * ``--hf-split``: Dataset split name (e.g., ``validation``)
    * ``--hf-config``: Dataset config/subset name for Hugging Face datasets
    * ``--hf-revision``: Hugging Face dataset revision

**Model and Agent Configuration**
    * ``--model``: Model name to use for inference
    * ``--provider``: Model provider (see supported providers below)
    * ``--endpoint``: Custom endpoint URL for model API
    * ``--agent``: Specific agent type to use

**Execution Control**
    * ``--limit``: Maximum number of scenarios to run per config
    * ``--max_concurrent_scenarios``: Control parallel execution (auto-detected by default)
    * ``--num_runs``: Number of times to run each scenario for variance analysis (default: 3)
    * ``--scenario_timeout``: Timeout for each scenario in seconds (default: 900)
    * ``--oracle``: Run in oracle mode where oracle events are executed
    * ``--noise``: Enable noise augmentation with tool and environment configs
    * ``--executor_type``: Type of executor to use for running scenarios (thread or process, default: process)
    * ``--enable_caching``: Enable caching of results

**Agent2Agent Configuration**
    * ``--a2a_app_prop``: Fraction of Apps to run in Agent2Agent mode (0.0-1.0, default: 0)
    * ``--a2a_app_agent``: Agent used for App agent instances (default: default)
    * ``--a2a_model``: Model used for App agent instances
    * ``--a2a_model_provider``: Provider for App agent model
    * ``--a2a_endpoint``: Endpoint URL for App agent models

**Output and Trace Management**
    * ``--output_dir``: Directory for saving scenario states and logs
    * ``--trace_dump_format``: Format for dumping traces ('hf', 'lite', or 'both', default: both)
    * ``--hf_upload``: Dataset name to upload traces to Hugging Face (if not specified, no upload)
    * ``--hf_public``: Upload as public dataset (default: false)

**Judge System Configuration**
    * ``--judge_model``: Model to use for judge system validation (default: "meta-llama/Meta-Llama-3.3-70B-Instruct")
    * ``--judge_provider``: Provider for the judge model (default: uses same provider as main model)
    * ``--judge_endpoint``: Custom endpoint URL for the judge model (optional)

.. note::
   **Reproducible Results**: For consistent and reproducible evaluation results, use **llama3.3-70B** as the judge model.
    You can use any provider that offers this model based on your preference, access, and cost considerations.

:octicon:`rocket` Basic Usage
-----------------------------

Simple Benchmark Run
~~~~~~~~~~~~~~~~~~~~

Run a basic benchmark with local scenarios:

.. code-block:: bash

   uvx --from meta-agents-research-environments are-benchmark run --dataset /path/to/scenarios --agent default --limit 10

This command:

* ``--dataset /path/to/scenarios``: Specifies the directory containing scenario files
* ``--agent default``: Uses the Meta OSS agent
* ``--limit 10``: Runs only the first 10 scenarios

With Hugging Face Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run benchmarks using Hugging Face datasets:

.. code-block:: bash

   uvx --from meta-agents-research-environments are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation --agent default

This uses:

* ``--hf-dataset meta-agents-research-environments/gaia2``: Hugging Face dataset path
* ``--hf-split validation``: Specific dataset split
* ``--agent default``: Agent implementation to use for the runs. `default` is the only one provided and you should use this one for gaia2 evaluation.

:octicon:`gear` Model Configuration
-----------------------------------

Supported Providers
~~~~~~~~~~~~~~~~~~~

the Agents Research Environments supports multiple model providers through liteLLM:

**API-Based Providers**
   * ``llama-api``: Llama models via API
   * ``anthropic``: Claude models
   * ``openai``: GPT-3.5, GPT-4, and variants
   * ``azure``: Azure OpenAI services

**Third-Party Providers**
   * ``huggingface``: Models from Hugging Face Hub
   * ``fireworks-ai``: Fireworks AI models
   * ``together``: Together AI models
   * ``replicate``: Replicate models
   * ...

**Local Deployments**
   * ``local``: Local model deployments with OpenAI-compatible APIs

Provider Examples
~~~~~~~~~~~~~~~~~

**Llama API**

.. code-block:: bash

   export LLAMA_API_KEY="your-api-key"
   uvx --from meta-agents-research-environments are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation \
     --model Llama-4-Maverick-17B-128E-Instruct-FP8 --provider llama-api --agent default

**OpenAI Models**

.. code-block:: bash

   export OPENAI_API_KEY="your-api-key"
   uvx --from meta-agents-research-environments are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation \
     --model gpt-4 --provider openai --agent default

**Local OpenAI-Compatible Endpoint**

.. code-block:: bash

   uvx --from meta-agents-research-environments are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation \
     --model your-local-model --provider local \
     --endpoint "http://0.0.0.0:4000" --agent default

**Hugging Face Models**

.. code-block:: bash

   export HUGGINGFACE_API_TOKEN="your-token"
   uvx --from meta-agents-research-environments are-benchmark run --hf-dataset meta-agents-research-environments/gaia2 --hf-split validation \
     --model meta-llama/llama3-70b-instruct --provider huggingface --agent default

**Complete Benchmark Run with Upload**

.. code-block:: bash

   export model="meta-llama/llama3-70b-instruct"
   export provider="huggingface"

   uvx --from meta-agents-research-environments are-benchmark gaia2-run --hf-dataset meta-agents-research-environments/gaia2 \
     --agent default --output_dir ./benchmark_results/ \
     --model $model --provider $provider \
     --hf_upload myhforg/${model////.} --hf_public


.. note::
   **Judge System LLM Independence**: The judge system uses its own separate LLM engine for validation, which is independent of your agent's model configuration. The judge's LLM is used for **semantic validation** of tool arguments, **soft comparison** of agent outputs, and **context-aware evaluation**. Hard validation (exact matching, scripted checks) runs without LLM inference.

   **Important**: The judge system does not use the --model or --provider settings, these are for the agent. For setting the judge LLM, use the --judge_model and --judge_provider settings.


Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Different providers require specific environment variables:

**Llama API**
   * ``LLAMA_API_KEY`` (required): Your Llama API key
   * ``LLAMA_API_BASE`` (optional): Custom base URL, defaults to ``https://api.llama.com/compat/v1``

**OpenAI**
   * ``OPENAI_API_KEY`` (required): Your OpenAI API key

**Azure OpenAI**
   * ``AZURE_API_KEY`` (required): Your Azure OpenAI API key
   * ``AZURE_API_BASE`` (required): Your Azure OpenAI endpoint URL

**Anthropic**
   * ``ANTHROPIC_API_KEY`` (required): Your Anthropic API key

**Hugging Face**
   * ``HUGGINGFACE_API_TOKEN`` (required for private models): Your Hugging Face token

Provider-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

**Llama API Configuration**
   The Llama API provider automatically configures the endpoint and authentication:

   * Uses OpenAI-compatible format internally
   * Requires ``LLAMA_API_KEY`` environment variable
   * Supports custom base URL via ``LLAMA_API_BASE``

   Get an API token at `Llama Developer <https://llama.developer.meta.com/>`


**Local Deployments**
   For local model deployments:

   * Use ``--provider local`` with ``--endpoint`` pointing to your server
   * Ensure your local server implements OpenAI-compatible API
   * Common local deployment tools: vLLM, text-generation-inference, Ollama

**Hugging Face Integration**
   Hugging Face provider supports:

   * Direct model loading from Hugging Face Hub
   * Private model access with authentication tokens
   * Various Hugging Face inference providers

Execution Control Options
~~~~~~~~~~~~~~~~~~~~~~~~~

**Concurrency Control**

.. code-block:: bash

   are-benchmark run -d /path/to/scenarios \
     --max_concurrent_scenarios 4 -a default

**Output Directory**

.. code-block:: bash

   are-benchmark run -d /path/to/scenarios \
     --output_dir ./benchmark_results -a default

**Scenario Limiting**

.. code-block:: bash

   are-benchmark run -d /path/to/scenarios \
     --limit 50 -a default

Result Management
~~~~~~~~~~~~~~~~~

**Save Results Locally**

.. code-block:: bash

   are-benchmark run --hf meta-agents-research-environments/gaia2 --hf-split validation \
     --output_dir ./benchmark_results -a default

**Upload to Hugging Face**

.. code-block:: bash

   are-benchmark gaia2-run --hf meta-agents-research-environments/gaia2 \
     --hf_upload my-org/gaia2-results -a default

**Public Dataset Upload**

.. code-block:: bash

   are-benchmark gaia2-run --hf meta-agents-research-environments/gaia2 \
     --hf_upload my-org/gaia2-results --hf_public -a default

:octicon:`workflow` Recommended Workflow
----------------------------------------

Development and Testing
~~~~~~~~~~~~~~~~~~~~~~~

1. **Validation Phase**

   Start with a small validation set to test your setup:

   .. code-block:: bash

      export LLAMA_API_KEY="your-api-key"
      are-benchmark run --hf meta-agents-research-environments/gaia2 --hf-split validation \
        --model Llama-3.1-70B-Instruct --provider llama-api \
        --agent default --limit 10 --output_dir ./validation_results

2. **Trace Analysis**

   Examine the generated traces in ``./validation_results`` to:

   * Verify agent behavior matches expectations
   * Check for errors or unexpected patterns
   * Validate scoring and evaluation metrics
   * Debug any issues before full evaluation

3. **Iterative Improvement**

   Based on validation results:

   * Adjust agent parameters
   * Modify scenario selection
   * Fine-tune model settings
   * Update evaluation criteria

Gaia2 Leaderboard Submission
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed Gaia2 evaluation guidance, see :doc:`gaia2_evaluation`.


:octicon:`shield` Scenario Validation
-------------------------------------

Judge Mode
~~~~~~~~~~

Use judge mode to validate scenarios without running agents:

.. code-block:: bash

   are-benchmark judge -d /path/to/traces

:octicon:`package` Dataset Integration
--------------------------------------

Hugging Face Dataset Format
~~~~~~~~~~~~~~~~~~~~~~~~~~~

ARE works with `Hugging Face datasets <https://huggingface.co/docs/datasets/en/index>`_ that follow this structure:

.. code-block:: python

   {
       "scenario_id": "unique_identifier",
       "data": "json_serialized_scenario",
       "metadata": {
           "difficulty": "easy|medium|hard",
           "domain": "email|calendar|file_system",
           "tags": ["tag1", "tag2"]
       }
   }

Dataset Configuration
~~~~~~~~~~~~~~~~~~~~~

The benchmark supports multiple dataset configurations:
   * ``execution``: Multi-step planning and state-changing operations
   * ``search``: Information gathering and combination from multiple sources
   * ``adaptability``: Dynamic adaptation to environmental changes
   * ``time``: Temporal reasoning with precise timing constraints
   * ``ambiguity``: Recognition and handling of ambiguous or impossible tasks
   * ``mini``: A 160 scenarios subset of the above configurations

Dataset Splits
~~~~~~~~~~~~~~

Common dataset splits:

* ``validation``: Main evaluation and development set, include the ground truth to run the judge

Result Format
~~~~~~~~~~~~~

Benchmark results are automatically formatted for Hugging Face upload:

.. code-block:: python

   {
       "scenario_id": "scenario_001",
       "success": true,
       "execution_time": 45.2,
       "steps_taken": 12,
       "validation_score": 0.95,
       "agent_trace": [...],
       "model_info": {
           "model": "Llama-3.1-70B-Instruct",
           "provider": "llama-api"
       }
   }


:octicon:`rocket` Gaia2 Submission Command
------------------------------------------

The ``gaia2-run`` command provides a comprehensive evaluation pipeline specifically designed for Gaia2 leaderboard submissions. This command automates the complex multi-phase evaluation process required for complete Gaia2 assessment.

Key Features
~~~~~~~~~~~~

**Automated Multi-Phase Evaluation**
   The command automatically executes three distinct evaluation phases:

   * **Standard Phase**: Base agent performance across all capability configurations (execution, search, adaptability, time, ambiguity)
   * **Agent2Agent Phase**: Multi-agent collaboration scenarios on the mini configuration
   * **Noise Phase**: Robustness evaluation with environment perturbations and tool augmentation on the mini configuration

**Standardized Evaluation Parameters**
   * **3 runs per scenario**: Ensures proper variance analysis for leaderboard requirements
   * **Hugging Face format**: Traces automatically formatted for submission compatibility
   * **Comprehensive reporting**: Generates validation reports and performance summaries

Submission-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Hugging Face Upload Configuration**
   * ``--hf_upload``: Dataset name for uploading consolidated results (required for submission)
   * ``--hf_public``: Upload as public dataset (default: false, making results private), you *can* submit to the leaderboard with private datasets.

**Example Usage**

.. code-block:: bash

   # Complete Gaia2 submission with public upload
   are-benchmark gaia2-run --hf meta-agents-research-environments/gaia2 \
     --model meta-llama/llama3-70b-instruct --provider huggingface \
     --agent default \
     --output_dir ./gaia2_submission \
     --hf_upload my-org/gaia2-llama3-70b-results \
     --hf_public

   # Validation run before full submission
   are-benchmark gaia2-run --hf meta-agents-research-environments/gaia2 \
     --split validation \
     --model your-model --provider your-provider \
     --agent default \
     --limit 20 \
     --output_dir ./gaia2_validation

For comprehensive Gaia2 evaluation guidance and submission process, see :doc:`gaia2_evaluation`.

:octicon:`graph` Variance Analysis
----------------------------------

The platform supports running each scenario multiple times to analyze performance variance and improve statistical confidence in results.

Multiple Runs Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``--num_runs`` parameter to specify how many times each scenario should be executed:

.. code-block:: bash

   # Run each scenario 5 times for better variance analysis
   are-benchmark run --hf meta-agents-research-environments/gaia2 --hf-split validation \
     --model gpt-4 --provider openai -a default --num_runs 5

.. note::
   The ``gaia2-run`` command automatically sets ``--num_runs 3`` to meet leaderboard requirements for variance analysis.

Benefits of Multiple Runs
~~~~~~~~~~~~~~~~~~~~~~~~~

**Statistical Confidence**
   Multiple runs provide more reliable performance metrics by reducing the impact of random variations.

**Variance Analysis**
   Understand the consistency of your agent's performance across identical scenarios.

**Robust Evaluation**
   Identify scenarios where performance is highly variable vs. consistently good/bad.

Result Structure
~~~~~~~~~~~~~~~~

When using multiple runs, the system automatically:

* Creates unique trace files for each run: ``scenario_123_run_1_[config hash]_[timestamp].json``, ``scenario_123_run_2_[config hash]_timestamp.json``, etc.
* Groups results by base scenario ID for variance calculations
* Provides comprehensive statistics in the final report

Report Structure
~~~~~~~~~~~~~~~~

The benchmark results include detailed reports for each configuration and overall summary:

.. code-block:: text

   === Validation Report ===
   Model: gemini-2-5-pro
   Provider: unknown

   === Time ===
   - Scenarios: 10 unique (30 total runs)
   - Success rate: 20.0% ± 0.0% (STD: 0.0%)
   - Pass@3: 3 scenarios (30.0%)
   - Pass^3: 1 scenarios (10.0%)
   - Average run duration: 93.4s (STD: 47.4s)

   === Ambiguity ===
   - Scenarios: 10 unique (30 total runs)
   - Success rate: 10.0% ± 0.0% (STD: 0.0%)
   - Pass@3: 2 scenarios (20.0%)
   - Pass^3: 0 scenarios (0.0%)
   - Average run duration: 94.9s (STD: 62.3s)

   === Execution ===
   - Scenarios: 10 unique (30 total runs)
   - Success rate: 53.3% ± 6.7% (STD: 11.5%)
   - Pass@3: 6 scenarios (60.0%)
   - Pass^3: 4 scenarios (40.0%)
   - Average run duration: 124.0s (STD: 78.3s)

   === Adaptability ===
   - Scenarios: 10 unique (30 total runs)
   - Success rate: 20.0% ± 0.0% (STD: 0.0%)
   - Pass@3: 2 scenarios (20.0%)
   - Pass^3: 2 scenarios (20.0%)
   - Average run duration: 95.1s (STD: 47.4s)

   === Search ===
   - Scenarios: 10 unique (30 total runs)
   - Success rate: 56.7% ± 3.3% (STD: 5.8%)
   - Pass@3: 8 scenarios (80.0%)
   - Pass^3: 3 scenarios (30.0%)
   - Average run duration: 168.0s (STD: 91.5s)

   === Global Summary ===
   - Scenarios: 50 unique (150 total runs)
   - Macro success rate: 32.0% ± 1.2% (STD: 2.0%)
   - Micro success rate: 32.0% ± 1.2% (STD: 2.0%)
   - Pass@3: 21 scenarios (42.0%)
   - Pass^3: 10 scenarios (20.0%)
   - Average run duration: 115.1s (STD: 72.7s)
   - Job duration: 277.5 seconds

Understanding Report Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The validation reports provide comprehensive statistical analysis:

**Per Config Metrics**
    * **Success rate:**: Percentage of individual runs that succeeded (counts each run separately)
    * **Pass^k**: Percentage of scenarios that succeed in all k runs
    * **Pass@k**: Percentage of scenarios that succeed in at least 1 out of k runs
    * **Average run duration**: Average time taken per run

**Global Metrics**
    * **Macro success rate:**: Average of per-capability success rates
    * **Micro success rate**: Average of per-run success rates
    * **Job duration**: Total time taken for all runs

**Variance Analysis**
    * **STD**: Standard deviation of the metric across runs
    * **±**: Standard error (STD / sqrt(n) where n is the number of runs)


Results Caching
~~~~~~~~~~~~~~~

Meta Agents Research Environments includes a caching system that stores scenario execution results to avoid re-running identical scenarios with the same configuration.
When enabled with ``--enable_caching``, the system generates unique cache keys based on both the scenario content and runner configuration (model, provider, agent settings, etc.).
Results are stored as JSON files in ``~/.cache/are/simulation/scenario_results/`` by default, or in a custom location specified by the ``Meta Agents Research Environments_CACHE_DIR`` environment variable.

Next Steps
----------

With benchmarking knowledge:

* **Run Systematic Evaluations**: Use benchmarks for comprehensive agent testing
* **Contribute Results**: Share findings with the research community
* **Iterate and Improve**: Use results to enhance agent capabilities
* **Develop Custom Benchmarks**: Create domain-specific evaluation suites

Ready to create your own scenarios? Continue to :doc:`../tutorials/scenario_development` for detailed guidance on scenario creation.
