..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`terminal` CLI Reference
=================================

The Agents Research Environments provides three main command-line interfaces for different use cases:

- **are-run**: Simple scenario runner for individual scenarios
- **are-benchmark**: Benchmark runner for dataset evaluation
- **are-gui**: GUI server for interactive scenario management

All CLIs share common parameters for consistency and ease of use.

.. note::
   **Recommended Usage**: We recommend using `uvx --from meta-agents-research-environments` to run these commands without installing the package locally:

   .. code-block:: bash

      # Instead of: are-run -s scenario_name
      uvx --from meta-agents-research-environments are-run -s scenario_name

      # Instead of: are-benchmark run --hf dataset
      uvx --from meta-agents-research-environments are-benchmark run --hf dataset

      # Instead of: are-gui -s scenario_name
      uvx --from meta-agents-research-environments are-gui -s scenario_name

   For users who want to dig deeper into the library or develop custom scenarios, local installation is available (see :doc:`../user_guide/installation`).

Common Parameters
-----------------

The following parameters are available across all Meta Agents Research Environments CLI tools:

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. option:: -m, --model <MODEL>

   Model name to use for the agent. This specifies which language model will be used
   to power the AI agent during scenario execution.

.. option:: -mp, --provider <PROVIDER>

   Provider of the model (e.g., 'openai', 'anthropic', 'meta'). This determines
   which API or service will be used to access the specified model.

.. option:: --endpoint <URL>

   URL of the endpoint to contact for running the agent's model. Use this when
   connecting to custom model endpoints or local model servers.

Agent Configuration
~~~~~~~~~~~~~~~~~~~

.. option:: -a, --agent <AGENT>

   Agent to use for running the scenarios. This specifies which agent implementation
   will be used to interact with the model and execute scenario actions.

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. option:: --log-level <LEVEL>

   Set the logging level. Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.
   Default: INFO

Runtime Configuration
~~~~~~~~~~~~~~~~~~~~~

.. option:: -o, --oracle

   Run scenarios in Oracle mode where oracle events (user-defined agent events) are executed.
   This is useful for testing and validation scenarios.

.. option:: --simulated_generation_time_mode <MODE>

   Mode for simulating LLM generation time. Available modes: measured, fixed, random.
   Default: measured

.. option:: --noise

   Enable noise augmentation with tool augmentation and environment events configs.
   This adds realistic variability to scenario execution.

.. option:: --max_concurrent_scenarios <NUMBER>

   Maximum number of concurrent scenarios to run. If not specified, automatically
   sets based on the number of CPUs available.

Output Configuration
~~~~~~~~~~~~~~~~~~~~

.. option:: --output_dir <DIRECTORY>

   Directory to dump the scenario states and logs.

JSON Configuration
~~~~~~~~~~~~~~~~~~

.. option:: --kwargs <JSON>

   Additional keyword arguments as a JSON string to pass to the scenario
   initialization function. Default: {}

.. option:: --scenario_kwargs <JSON>

   Additional keyword arguments as a JSON string to pass to initialize the scenario.
   Default: {}

are-run CLI
-----------

The main scenario runner for executing individual scenarios.

.. click:: are.simulation.main:main
   :prog: are-run
   :nested: full

Run Usage Examples
~~~~~~~~~~~~~~~~~~

Run a scenario by ID::

    are-run --scenario-id example_scenario --model Llama-3.1-70B-Instruct --provider llama-api

Run scenarios from JSON files::

    are-run --scenario-file scenario1.json --scenario-file scenario2.json --model Llama-3.1-70B-Instruct --provider llama-api

Run with custom output directory and oracle mode::

    are-run --scenario-id test_scenario --model Llama-3.1-70B-Instruct --provider llama-api --oracle --output_dir ./results

are-benchmark CLI
-----------------

The benchmark runner for evaluating your agent against a datasets (e.g. Gaia2).

**For comprehensive documentation, examples, and best practices, see:** :doc:`../user_guide/benchmarking`

are-gui CLI
-----------

The GUI server for interactive scenario management and execution.

.. click:: are.simulation.gui.cli:main
   :prog: are-gui
   :nested: full

GUI Usage Examples
~~~~~~~~~~~~~~~~~~

Start GUI server on default port::

    are-gui --model Llama-3.1-70B-Instruct --provider llama-api

Start with custom hostname and port::

    are-gui --hostname 0.0.0.0 --port 8080 --model Llama-3.1-70B-Instruct --provider llama-api

Start with SSL support::

    are-gui --certfile cert.pem --keyfile key.pem --model Llama-3.1-70B-Instruct --provider llama-api

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Parameter Conflicts**

If you see errors about conflicting parameters, make sure you're not mixing old and new parameter names::

    # Wrong - mixing old and new names
    are-run --scenario-id test --scenario_id backup

    # Correct - use consistent naming
    are-run --scenario-id test backup

**Model Provider Issues**

Ensure your model provider is correctly specified::

    # For OpenAI models
    are-run --model gpt-4 --provider openai

    # For Anthropic models
    are-run --model claude-3-sonnet --provider anthropic

**Dataset Loading Issues**

For Hugging Face datasets, make sure to specify the split::

    # Wrong - missing split
    are-benchmark run --hf-dataset my_dataset

    # Correct - with split specified
    are-benchmark run --hf-dataset my_dataset --hf-split test

Getting Help
~~~~~~~~~~~~

Use the ``--help`` flag with any CLI command to see detailed usage information::

    are-run --help
    are-benchmark --help
    are-gui --help

For specific subcommands::

    are-benchmark run --help
    are-benchmark judge --help
