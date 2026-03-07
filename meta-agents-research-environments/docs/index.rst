..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`home` Welcome to Meta Agents Research Environments
============================================================

**Meta Agents Research Environments (ARE)** is a dynamic simulation research platform for training and evaluating AI agents on complex, multi-step tasks that mirror real-world challenges. Unlike static benchmarks, ARE creates evolving environments where agents must adapt their strategies as new information emerges and conditions change over time. In particular, ARE runs the :doc:`Gaia2 Benchmark <user_guide/gaia2_evaluation>`, a follow-up to `Gaia <https://arxiv.org/abs/2311.12983>`_, evaluating a broader range of agent capabilities.

:octicon:`zap` What Can ARE Do?
-------------------------------

:octicon:`sync` **Dynamic Simulations**
    Create realistic scenarios that evolve over minutes, hours, or days - simulating complex workflows that require persistent reasoning and adaptation.

:octicon:`beaker` **Agent Evaluation**
    Test AI agents on multi-step tasks with comprehensive benchmarking tools, including the Gaia2 benchmark with 800 scenarios across 10 universes.

:octicon:`apps` **Interactive Applications**
    Agents interact with realistic apps like email, calendars, file systems, and messaging - each with domain-specific data and behaviors.

:octicon:`graph` **Research & Benchmarking**
    Systematic evaluation with parallel execution, multiple model support, and automatic result collection for the research community.

:octicon:`telescope` Next Steps
-------------------------------

.. grid:: 3

   .. grid-item-card:: :doc:`quickstart`

      For a step-by-step guide to use ARE to evaluation your agents, see the :doc:`quickstart` page.

   .. grid-item-card:: :doc:`foundations/index`

      **Learn More**: Dive deeper into the core concepts of agents, environments, apps, events, and scenarios.

   .. grid-item-card:: `Demo <https://huggingface.co/spaces/meta-agents-research-environments/demo>`_

      `Try the ARE Demo on Hugging Face <https://huggingface.co/spaces/meta-agents-research-environments/demo>`_ — Play around with the agent platform directly in your browser, no installation required!

   .. grid-item-card:: :doc:`user_guide/gaia2_evaluation`

      Build and evaluate your agents on the Gaia2 benchmark, a comprehensive suite of 800 dynamic scenarios across 10 universes.

   .. grid-item-card:: :doc:`tutorials/working_with_scenarios`

      Explore other ways to create and work with scenarios, including using the CLI, Python, and JSON.

   .. grid-item-card:: `Explore Code <https://github.com/facebookresearch/meta-agents-research-environments>`_

      Explore the ARE code on GitHub.

:octicon:`book` Documentation Structure
---------------------------------------

.. toctree::
   :hidden:

   quickstart
   foundations/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/docker
   user_guide/understanding_ui
   user_guide/llm_configuration
   user_guide/benchmarking
   user_guide/gaia2_evaluation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
   tutorials/working_with_scenarios
   tutorials/scenario_development
   tutorials/apps_tutorial

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference/index
   api_reference/events
   api_reference/environment
   api_reference/apps
   api_reference/scenarios
   api_reference/json_format
   api_reference/agents
   api_reference/mcp_app
   api_reference/are_simulation_mcp_server
   api_reference/cli_reference
   api_reference/validation

Indices and tables
==================

* :ref:`genindex`
