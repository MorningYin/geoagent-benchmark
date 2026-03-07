..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


API Reference
=============

This section provides detailed API documentation for Meta Agents Research Environments core components.
The API is organized into four main areas:

* **Environment**: Core simulation orchestration and time management
* **Apps**: Building blocks for creating interactive applications
* **Scenarios**: Framework for defining and running simulation tasks
* **Agents**: Base classes and utilities for agent implementation


Overview
--------

Meta Agents Research Environments provides a comprehensive API for creating dynamic simulations.
The main components work together to create realistic, interactive environments
where AI agents can perform complex tasks.

**Key Design Principles**

* **Modularity**: Components are designed to work independently and together
* **Extensibility**: Easy to add new apps, scenarios, and agent types
* **Type Safety**: Comprehensive type hints throughout the codebase
* **Documentation**: Detailed docstrings for all public APIs

**Architecture Overview**

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │     Agents      │    │   Scenarios     │    │      Apps       │
   │                 │    │                 │    │                 │
   │ • BaseAgent     │◄──►│ • Scenario      │◄──►│ • App           │
   │ • AgentConfig   │    │ • Events        │    │ • Tools         │
   │ • Tools         │    │ • Validation    │    │ • State         │
   └─────────────────┘    └─────────────────┘    └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                            ┌─────────────────┐
                            │  Environment    │
                            │                 │
                            │ • Event Loop    │
                            │ • State Mgmt    │
                            │ • Coordination  │
                            └─────────────────┘

Quick Reference
---------------

**Common Imports**

.. code-block:: python

   # Core components
   from are.simulation.environment import Environment
   from are.simulation.apps.app import App
   from are.simulation.scenarios.scenario import Scenario
   from are.simulation.agents.default_agent.base_agent import BaseAgent

   # Decorators and utilities
   from are.simulation.core.decorators import event_registered
   from are.simulation.core.events import Event, Action
   from are.simulation.core.types import EventType

**Basic Usage Pattern**

.. code-block:: python

   # Create an environment
   env = Environment()

   # Add apps
   email_app = EmailApp()
   env.register_app(email_app)

   # Create and run scenario
   scenario = MyScenario()
   result = scenario.run(env, agent)

**Event Registration**

.. code-block:: python

   class MyApp(App):
       @event_registered
       def my_action(self, param: str) -> str:
           """This method can be called by agents as a tool."""
           return f"Processed: {param}"

For detailed examples and tutorials, see the :doc:`../tutorials/index`.

Getting Help
------------

* **Type Hints**: All APIs include comprehensive type annotations
* **Docstrings**: Detailed documentation in code
* **Examples**: See scenario implementations for usage patterns
* **Community**: Check the contributing guide for support channels

The API documentation is automatically generated from the source code, ensuring it stays up-to-date with the latest implementation.
