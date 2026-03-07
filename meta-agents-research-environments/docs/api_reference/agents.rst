..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.

    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`dependabot` Agents API
================================

Agents are the AI entities that interact with the Agents Research Environments to complete tasks.
This section documents the core Agent API and related classes for creating and customizing
agent behavior.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Overview
--------

Agents in Meta Agents Research Environments are responsible for:

* **Task Execution**: Understanding and completing assigned tasks
* **Tool Usage**: Interacting with apps through their exposed APIs
* **Reasoning**: Making decisions based on available information
* **Learning**: Adapting behavior based on feedback and results

The agent system is built around a ReAct (Reasoning + Acting)
framework that allows agents to think, act, and observe in iterative cycles.

Creating Custom Agents
----------------------

Agents used on the CLI are built through `are/simulation/agents/default_agent/agent_factory.py`.
The ``BaseAgent`` class describes the "inner loop" of the agent - the core reasoning and action execution logic.
The environment event reaction loop (handling user messages, notifications, and the overall agent lifecycle)
is implemented in `are/simulation/agents/default_agent/are_simulation_main.py`.

When creating custom agents, you should focus on configuring the ``BaseAgent`` and should not need to
modify the environment event handling in ``are_simulation_main.py``.

To create a custom agent, inherit from the base ``BaseAgent`` class and customize the behavior:

Base Agent Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.agents.default_agent.base_agent import BaseAgent, ConditionalStep
   from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor

   class MyCustomAgent(BaseAgent):
       def __init__(self, llm_engine, **kwargs):
           # Custom system prompts
           system_prompts = {
               "system_prompt": """You are a helpful AI assistant specialized in email management.
               Your goal is to help users organize and respond to emails efficiently.

               Available tools: <<tool_descriptions>>

               Always think step by step and explain your reasoning."""
           }

           # Custom action executor
           action_executor = JsonActionExecutor(llm_engine=llm_engine)

           # Custom conditional steps
           pre_steps = [
               ConditionalStep(
                   condition=lambda agent: agent.iterations == 0,
                   function=self.initial_setup,
                   name="initial_setup"
               )
           ]

           super().__init__(
               llm_engine=llm_engine,
               system_prompts=system_prompts,
               action_executor=action_executor,
               conditional_pre_steps=pre_steps,
               max_iterations=15,
               **kwargs
           )

           self.name = "email_specialist_agent"

       def initial_setup(self):
           """Custom initialization logic."""
           self.logger.info("Initializing email specialist agent...")
           # Add custom setup logic here

Agent Customization Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Custom System Prompts**

.. code-block:: python

   system_prompts = {
       "system_prompt": """You are an expert data analyst.

       Your capabilities include:
       - Analyzing datasets and identifying patterns
       - Creating visualizations and reports
       - Providing statistical insights

       Available tools: <<tool_descriptions>>

       Always provide detailed explanations for your analysis.""",
   }


**Conditional Steps**

.. code-block:: python

   def create_monitoring_agent():
       def check_progress(agent):
           """Check if agent is making progress."""
           return agent.iterations > 5 and agent.planning_counter == 0

       def ask_user_help(agent):
           """Ask the user to help when agent is stuck."""
           agent.send_message_to_user(
               "I might be stuck. Can you help me with this task?"
           )

       monitoring_step = ConditionalStep(
           condition=check_progress,
           function=ask_user_help,
           name="progress_monitoring"
       )

       return BaseAgent(
           llm_engine=my_llm_engine,
           conditional_post_steps=[monitoring_step]
       )

**Custom Termination Conditions**

.. code-block:: python

   from are.simulation.agents.default_agent.base_agent import TerminationStep

   def success_based_termination(agent):
       """Terminate when task is successfully completed."""
       # Check if the last action was successful
       last_log = agent.get_last_log_of_type(ObservationLog)
       if last_log and "success" in last_log.content.lower():
           return True
       return agent.iterations >= agent.max_iterations

   def cleanup_on_termination(agent):
       """Clean up resources when terminating."""
       agent.logger.info("Task completed, cleaning up...")
       # Add cleanup logic here
       return "Task completed successfully"

   custom_termination = TerminationStep(
       condition=success_based_termination,
       function=cleanup_on_termination,
       name="success_termination"
   )

Agent Configuration
-------------------


Tool Integration
~~~~~~~~~~~~~~~~

Sometimes agents need to be configured with tools, in addition to the apps provided by the environment. Think
about an agent that knows how to browse the web, or needs to execute code, in addition to the apps provided by the environment.

.. code-block:: python

   from are.simulation.tools import Tool

   class CustomTool(Tool):
       name = "custom_calculator"
       description = "Performs custom mathematical calculations"

       def __call__(self, a: float, b: float) -> str:
           # Implement custom addition logic
           try:
               result = a + b
               return f"Result: {result}"
           except Exception as e:
               return f"Error: {str(e)}"

   # Add custom tools to agent
   custom_tools = {"custom_calculator": CustomTool()}

   agent = BaseAgent(
       llm_engine=llm_engine,
       tools=custom_tools
   )

Logging and Monitoring
----------------------

Custom Log Callbacks
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def custom_log_callback(log: BaseAgentLog):
       """Custom logging callback for monitoring agent behavior."""
       if isinstance(log, ErrorLog):
           print(f"ERROR: {log.exception}")
       elif isinstance(log, TaskLog):
           print(f"NEW TASK: {log.content}")
       elif isinstance(log, StepLog):
           print(f"STEP {log.iteration}: Starting...")

   agent = BaseAgent(
       llm_engine=llm_engine,
       log_callback=custom_log_callback
   )

Log Analysis
~~~~~~~~~~~~

.. code-block:: python

   def analyze_agent_performance(agent: BaseAgent):
       """Analyze agent performance from logs."""
       logs = agent.get_agent_logs()

       # Count different log types
       log_counts = {}
       for log in logs:
           log_type = type(log).__name__
           log_counts[log_type] = log_counts.get(log_type, 0) + 1

       # Calculate success rate
       error_count = log_counts.get('ErrorLog', 0)
       total_steps = log_counts.get('StepLog', 0)
       success_rate = (total_steps - error_count) / total_steps if total_steps > 0 else 0

       return {
           'total_steps': total_steps,
           'errors': error_count,
           'success_rate': success_rate,
           'log_distribution': log_counts
       }


For more examples of agent implementation, see the built-in agents in the ``are.simulation.agents`` module.


Core Agent Classes
------------------

.. automodule:: are.simulation.agents.default_agent.base_agent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Base Agent
~~~~~~~~~~

.. autoclass:: are.simulation.agents.default_agent.base_agent.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Key Methods
~~~~~~~~~~~

**Initialization and Setup**

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.initialize

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.init_tools

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.is_initialized

**Execution Control**

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.run

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.step

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.stop

**Logging and History**

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.get_agent_logs

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.append_agent_log

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.build_history_from_logs

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.replay

**Communication**

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.send_message_to_user

.. automethod:: are.simulation.agents.default_agent.base_agent.BaseAgent.log_error

Configuration Classes
---------------------

.. autoclass:: are.simulation.agents.default_agent.base_agent.ConditionalStep
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.agents.default_agent.base_agent.TerminationStep
   :members:
   :undoc-members:
   :no-index:


Agent Logging System
--------------------

.. automodule:: are.simulation.agents.agent_log
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: are.simulation.agents.multimodal
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ManifoldAttachment
   :no-index:

Multimodal Classes
~~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.agents.multimodal.Attachment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Base Log Classes
~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.agents.agent_log.BaseAgentLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.TaskLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.StepLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.ObservationLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.ActionLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.LLMOutputThoughtActionLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.agent_log.ErrorLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Action Executors
~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.agents.default_agent.tools.action_executor.BaseActionExecutor
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.agents.default_agent.tools.json_action_executor.JsonActionExecutor
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Action Types
~~~~~~~~~~~~

.. autoclass:: are.simulation.agents.default_agent.tools.action_executor.AgentAction
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.agents.default_agent.tools.action_executor.ParsedAction
   :members:
   :undoc-members:
   :no-index:
