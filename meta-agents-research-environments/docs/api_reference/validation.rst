..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`shield-check` Judge System for Scenario Validation
============================================================

The Agents Research Environments judge system provides comprehensive validation capabilities for evaluating agent performance against ground truth oracle scenarios.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

:octicon:`info` Overview
------------------------

The judge system compares agent execution traces against oracle (ground truth) scenarios to determine success or failure. It operates in two main modes:

* **Online Validation**: Real-time validation during scenario execution
* **Offline Validation**: Post-execution validation using the ``judge`` command

The system uses a hierarchical approach with multiple types of judges that evaluate different aspects of agent behavior:

* **Event-level validation**: Comparing individual agent actions against oracle actions
* **Tool-level validation**: Verifying correct tool usage and parameters
* **Temporal validation**: Ensuring actions occur within acceptable time windows
* **Causal validation**: Verifying proper dependency ordering between actions

:octicon:`gear` Judge Architecture
----------------------------------

Base Judge Classes
~~~~~~~~~~~~~~~~~~

**BaseJudge**
   Abstract base class for all judges that compare agent and oracle event logs for a given scenario.

**ToolJudge**
   Base class for judges that compare individual tool calls between agent and oracle events.

**EventJudge**
   Base class for judges that compare agent and oracle events and determine if they match.

Judge State Management
~~~~~~~~~~~~~~~~~~~~~~

The judge system maintains state throughout validation:

.. code-block:: python

   @dataclass
   class BaseJudgeState:
       # Initialization flag
       initialized: bool = False

       # Turn tracking
       nb_turns: int = -1
       turn_idx: int = -1
       last_turn_success: bool = True
       last_turn_rationale: str = ""

      # Scenario data
      scenario_start_time: float = 0.0
      scenario_tasks: list[str] = field(default_factory=list)
      user_details: Contact | None = None

      # Oracle events
      turn_to_oracle_events: list[list[CompletedOracleEvent]] = field(
         default_factory=list
      )
      turn_to_oracle_graph: list[dict[str, list[str]]] = field(default_factory=list)
      oracle_event_id_to_turn_idx: dict[str, int] = field(default_factory=dict)

      # Agent events
      turn_to_agent_events: list[list[CompletedEvent]] = field(default_factory=list)

:octicon:`workflow` Judge Types
-------------------------------

GraphPerEventJudge
~~~~~~~~~~~~~~~~~~

The primary judge implementation that performs comprehensive validation by:

1. **Preliminary Checks**: Verifies tool call counts match between agent and oracle
2. **Event Matching**: Attempts to match each oracle event with an agent event
3. **Causality Verification**: Ensures proper dependency ordering
4. **Tool Validation**: Uses specialized tool judges for event comparison

**Key Features:**

* Topological ordering of oracle events based on dependencies
* Support for extra ``send_message_to_user`` calls from agents
* Detailed failure reporting with specific mismatch information

**Usage Example:**

.. code-block:: bash

   # Run judge mode for oneline validation
   uvx --from meta-agents-research-environments are-benchmark run -d /path/to/scenarios --limit 10

   # Run judge mode for offline validation
   uvx --from meta-agents-research-environments are-benchmark judge -d /path/to/scenarios --limit 10

ScriptedGraphPerEventJudge
~~~~~~~~~~~~~~~~~~~~~~~~~~

A specialized implementation of the GraphPerEventJudge that uses purely deterministic, scripted validation without LLM-based soft validation.
This judge is ideal for scenarios where you need predictable, reproducible validation results and have well-defined validation criteria.

**Core Characteristics:**

* **LLM-Free Operation**: Completely deactivates soft judges and relies only on hard, scripted checkers
* **Event-Specific Validation**: Uses custom validation rules per oracle event

**Key Configuration:**

The judge requires an ``event_id_to_checker_params`` mapping that defines specific validation rules for each oracle event:

.. code-block:: python

   # Example configuration for scripted validation
   scripted_config = ScriptedGraphPerEventJudgeConfig(
       event_id_to_checker_params={
           "oracle_send_email": [
               ToolCheckerParam(
                   arg_name="recipient",
                   checker_type=CheckerType.eq_checker,
                   tool_name="EmailApp__send_email"
               ),
               ToolCheckerParam(
                   arg_name="subject",
                   checker_type=CheckerType.contain_any_checker,
                   tool_name="EmailApp__send_email",
                   checker_args={"targets": ["urgent", "important"]}
               )
           ],
           "oracle_send_message": [
               ToolCheckerParam(
                   arg_name="content",
                   checker_type=CheckerType.contain_all_checker,
                   tool_name="MessagingApp__send_message",
                   checker_args={"targets": ["meeting", "2pm"]}
               )
           ]
       },
       extra_send_message_to_user_allowed=0,
       pre_event_tolerance_seconds=5.0,
       post_event_tolerance_seconds=20.0
   )

**ToolCheckerParam Structure:**

Each ``ToolCheckerParam`` defines validation rules for specific tool arguments:

* ``arg_name``: The argument name to validate (e.g., "content", "recipient")
* ``checker_type``: The type of validation to perform (see available checkers below)
* ``tool_name``: The full tool name including app prefix (e.g., "EmailApp__send_email")
* ``checker_args``: Additional parameters for the checker (optional)

**Available Checker Types:**

* ``CheckerType.eq_checker``: Exact equality comparison
* ``CheckerType.contain_any_checker``: Checks if argument contains any of the target strings
* ``CheckerType.contain_all_checker``: Checks if argument contains all target strings
* ``CheckerType.unordered_list_checker``: Set-based list comparison ignoring order
* ``CheckerType.path_checker``: Normalized file path comparison
* ``CheckerType.phone_number_checker``: Phone number format validation
* ``CheckerType.datetime_checker``: Date/time format validation

**Example Scenario Integration:**

.. code-block:: python

   class MyScenario(Scenario):
       def __init__(self):
           # Define checker parameters for each oracle event
           self.d_checker_params = {
               "oracle_book_restaurant": [
                   ToolCheckerParam(
                       arg_name="restaurant_name",
                       checker_type=CheckerType.contain_any_checker,
                       tool_name="BookingApp__make_reservation",
                       checker_args={"targets": ["Italian", "Chinese"]}
                   ),
                   ToolCheckerParam(
                       arg_name="party_size",
                       checker_type=CheckerType.eq_checker,
                       tool_name="BookingApp__make_reservation"
                   )
               ]
           }

       def initialize(self, **kwargs):
           super().initialize(**kwargs)
           self.judge = JudgeFactory()(
               ScriptedGraphPerEventJudgeConfig(
                   event_id_to_checker_params=self.d_checker_params,
                   extra_send_message_to_user_allowed=1
               )
           )

InContextJudge
~~~~~~~~~~~~~~

A baseline judge that uses LLM-based evaluation by providing all agent and oracle events in context to a model for comparison.

**Features:**

* LLM-powered validation using configurable evaluation criteria
* Support for tool-specific evaluation templates
* Time-based validation with configurable tolerance windows

:octicon:`tools` Tool Judges
----------------------------

The system includes three types of tool judges for validating individual tool calls:

HardToolJudge
~~~~~~~~~~~~~

Performs scripted, deterministic checks on tool arguments using predefined checkers:

**Available Checkers:**

* ``eq_checker``: Exact equality comparison
* ``unordered_list_checker``: Set-based list comparison
* ``datetime_checker``: Date/time format validation
* ``phone_number_checker``: Phone number format validation
* ``path_checker``: File path normalization and comparison
* ``contain_any_checker``: Substring containment validation
* ``contain_all_checker``: Multiple substring validation

**Example Configuration:**

.. code-block:: python

   # Hard validation for exact matches
   arg_to_checker_type = {
       "recipient": CheckerType.eq_checker,
       "file_paths": CheckerType.unordered_path_list_checker,
       "phone": CheckerType.phone_number_checker
   }

SoftToolJudge
~~~~~~~~~~~~~

The SoftToolJudge uses specialized LLM-based validation for semantic comparison of tool arguments.
It employs multiple targeted checkers, each optimized for specific validation scenarios.

**Architecture:**

The judge operates in a two-phase approach:

1. **Equality Pre-check**: Quick comparison to avoid unnecessary LLM calls when arguments are identical
2. **Specialized Soft Checkers**: LLM-powered validation using domain-specific checkers

**Available Soft Checkers:**

* ``content_checker``: Validates semantic equivalence of content against oracle and task context
* ``signature_checker``: Ensures proper user name/signature usage in communications
* ``sanity_checker``: Performs basic reasonableness checks on agent outputs
* ``placeholder_checker``: Detects and rejects placeholder text (e.g., "[User's Name]", "[Your Name]")
* ``cab_checker``: Validates cab/ride booking details against user address
* ``email_checker``: Specialized validation for email compositions
* ``message_checker``: Validates message content and formatting
* ``user_message_checker``: Validates user-directed messages for appropriateness
* ``event_checker``: Validates event details against context and user information
* ``tone_checker``: Ensures appropriate communication tone

**Key Features:**

* **Equality Pre-check**: Avoids LLM calls when agent and oracle arguments are identical
* **Subtask Extraction**: Automatically extracts relevant subtasks from the broader task context
* **Context-Aware Validation**: Uses user details, dates, and task context for validation
* **Placeholder Detection**: Built-in detection for common placeholder text patterns

**Validation Process:**

1. **Initial Setup**: Identifies arguments marked for LLM checking
2. **Equality Check**: Compares normalized agent and oracle arguments
3. **Context Preparation**: Extracts user details, dates, and subtasks as needed
4. **Checker Execution**: Runs configured soft checkers in sequence
5. **Result Aggregation**: Returns success only if all checkers pass

**Example Configuration:**

.. code-block:: python

    # Configure split soft judge with multiple checkers
    config = SplitSoftToolJudgeConfig(
        tool_name="send_email",
        arg_to_checker_type={
            "subject": CheckerType.llm_checker,
            "body": CheckerType.llm_checker,
            "recipient": CheckerType.eq_checker  # Hard check for exact match
        },
        soft_checker_types=[
            SoftCheckerType.placeholder_checker,  # Reject placeholder text
            SoftCheckerType.content_checker,      # Semantic content validation
            SoftCheckerType.tone_checker,         # Appropriate communication tone
            SoftCheckerType.signature_checker     # Proper user signature
        ],
        engine=llm_engine
    )

**Context Extraction:**

The judge automatically extracts relevant context for validation:

* **User Details**: Name and address from scenario user information
* **Temporal Context**: Event date/time formatted for validation
* **Task Context**: Extracts relevant subtasks using LLM-based extraction
* **Previous Tasks**: Maintains context from prior scenario steps

**Placeholder Detection:**

Built-in detection for common placeholder patterns:

.. code-block:: text

    Detected Placeholders (automatically rejected):
    - "[User's Name]", "[User Name]", "[User]"
    - "[Your Name]", "[My Name]"
    - "Best regards,\nYour Name"
    - "Best,\nYour Name"

MildToolJudge
~~~~~~~~~~~~~

Combines hard and soft validation approaches:

1. **Hard Validation**: Performs scripted checks first
2. **Soft Validation**: Falls back to LLM validation for remaining arguments

This approach provides both reliability (hard checks) and flexibility (soft checks).

:octicon:`clock` Temporal Validation
------------------------------------

The judge system includes sophisticated time-based validation:

Time Comparison Types
~~~~~~~~~~~~~~~~~~~~~

**EQUAL** (default)
   Agent event must occur within tolerance window around oracle time

**LESS_THAN**
   Agent event must occur before oracle time (plus post-tolerance)

**GREATER_THAN**
   Agent event must occur after oracle time (minus pre-tolerance)

**Configuration Options:**

.. code-block:: python

   # Time validation settings
   check_time_threshold_seconds = 30.0      # Minimum time gap to check
   pre_event_tolerance_seconds = 5.0        # Allowed time before oracle
   post_event_tolerance_seconds = 20.0      # Allowed time after oracle

Absolute vs Relative Time
~~~~~~~~~~~~~~~~~~~~~~~~~

**Absolute Time**
   Direct comparison against wall-clock time

**Relative Time**
   Comparison relative to parent event completion times

:octicon:`link` Causality Validation
------------------------------------

The judge system enforces proper dependency ordering:

**Dependency Graph**
   Oracle events include parent-child relationships

**Causality Rules**
   * All parent events must be matched before child events
   * Agent events must respect the same ordering constraints
   * Violations result in validation failure

**Example:**

.. code-block:: text

   Oracle Event Graph:
   A → B → D
   A → C → D

   Valid Agent Sequence: A, B, C, D (or A, C, B, D)
   Invalid Agent Sequence: B, A, C, D (B before A violates causality)

:octicon:`bug` Failure Types and Diagnostics
--------------------------------------------

The judge system provides detailed failure information:

ToolCallCountsFailure
~~~~~~~~~~~~~~~~~~~~~

Indicates mismatched tool usage counts between agent and oracle:

.. code-block:: text

   Agent and oracle counters do not match for the following tools:
   - Tool 'send_email': Agent count 2, Oracle count 1
   - Tool 'read_file': Agent count 0, Oracle count 1

EventComparisonFailure
~~~~~~~~~~~~~~~~~~~~~~

Specific event matching failures with detailed context:

* **CAUSALITY**: Dependency ordering violation
* **ALREADY_MATCHED**: Agent event already matched to another oracle event
* **TOOL_JUDGE_REJECT**: Tool-specific validation failed

OracleEventMatchingFailure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Comprehensive failure when no agent event matches an oracle event:

.. code-block:: text

   Failure: Agent did not perform the following oracle tool call:
   tool name: send_email
   tool args:
   - recipient: john@example.com
   - subject: Meeting Reminder
   - body: Don't forget about our 2pm meeting

   List of matching attempts:
   - Failure matching agent event (ID: evt_123) with oracle event (ID: oracle_456), reason: tool judge reject

:octicon:`terminal` Command Line Usage
--------------------------------------

Basic Judge Command
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run judge on local scenarios
   are-benchmark judge -d /path/to/scenarios

   # Run judge on Hugging Face dataset
   are-benchmark judge --hf are-benchmark/gaia2 --hf-split validation


Judge System LLM Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The judge system uses its own separate LLM engine for soft validation (semantic comparison of tool arguments). This LLM engine is independent of the model configuration you specify for your main agent, and can be customized for cost control and performance optimization.

**Default Judge Configuration**

.. code-block:: bash

   # Judge command - uses default judge model configuration
   uvx --from meta-agents-research-environments are-benchmark judge --hf are-benchmark/gaia2 --hf-split validation \
     --output_dir ./judge_results

**Custom Judge Model Configuration**

You can specify custom judge model settings to control costs and performance:

.. code-block:: bash

   # Use custom judge model and provider
   uvx --from meta-agents-research-environments are-benchmark judge --hf are-benchmark/gaia2 --hf-split validation \
     --judge_model custom-juge-model --judge_provider custom-provider \
     --output_dir ./judge_results

   # Use different provider for judge vs main agent
   uvx --from meta-agents-research-environments are-benchmark run --hf are-benchmark/gaia2 --hf-split test \
     --model custom-model_model --model_provider custom-model-provider\
     --judge_model custom-juge-model --judge_provider custom-judge-model-provider

   # Use custom endpoint for judge model
   uvx --from meta-agents-research-environments are-benchmark judge --hf are-benchmark/gaia2 --hf-split validation \
     --judge_model custom-judge-model --judge_provider custom-provider \
     --judge_endpoint http://localhost:8000

**Judge Model Configuration Options**

**--judge_model**
   Model to use for the judge system validation. Use a capable model for best evaluation quality.

   * Default: "meta-llama/Meta-Llama-3.3-70B-Instruct"
   * Examples: "gpt-4", "claude-3-opus", "meta-llama/llama3-70b-instruct"

**--judge_provider**
   Provider for the judge model. If not specified, uses the same provider as the main model.

   * Supports all LiteLLM providers: openai, anthropic, huggingface, llama-api, etc.
   * Allows separate billing control from your main agent model

**--judge_endpoint**
   Custom endpoint URL for the judge model (optional).

   * Useful for local deployments or custom inference servers
   * Must be OpenAI-compatible API format

.. note::
   **Reproducible Results**: For consistent and reproducible evaluation results, use **llama3.3-70B** as the judge model.

.. note::
   **Judge LLM Independence**: The judge system uses its own configurable LLM engine, which is separate from and independent of the model configuration you specify for your main agent (--model, --model_provider, etc.). The judge's LLM is used for:

   * **SoftToolJudge**: Semantic comparison of tool arguments when exact matching isn't sufficient
   * **InContextJudge**: LLM-based evaluation of entire agent traces

   Hard validation (exact matching, scripted checks) does not require LLM inference and runs regardless of any model configuration.

:octicon:`gear` Configuration
-----------------------------

Judge Configuration Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GraphPerEventJudgeConfig**
   Primary judge configuration with tool-specific validation rules

.. note::
   **Reproducible Results**: For consistent and reproducible evaluation results, use default ``GraphPerEventJudgeConfig`` config.

**InContextJudgeConfig**
   Baseline LLM-based judge configuration with evaluation criteria

**AgentEventJudgeConfig**
   Agent event validation with time tolerance settings

**Tool Judge Configs**
   * ``HardToolJudgeConfig``: Scripted validation rules
   * ``SoftToolJudgeConfig``: LLM validation settings
   * ``MildToolJudgeConfig``: Combined validation approach

Example Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Configure judge for email scenario validation
   judge_config = GraphPerEventJudgeConfig(
       # Time validation
       check_time_threshold_seconds=30.0,
       pre_event_tolerance_seconds=5.0,
       post_event_tolerance_seconds=20.0,

       # Checker types for each tool
       per_tool_arg_to_checker_type={
           "send_email": {
               "recipient": CheckerType.eq_checker,
               "subject": CheckerType.llm_checker,
               "body": CheckerType.llm_checker
           },
           "read_file": {
               "file_path": CheckerType.path_checker
           }
       },

       # Soft checkers
       per_tool_soft_checker_types={
           "send_email": [
               SoftCheckerType.placeholder_checker,
               SoftCheckerType.content_checker,
               SoftCheckerType.tone_checker,
               SoftCheckerType.signature_checker
            ],
       },

       # Allow extra user messages
       extra_send_message_to_user_allowed=1
   )

Next Steps
----------

For comprehensive benchmarking workflows that incorporate judge validation, see :doc:`../user_guide/benchmarking`.
Ready to develop custom scenarios? Continue to :doc:`../tutorials/scenario_development` for detailed guidance on creating scenarios with proper oracle events for validation.


Judge System Classes
--------------------

.. autoclass:: are.simulation.validation.base.BaseJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judge.GraphPerEventJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judge.InContextJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Judge States
------------

.. autoclass:: are.simulation.validation.base.BaseJudgeState
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.validation.judge_states.GraphPerEventJudgeState
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


.. autoclass:: are.simulation.validation.judge_states.InContextJudgeState
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Event Judge Classes
-------------------

.. autoclass:: are.simulation.validation.base.EventJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.event_judge.EnvUserEventJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.event_judge.AgentEventJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Tool Judge Classes
------------------

.. autoclass:: are.simulation.validation.base.ToolJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.tool_judge.HardToolJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.tool_judge.SoftToolJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.tool_judge.MildToolJudge
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Configs Classes
---------------

.. autoclass:: are.simulation.validation.configs.BaseJudgeConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.configs.GraphPerEventJudgeConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.configs.ScriptedGraphPerEventJudgeConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.configs.InContextJudgeConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:


Factory classes
---------------

.. autoclass:: are.simulation.validation.factory.JudgeFactory
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Judgment Classes
----------------
.. autoclass:: are.simulation.validation.judgment.Failure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.ToolCallCountsFailure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.EventComparisonFailureType
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.EventComparisonFailure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.OracleEventMatchingFailure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.EnvOracleMatchingFailure
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.validation.judgment.Judgment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
