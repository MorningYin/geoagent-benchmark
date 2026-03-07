..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`file-code` Scenario JSON Format
=========================================

The Agents Research Environments scenarios can be exported and imported using a structured JSON format defined by Pydantic models.
All Gaia2 scenarios are available in this standardized JSON format.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

:octicon:`info` Overview
------------------------

The scenario format is built around the ``ExportedTrace`` structure, which contains all the information needed to define, execute, and validate a scenario.
This includes metadata, applications, events, and validation criteria.

:octicon:`package` Core Structure
---------------------------------

ExportedTrace
~~~~~~~~~~~~~

The root structure for all Meta Agents Research Environments scenarios:

.. code-block:: python

   class ExportedTraceBase(BaseModel):
       world_logs: list[str] = []
       apps: list[ExportedApp] = []
       events: list[ExportedEvent | ExportedOracleEvent] = []
       completed_events: list[ExportedCompletedEvent] = []
       version: str
       context: str | None = None
       augmentation: dict | None = None

   class ExportedTrace(ExportedTraceBase):
      metadata: ExportedTraceMetadata

Fields:

* ``metadata``: Comprehensive metadata about the scenario
* ``world_logs``: Agent and environment log from the simulation
* ``apps``: List of applications available in the scenario
* ``events``: Scheduled events
* ``completed_events``: Events that have been executed
* ``version``: Scenario format version
* ``context``: Optional additional context information
* ``augmentation``: Optional additional data for augmentation

:octicon:`tag` Metadata Components
----------------------------------

The metadata section contains three main categories of information:

Definition Metadata
~~~~~~~~~~~~~~~~~~~

Core scenario definition:

.. code-block:: python

   class ExportedTraceDefinitionMetadata(BaseModel):
      scenario_id: str
      seed: int | None = None
      duration: float | None = None
      time_increment_in_seconds: int | None = None
      start_time: float | None = None
      run_number: int | None = None  # Run number for multiple runs of the same scenario
      hints: list[ExportedHint] = []
      config: str | None = None
      has_a2a_augmentation: bool = False
      has_tool_augmentation: bool = False
      has_env_events_augmentation: bool = False
      has_exception: bool = False
      exception_type: str | None = None
      exception_message: str | None = None
      tags: list[str] | None = None
      hf_metadata: ExportedHugging FaceMetadata | None = None

:octicon:`key` Fields:

* ``scenario_id``: Unique identifier for the scenario
* ``seed``: Random seed for reproducibility
* ``duration``: Maximum scenario runtime in seconds
* ``time_increment_in_seconds``: Time increment used to run the scenario
* ``start_time``: Scenario start timestamp (Unix timestamp)
* ``hints``: List of hints to guide agent behavior
* ``tags``: Capability tags for categorization (e.g., "execution", "search")

Simulation Metadata
~~~~~~~~~~~~~~~~~~~

Information about the execution environment:

.. code-block:: python

   class ExportedTraceSimulationMetadata(BaseModel):
       agent_id: str
       model_id: str

:octicon:`key` Fields:

* ``agent_id``: Identifier of the agent used for execution
* ``model_id``: Model identifier for tracking and reproducibility

Annotation Metadata
~~~~~~~~~~~~~~~~~~~

Human validation and quality assurance:

.. code-block:: python

   class ExportedTraceAnnotationMetadata(BaseModel):
      annotation_id: str | None = None
      annotator: str | None = None
      validation_decision: str | None = None
      comment: str | None = None
      date: float = 0.0

:octicon:`key` Fields:

* ``validation_decision``: Human validation status ("valid", "invalid", "needs_review")
* ``comment``: Additional notes or feedback from human reviewers
* ``annotator``: Person who validated the scenario

:octicon:`apps` Applications
----------------------------

Scenarios include multiple applications that agents can interact with:

ExportedApp Structure
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedApp(BaseModel):
       name: str
       class_name: str
       app_state: dict[str, Any]

:octicon:`key` Fields:

* ``name``: Human-readable application name
* ``class_name``: Python class implementing the application
* ``app_state``: Initial state data for the application

Common Application Types
~~~~~~~~~~~~~~~~~~~~~~~~

**Email Applications**
   * Sending and receiving emails
   * Email filtering and organization tools

**Messaging Systems**
   * Real-time communication platforms
   * Chat and collaboration tools

**File Systems**
   * File browsers and managers
   * Document storage and retrieval systems
   * Search and file organization tools


:octicon:`zap` Events and Actions
---------------------------------

Events define what happens during scenario execution:

Event Structure
~~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedEvent(BaseModel):
       class_name: str
       event_type: str
       event_time: float | None
       event_id: str
       dependencies: list[str]
       event_relative_time: float | None
       action: ExportedAction | None = None

:octicon:`key` Fields:

* ``class_name``: Event type class (e.g., "Event", "ConditionCheckEvent")
* ``event_type``: Category of event ("AGENT", "ENV", "USER")
* ``event_time``: Timestamp when the event occurs
* ``event_id``: Unique event identifier
* ``dependencies``: IDs of prerequisite events that must complete first
* ``event_relative_time``: Relative time to parent events
* ``action``: The action to be executed when the event fires

Event Categories
~~~~~~~~~~~~~~~~

**AGENT Events**
   Actions initiated by the AI agent during execution

**ENV Events**
   Environmental changes scheduled by the scenario

**USER Events**
   Simulated user interactions and inputs

Action Structure
~~~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedAction(BaseModel):
       action_id: str
       app: str | None = None
       function: str | None = None
       operation_type: str | None = None
       args: list[ExportedActionArg] | None = None

:octicon:`key` Fields:

* ``action_id``: Unique action identifier
* ``app``: Target application name
* ``function``: Tool name
* ``operation_type``: Type of operation ("READ", "WRITE", "SEARCH", etc.)
* ``args``: List of arguments to pass to the tool

Action Arguments
~~~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedActionArg(BaseModel):
       name: str
       value: str | None = None
       value_type: str | None = None

:octicon:`key` Fields:

* ``name``: Parameter name
* ``value``: Parameter value (can be any JSON-serializable type)
* ``value_type``: Type hint for the parameter

:octicon:`light-bulb` Hints and Guidance
----------------------------------------

Hints provide guidance to agents during execution:

Hint Structure
~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedHint(BaseModel):
       hint_type: str
       content: str
       associated_event_id: str | None

:octicon:`key` Fields:

* ``hint_type``: Category of hint ("task", "context", "warning", "tip")
* ``content``: The actual hint text
* ``associated_event_id``: Optional reference to a related event

Hint Types
~~~~~~~~~~

**Task Hints**
   Direct guidance about the main objective

**Environment Hints**
   Contextual information about the environment

:octicon:`shield` Oracle Events
-------------------------------

Oracle events define expected agent behaviors for validation and testing:

Oracle Event Structure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ExportedOracleEvent(ExportedEvent):
       event_time_comparator: str

:octicon:`key` Fields:

* ``event_time_comparator``: Comparator for event time validation ("LESS_THAN", "GREATER_THAN", "EQUAL")


:octicon:`database` Complete Example
------------------------------------

Here's a complete example of a scenario JSON structure:

.. code-block:: json

   {
     "metadata": {
       "scenario_id": "email_apology_001",
       "seed": 12345,
       "duration": 300.0,
       "start_time": 1640995200,
       "hints": [
         {
           "hint_type": "task",
           "content": "Check your email for client complaints",
           "associated_event_id": null
         }
       ],
       "tags": ["email", "communication", "customer-service"]
     },
     "world_logs": [],
     "apps": [
       {
         "name": "EmailApp",
         "class_name": "are.simulation.apps.email.EmailApp",
         "app_state": {
           "inbox": [
             {
               "from": "client@company.com",
               "subject": "Project Delay Concerns",
               "body": "We're concerned about the project timeline...",
               "timestamp": 1640995100
             }
           ]
         }
       }
     ],
     "events": [
       {
         "class_name": "ScheduledEvent",
         "event_type": "ENV",
         "event_time": 1640995260,
         "event_id": "reminder_001",
         "dependencies": [],
         "action": {
           "action_id": "notify_001",
           "app": "NotificationApp",
           "function": "show_reminder",
           "operation_type": "NOTIFY",
           "args": [
             {
               "name": "message",
               "value": "Don't forget to respond to the client",
               "type": "str"
             }
           ]
         }
       }
     ],
     "completed_events": [],
     "version": "2.0",
     "context": "Customer service scenario focusing on professional communication"
   }

:octicon:`gear` Schema Validation
---------------------------------

The JSON format is validated using Pydantic models, which provide:

**Type Safety**
   Ensures all fields have correct types

**Data Validation**
   Validates field values and constraints

**Serialization**
   Consistent conversion between Python objects and JSON

**Error Reporting**
   Clear error messages for invalid data

:octicon:`tools` Working with the Format
----------------------------------------

Programmatic Access
~~~~~~~~~~~~~~~~~~~

Use the ``JsonScenarioImporter`` to load and work with scenarios from JSON files:

.. code-block:: python

   from are.simulation.data_handler.importer import JsonScenarioImporter

   # Initialize the importer
   importer = JsonScenarioImporter()

   # Load scenario from JSON file
   with open('scenario.json', 'r') as f:
       json_content = f.read()

   # Import the scenario
   scenario, completed_events, world_logs = importer.import_from_json(
       json_str=json_content,
       load_completed_events=True
   )

   # Access scenario properties
   print(f"Scenario ID: {scenario.scenario_id}")
   print(f"Duration: {scenario.duration} seconds")
   print(f"Number of hints: {len(scenario.hints)}")
   print(f"Tags: {scenario.tags}")
   print(f"Number of completed events: {len(completed_events)}")

Filtering Applications
~~~~~~~~~~~~~~~~~~~~~~

Control which applications are loaded when importing scenarios:

.. code-block:: python

   from are.simulation.data_handler.importer import JsonScenarioImporter

   importer = JsonScenarioImporter()

   # Skip specific applications
   scenario, _, _ = importer.import_from_json(
       json_str=json_content,
       apps_to_skip=["EmailClientApp", "SlackApp"]
   )

   # Or keep only specific applications
   scenario, _, _ = importer.import_from_json(
       json_str=json_content,
       apps_to_keep=["FileSystemApp", "BrowserApp"]
   )

Benchmark Scenarios
~~~~~~~~~~~~~~~~~~~

Import scenarios specifically for benchmarking with additional metadata handling:

.. code-block:: python

   from are.simulation.data_handler.importer import JsonScenarioImporter

   importer = JsonScenarioImporter()

   # Import as benchmark scenario
   benchmark_scenario, completed_events, world_logs = importer.import_from_json_to_benchmark(
       json_str=json_content,
       load_completed_events=True
   )

   # Benchmark scenarios include additional Hugging Face metadata handling
   if hasattr(benchmark_scenario, 'hf_metadata'):
       print(f"Hugging Face dataset: {benchmark_scenario.hf_metadata.dataset}")

Running Scenarios
~~~~~~~~~~~~~~~~~

Run scenarios from local JSON files:

.. code-block:: bash

   # Directory of scenarios
   uvx --from meta-agents-research-environments are-benchmark -d /path/to/scenarios/ -a default
   # JSONL file with multiple scenarios
   are-benchmark -d /path/to/scenarios.jsonl -a default

Run scenarios from Hugging Face:

.. code-block:: bash

   # Basic usage
   uvx --from meta-agents-research-environments are-benchmark --hf dataset_name --hf-split split_name
   # With specific parameters
   uvx --from meta-agents-research-environments are-benchmark --hf meta-agents-research-environments/gaia2 \
            --hf-split validation \
            --limit 5 \
            --agent default \
            --model your-model \
            --model_provider your-provider

Scenario Export
~~~~~~~~~~~~~~~

Use the ``JsonScenarioExporter`` to export scenarios and environments to JSON:

.. code-block:: python

   from are.simulation.data_handler.exporter import JsonScenarioExporter
   from are.simulation.environment import Environment

   # Initialize the exporter
   exporter = JsonScenarioExporter()

   # Export scenario and environment to JSON string
   json_output = exporter.export_to_json(
       env=environment,
       scenario=scenario,
       scenario_id="my_scenario_001",
   )

   # Save to file
   with open('exported_scenario.json', 'w') as f:
       f.write(json_output)

Export to File
~~~~~~~~~~~~~~

Directly export scenarios to files with automatic naming:

.. code-block:: python

   from are.simulation.data_handler.exporter import JsonScenarioExporter

   exporter = JsonScenarioExporter()

   # Export directly to file
   success, file_path = exporter.export_to_json_file(
       env=environment,
       scenario=scenario,
       output_dir="/path/to/output",
       export_apps=True  # Include app states in export
   )

   if success:
       print(f"Scenario exported to: {file_path}")
   else:
       print("Export failed")

Controlling App Export
~~~~~~~~~~~~~~~~~~~~~~

Control whether application states are included in exports:

.. code-block:: python

   from are.simulation.data_handler.exporter import JsonScenarioExporter

   exporter = JsonScenarioExporter()

   # Export without apps (useful when apps are stored separately)
   json_output = exporter.export_to_json(
       env=environment,
       scenario=scenario,
       scenario_id="lightweight_scenario",
       export_apps=False  # Exclude app states to reduce file size
   )

   # Export with custom app states
   custom_app_states = {
       "EmailApp": {
           "class_name": "are.simulation.apps.email.EmailApp",
           "serialized_state": '{"inbox": [], "sent": []}'
       }
   }

   json_output = exporter.export_to_json(
       env=environment,
       scenario=scenario,
       scenario_id="custom_app_scenario",
       apps_state=custom_app_states
   )

Working with World Logs
~~~~~~~~~~~~~~~~~~~~~~~

Include agent logs and world state information in exports:

.. code-block:: python

   from are.simulation.data_handler.exporter import JsonScenarioExporter
   from are.simulation.agents.are_simulation_agent import BaseAgentLog

   exporter = JsonScenarioExporter()

   # Create world logs
   world_logs = [
       BaseAgentLog.from_dict({
           "timestamp": 1640995200,
           "level": "INFO",
           "message": "Agent started scenario execution"
       })
   ]

   # Export with world logs
   json_output = exporter.export_to_json(
       env=environment,
       scenario=scenario,
       scenario_id="scenario_with_logs",
       world_logs=world_logs,
   )

Next Steps
----------
For hands-on scenario creation, continue to :doc:`../tutorials/working_with_scenarios`
for detailed development guidance.

For benchmarking with Meta Agents Research Environments, refer to :doc:`../user_guide/benchmarking`

Core Scenario JSON Format Classes
---------------------------------

ExportedTrace Classes
---------------------

.. autoclass:: are.simulation.data_handler.models.ExportedTraceBase
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedTrace
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedTraceMetadata
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedTraceDefinitionMetadata
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedApp
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedCompletedEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedOracleEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedAction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedActionArg
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.data_handler.models.ExportedHint
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Exporter class
--------------

.. autoclass:: are.simulation.data_handler.exporter.JsonScenarioExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Importer class
--------------

.. autoclass:: are.simulation.data_handler.importer.JsonScenarioImporter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

JSON Scenario classes
---------------------

.. autoclass:: are.simulation.scenarios.scenario_imported_from_json.scenario.ScenarioImportedFromJson
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario.BenchmarkScenarioImportedFromJson
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
