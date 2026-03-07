..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`calendar` Events API Reference
========================================

This page provides detailed technical information about the Events system in Meta Agents Research Environments.
For a higher-level overview, see :doc:`../foundations/index`.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Event Structure
---------------

All events inherit from the ``AbstractEvent`` class, which contains shared fields between completed and future events:

.. code-block:: python

   @dataclass(order=True)
   class AbstractEvent(ABC):
       """
       Abstract event class, that contains shared field between completed and future events.
       """
       # The action that will be executed when the event happens, which is an App API call
       action: Action
       # The type of the event, either AGENT, ENV or USER.
       event_type: EventType = field(default=EventType.ENV)
       # The time at which the event will happen, this can get overridden in various places for e.g. in case of conditional triggers.
       event_time: float = field(default_factory=lambda: global_time.time())
       # The relative time wrt the simulation start time
       # WARNING when the event is going to be added to the queue, this information is going to be used to set event_time
       event_relative_time: float | None = field(default=None)
       # The unique id of the event, this is used to identify the event in the logs.
       # This is created automatically and does NOT need to be handled by the user
       event_id: str = field(default=None)

.. note::
   Events have an ``event_type`` which describes the origin of the event. By default, each ``app_tool`` call made by the agent will be of type ``EventType.AGENT``, each event definition in the scenario script will be of type ``EventType.ENV``. You can also decide to manually add the type ``EventType.USER`` to events when defining a scenario to simulate that the user is the source of the event. This field is only used for logging purposes and does not have any impact on the execution of the scenario.

Event Types by Origin
---------------------

* ``EventType.AGENT``: Events initiated by agent tool calls
* ``EventType.ENV``: Events defined in scenario scripts
* ``EventType.USER``: Events simulating user interactions

Actions
-------

Events execute **Actions**, which define what actually happens:

.. code-block:: python

   @dataclass
   class Action:
       function: Callable               # Function to call
       args: dict[str, Any]            # Function arguments
       app: App | None = None          # App instance
       action_id: str = None           # Unique identifier

Event Management Overview
-------------------------

The environment manages events through two main data structures:

* **Event Queue**: Stores future events waiting to be processed
* **Event Log**: Contains the history of completed events

.. code-block:: python

   # Schedule an event
   env.schedule(event)

   # Access event history
   completed_events = env.event_log.list_view()

Event Graph Nodes
-----------------

Event graphs consist of three types of nodes:

Event Nodes
~~~~~~~~~~~

Standard events as described above

ConditionCheckEvent Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~

Check functions that return boolean values

ValidationEvent Nodes
~~~~~~~~~~~~~~~~~~~~~

Validation functions with checklists of expected agent actions

Advanced Event Properties
-------------------------

``ConditionCheckEvent`` and ``ValidationEvent`` nodes support additional properties:

* ``timeout_tick``: Maximum time to wait for condition/validation
* ``every_tick``: Frequency of condition/validation checks

Example ValidationEvent
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def validation_function(env: AbstractEvent, event: AbstractEvent) -> bool:
       aui = event.get_app(AgentUserInterface.__name__)
       return all([
           event.action.app == aui,
           event.function_name() == "send_message_to_user",
       ])

   validation = ValidationEvent(
       check_list=[validation_function],
       schedule_every_ticks=1,
       timeout=20,
   )

Event Dependencies
------------------

Events can be chained with dependencies and delays:

.. code-block:: python

   e1 = Event(...)
   e2 = Event(...)

   e1.depends_on(None, delay_seconds=30)  # Start after 30 seconds
   e2.depends_on(e1, delay_seconds=5)     # Start 5 seconds after e1

Here ``e1`` is delayed by 30 seconds after the start of the simulation and ``e2`` will happen 5 seconds after ``e1`` is executed.

API Methods
-----------

Event Creation
~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.core.events import Event
   from are.simulation.core.action import Action

   # Create a basic event
   action = Action(function=my_function, args={"param": "value"})
   event = Event(action=action, event_time=simulation_time + 30)

   # Schedule the event
   environment.schedule(event)

Event Scheduling
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Schedule with relative time
   event.event_relative_time = 60  # 60 seconds from simulation start
   environment.schedule(event)

   # Schedule with dependencies
   event1 = Event(...)
   event2 = Event(...)
   event2.depends_on(event1, delay_seconds=10)

Event Querying
~~~~~~~~~~~~~~

.. code-block:: python

   # Get all completed events
   completed = environment.event_log.list_view()

   # Filter events by type
   agent_events = [e for e in completed if e.event_type == EventType.AGENT]

Validation Events
-----------------

Validation events are special events that check whether certain conditions are met during scenario execution:

.. code-block:: python

    def both_responses_validator(env: AbstractEnvironment) -> bool:
        """Check if agent gave both responses."""
        aui_app = env.get_app("AgentUserInterface")
        agent_messages = aui_app.get_all_messages_from_agent()
        has_correct = any(msg.content == "I am a robot" for msg in agent_messages)
        has_incorrect = any(
            msg.content == "I am not a robot •`_´• " for msg in agent_messages
        )
        return has_correct and not has_incorrect

    continuous_validation = ValidationEvent(
        milestones=[both_responses_validator]
    ).with_id("continuous_validation")
    continuous_validation.depends_on(user_request, delay_seconds=1)
    continuous_validation.schedule(every_ticks=1, timeout=15)
    self.events.append(continuous_validation)

Condition Check Events
----------------------

Condition check events monitor the environment state and trigger other events when conditions are met:

.. code-block:: python

   from are.simulation.core.events import ConditionCheckEvent

   def check_file_exists(env, event):
       """Check if a specific file exists"""
       fs_app = env.get_app("FileSystemApp")
       return fs_app.file_exists("/important/document.txt")

   condition = ConditionCheckEvent(
       condition_function=check_file_exists,
       schedule_every_ticks=5,  # Check every 5 ticks
       timeout=600,             # Timeout after 10 minutes
   )

   # Trigger another event when condition is met
   followup_event = Event(...)
   followup_event.depends_on(condition)

For practical examples of event usage, see the tutorials and scenario development guides.

Oracle Events
-------------

Oracle Events are specialized events that represent actions the agent should perform during scenario execution. They serve as templates or blueprints for generating actual executable events at runtime. Oracle Events are particularly useful for defining expected agent behaviors in scenarios and for creating validation trajectories.

Key Characteristics
~~~~~~~~~~~~~~~~~~~

* **Agent-focused**: Oracle Events default to ``EventType.AGENT``, indicating they represent actions the agent should take
* **Dynamic creation**: They use a ``make_event`` callable to generate the actual event when needed
* **Timing flexibility**: Support event time comparators for flexible scheduling
* **Action descriptions**: Store metadata about the action to be performed

Creating Oracle Events
~~~~~~~~~~~~~~~~~~~~~~

Oracle Events can be created in several ways:

**From Existing Events**

.. code-block:: python

    from are.simulation.core.events import Event, OracleEvent
    from are.simulation.core.action import Action

    # Create a regular event
    action = Action(function=email_app.send_email, args={"to": "user@example.com"})
    regular_event = Event(action=action)

    # Convert to Oracle Event
    oracle_event = OracleEvent.from_event(regular_event)

    # Or use the convenience method
    oracle_event = regular_event.oracle()

**Direct Creation**

.. code-block:: python

    def make_email_event(env):
        email_app = env.get_app("EmailClientApp")
        return Event.from_function(
            email_app.send_email,
            recipients=["client@company.com"],
            subject="Project Update"
        )

    oracle_event = OracleEvent(
        make_event=make_email_event,
        event_type=EventType.AGENT,
        event_relative_time=30  # Execute 30 seconds after dependencies
    )

Generating Executable Events
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Oracle Events generate actual executable events through the ``make()`` method:

.. code-block:: python

    # Generate the executable event
    executable_event = oracle_event.make(environment)

    # The generated event inherits timing and dependencies
    assert executable_event.event_time == oracle_event.event_time
    assert executable_event.dependencies == oracle_event.dependencies

Event Time Comparators
~~~~~~~~~~~~~~~~~~~~~~

Oracle Events support time-based filtering through event time comparators:

.. code-block:: python

    from are.simulation.types import EventTimeComparator

    oracle_event = OracleEvent(
        make_event=make_email_event,
        event_time_comparator=EventTimeComparator.LESS_THAN
    )

    # This oracle event can be used to find events that occur before a certain time

Action Descriptions
~~~~~~~~~~~~~~~~~~~

Oracle Events automatically capture action metadata when created from existing events:

.. code-block:: python

    # The action description includes app name, function name, and arguments
    oracle_event = regular_event.oracle()

    action_desc = oracle_event.action_desc
    print(f"App: {action_desc.app}")
    print(f"Function: {action_desc.function}")
    print(f"Args: {action_desc.args}")

**Completed Oracle Events**

When Oracle Events are executed, they become ``CompletedOracleEvent`` instances that preserve timing information:

.. code-block:: python

    from are.simulation.types import CompletedOracleEvent

    # After execution, create a completed oracle event
    completed_oracle = CompletedOracleEvent.from_completed_event_and_oracle_event(
        completed_event, original_oracle_event
    )

    # Access timing metadata
    print(f"Original scheduled time: {completed_oracle.absolute_event_time}")
    print(f"Time comparator: {completed_oracle.event_time_comparator}")

Integration with Event System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Oracle Events integrate seamlessly with the standard event system:

* They inherit from ``AbstractEvent`` and support all standard event operations
* They can have dependencies and successors like regular events
* They participate in event scheduling and timing calculations
* They can be serialized and deserialized for persistence

Event Registration
------------------

The ``@event_registered`` decorator and ``EventRegisterer`` class are used for automatic event registration when app methods are called:

**Event Registerer**

.. autoclass:: are.simulation.types.EventRegisterer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

**Event Registration Decorator**

The ``@event_registered`` decorator is used to automatically register events when app methods are called:

.. autofunction:: are.simulation.types.event_registered

This decorator wraps app methods to automatically create and log events when they are executed.

It supports different operation types and event types, and can be used in capture mode for testing and validation.

Example usage:

.. code-block:: python

   from are.simulation.types import event_registered, OperationType, EventType
   from are.simulation.apps.app import App

   class MyApp(App):
       @event_registered(operation_type=OperationType.WRITE, event_type=EventType.AGENT)
       def send_message(self, message: str):
           # This method call will automatically create and log an event
           return f"Sent: {message}"



Core Event Classes
------------------

All event classes are defined in the ``are.simulation.types`` module:

.. automodule:: are.simulation.types
   :members: AbstractEvent, Event, ValidationEvent, ConditionCheckEvent, CompletedEvent, OracleEvent, StopEvent, AgentValidationEvent
   :undoc-members:
   :show-inheritance:
   :no-index:

Abstract Event
~~~~~~~~~~~~~~

.. autoclass:: are.simulation.types.AbstractEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Event
~~~~~

.. autoclass:: are.simulation.types.AbstractEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.types.Event
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.types.ValidationEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.types.ConditionCheckEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.types.CompletedEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.types.OracleEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Action Class
------------

.. autoclass:: are.simulation.types.Action
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Event Types and Enums
---------------------

.. autoclass:: are.simulation.types.EventType
   :members:
   :undoc-members:
   :no-index:

Event Management
----------------

**Event Queue**

.. autoclass:: are.simulation.types.EventQueue
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

**Event Log**

.. autoclass:: are.simulation.types.EventLog
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Additional Types and Classes
----------------------------

.. autoclass:: are.simulation.types.AbstractEnvironment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.ActionDescription
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.types.AgentActionValidator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.CapabilityTag
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.types.ConditionCheckAction
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.EventMetadata
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.EventTimeComparator
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
