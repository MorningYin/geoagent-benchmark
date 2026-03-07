..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`gear` Environment API
===============================

The Environment system is the core orchestration layer of Meta Agents Research Environments simulations. It manages the simulation lifecycle, event scheduling, time management, and coordination between apps and agents. This section documents the Environment API and configuration options.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Overview
--------

The Environment acts as the central controller for Meta Agents Research Environments simulations. It provides:

* **Time Management**: Controls simulation time, including start time, duration, and time increments
* **Event Orchestration**: Manages the event loop, scheduling, and execution
* **App Registration**: Coordinates registration and interaction between apps
* **State Management**: Maintains simulation state and provides persistence
* **Lifecycle Control**: Start, pause, resume, and stop simulation execution

Key Components
~~~~~~~~~~~~~~

* **Environment**: Main simulation controller
* **EnvironmentConfig**: Configuration settings for environment behavior
* **Event Loop**: Background thread managing event execution
* **Time Manager**: Unified time management across the simulation with pause/resume capabilities
* **Notification System**: Message queuing and notification handling for agent communication

Basic Usage
-----------

Simple Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from are.simulation.environment import Environment, EnvironmentConfig

    # Create configuration
    config = EnvironmentConfig(
        start_time=0,                    # Start at time 0 (or use datetime.now().timestamp())
        duration=60,                     # Run for 60 seconds
        time_increment_in_seconds=1      # 1-second time steps
    )

    # Create and start environment
    env = Environment(config=config)
    env.start()  # Non-blocking, starts event loop in separate thread

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import datetime
    from are.simulation.environment import EnvironmentConfig

    config = EnvironmentConfig(
        # Absolute start time - use datetime for specific times
        start_time=datetime.datetime(2024, 1, 1, 9, 0, 0).timestamp(),

        # Duration in seconds (None = run until main thread exits)
        duration=3600,  # 1 hour simulation

        # Time increment between ticks
        time_increment_in_seconds=0.5,
    )


Lifecycle Management
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    env = Environment(config=config)

    # Start the simulation (non-blocking)
    env.start()

    # Simulation is now running in background
    # Main thread can continue with other work
    time.sleep(10)

    # Pause simulation
    env.pause()

    # Do other work while paused
    time.sleep(5)

    # Resume simulation
    env.resume()

    # Keep main thread alive until simulation completes
    time.sleep(config.duration)

App Registration
~~~~~~~~~~~~~~~~

.. code-block:: python

    from are.simulation.apps.email_client import EmailClientApp
    from are.simulation.apps.calendar import CalendarApp

    # Create apps
    email_app = EmailClientApp()
    calendar_app = CalendarApp()

    # Register apps with environment
    env.register_apps([email_app, calendar_app])

    # Get registered app by name
    email_app = env.get_app("EmailClientApp")


Event Scheduling
~~~~~~~~~~~~~~~~

.. code-block:: python

    from are.simulation.types import Event, Action

    # Create an action
    action = Action(
        function=email_app.add_email,
        args={"email": email_object}
    )

    # Create and schedule event
    event = Event(action=action, event_time=env.start_time + 30)
    env.schedule(event)

    # Schedule multiple events
    events = [event1, event2, event3]
    env.schedule(events)

Advanced Features
-----------------

Time Management
~~~~~~~~~~~~~~~

.. code-block:: python

    # Get current simulation time
    current_time = env.time_manager.time()

    # Check if simulation is running
    if env.is_running():
        print("Simulation is active")

    # Get simulation duration
    total_duration = env.config.duration

    # Get elapsed time
    elapsed = env.time_manager.time_passed()


Event Access and Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Access event history
    completed_events = env.event_log.list_view()

    # Monitor specific event types
    agent_events = [e for e in env.event_log.list_view() if e.event_type == EventType.AGENT]


Threading and Concurrency
-------------------------

The Environment runs its event loop in a separate background thread, allowing the main thread to continue other work:

.. code-block:: python

    env = Environment(config)
    env.start()  # Returns immediately, event loop runs in background

    # You can do other stuff while event loop is running e.g. do nothing for a few seconds
    time.sleep(2)

    # You can also pause the simulation here for 3 seconds
    env.pause()
    time.sleep(3)

    # And then resume it
    env.resume()

For practical examples, see the environment tutorial :doc:`../tutorials/index`.

Core Environment Classes
------------------------

.. automodule:: are.simulation.environment
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Environment
~~~~~~~~~~~

.. autoclass:: are.simulation.environment.Environment
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

EnvironmentConfig
~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.environment.EnvironmentConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Key Methods
~~~~~~~~~~~

**Lifecycle Management**

.. automethod:: are.simulation.environment.Environment.run

.. automethod:: are.simulation.environment.Environment.start

.. automethod:: are.simulation.environment.Environment.pause

.. automethod:: are.simulation.environment.Environment.resume

.. automethod:: are.simulation.environment.Environment.stop

.. automethod:: are.simulation.environment.Environment.join

**App Management**

.. automethod:: are.simulation.environment.Environment.register_apps

.. automethod:: are.simulation.environment.Environment.get_app

.. automethod:: are.simulation.environment.Environment.get_tools_by_app

**Event Management**

.. automethod:: are.simulation.environment.Environment.schedule

.. automethod:: are.simulation.environment.Environment.get_currently_active_validation_events


Event Log
---------

The EventLog is a crucial component of the Environment that stores all completed events during simulation execution. It maintains a chronological record of all events that have occurred, enabling monitoring, analysis, and debugging of simulation behavior.

The event log is automatically populated as events are processed by the environment. Each time an event completes (successfully or with errors), it's added to the event log as a CompletedEvent instance.

EventLog Class
~~~~~~~~~~~~~~

.. automodule:: are.simulation.types
   :no-index:

.. autoclass:: are.simulation.types.EventLog
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__, __len__
   :no-index:

Event Log Usage
~~~~~~~~~~~~~~~

.. code-block:: python

    # Access event log from environment
    env = Environment(config)

    # Get all completed events
    completed_events = env.event_log.list_view()

    # Get event count
    event_count = len(env.event_log)

    # Filter events by type
    agent_events = [e for e in env.event_log.list_view() if e.event_type == EventType.AGENT]
    env_events = [e for e in env.event_log.list_view() if e.event_type == EventType.ENV]

    # Access specific event details
    for event in completed_events:
        print(f"Event ID: {event.event_id}")
        print(f"Event Type: {event.event_type}")
        print(f"Event Time: {event.event_time}")
        print(f"App: {event.app_name()}")
        print(f"Function: {event.function_name()}")
        print(f"Return Value: {event.metadata.return_value}")
        if event.failed():
            print(f"Exception: {event.metadata.exception}")

Key Features
~~~~~~~~~~~~

* **Chronological Storage**: Events are stored in order by their execution time using a priority queue
* **Immutable Records**: Events are copied when added to prevent mutation of logged events
* **Rich Metadata**: Each logged event includes execution metadata like return values, exceptions, and timing
* **Flexible Access**: Provides both list view and dictionary serialization for different use cases
* **Memory Efficient**: Uses priority queue for efficient storage and retrieval

The event log is essential for:

* **Monitoring**: Track what events have occurred during simulation
* **Debugging**: Examine event execution order and results
* **Analysis**: Post-simulation analysis of agent behavior and environment state changes
* **Validation**: Verify that expected events occurred during simulation

Time Manager
------------

The TimeManager class provides sophisticated time management for simulations, including pause/resume capabilities and time offset handling.

.. automodule:: are.simulation.time_manager
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

TimeManager
~~~~~~~~~~~

.. autoclass:: are.simulation.time_manager.TimeManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Time Management Usage
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from are.simulation.time_manager import TimeManager

    # Create time manager
    time_manager = TimeManager()

    # Get current simulation time
    current_time = time_manager.time()

    # Get time passed since start
    elapsed = time_manager.time_passed()

    # Pause simulation time
    time_manager.pause()

    # Resume simulation time
    time_manager.resume()

    # Reset time with custom start time
    import time
    time_manager.reset(time.time())

    # Add time offset (e.g., for time jumps)
    time_manager.add_offset(3600)  # Jump forward 1 hour

Notification System
-------------------

The Notification System provides message queuing and notification handling for agent communication and environment events.

.. automodule:: are.simulation.notification_system
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Core Classes
~~~~~~~~~~~~

BaseNotificationSystem
^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.BaseNotificationSystem
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

VerboseNotificationSystem
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.VerboseNotificationSystem
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Configuration Classes
~~~~~~~~~~~~~~~~~~~~~

NotificationSystemConfig
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.NotificationSystemConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Message Classes
~~~~~~~~~~~~~~~

Message
^^^^^^^

.. autoclass:: are.simulation.notification_system.Message
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

MessageQueue
^^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.MessageQueue
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Enumerations
~~~~~~~~~~~~

MessageType
^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.MessageType
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

VerbosityLevel
^^^^^^^^^^^^^^

.. autoclass:: are.simulation.notification_system.VerbosityLevel
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Notification System Usage
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from are.simulation.notification_system import (
        BaseNotificationSystem,
        VerboseNotificationSystem,
        VerbosityLevel,
        NotificationSystemConfig
    )

    # Create a notification system with specific verbosity
    notification_system = VerboseNotificationSystem(
        verbosity_level=VerbosityLevel.MEDIUM
    )

    # Or create with custom configuration
    config = NotificationSystemConfig(
        notified_tools={
            "EmailClientApp": ["send_email", "receive_email"],
            "CalendarApp": ["add_event", "delete_event"]
        }
    )
    base_notification_system = BaseNotificationSystem(config)

    # Initialize with time manager
    from are.simulation.time_manager import TimeManager
    time_manager = TimeManager()
    notification_system.initialize(time_manager)

    # Handle events
    notification_system.handle_event(some_event)

    # Get messages for current time
    current_time = datetime.now(timezone.utc)
    messages = notification_system.message_queue.get_by_timestamp(current_time)

Verbosity Levels
^^^^^^^^^^^^^^^^

The notification system supports three verbosity levels:

* **LOW** (VerbosityLevel.LOW): Only user messages and system notifications
* **MEDIUM** (VerbosityLevel.MEDIUM): Environment events that are consequences of agent actions
* **HIGH** (VerbosityLevel.HIGH): Most environment events, including independent events

Key Functions
~~~~~~~~~~~~~

.. autofunction:: are.simulation.notification_system.get_notification_tools

.. autofunction:: are.simulation.notification_system.get_content_for_message
