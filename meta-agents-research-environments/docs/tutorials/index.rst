..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`mortar-board` Tutorials
=================================

This section provides hands-on tutorials that teach you the core concepts of Meta Agents Research Environments through practical, runnable examples.
Each tutorial is implemented as a complete scenario that you can run, modify, and learn from.

:octicon:`info` Tutorial Overview
---------------------------------

The Meta Agents Research Environments tutorials are organized into focused lessons, each demonstrating specific concepts:

**Core Concepts**
   Learn the fundamental building blocks of Meta Agents Research Environments scenarios

**Event Systems**
   Master the different types of events and their relationships

**Validation Patterns**
   Understand how to validate agent behavior and scenario outcomes

**Advanced Techniques**
   Explore sophisticated scenario patterns and best practices

Each tutorial includes:

* **Runnable Code**: Complete scenario implementations you can execute
* **Detailed Comments**: Line-by-line explanations of key concepts
* **Practical Examples**: Real-world patterns you can adapt
* **Validation Logic**: How to measure success and debug issues

:octicon:`rocket` Getting Started with Tutorials
------------------------------------------------

Running a Tutorial
~~~~~~~~~~~~~~~~~~

Each tutorial can be run independently:

.. code-block:: bash

   # Use the ARE CLI to run with an agent
   are-run -s scenario_tutorial -a default --model Llama-4-Scout-17B-16E-Instruct-FP8 --provider llama-api

   # Use the ARE CLI to run in oracle mode (without an agent)
   are-run -s scenario_tutorial -o

   # Run locally with Python in oracle mode
   cd /path/to/are.simulation
   python -m are.simulation.scenarios.scenario_tutorial.scenario

Modifying Tutorials
~~~~~~~~~~~~~~~~~~~

The tutorial files are designed to be modified and experimented with:

1. **Copy the tutorial**: Make your own version to experiment with
2. **Change parameters**: Modify timing, content, or behavior
3. **Add new events**: Extend the scenarios with your own ideas
4. **Test variations**: See how changes affect agent behavior

:octicon:`book` Core Tutorials
------------------------------

Scenario Development Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``are/simulation/scenarios/scenario_tutorial/scenario.py``

**Concepts Covered**:
- Complete scenario structure and lifecycle
- App initialization and population
- Event creation and dependencies
- File attachments and messaging
- Scenario validation patterns

**What You'll Learn**:
This comprehensive tutorial demonstrates a complete scenario workflow involving multiple applications.
You'll see how to create a realistic scenario where the agent's task is to forward an email received by the user.

**Key Takeaways**:
- How to structure a complete scenario class
- Proper app initialization and data population
- Creating realistic event flows with dependencies
- Implementing robust validation logic

.. code-block:: python

   # Example from the tutorial
   @register_scenario("scenario_tutorial")
   @dataclass
   class ScenarioTutorial(Scenario):
       start_time: float | None = 0
       duration: float | None = 20

       def init_and_populate_apps(self, *args, **kwargs) -> None:
           # Initialize multiple apps
           agui = AgentUserInterface()
           email_client = EmailClientApp()
           messaging = MessagingApp()
           # ... populate with realistic data

Environment Tutorial
~~~~~~~~~~~~~~~~~~~~

**File**: ``are/simulation/tutorials/environment.py``

**Concepts Covered**:
- Environment configuration and timing
- Simulation time vs real time
- Environment control (start/pause/resume)
- Time-based event scheduling

**What You'll Learn**:
Understanding how Meta Agents Research Environments environments work is crucial for creating effective scenarios.
This tutorial shows you how to configure timing, control execution, and understand the relationship between simulation time and real time.

**Key Takeaways**:
- How to configure environment timing parameters
- The difference between simulation time and real time
- How to create time-sensitive scenarios
- Environment state monitoring and control

:octicon:`zap` Event System Tutorials
-------------------------------------

Events Tutorial
~~~~~~~~~~~~~~~

**File**: ``are/simulation/scenarios/scenario_events_tutorial/scenario.py``

**Concepts Covered**:
- Scheduled events with specific timing
- Conditional events triggered by environment state
- Event dependencies and chaining
- Different event creation patterns

**What You'll Learn**:
Events are the heart of dynamic scenarios.
This tutorial demonstrates the three main types of events and various patterns for creating and scheduling them.

**Event Types Demonstrated**:

.. code-block:: python

   # Scheduled Event - happens at a specific time
   scheduled_event = Event.from_function(
       app.some_function,
       param="value"
   ).depends_on(None, delay_seconds=5)

   # Conditional Event - triggers when condition is met
   def condition(env):
       return len(env.get_app("SomeApp").items) > 3

   condition_check = ConditionCheckEvent.from_condition(condition)
   conditional_event = Event.from_function(
       app.other_function
   ).depends_on(condition_check)

   # Dependent Event - happens after other events
   dependent_event = Event.from_function(
       app.final_function
   ).depends_on([scheduled_event, conditional_event])

DAG (Event Graph) Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``are/simulation/tutorials/event_dag.py``

**Concepts Covered**:
- Complex event dependency graphs
- Parallel and sequential event execution
- Random timing with realistic delays
- Agent validation in complex scenarios

**What You'll Learn**:
Real scenarios often involve complex event relationships.
This tutorial shows you how to create sophisticated event graphs where events depend on multiple predecessors,
execute in parallel, or converge at specific points.

**DAG Pattern Example**:

.. code-block:: text

       user_request
            ↓
         email_1
            ↓
         email_2
            ↙ ↘
      email_3  email_4
            ↘ ↙
         email_5
            ↓
      condition_check
            ↓
      agent_validation

:octicon:`shield` Validation Tutorials
--------------------------------------

Validation Tutorial
~~~~~~~~~~~~~~~~~~~

**File**: ``are/simulation/scenarios/scenario_validation_tutorial/scenario.py``

**Concepts Covered**:
- Environment state validation
- Real-time agent validation events
- Milestone and minefield patterns
- Timeout-based validation

**What You'll Learn**:
Validation is crucial for measuring agent performance.
This tutorial demonstrates multiple validation patterns, from simple state checks to complex real-time monitoring.

**Validation Patterns**:

.. code-block:: python

   # State Validation - check environment at specific time
   def state_validator(env):
       app = env.get_app("SomeApp")
       return len(app.items) > expected_count

   validation = ValidationEvent(milestones=[state_validator])

   # Agent Validation - monitor agent actions in real-time
   def agent_validator(env, event):
       return (event.function_name() == "expected_function" and
               event.action.args["param"] == "expected_value")

   agent_validation = AgentValidationEvent(
       milestones=[agent_validator],
       minefields=[unsafe_action_validator],
       timeout=30
   )



:octicon:`tools` Practical Exercises
------------------------------------

Try These Modifications
~~~~~~~~~~~~~~~~~~~~~~~

After working through the tutorials, try these exercises to deepen your understanding:

**Beginner Exercises**:

1. Modify the scenario tutorial to use different apps
2. Change the timing of events in the events tutorial
3. Add new validation criteria to the validation tutorial

**Intermediate Exercises**:

1. Create a new scenario combining concepts from multiple tutorials
2. Implement a scenario with branching event paths
3. Add probabilistic events that sometimes succeed or fail

**Advanced Exercises**:

1. Create a multi-agent scenario with complex interactions
2. Implement a scenario with dynamic event generation
3. Build a scenario that adapts based on agent performance

:octicon:`link` Next Steps
--------------------------

After completing the tutorials:

1. **Explore Working with Scenarios**: :doc:`working_with_scenarios` for usage patterns
2. **Read the Scenario Development Guide**: :doc:`scenario_development` for comprehensive scenario creation
3. **Build Your Own Apps**: :doc:`apps_tutorial` for creating custom apps
4. **Try Benchmarking**: :doc:`../user_guide/benchmarking` for systematic evaluation
5. **Try Gaia2 Evaluation**: :doc:`../user_guide/gaia2_evaluation` for benchmark evaluation
6. **Create Your Own**: Start building scenarios for your specific use cases

The tutorials provide a solid foundation, but the real learning happens when you start creating your own scenarios.
Don't hesitate to experiment, modify, and build upon these examples!
