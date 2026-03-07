..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`play` Working with Scenarios
======================================

Scenarios are the heart of Meta Agents Research Environments - they define the tasks that agents need to complete within the simulated environment. This guide covers how to understand, run, and work with scenarios effectively.

:octicon:`info` Understanding Scenarios
---------------------------------------

A scenario in Meta Agents Research Environments is more than just a task description. It's a complete simulation setup that includes:

* **Initial Environment State**: How the world looks when the scenario starts
* **Available Applications**: Which tools the agent can use
* **Dynamic Events**: Things that happen during the scenario execution
* **Task Definition**: What the agent needs to accomplish
* **Validation Logic**: How success is measured

:octicon:`package` Scenario Structure
-------------------------------------

Anatomy of a Scenario
~~~~~~~~~~~~~~~~~~~~~

Every Meta Agents Research Environments scenario follows a consistent structure by inheriting from the ``Scenario`` base class and implementing key methods:

.. code-block:: python

   from are.simulation.scenarios.scenario import Scenario
   from are.simulation.scenarios.utils.registry import register_scenario
   from dataclasses import dataclass

   @register_scenario("my_scenario")
   @dataclass
   class MyScenario(Scenario):
       # Define scenario timing
       start_time: float | None = 0
       duration: float | None = 30  # Duration in seconds

       def init_and_populate_apps(self, *args, **kwargs) -> None:
           # 1. Initialize applications
           email_app = EmailClientApp()
           calendar_app = CalendarApp()
           messaging_app = MessagingApp()

           # 2. Populate initial data
           email_app.add_email(...)
           calendar_app.add_event(...)

           # 3. Register apps with scenario
           self.apps = [email_app, calendar_app, messaging_app]

       def build_events_flow(self) -> None:
           messaging_app = self.get_typed_app(MessagingApp)
           # 4. Define dynamic events that occur during execution
           event1 = Event.from_function(
               messaging_app.add_message,
               conversation_id="conv1",
               sender="John Doe",
               content="Hello!"
           ).depends_on(None, delay_seconds=5)

           self.events = [event1]

       def validate(self, env: AbstractEnvironment) -> ScenarioValidationResult:
           # 5. Check if task was completed successfully
           try:
               # Validation logic here
               success = self.check_completion_criteria(env)
               return ScenarioValidationResult(success=success)
           except Exception as e:
               return ScenarioValidationResult(success=False, exception=e)

Key Components
~~~~~~~~~~~~~~

**Applications (Apps)**
   The tools available to the agent, such as email clients, file systems, calendars, etc.

**Initial Data**
   The starting state of all applications - emails in the inbox, files on disk, calendar appointments, etc.

**Events**
   Dynamic occurrences that happen during scenario execution - new emails arriving, calendar reminders, system notifications.

**Task Description**
   A clear, natural language description of what the agent needs to accomplish.

**Validation Function**
   Logic that determines whether the agent successfully completed the task.

:octicon:`rocket` Running Scenarios
-----------------------------------

Basic Execution
~~~~~~~~~~~~~~~

Run a scenario using the command line:

.. code-block:: bash

   are-run -s scenario_name -a agent_name --provider model_provider -m model_name

For example:

.. code-block:: bash

   # Run scenario "scenario_find_image_file" with agent "default" using model "Llama-3.1-70B"
   are-run -s scenario_find_image_file -a default --model Llama-3.1-70B-Instruct --provider llama-api

   # Run scenario "scenario_find_image_file" in oracle mode
   are-run -s scenario_find_image_file -o


Interactive Mode
~~~~~~~~~~~~~~~~

Use the GUI for interactive scenario execution:

.. code-block:: bash

   are-gui -s scenario_find_image_file -a default --model Llama-3.1-70B-Instruct --provider llama-api

This opens a web interface where you can:

* Watch the agent's reasoning process
* See real-time environment updates
* Interact with the simulation
* Debug agent behavior

:octicon:`database` Working with Scenario Data
----------------------------------------------

Understanding Initial State
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each scenario starts with a specific initial state. To understand what's available:

1. **Check App States**: Look at what data is pre-populated in each app
2. **Review Events**: Understand what will happen during execution
3. **Analyze Task Requirements**: Determine what information the agent needs

Example: Email Scenario
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Initial state might include:
   email_app = EmailClientApp()
   email_app.inbox = [
       Email(sender="client@company.com", subject="Project Delay", ...),
       Email(sender="boss@company.com", subject="Urgent: Client Response", ...),
   ]

:octicon:`shield` Scenario Validation
-------------------------------------

Understanding Validation Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation functions determine scenario success. They typically check:

* **Required Actions**: Did the agent perform necessary actions?
* **Output Quality**: Is the result appropriate and correct?
* **Process Compliance**: Did the agent follow proper procedures?

Common Validation Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~


**Validation Event**

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


**Agent Validation Event**

.. code-block:: python

    def milestone_correct_response(
        env: AbstractEnvironment, event: AbstractEvent
    ) -> bool:
        """Milestone: Agent should say 'I am a robot'."""
        if not isinstance(event, Event):
            return False
        return (
            event.action.app.__class__ == AgentUserInterface
            and event.function_name() == "send_message_to_user"
            and event.action.args["content"] == "I am a robot"
        )

    def minefield_unsafe_response(
        env: AbstractEnvironment, event: AbstractEvent
    ) -> bool:
        """Minefield: Agent should NOT say unsafe things."""
        if not isinstance(event, Event):
            return False
        return (
            event.action.app.__class__ == AgentUserInterface
            and event.function_name() == "send_message_to_user"
            and event.action.args["content"] == "I am doing something unsafe!"
        )

    agent_validation = AgentValidationEvent(
        milestones=[milestone_correct_response],
        minefields=[minefield_unsafe_response],
        timeout=15,
    ).with_id("agent_validation")
    agent_validation.depends_on(user_request)
    self.events.append(agent_validation)


**Action-Based Validation**
   Check if specific actions were performed:

   .. code-block:: python

      def validate(self, env):
          # Check if email was sent
          sent_emails = env.get_app("EmailApp").sent_emails
          return len(sent_emails) > 0 and "apology" in sent_emails[0].body.lower()

**State-Based Validation**
   Check if the environment reached the desired state:

   .. code-block:: python

      def validate(self, env):
          # Check if calendar was updated
          calendar = env.get_app("CalendarApp")
          return calendar.has_appointment_on_date("2024-01-15")

**Content-Based Validation**
   Analyze the quality of generated content:

   .. code-block:: python

      def validate(self, env):
          # Check email content quality
          email = env.get_app("EmailApp").sent_emails[0]
          return self.check_professional_tone(email.body) and \
                 self.check_apology_present(email.body)

:octicon:`star` Best Practices for Scenario Usage

Debugging Scenarios
~~~~~~~~~~~~~~~~~~~

When scenarios don't work as expected:

1. **Check Initial State**: Verify the scenario setup is correct
2. **Review Events**: Ensure events fire at the right times
3. **Validate Logic**: Confirm validation functions work correctly
4. **Test Incrementally**: Start with simpler versions of complex scenarios

For detailed guidance on creating scenarios, see :doc:`scenario_development`.

:octicon:`mortar-board` Practical Examples and Tutorials
--------------------------------------------------------

The Meta Agents Research Environments repository includes several hands-on tutorials that demonstrate key concepts:

**Environment Tutorial** (``are/simulation/tutorials/environment.py``)
   Learn how to create and configure environments, manage time, and control simulation flow.

**Events Tutorial** (``are/simulation/scenarios/scenario_events_tutorial/scenario.py``)
   Understand how to create, schedule, and manage events in your scenarios.

**Advanced Events Tutorial** (``are/simulation/tutorials/events_advanced.py``)
   Explore complex event patterns, triggers, and conditional logic.

**DAG Scenario Tutorial** (``are/simulation/tutorials/event_dag.py``)
   Build sophisticated scenarios using Event Graphs with dependencies and complex flows.

**Scenario Development Tutorial** (``are/simulation/scenarios/scenario_tutorial/scenario.py``)
   Step-by-step guide to creating complete scenarios from scratch.

**Validation Tutorial** (``are/simulation/scenarios/scenario_validation_tutorial/scenario.py``)
   Learn how to implement robust validation logic for your scenarios.

These tutorials provide practical, runnable examples that complement the documentation and help you understand how to apply the concepts in real scenarios.

Next Steps
----------

Now that you understand how scenarios work:

* **Experiment**: Try running different scenarios with various agents
* **Analyze**: Study successful and failed scenario executions
* **Benchmark**: Use scenarios for systematic agent evaluation
* **Create**: Develop your own scenarios for specific use cases
* **Learn by Example**: Work through the tutorials in the ``are/simulation/tutorials/`` directory

Ready to create your own scenarios?

Check out the :doc:`scenario_development` guide for detailed instructions.
