..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`codespaces` Scenario Development
==========================================

This guide explains how to create and register custom scenarios for Meta Agents Research Environments
using the entry point-based plugin system. Whether you're creating simple test scenarios
or complex multi-step simulations, this guide will help you get started.


Overview
--------

Meta Agents Research Environments uses Python's entry point mechanism to discover and load scenarios.

Architecture
~~~~~~~~~~~~

The scenario system works through:

1. **Entry Points**: Python packaging mechanism for plugin discovery
2. **Registration Decorators**: Decorators that register scenarios with the system
3. **Registry System**: Central registry that manages all available scenarios
4. **Dynamic Loading**: Scenarios are loaded on-demand when needed

Creating a Scenario
-------------------

Basic Scenario Structure
~~~~~~~~~~~~~~~~~~~~~~~~

First, create a class that inherits from the ``Scenario`` base class and decorate it
 with ``@register_scenario``, choose a unique meaningful scenario ID, and implement:

.. code-block:: python

   from are.simulation.scenarios.scenario import Scenario
   from are.simulation.scenarios.utils.registry import register_scenario
   from are.simulation.apps.email_client import EmailClientApp
   from are.simulation.apps.calendar import CalendarApp
   from are.simulation.types import EventType

   @register_scenario("my_custom_scenario")
   class MyCustomScenario(Scenario):
       """
       A custom scenario that demonstrates email and calendar integration.
       """
       start_time: float | None = 0
       duration: float | None = 600  # Duration in seconds

       def init_and_populate_apps(self):
           """Initialize and populate apps with data."""
           # Create apps
           aui = AgentUserInterface()
           email_app = EmailClientApp()
           calendar_app = CalendarApp()

           # Populate with initial data
           email_app.add_email(
                email=Email(
                    sender="boss@company.com",
                    recipients=[self.email_app.user_email],
                    subject="Important Meeting",
                    content="Please schedule a 'Team Standup' for next week same time.",
                ),
           )

           # Add calendar events
           calendar_app.add_calendar_event(
                title="Team Standup"
                start_datetime="2024-01-15 09:00:00"
                end_datetime="2024-01-15 09:30:00"
           )

           # Register apps
           self.apps = [email_app, calendar_app, aui]

       def build_events_flow(self):
           """Define the sequence of events in the scenario."""
            aui = self.get_typed_app(AgentUserInterface)
            calendar_app = self.get_typed_app(CalendarApp)

            with EventRegisterer.capture_mode():
                # Add user task to read email
                aui.send_message_to_agent(
                    content="Read my email and schedule the meeting my boss requested."
                ).depends_on(None, delay_seconds=5)
                # Add the calenddar event
                calendar_app.add_calendar_event(
                    title="Team Standup"
                    start_datetime="2024-01-22 09:00:00"
                    end_datetime="2024-01-22 09:30:00"
                )


       def validate(self, env) -> ScenarioValidationResult:
           """Validate that the scenario was completed successfully."""
           try:
               # Check if a meeting was scheduled
               success = any(
                    event.event_type == EventType.AGENT
                    and isinstance(event.action, Action)
                    and event.action.function_name == "add_calendar_event"
                    and event.action.class_name == "CalendarApp"
                    and event.action.args["start_datetime"] == "2024-01-22 09:00:00"
                    and event.action.args["end_datetime"] == "2024-01-22 09:30:00"
                    and event.action.args["title"] == "Team Standup"
                    for event in env.event_log.list_view()
                )

               return ScenarioValidationResult(success=success)
           except Exception as e:
               return ScenarioValidationResult(success=False, exception=e)

Registration Function
~~~~~~~~~~~~~~~~~~~~~

Create a function that will import your scenario modules to trigger the decorators:

.. code-block:: python

   def register_scenarios(registry):
       """
       Register all scenarios in this package with the provided registry.

       Args:
           registry: The ScenarioRegistry instance to register with
       """
       # Simply import the modules containing the scenarios
       # The decorators will handle the registration
       import my_scenarios.email_scenario
       import my_scenarios.calendar_scenario
       import my_scenarios.complex_scenario

Packaging Your Scenarios
------------------------

Package Structure
~~~~~~~~~~~~~~~~~

Organize your code as a Python package:

.. code-block:: text

   my_scenarios/
   ├── __init__.py
   ├── email_scenario.py
   ├── calendar_scenario.py
   ├── complex_scenario.py
   └── registration.py  # Contains the register_scenarios function

Project Configuration
~~~~~~~~~~~~~~~~~~~~~

Create a ``pyproject.toml`` file that defines your package and its entry points:

.. code-block:: toml

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [project]
   name = "my-are-scenarios"
   version = "0.1.0"
   description = "My custom scenarios for Meta Agents Research Environments"
   readme = "README.md"
   authors = [
       {name = "Your Name", email = "your.email@example.com"}
   ]
   license = {text = "MIT License"}
   requires-python = ">=3.10"
   dependencies = [
       "are.simulation",  # Depend on the main Meta Agents Research Environments package
   ]

   [project.entry-points."are.simulation.scenarios"]
   my_scenarios = "my_scenarios.registration:register_scenarios"

   [tool.hatch.build]
   include = [
       "my_scenarios/**/*.py",
   ]
   exclude = [
       "**/__pycache__/**",
       "**/*.pyc",
   ]

   [tool.hatch.build.targets.wheel]
   packages = ["my_scenarios"]

Entry Point Format
~~~~~~~~~~~~~~~~~~

The entry point format is:

* **Group name**: ``are.simulation.scenarios`` (required for Meta Agents Research Environments to discover your scenarios)
* **Entry point name**: A unique identifier for your scenario package (e.g., ``my_scenarios``)
* **Object reference**: Path to your registration function (e.g., ``my_scenarios.registration:register_scenarios``)

Installation
~~~~~~~~~~~~

Install your package in development mode:

.. code-block:: bash

   pip install -e /path/to/your/package

Or build and install it:

.. code-block:: bash

   cd /path/to/your/package
   pip install .

Advanced Scenario Patterns
--------------------------

Multi-App Scenarios
~~~~~~~~~~~~~~~~~~~

Create scenarios that use multiple apps working together:

.. code-block:: python

   @register_scenario("multi_app_workflow")
   class MultiAppWorkflowScenario(Scenario):
       """Scenario involving email, calendar, and file system."""

       scenario_id = "multi_app_workflow"

       def init_and_populate_apps(self):
           email_app = EmailClientApp()
           calendar_app = CalendarApp()
           file_system = VirtualFileSystemApp()

           # Create initial files
           file_system.create_file(
               "/documents/project_plan.txt",
               "Project timeline and milestones"
           )

           # Add initial email
           email_app.add_email(
                email=Email(
                    sender="client@company.com",
                    recipients=[self.email_app.user_email],
                    subject="Report Request",
                    content="Please send the updated project plan.",
                ),
           )

           self.apps = [email_app, calendar_app, file_system]

       def build_events_flow(self):
           aui = self.get_typed_app(AgentUserInterface)
           email_app = self.get_typed_app(EmailClientApp)
           calendar_app = self.get_typed_app(CalendarApp)

            with EventRegisterer.capture_mode():
                # User event - send task to agent
                send_task_event = (
                    aui.send_message_to_agent(
                        content="Read the email from the client and fulfill their request."
                    )
                ).depends_on(None, delay_seconds=5)
                # Oracle event - send email
                send_email_event = (
                    email_app.send_email(
                        recipients=["agent@company.com"],
                        subject="Project Update Request",
                        content="Please find attached the updated project plan."
                        attachment_paths=["/documents/project_plan.txt"]
                    )
                    .oracle()
                    .depends_on(send_task_event, delay_seconds=5)
                )

                # Oracle event - schedule meeting
                schedule_meeting_event = (
                    calendar_app.add_calendar_event(
                        title="Project Review",
                        start_datetime="2024-01-15 10:00:00",
                        end_datetime="2024-01-15 11:00:00",
                    )
                    .oracle()
                    .depends_on(send_email_event, delay_seconds=5)
                )

            self.events = [send_task_event, send_email_event, schedule_meeting_event]


Complex Validation Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implement sophisticated validation logic:

.. code-block:: python

   @register_scenario("complex_validation_scenario")
   class ComplexValidationScenario(Scenario):
       """Scenario with multi-criteria validation."""

       scenario_id = "complex_validation_scenario"

       def validate(self, env) -> ScenarioValidationResult:
           """Multi-step validation with detailed feedback."""
           try:
               validation_results = []

               # Check email was sent
               email_app = self.get_typed_app(EmailClientApp)
               sent_emails= email_app.folders[EmailFolderName.SENT].emails

               email_sent = len(sent_emails) > 0
               validation_results.append(("email_sent", email_sent))

               # Check email content quality
               if email_sent:
                   last_email = sent_emails[-1]
                   professional_tone = self._check_professional_tone(last_email.content)
                   has_greeting = any(greeting in last_email.content.lower()
                                    for greeting in ["hello", "hi", "dear"])
                   has_closing = any(closing in last_email.content.lower()
                                   for closing in ["regards", "sincerely", "thanks"])

                   validation_results.extend([
                       ("professional_tone", professional_tone),
                       ("has_greeting", has_greeting),
                       ("has_closing", has_closing)
                   ])

               # Overall success requires all criteria
               overall_success = all(result[1] for result in validation_results)

               # Log detailed results for debugging
               rationale = ""
               for criterion, passed in results:
                   status = "PASS" if passed else "FAIL"
                   rationale += f"Validation {criterion}: {status}\n"

               return ScenarioValidationResult(success=overall_success, rationale=rationale)

           except Exception as e:
               return ScenarioValidationResult(success=False, exception=e)

       def _check_professional_tone(self, text: str) -> bool:
           """Check if text maintains professional tone."""
           unprofessional_words = ["hey", "sup", "lol", "omg", "wtf"]
           return not any(word in text.lower() for word in unprofessional_words)

Testing Your Scenarios
----------------------

Check success of a run in oracle mode:

.. code-block:: python

   from are.simulation.scenarios.utils.cli_utils import run_and_validate

   # Run scenario in oracle mode
    run_and_validate(MyCustomScenario())

Basic Testing
~~~~~~~~~~~~~

Test that your scenarios are properly registered:

.. code-block:: python

   from are.simulation.scenarios.utils.registry import registry

   # This will trigger scenario discovery via entry points
   all_scenarios = registry.get_all_scenarios()

   # Check if your scenario is in the registry
   if "my_custom_scenario" in all_scenarios:
       print("Scenario successfully registered!")

       # You can also instantiate your scenario
       scenario_class = all_scenarios["my_custom_scenario"]
       scenario = scenario_class()
   else:
       print("Scenario not found in registry")

Unit Testing
~~~~~~~~~~~~

Create comprehensive unit tests for your scenarios:

.. code-block:: python

   import unittest
   from unittest.mock import Mock, patch
   from my_scenarios.email_scenario import MyCustomScenario

   class TestMyCustomScenario(unittest.TestCase):

       def setUp(self):
           self.scenario = MyCustomScenario()
           self.scenario.initialize()

       def test_apps_initialization(self):
           """Test that apps are properly initialized."""
           self.assertIsNotNone(self.scenario.email_app)
           self.assertIsNotNone(self.scenario.calendar_app)
           self.assertEqual(len(self.scenario.apps), 2)

       def test_initial_data(self):
           """Test that initial data is properly set up."""
           emails = self.scenario.email_app.get_emails()
           self.assertGreater(len(emails), 0)

           events = self.scenario.calendar_app.get_events()
           self.assertGreater(len(events), 0)

       def test_validation_success(self):
           """Test validation with successful completion."""
           # Mock environment with successful state
           mock_env = Mock()
           mock_calendar = Mock()
           mock_calendar.get_upcoming_meetings.return_value = ["meeting1", "meeting2"]
           mock_env.get_app.return_value = mock_calendar

           result = self.scenario.validate(mock_env)
           self.assertTrue(result.success)

       def test_validation_failure(self):
           """Test validation with failed completion."""
           # Mock environment with failed state
           mock_env = Mock()
           mock_calendar = Mock()
           mock_calendar.get_upcoming_meetings.return_value = ["meeting1"]  # Only original
           mock_env.get_app.return_value = mock_calendar

           result = self.scenario.validate(mock_env)
           self.assertFalse(result.success)

   if __name__ == "__main__":
       unittest.main()

Complete Example Package
------------------------

Here's a complete example of a scenario package structure:

.. code-block:: text

   my_are_simulation_scenarios/
   ├── pyproject.toml
   ├── README.md
   ├── tests/
   │   ├── __init__.py
   │   └── test_scenarios.py
   └── my_scenarios/
       ├── __init__.py
       ├── email_scenario.py
       ├── calendar_scenario.py
       ├── complex_scenario.py
       └── registration.py

**my_scenarios/__init__.py**:

.. code-block:: python

   """My custom Meta Agents Research Environments scenarios package."""
   __version__ = "0.1.0"

**my_scenarios/email_scenario.py**:

.. code-block:: python

   from are.simulation.scenarios.scenario import Scenario, ScenarioValidationResult
   from are.simulation.scenarios.utils.registry import register_scenario
   from are.simulation.apps.email_client import EmailClientApp
   from are.simulation.types import EventType

   @register_scenario("email_management_scenario")
   class EmailManagementScenario(Scenario):
       """Scenario focused on email management tasks."""

       scenario_id = "email_management_scenario"

       def init_and_populate_apps(self):
           self.email_app = EmailClientApp()

           # Add sample emails
           self.email_app.add_email(
                email=Email(
                    sender="boss@company.com",
                    recipients=[self.email_app.user_email],
                    subject="Urgent: Project Deadline",
                    content="The project deadline has been moved up. Please confirm receipt.",
                ),
           )

           self.email_app.add_email(
                email=Email(
                    sender="spam@marketing.com",
                    recipients=[self.email_app.user_email],
                    subject="Amazing Deal! Click Now!",
                    content="Limited time offer! Buy now and save 90%!",
                ),
           )

           self.apps = [self.email_app]

       def build_events_flow(self):
            aui = self.get_typed_app(AgentUserInterface)
            email_client = self.get_typed_app(EmailClientApp)

            with EventRegisterer.capture_mode():
                # User asks agent to read emails
                event1 = aui.send_message_to_agent(
                    content="Read my emails and respond to the urgent one."
                ).depends_on(None, delay_seconds=5)
                # Agent respond to urgent email
                event2 =  email_client.send_email(
                    recipients=["boss@company.com"],
                    subject="Re: Urgent: Project Deadline",
                    body="Received and understood. Will adjust timeline accordingly.",
                ).oracle().depends_on(event1, delay_seconds=5)

       def validate(self, env) -> ScenarioValidationResult:
           try:
               email_app = env.get_app("EmailClientApp")
               sent_emails = email_app.folders[EmailFolderName.SENT].emails

               # Check if response was sent
               success = any("Re: Urgent" in email.subject for email in sent_emails)
               return ScenarioValidationResult(success=success)
           except Exception as e:
               return ScenarioValidationResult(success=False, exception=e)

**my_scenarios/registration.py**:

.. code-block:: python

   def register_scenarios(registry):
       """Register all scenarios in this package."""
       # Import modules containing scenario classes
       # The decorators will handle the registration
       import my_scenarios.email_scenario
       import my_scenarios.calendar_scenario
       import my_scenarios.complex_scenario

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Scenarios Not Being Discovered**

1. **Check Package Installation**: Make sure your package is properly installed

   .. code-block:: bash

      pip list | grep my-are-scenarios

2. **Verify Entry Point**: Check that your entry point is correctly defined in ``pyproject.toml``

3. **Test Registration Function**: Verify that your registration function is being called

   .. code-block:: python

      def register_scenarios(registry):
           print("Registration function called!")  # Add debug logging
           import my_scenarios.email_scenario

4. **Check for Import Errors**: Make sure there are no import errors in your scenario modules

**Registration Decorator Not Working**

1. **Import Order**: Ensure modules are imported after the registry is set up
2. **Decorator Syntax**: Verify the ``@register_scenario`` decorator is used correctly
3. **Unique IDs**: Make sure scenario IDs are unique across all packages

**Validation Errors**

1. **Environment Access**: Ensure you're accessing the environment correctly in validation
2. **App Names**: Verify app names match exactly (case-sensitive)
3. **Exception Handling**: Always wrap validation logic in try-catch blocks

Debugging Tools
~~~~~~~~~~~~~~~

**Inspect Available Entry Points**:

.. code-block:: python

   import pkg_resources
   for entry_point in pkg_resources.iter_entry_points('are.simulation.scenarios'):
       print(f"Found entry point: {entry_point.name} -> {entry_point.dist}")

**List Registered Scenarios**:

.. code-block:: python

   from are.simulation.scenarios.utils.registry import registry
   scenarios = registry.get_all_scenarios()
   for scenario_id, scenario_class in scenarios.items():
       print(f"Scenario: {scenario_id} -> {scenario_class}")

**Test Scenario Loading**:

.. code-block:: python

   try:
       scenario_class = registry.get_all_scenarios()["my_custom_scenario"]
       scenario = scenario_class()
       print("Scenario loaded successfully!")
   except KeyError:
       print("Scenario not found in registry")
   except Exception as e:
       print(f"Error loading scenario: {e}")

Hands-On Learning Resources
---------------------------

The Meta Agents Research Environments repository includes practical tutorials that complement this guide:

**Scenario Development Tutorial** (``are/simulation/scenarios/scenario_tutorial/scenario.py``)
   A complete walk-through of creating a scenario from scratch, including all the steps covered in this guide.

**DAG Scenario Tutorial** (``are/simulation/tutorials/event_dag.py``)
   Learn how to build complex scenarios using Event Graphs with dependencies, as demonstrated in the Advanced Scenario Patterns section.

**Validation Tutorial** (``are/simulation/scenarios/scenario_validation_tutorial/scenario.py``)
   Practical examples of implementing robust validation logic for different types of scenarios.

**Environment Tutorial** (``are/simulation/tutorials/environment.py``)
   Understand how to work with environments and configure them for your scenarios.

**Events Tutorial** (``are/simulation/scenarios/scenario_events_tutorial/scenario.py``)
   Learn the fundamentals of creating and managing events in scenarios.

**Advanced Events Tutorial** (``are/simulation/tutorials/events_advanced.py``)
   Explore sophisticated event patterns and conditional logic for complex scenarios.

These tutorials provide runnable code examples that you can study, modify, and use as starting points for your own scenarios.


Next Steps
----------

Now that you've learned the basics of creating and registering scenarios, you might be interested in developing your own apps.

Continue to the :doc:`apps_tutorial` to learn how to create and register new apps.
