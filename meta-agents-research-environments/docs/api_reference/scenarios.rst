..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`book` Scenarios API
=============================

Scenarios define the tasks and environments that agents operate within. This section documents the core Scenario API and related classes for creating and managing simulation scenarios.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Overview
--------

Scenarios in Meta Agents Research Environments are complete simulation setups that include:

* **Apps and Initial State**: The applications and data available to agents
* **Events and Timeline**: Dynamic occurrences that happen during execution
* **Task Definition**: What the agent needs to accomplish
* **Validation Logic**: How success is measured and evaluated


Creating Custom Scenarios
-------------------------

For a comprehensive guide on creating and registering custom scenarios, see :doc:`../tutorials/scenario_development`.


To create a custom scenario, inherit from the base ``Scenario`` class and implement the required methods:

Basic Scenario Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.scenarios.scenario import Scenario
   from are.simulation.scenarios.utils.registry import register_scenario
   from are.simulation.apps.email_client import EmailClientApp, Email
   from are.simulation.apps.calendar import CalendarApp
   from are.simulation.types import EventType

   @register_scenario("my_custom_scenario")
   class MyCustomScenario(Scenario):
       """
       A custom scenario that demonstrates email and calendar integration.
       """
       scenario_id: str = "my_custom_scenario"
       start_time: float | None = 0
       duration: float | None = 600  # Duration in seconds

       def init_and_populate_apps(self):
           """Initialize and populate apps with data."""
           # Create apps
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
           self.apps = [email_app, calendar_app]

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


App Initialization Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Simple App Setup**

.. code-block:: python

   def init_and_populate_apps(self):
       # Create apps
       file_system = VirtualFileSystemApp()
       email_app = EmailClientApp()

       # Populate file system
       file_system.create_file("/documents/report.txt", "Quarterly report content")
       file_system.mkdir("/images")

       # Populate email
       email_app.add_email(
           email_app.add_email(
                email=Email(
                    sender="client@company.com",
                    recipients=[self.email_app.user_email],
                    subject="Report Request",
                    content="Please send the quarterly report.",
                ),
           )
       )

       self.apps = [file_system, email_app]

**Complex App Configuration**

.. code-block:: python

   def init_and_populate_apps(self):
       # Create apps with custom configurations
       shopping_app = ShoppingApp()
       messaging = MessagingApp()

       # Set up product catalog
       products = {
            "6938111410": {
                "name": "Running Shoes",
                "product_id": "6938111410",
                "variants": {
                    "4153505238": {
                        "item_id": "4153505238",
                        "options": {
                            "size": "8",
                            "color": "red",
                            "material": "leather",
                            "sole": "EVA",
                        },
                        "available": True,
                        "price": 158.67,
                    },
                    "1775591963": {
                        "item_id": "1775591963",
                        "options": {
                            "size": "10",
                            "color": "white",
                            "material": "leather",
                            "sole": "EVA",
                        },
                        "available": True,
                        "price": 154.75,
                    },
                },
            },
            "8310926033": {
                "name": "Water Bottle",
                "product_id": "8310926033",
                "variants": {
                    "1434748144": {
                        "item_id": "1434748144",
                        "options": {
                            "capacity": "1000ml",
                            "material": "glass",
                            "color": "red",
                        },
                        "available": False,
                        "price": 49.72,
                    },
                    "4579334072": {
                        "item_id": "4579334072",
                        "options": {
                            "capacity": "750ml",
                            "material": "glass",
                            "color": "black",
                        },
                        "available": True,
                        "price": 54.85,
                    },
                },
            },
        }
       shopping_app.load_products_from_dict(products)

       # Add conversation
        messaging.add_conversations(
            [
                Conversation(
                    participants=["Dad", "Me"],
                    title="Dad and I",
                    conversation_id="23",
                    messages=[
                        Message(
                            sender="Dad",
                            message_id="0",
                            timestamp=1708550557,
                            content="Hey kid, how are you doing?",
                        ),
                        Message(
                            sender="Me",
                            message_id="1",
                            timestamp=1708550560,
                            content="I am doing good dad. How are you?",
                        ),
                        Message(
                            sender="Dad",
                            message_id="2",
                            timestamp=1708550570,
                            content="I am doing good son. Just came back home from camping trip. It was good. Missed you. What about you?",
                        ),
                        Message(
                            sender="Me",
                            message_id="3",
                            timestamp=1708550580,
                            content="Great dad! I missed it.",
                        ),
                        Message(
                            sender="Dad",
                            message_id="4",
                            timestamp=1708550590,
                            content="Btw son, what car does aunt Linda drive?",
                        ),
                        Message(
                            sender="Me",
                            message_id="5",
                            timestamp=1708550600,
                            content="It's Toyota.",
                        ),
                        Message(
                            sender="Dad",
                            message_id="6",
                            timestamp=1708550620,
                            content="Thanks kid. I will remember it. See you soon. Bye",
                        ),
                        Message(
                            sender="Me",
                            message_id="7",
                            timestamp=1708550650,
                            content="Bye Dad",
                        ),
                    ],
                ),
            ]
        )

       self.apps = [shopping_app, payment_app]

Event Flow Patterns
~~~~~~~~~~~~~~~~~~~

**Sequential Events**

.. code-block:: python

   def build_events_flow(self):
        with EventRegisterer.capture_mode():
            # Event 1: Send initial message
            event1 = (
                aui.send_message_to_user(
                    content="Task started",
                )
                .oracle()
                .depends_on(None, delay_seconds=5)
            )

            # Event 2: Follow-up after 60 seconds
            event2 = (
                aui.send_message_to_user(
                    content="Task done",
                )
                .oracle()
                .depends_on(event1, delay_seconds=60)
            )

**Conditional Events**

.. code-block:: python

    # These events trigger when certain conditions are met

    def enough_emails_condition(env: AbstractEnvironment) -> bool:
        """Condition: trigger when we have at least 2 emails"""
        email_app = env.get_app("EmailClientApp")
        return len(email_app.folders[EmailFolderName.INBOX].emails) >= 2

    # Create a condition check event
    condition_check = ConditionCheckEvent.from_condition(enough_emails_condition)

    # Event that triggers when condition is met
    conditional_email = Event.from_function(
        self.email_app.add_email,
        email=Email(
            sender="system@example.com",
            recipients=[self.email_app.user_email],
            subject="Conditional Event Triggered!",
            content="This email was sent because the condition (2+ emails) was met!",
        ),
    ).depends_on(condition_check, delay_seconds=1)

Validation Patterns
~~~~~~~~~~~~~~~~~~~

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

**State-Based Validation**

.. code-block:: python

    def validate(self, env: AbstractEnvironment) -> ScenarioValidationResult:
        """
        Final scenario validation - checking the overall
        state after all events have completed.
        """
        try:
            aui_app = env.get_app("AgentUserInterface")

            # Check that agent provided responses
            agent_messages = aui_app.get_all_messages_from_agent()
            has_one_responses = len(agent_messages) == 1

            # Check that the correct response was given
            has_correct_response = any(
                msg.content == "I am a robot" for msg in agent_messages
            )

            # Check that user made the initial request
            user_messages = aui_app.get_all_messages_from_user()
            has_user_request = len(user_messages) > 0

            # Overall success criteria
            success = has_one_responses and has_correct_response and has_user_request

            return ScenarioValidationResult(success=success)

        except Exception as e:
            return ScenarioValidationResult(success=False, exception=e)

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

    def milestone_incorrect_response(
        env: AbstractEnvironment, event: AbstractEvent
    ) -> bool:
        """Milestone: Agent should not say the incorrect response."""
        if not isinstance(event, Event):
            return False
        return not (
            event.action.app.__class__ == AgentUserInterface
            and event.function_name() == "send_message_to_user"
            and event.action.args["content"] == "I am not a robot •`_´• "
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
        milestones=[milestone_correct_response, milestone_incorrect_response],
        minefields=[minefield_unsafe_response],
        timeout=15,
    ).with_id("agent_validation")
    agent_validation.depends_on(user_request)
    self.events.append(agent_validation)


Advanced Features
-----------------

Hints
~~~~~

.. code-block:: python

   from are.simulation.types import Hint, HintType

   def build_events_flow(self):
       # Add event with associated hint
        with EventRegisterer.capture_mode():
            event1 = aui.send_message_to_agent(
                content="Send an email to the client for the delay."
            ).depends_on(None, delay_seconds=5).with_id("event1")

       # Add hint for this event
       if not self.hints:
           self.hints = []

       self.hints.append(Hint(
           hint_type=HintType.TASK_HINT,
           content="Remember to include an apology in your email",
           associated_event_id="event1"
       ))

Tool Augmentation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.types import ToolAugmentationConfig

   class UnreliableScenario(Scenario):
       def __init__(self):
           super().__init__()
           # Make tools fail 20% of the time
           self.tool_augmentation_config = ToolAugmentationConfig(
               tool_failure_probability=0.2,
           )

Environment Events
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.scenarios.utils.scenario_expander import EnvEventsConfig

   class DynamicScenario(Scenario):
       def __init__(self):
           super().__init__()
           # Add random environment events
           self.env_events_config = EnvEventsConfig(
               num_env_events_per_minute=3,
               env_events_seed=42,
               weight_per_app_class = {
                    "EmailClientApp": 1.0,
                    "ApartmentListingApp": 1.0,
                },
           )


Core Scenario Classes
---------------------

.. automodule:: are.simulation.scenarios.scenario
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.scenarios.scenario.Scenario
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: are.simulation.scenarios.validation_result.ScenarioValidationResult
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Configuration Classes
---------------------

.. autoclass:: are.simulation.scenarios.config.ScenarioRunnerConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.scenarios.utils.scenario_expander.EnvEventsConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.ToolAugmentationConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Utility Classes
---------------

.. autoclass:: are.simulation.types.Hint
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.HintType
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.ValidationEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.types.AgentValidationEvent
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
