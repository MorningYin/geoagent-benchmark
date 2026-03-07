..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`apps` Apps
--------------------

Environment: a Family of Apps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of Agents Research Environments' core principles is "apps." As explained in the previous section, environments are based on a major concept: applications.
To train agents to solve daily and tedious tasks for users, it is essential to get as close as possible to the real deployment environment of these agents.
A first env we introduce primarily aims at training agents capable of evolving in environments similar to those of operating systems such as iOS,
Android, Meta RayBan OS, QuestOS, etc. It is therefore natural to design this environment by integrating applications, much like a mobile phone.

Definition: An application, in the platform, is a coherent set of APIs. Its purpose is to interact with a data source.
For example, a Mails app will have tools such as ``send_email`` or ``delete_email``, as well as some database of emails.
These applications are "stateful," meaning that the result of two consecutive calls to the same API with the same arguments is
not guaranteed to be identical, as it depends on the internal state of the application. This design aims to train agents to reason
by taking into account a global context.

Applications
~~~~~~~~~~~~

.. dropdown:: :octicon:`database` Stateful by Design
    :animate: fade-in

    Each App is stateful, meaning it maintains an internal state that evolves with every interaction.
    Unlike conventional stateless APIs, Apps store all their state directly inside the environment, with no reliance on an external database.
    This ensures that experiments and scenarios can be replayed exactly as they were run, which is essential for benchmarking and debugging agents.

    By fully controlling and validating the state, the system guarantees that an agent's actions produce verifiable side effects.
    This also enables richer, more realistic interactions: an agent can manage emails, calendars, files, or messages that change over time,
    just like in a real operating system.

    Technically, each App must implement a ``get_state()`` method, which serializes its internal state. This snapshot can then be used to monitor the environment,
    compare states during evaluation, or reset the App to its initial configuration.

.. dropdown:: :octicon:`command-palette` Exposing APIs via Tools
    :animate: fade-in

    Inside the environment, an App's functionality is exposed through methods in its Python class. Any method can be decorated to become a tool that the agent can call. The ``@app_tool`` decorator (see :doc:`../../api_reference/apps`) marks a method as an available tool, while an additional ``@event_registered`` decorator records whether the action is a read or write operation.

    Read actions retrieve information without modifying state — for example, reading an email or listing files — whereas write actions directly update the App's state, such as sending a message, creating a new calendar event, or saving a log. This distinction is crucial for tracking what the agent does and validating its behavior.

    Every decorated method automatically generates clear documentation describing its name, parameters, and return type, which is provided to the agent at runtime to guide its use of the tool.

.. dropdown:: :octicon:`tools` Unified Tool Creation
    :animate: fade-in

    The platform enforces a unified, standardized declaration for all App APIs. By following this structure and using the provided decorators, any valid method can be
    automatically converted into a tool that appears in the agent's toolset, with no extra manual configuration required.

    This automatic conversion means developers only need to write standard Python methods. The system handles tool registration, documentation,
    event logging, and integration with the agent's environment interface, which greatly simplifies scaling the system as new Apps are added.

.. dropdown:: :octicon:`apps` Creating/Wrapping your own App
    :animate: fade-in

    The App metaclass is deliberately designed to be abstract and highly extensible. This allows developers to wrap existing applications or APIs and
    integrate them seamlessly into the platform as Apps.

    For example, a developer could encapsulate a Spotify API to control music playbook, a phone app to simulate calls or messages, or even a weather app
    that pulls real-time weather data but stores it inside the sandbox for consistency and replayability.

    This flexibility allows researchers to build complex, realistic environments while preserving the fully controlled, isolated nature of the simulation.
    In short, Applications transform APIs into stateful, testable, and observable tools, forming a robust bridge between static language models and dynamic,
    interactive agent behavior.

.. dropdown:: :octicon:`list-unordered` Non-exhaustive list of apps native to the Agents Research Environments
    :animate: fade-in

    * Contacts
    * EmailClient
    * Messaging
    * FileSystem
    * Calendar
    * Shopping
    * Cab
    * City
    * ApartmentListing

    Among these apps, a couple play a crucial role in the execution of the environment.

Some `apps` in the initial environment play a crucial role in the execution of scenarios. We give more details below about the most important ones,
giving a glimpse of the possibilities offered by the platform. Thanks to the high flexibility of the system, it is also possible to integrate MCP interfaces (:doc:`../../api_reference/mcp_app`)
and any other interface as long as they are wrapped to match the platform's standards. For instance, some recent efforts have integrated SQL databases to test agents'
compatibility with different search strategies.

**Agent-User Interface**

The Agent-User Interface (AUI) app is a required component in every environment, serving as the communication bridge between users and AI agents.
Unlike traditional chat models where interaction happens through direct messaging, our system models user-agent communication through a dedicated ARE application.

This design requires the agent to make specific tool calls when reporting information to the user, rather than simply generating responses.
When users send messages to the agent through the AUI, their content is automatically injected into the agent's memory for processing.
We highlight that this is the current implementation choice and that better alternatives may exist, like using the
Notification System to surface user messages to the agent (see :doc:`notifications`).

This tool-based communication model provides several advantages:

* It allows precise control over when the agent should communicate with the user versus when it should continue working silently on tasks
* It enables users to interact with the agent even while the agent is actively engaged in completing other objectives

The AUI thus creates a more realistic interaction pattern that mirrors how people actually work with digital assistants in complex, multi-tasking environments.

.. note::
   Code Pointer: More information can be found in the AUI App (see ``apps/agent_user_interface.py``).

**System App**

Some applications operate at a higher abstraction level than conventional data sources. For instance, the System app offers agents essential tooling for temporal
operations, including:

* Retrieving the current simulation time
* Waiting for specified durations
* Suspending execution until new events arrive in the system

The waiting APIs represent a particularly sophisticated feature set that directly interfaces with the simulation's core execution model.
These APIs possess the unique capability to modify environment execution flow, allowing agents to pause their operations in a controlled manner that
integrates seamlessly with the event-driven architecture.

Crucially, these waiting mechanisms serve critical evaluation purposes by enabling the acceleration of long-horizon task execution,
allowing researchers to efficiently evaluate agent performance across extended temporal scenarios without requiring real-time execution delays.

This system-level functionality demonstrates how the platform not only supports application-specific interactions but also fundamental simulation control mechanisms
that enable complex agent behaviors and efficient evaluation methodologies.

.. note::
   Code Pointer: More information can be found in the System App (see ``apps/system.py``).


**Next Steps**
    * Keep reading the Foundations guide to learn more about :doc:`events`.
    * Check the technical details of Apps in :doc:`../../api_reference/apps`.
