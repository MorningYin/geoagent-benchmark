..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`calendar` Events
==========================

Event-Based Environment: Foundational Concepts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The Meta Agents Research Environments platform is built on the foundational principle that "everything is an event"**. Existing agentic and tool-use benchmarks have demonstrated that
validation presents significant challenges when evaluating agentic trajectories—whether through environment state validation, agent action validation,
LLM-as-Judge approaches, or identifying potential failure modes. To address these complexities, we designed the Environment to be maximally general,
capturing all system interactions as events. This comprehensive event logging enables the application of diverse validation methodologies and ensures forward
compatibility with emerging validation techniques. This event-driven architecture forms the core of the platform's extensible design.


Overview
..........

Our simulation environment is built on a comprehensive event-driven architecture that models interactions between agents, applications, and the environment itself.
This system provides a structured way to orchestrate complex scenarios while maintaining complete observability and control over the simulation timeline.


.. thumbnail:: /_static/main-diagram.png
   :alt: Diagram of the Events Flow
   :width: 100%
   :align: center
   :group: events
   :title: Event-Based Environment Flow - Comprehensive diagram showing how all system interactions are modeled as events in the simulation environment

Core Philosophy
................

The environment operates on the principle that everything changing the state of the environment is an event. Whether an agent sends an email, a timer expires,
or a validation check occurs, these actions are all represented as events in a unified system. This approach provides several key benefits:

    *   **Deterministic execution**: Events are processed in a predictable order based on their scheduled time
    *   **Complete auditability**: Every action is logged and can be replayed or analyzed
    *   **Flexible scheduling**: Events can be scheduled at specific times or relative to other events
    *   **Transactional semantics**: Events either complete successfully or fail atomically


Event Lifecycle
................

Events flow through a well-defined four-stage lifecycle within the environment (see ``environment.py``),
ensuring deterministic execution and comprehensive traceability.

.. dropdown:: Event Creation
    :animate: fade-in

    Events are instantiated to represent future or instant actions requiring execution. The system supports multiple creation pathways, including agent-driven
    events that are generated when agents decide to perform operations such as tool calls and environment interactions. Environmental triggers constitute another
    major category, comprising time-based events and scenario-defined milestones that advance simulation state. The system also supports conditional events,
    which are rule-based events that activate when specific system conditions are satisfied (i.e. a message has been sent to the user).


.. dropdown:: Event Scheduling
    :animate: fade-in

    Created events enter a priority queue system ordered by execution time, that support both absolute and relative timing models. Absolute timing
    allows events to be scheduled for precise simulation timestamps, while relative timing enables events to be scheduled with temporal offsets relative to other events.
    The scheduler incorporates dependency management, storing events as a DAG, ensuring that events await prerequisite completion before becoming eligible for execution.
    Conditional scheduling further extends this capability by allowing events to include runtime conditions that determine execution eligibility.


.. dropdown:: Event Execution
    :animate: fade-in

    When an event reaches its scheduled time, or when an agent calls a tool (instant execution),  the environment processes it through a standardized workflow.
    The system then captures results, collecting return values, state changes, and any exceptions that occur during execution. Finally, completion recording
    generates a completed event record for persistent logging. This execution model ensures consistent handling regardless of event type or complexity.

.. dropdown:: Event Logging
    :animate: fade-in

    All executed events are stored in am EventLog that maintains detailed execution metadata, including success/failure status, return values, and exception details.
    The logging system captures temporal information such as creation timestamps, scheduled times, and actual execution times. Dependency tracking maintains causal
    relationships between events and their prerequisites, while state snapshots capture environment state before and after event execution when applicable.
    This logging framework enables post-hoc analysis, debugging, and validation of agent trajectories.


The Event Loop thus ensures that every interaction with the environment is correctly scheduled, processed and logged to ensure a complete traceability of events
for validation.


Event Types
............

The environment defines several distinct categories of events based on their origin and function within the simulation environment.
In particular, to maintain some coherence at the system level, different actions wrt the environment are subclasses of the Event abstract class.


.. dropdown:: :octicon:`dependabot` Agent Events
    :animate: fade-in

    Agent Events actions initiated by autonomous agents within the simulation. These events encompass API calls to applications such as sending emails,
    creating files, or making web requests. They also include decision-making processes that affect the environment and responses to environmental stimuli
    or user instructions. Agent events form the core of autonomous behavior within the simulation, capturing the full spectrum of agent-initiated interactions.


.. dropdown:: :octicon:`devices` Environment Events
    :animate: fade-in

    Environment Events are controlled by the simulation itself and include timer-based triggers that advance the scenario according to predefined schedules.
    These events handle initial setup conditions and scenario initialization, ensuring proper simulation state at startup.
    Environmental state changes that affect available actions also fall into this category, providing the dynamic backdrop against which agents operate.

.. dropdown:: :octicon:`person` User Events
    :animate: fade-in

    User Events are a specialized subset of Environmental Events, distinguished by their human-initiated origin.
    While User Events capture deliberate actions performed by the user, they contrast with broader Environmental Events that
    represent information or stimuli received by the user's environment from external sources, such as receiving a message from a friend or system notifications.
    This distinction ensures clear attribution of agency within the simulation, separating user-driven actions from externally-originated environmental
    changes that affect the user's context. When the platform operates in "demo" mode, all user interactions with the interface are automatically captured
    and logged as User Events, providing comprehensive tracking of human participation in the simulation for analysis and replay purposes


.. dropdown:: :octicon:`checklist` Conditional Events
    :animate: fade-in

    Conditional Events serve as monitoring mechanisms that periodically check conditions and can trigger other events when specific criteria are met.
    These events enable reactive behaviors based on environmental state, allowing the simulation to respond dynamically to changing conditions.
    They facilitate milestone detection and scenario progression, while also handling timeout scenarios and failure recovery mechanisms.

.. dropdown:: :octicon:`check-circle` Validation Events
    :animate: fade-in

    Validation Events are a special category of events that assesses whether the simulation is progressing as expected.
    These events perform milestone achievement checking, ensuring that key scenario objectives are being met at a certain timestamp.
    They also handle constraint violation detection, including the identification of problematic behaviors or "minefields" that should be avoided.
    Overall scenario success and failure determination is managed through these validation events.


Other types of events exist, such as ``OracleEvent``, which will be further discussed and explained in the :doc:`scenarios` section.


Dependencies and Scheduling
........................................

A powerful feature of the system is its sophisticated orchestration capability, which enables the creation of complex, realistic scenarios with proper
temporal and conditional relationships. This is achieved through the modeling of dependencies between events as Directed Acyclic Graphs (DAGs) of Events.

**Sequentiasl Dependencies**

Sequential Dependencies is the chaining of events, where an event only executes upon the successful completion of its prerequisites.
This mechanism supports multi-step workflows that necessitate proper execution order, ensuring logical progression through intricate processes.
Conditional execution, based on prior results, is also supported, enabling scenarios to adapt to intermediate outcomes.
For instance, in the example below, Event e1 is processed immediately at the simulation's start, and e2 will be processed 30 seconds after e1.


Here's an example of a simple event dependency chain after 30 seconds of simulation:

.. thumbnail:: /_static/basic_graph_tuto.png
    :alt: Basic Event Graph Tutorial
    :width: 100%
    :align: center
    :group: events
    :title: Sequential Event Dependencies - Simple event chain showing e1 executing immediately followed by e2 scheduled 30 seconds later, demonstrating temporal sequencing

**Parallel Execution**

Parallel Execution is also enabled by the environment. While most scenarios are modeled as sequential dependencies,
multiple branches of events can be executed in parallel to model multiple things happening simultaneously.

**Conditional Execution**

Conditional Execution allows events to be configured to execute only when certain conditions are met, enabling sophisticated scenario branching.
This feature supports branching scenario paths that can adapt to different circumstances, adaptive responses to agent behavior that maintain scenario coherence,
and dynamic scenario modification based on runtime conditions. The conditional execution framework enables the creation of scenario templates to scale data
generation for agents (more info in next sections).



.. thumbnail:: /_static/complex_dag.png
    :alt: Complex Event Graph DAG
    :width: 100%
    :align: center
    :group: events
    :title: Complex Event DAG Structure - Sophisticated interconnected event graph showing multiple dependencies, timing constraints, and validation pathways for realistic scenario design

This complex graph shows how multiple events can be interconnected with various dependencies, timing constraints and validation, allowing for realistic and sophisticated scenario design.

For detailed technical information about events, see :doc:`../../api_reference/events`.


**Next Steps**
    * Keep reading the Foundations guide to learn more about :doc:`notifications`.
    * Check the technical details of Events in :doc:`../../api_reference/events`.
