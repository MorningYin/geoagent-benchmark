..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`bell` Notifications
=============================

Notification System: The second interface with Agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the Agents Research Environments, the Notification System is a core environmental component and serves as the secondary interface between agents and their environment,
complementing the primary tool-based interaction model. This system is analogous to a notification framework found on a mobile device, which alerts
users to important changes across their device ecosystem without requiring them to manually monitor each individual application (see ``notification_system.py``).

The Notification System operates as a selective observability mechanism within the environment, implementing partial rather than complete observability
of the environment changes. This design choice reflects the realistic constraint that not every event or state transition warrants immediate attention or
generates a corresponding notification. Instead, the system filters events based on relevance, urgency, and configured notification policies to present
agents with actionable information streams.

Notifications represent a curated subset of the broader event ecosystem, transforming raw environmental events into structured alerts that can influence
agent decision-making and behavior.

**Notification Policy**

The Notification System operates under a configurable Notification Policy (verbosity level), maintaining a whitelist of tools authorized to emit notifications.
This policy determines which events become visible to agents, effectively controlling the information flow between the environment and agent awareness.

The current implementation provides three verbosity levels: ``LOW``, ``MEDIUM``, and ``HIGH``, creating a graduated spectrum of environmental transparency.
Each successive level provides increasingly comprehensive details about environmental activities, from critical events only (``LOW``) to near-comprehensive
visibility (``HIGH``).

The Notification Policy framework offers maximum flexibility, allowing researchers to dynamically adjust verbosity levels, create custom notification tiers,
and experiment with varying degrees of environmental transparency. This configurability enables systematic investigation of how information availability
affects agent behavior, supporting research questions such as whether agents behave differently under limited notification conditions or exhibit more proactive
versus reactive strategies based on environmental observability.

**How do Agents interact with the Notification System?**

Agents interact with the Notification System through a structured pull-based mechanism that integrates seamlessly with the agent's work cycle.
At each environment tick, a subset of processed events can trigger notifications, based on the configured Notification Policy.
These notifications are queued in the Notification System's MessageQueue, which operates as a priority queue indexed by timestamp to ensure temporal ordering.

At the beginning of each agent step, (Thought, Action, and Observation), the agent orchestration layer manually retrieves pending notification
from the Notification System's queue. These retrieved notifications are then injected into the agent's context, effectively expanding the agent'
knowledge base with relevant information about recent environmental changes that occurred since the last interaction cycle.

This notification integration mechanism represents one proposed approach to managing environment updates from the agents modeling side.
The current implementation provides a foundation for notification handling, but alternative approaches may prove more effective depending on specific
use cases and agent architectures. The flexibility of this design allows agent modeling researchers to experiment with different strategies, enabling
exploration of how notification timing, prioritization, and context integration affect agent performance and decision-making processes.


.. thumbnail:: /_static/notifications_in_agents.png
   :alt: Notifications are injected in the agent's context at each step.
   :width: 100%
   :align: center
   :group: notifications
   :title: Notification System Integration - Flow diagram showing how notifications are injected into the agent's context at each ReAct step to provide environmental updates


**Next Steps**
    * Keep reading the Foundations guide to learn more about :doc:`scenarios`.
