..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`gear` App Implementation Tutorial
===========================================

**File**: ``are/simulation/scenarios/scenario_apps_tutorial/scenario.py``

**Concepts Covered**:

- Data model design with dataclasses and enums
- App class structure and inheritance patterns
- Tool method registration and decorators
- State management and persistence
- Event integration and validation
- CRUD operations and error handling

**What You'll Learn**:
Building custom apps is essential for extending Meta Agents Research Environments functionality.
This comprehensive tutorial walks you through creating a complete task management app,
demonstrating all the key patterns and best practices for app development.

**States and Simulation Environment**:

App States in the simulations should be easy to serialize and load quickly for repeated, reproducible
experiments and training.
Meta Agents Research Environments scenarios are run one by one and app state can be easily kept in memory usually. When using
the GUI, some assumption is made that the app state can be sent serialized to the client.

You usually do not need to store the state through an API or in a DB, keep it simple unless your
app state is very large.

**State Reproducibility**:

We want to generate reproducible scenarios that can start from an initial state and run
predictably at every evaluation/training runs. You should make sure that your app state and
tools perform in a deterministic way. If you need randomness, make sure to use a seed that
can be reset when the scenario is reset.

**Real Apps**:

There is nothing stopping you from using this framework to build real apps that can connect
to the real world so you can test your agent in the real world. However you should be careful
as this can open up a lot of security concerns. Meta Agents Research Environments does not provide any security guarantees
outside of the simulated environment, in particular, it does not protect against prompt injection
attacks.

Note that real apps are not really reproducible, they are fun, but do not use them for reproducible
benchmarks.

**Key Components Demonstrated**:

**Data Models**:

.. code-block:: python

   @dataclass
   class Task:
       title: str
       description: str = ""
       task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
       priority: Priority = Priority.MEDIUM
       completed: bool = False

       def __post_init__(self):
           # Validation logic ensures data integrity
           if not self.title or len(self.title.strip()) == 0:
               raise ValueError("Task title cannot be empty")

**App Class Structure**:

.. code-block:: python

   @dataclass
   class SimpleTaskApp(App):
       name: str | None = "SimpleTaskApp"
       tasks: dict[str, Task] = field(default_factory=dict)

       def __post_init__(self):
           super().__init__(self.name)

       def get_state(self) -> dict[str, Any]:
           return get_state_dict(self, ["tasks"])

       def load_state(self, state_dict: dict[str, Any]):
           # Restore app state from saved data

**Tool Method Registration**:

.. code-block:: python

   @type_check  # Runtime type validation
   @app_tool()  # Agent-accessible tool
   @event_registered(operation_type=OperationType.WRITE)  # Write action
   def create_task(self, title: str, description: str = "",
                  priority: str = "Medium") -> str:
       """Create a new task with validation and storage"""

**Architecture Patterns**:

1. **Data Layer**: Clean dataclass models with validation
2. **Business Logic**: App class managing state and operations
3. **Interface Layer**: Decorated tool methods for agent access
4. **Integration Layer**: Event registration and environment hooks

**Key Takeaways**:

- How to design robust data models with proper validation
- App lifecycle management (initialization, state, reset)
- Tool registration patterns and decorator usage
- State persistence for scenario reproducibility
- Integration with the Meta Agents Research Environments event system
- Error handling and data integrity patterns
