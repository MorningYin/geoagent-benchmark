..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`apps` Apps API
========================

Apps are the building blocks of Meta Agents Research Environments simulations. They provide interactive functionality that agents can use as tools to accomplish tasks. This section documents the core App API and how to create custom applications.

.. contents:: In this section
   :local:
   :depth: 2
   :class: this-will-duplicate-information-and-it-is-still-useful-here

Overview
--------

Apps in Meta Agents Research Environments function similarly to applications on your phone or computer. Each app:

* Provides specific functionality (email, file system, calendar, etc.)
* Exposes APIs that agents can call as tools
* Maintains internal state that evolves during simulation
* Generates events when actions are performed


Creating Custom Apps
--------------------

To create a custom app, inherit from the base ``App`` class and implement the required functionality:

Basic App Structure
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from are.simulation.apps.app import App
   from are.simulation.tool_utils import app_tool

   class MyCustomApp(App):
       def __init__(self, name: str = None):
           super().__init__(name)
           # Initialize app-specific state
           self.data = {}

       @app_tool
       def my_action(self, param: str) -> str:
           """
           Perform a custom action.

           Args:
               param: Input parameter

           Returns:
               Result of the action
           """
           # Implement your logic here
           result = f"Processed: {param}"
           self.data[param] = result
           return result

       def get_state(self) -> dict:
           """Return the current state of the app."""
           return {"data": self.data}

       def load_state(self, state: dict) -> None:
           """Load state into the app."""
           self.data = state.get("data", {})

Tool Decorators
~~~~~~~~~~~~~~~

Meta Agents Research Environments provides several decorators for registering methods as tools:

.. code-block:: python

   from are.simulation.tool_utils import app_tool, user_tool, env_tool, data_tool

   class MyApp(App):
       @app_tool # Available to agents as a tool
       def agent_action(self, param: str) -> str:
           """Available to agents as a tool."""
           pass

       @user_tool # Only available to the user
       def user_action(self, param: str) -> str:
           """Available for user interactions."""
           pass

       @env_tool # To be used for environment events
       def env_action(self, param: str) -> str:
           """Available for environment events."""
           pass

       @data_tool # To populate data
       def data_action(self, param: str) -> str:
           """Available for data operations."""
           pass

State Management
~~~~~~~~~~~~~~~~

Apps should implement proper state management for simulation consistency:

.. code-block:: python

   class StatefulApp(App):
       def __init__(self):
           super().__init__()
           self.items = []
           self.counter = 0

       def get_state(self) -> dict:
           """Serialize app state."""
           return {
               "items": self.items,
               "counter": self.counter
           }

       def load_state(self, state: dict) -> None:
           """Restore app state."""
           self.items = state.get("items", [])
           self.counter = state.get("counter", 0)

       def reset(self) -> None:
           """Reset app to initial state."""
           super().reset()
           self.items = []
           self.counter = 0

Event Integration
~~~~~~~~~~~~~~~~~

Apps can generate events that affect the simulation:

.. code-block:: python

   from are.simulation.core.events import Event, Action

   class EventGeneratingApp(App):
       @app_tool
       @event_registered(operation_type=OperationType.WRITE) # Register the event as a write action
       def send_message(self, message: str) -> str:
           """Action that sends a message."""
           self.elf._send_message(message, sender=self.name)



Protocol Implementation
~~~~~~~~~~~~~~~~~~~~~~~

Apps can implement protocols to interact with other apps:

.. code-block:: python

   from are.simulation.apps.app import Protocol

   class FileSystemApp(App):
       def get_implemented_protocols(self) -> list[Protocol]:
           """Declare which protocols this app implements."""
           return [Protocol.FILE_SYSTEM]

       @app_tool
       def read_file(self, path: str) -> str:
           """Read a file from the file system."""
           # Implementation here
           pass

   class DocumentApp(App):
       def connect_to_protocols(self, protocols: dict[Protocol, Any]) -> None:
           """Connect to other apps via protocols."""
           if Protocol.FILE_SYSTEM in protocols:
               self.file_system = protocols[Protocol.FILE_SYSTEM]

       @app_tool
       def open_document(self, path: str) -> str:
           """Open a document using the file system."""
           content = self.file_system.read_file(path)
           return f"Opened document: {content}"

Best Practices
--------------

App Design Guidelines
~~~~~~~~~~~~~~~~~~~~~

* **Single Responsibility**: Each app should have a clear, focused purpose
* **Realistic Behavior**: Apps should behave like their real-world counterparts
* **Comprehensive APIs**: Provide all necessary functionality for realistic scenarios
* **Error Handling**: Handle edge cases and invalid inputs gracefully
* **Documentation**: Include detailed docstrings for all agent methods

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **State Efficiency**: Keep state minimal and serializable
* **Tool Registration**: Use lazy initialization for tool registries
* **Memory Management**: Clean up resources in reset methods
* **Concurrency**: Ensure thread-safety for multi-scenario execution

Testing Apps
~~~~~~~~~~~~

* **Unit Tests**: Test individual app methods in isolation
* **Integration Tests**: Test app behavior within scenarios
* **State Tests**: Verify state serialization and restoration
* **Tool Tests**: Ensure tools are properly registered and callable

For more examples of app implementation, see the built-in apps in the ``are.simulation.apps`` module.


Core App Class
--------------

.. automodule:: are.simulation.apps.app
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Key Methods
~~~~~~~~~~~

**Tool Registration**

.. automethod:: are.simulation.apps.app.App.get_tools

.. automethod:: are.simulation.apps.app.App.get_user_tools

.. automethod:: are.simulation.apps.app.App.get_env_tools

.. automethod:: are.simulation.apps.app.App.get_data_tools

**State Management**

.. automethod:: are.simulation.apps.app.App.get_state

.. automethod:: are.simulation.apps.app.App.load_state

.. automethod:: are.simulation.apps.app.App.reset

**Environment Integration**

.. automethod:: are.simulation.apps.app.App.register_to_env

.. automethod:: are.simulation.apps.app.App.add_event

Tool Utilities
--------------

.. automodule:: are.simulation.tool_utils
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Tool Classes
~~~~~~~~~~~~

.. autoclass:: are.simulation.tool_utils.AppTool
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.tool_utils.ToolAttributeName
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.tool_utils.OperationType
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.tool_utils.AppToolArg
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Enums and Types
~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.apps.app.Protocol
   :members:
   :undoc-members:

.. autoclass:: are.simulation.apps.app.ToolType
   :members:
   :undoc-members:

Built-in Apps
-------------

Meta Agents Research Environments includes a variety of pre-built apps for common use cases:

Communication Apps
~~~~~~~~~~~~~~~~~~

**Email Client**

.. automodule:: are.simulation.apps.email_client
   :members: EmailClientApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.email_client.Email
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.email_client.ReturnedEmails
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.email_client.EmailFolder
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.email_client.EmailFolderName
   :members:
   :undoc-members:
   :no-index:

**Messaging**

.. automodule:: are.simulation.apps.messaging
   :members: MessagingApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.messaging.Conversation
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.messaging.Message
    :members:
    :undoc-members:
    :no-index:

**Messaging V2**

.. automodule:: are.simulation.apps.messaging_v2
    :members: MessagingAppV2
    :undoc-members:
    :show-inheritance:
    :no-index:

.. autoclass:: are.simulation.apps.messaging_v2.ConversationV2
    :members:
    :undoc-members:
    :no-index:

.. autoclass:: are.simulation.apps.messaging_v2.MessageV2
    :members:
    :undoc-members:
    :no-index:

**Agent User Interface**

.. automodule:: are.simulation.apps.agent_user_interface
    :members: AgentUserInterface
    :undoc-members:
    :show-inheritance:
    :no-index:

System Apps
~~~~~~~~~~~

**File System**

.. automodule:: are.simulation.apps.sandbox_file_system
   :members: SandboxLocalFileSystem
   :undoc-members:
   :show-inheritance:
   :no-index:

**Virtual File System**

.. automodule:: are.simulation.apps.virtual_file_system
   :members: VirtualFileSystem
   :undoc-members:
   :show-inheritance:
   :no-index:

**Calendar**

.. automodule:: are.simulation.apps.calendar
   :members: CalendarApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.calendar.CalendarEvent
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.calendar.CalendarEventsResult
   :members:
   :undoc-members:
   :no-index:

**Contact**

.. automodule:: are.simulation.apps.contacts
   :members: ContactsApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.contacts.Contact
   :members:
   :undoc-members:
   :no-index:

Utility Apps
~~~~~~~~~~~~

**Shopping**

.. automodule:: are.simulation.apps.shopping
   :members: ShoppingApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.Product
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.ProductListResult
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.Order
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.CartItem
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.Item
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.shopping.ProductMetadata
   :members:
   :undoc-members:
   :no-index:

**Cab Service**

.. automodule:: are.simulation.apps.cab
   :members: CabApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.cab.Ride
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.cab.RideHistory
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.cab.OnGoingRide
   :members:
   :undoc-members:
   :no-index:

**Apartment Listing**

.. automodule:: are.simulation.apps.apartment_listing
   :members: ApartmentListingApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.apartment_listing.Apartment
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: are.simulation.apps.apartment_listing.RentAFlat
   :members:
   :undoc-members:
   :no-index:

**City Information**

.. automodule:: are.simulation.apps.city
   :members: CityApp
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.apps.city.CrimeDataPoint
   :members:
   :undoc-members:
   :no-index:

Integration Apps
~~~~~~~~~~~~~~~~

**Model Context Protocol (MCP)**

For detailed documentation on MCPApp usage, configuration, and examples, see :doc:`mcp_app`.

MCP Tools vs Meta Agents Research Environments Apps: Comparison and Integration
-------------------------------------------------------------------------------

The Model Context Protocol (MCP) and Meta Agents Research Environments Apps represent two different but complementary approaches to providing functionality to AI agents.
Understanding their differences and similarities helps in choosing the right approach for your use case.

Conceptual Differences
~~~~~~~~~~~~~~~~~~~~~~

**MCP Tools**

MCP tools are individual, executable functions exposed by MCP servers to clients. They represent a **function-centric** approach where:

* Each tool is a single, atomic operation
* Tools are discovered dynamically through the MCP protocol
* Tools are stateless from the protocol perspective
* Tools are designed to be **model-controlled** (AI can invoke them automatically with human approval)
* Tools focus on specific, well-defined operations

**Meta Agents Research Environments Apps**

Meta Agents Research Environments Apps represent an **application-centric** approach where:

* Apps encapsulate related functionality as a cohesive unit
* Apps maintain internal state throughout simulation lifecycle
* Apps provide multiple related tools grouped by application domain
* Apps model real-world applications (email, calendar, file system, etc.)
* Apps can implement protocols for inter-app communication

Key Similarities
~~~~~~~~~~~~~~~~

Both MCP tools and Meta Agents Research Environments Apps share several important characteristics:

* **Tool Discovery**: Both support dynamic discovery of available functionality
* **Structured Parameters**: Both use JSON Schema for parameter definition and validation
* **Rich Descriptions**: Both provide detailed descriptions to guide AI usage
* **Error Handling**: Both implement structured error reporting
* **Extensibility**: Both allow for custom implementations and extensions

Architectural Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

**Granularity**

.. code-block:: text

   MCP Server                    Meta Agents Research Environments App
   ├── search_web               ├── BrowserApp
   ├── read_file                │   ├── navigate_to_url
   ├── write_file               │   ├── search_web
   ├── send_email               │   ├── click_element
   └── get_weather              │   └── get_page_content
                                ├── FileSystemApp
                                │   ├── read_file
                                │   ├── write_file
                                │   └── list_directory
                                └── EmailApp
                                    ├── send_email
                                    ├── read_inbox
                                    └── delete_email

**State Management**

* **MCP Tools**: Stateless at the protocol level; any state is managed internally by the server
* **Meta Agents Research Environments Apps**: Explicit state management with serialization/deserialization for simulation consistency

Integration in Meta Agents Research Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Meta Agents Research Environments provides bidirectional integration with MCP:

**Using MCP Tools in Meta Agents Research Environments**

The :doc:`mcp_app` discusses  connecting to external MCP servers and use their tools within Meta Agents Research Environments simulations.
This allows you to leverage the growing ecosystem of MCP servers while maintaining Meta Agents Research Environments's application-centric approach.

**Exposing Meta Agents Research Environments as MCP Server**

The :doc:`are_simulation_mcp_server` allows you to expose Meta Agents Research Environments apps and scenarios as MCP servers, making them accessible to external agentic tools like Claude Desktop, Cursor, and other MCP-compatible clients.

**Benefits of MCP Integration**

* **Ecosystem Access**: Leverage the growing ecosystem of MCP servers
* **Rapid Integration**: Connect to external services without custom app development
* **Protocol Standardization**: Benefit from MCP's standardized communication protocol
* **Community Tools**: Access community-developed MCP servers
* **Multi-Agent Testing**: Test scenarios with different AI agents and tools

**Meta Agents Research Environments App Advantages**

* **Simulation Fidelity**: Apps model real-world applications more accurately
* **State Consistency**: Explicit state management ensures simulation reproducibility
* **Inter-App Communication**: Apps can interact through protocols
* **Domain Modeling**: Natural grouping of related functionality
