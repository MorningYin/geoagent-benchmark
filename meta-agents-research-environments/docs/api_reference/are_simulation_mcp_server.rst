..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`server` MCP Server - Expose Scenarios
===============================================

The Meta Agents Research Environments MCP Server enables you to expose any Meta Agents Research Environments app or complete scenario as a Model Context Protocol (MCP) server, making Meta Agents Research Environments functionality accessible to external agentic tools like Cursor, Claude Desktop, and other MCP-compatible clients.


Overview
--------

While the :doc:`mcp_app` allows Meta Agents Research Environments to connect to external MCP servers, the Meta Agents Research Environments MCP Server does the opposite - it exposes Meta Agents Research Environments apps and scenarios as MCP servers that can be accessed by any MCP client.

This powerful capability enables you to:

* **Test scenarios with different agents**: Use Claude Desktop, Cursor, or other agentic tools to interact with Meta Agents Research Environments scenarios
* **Integrate Meta Agents Research Environments into development workflows**: Access Meta Agents Research Environments apps directly from your IDE or development tools
* **Create hybrid agent systems**: Combine Meta Agents Research Environments simulated environments with external AI agents
* **Validate scenario quality**: Test scenarios with multiple different agent implementations

Key Features
~~~~~~~~~~~~

* **Universal App Exposure**: Expose any Meta Agents Research Environments app as an MCP server
* **Complete Scenario Support**: Load entire scenarios with all their apps and state
* **Automatic Tool Discovery**: All Meta Agents Research Environments app tools are automatically exposed as MCP tools
* **State Monitoring**: Real-time access to app states through MCP resources
* **Rich Metadata**: Detailed tool information and parameter descriptions
* **Multiple Transports**: Support for both stdio and Server-Sent Events (SSE) connections
* **State Change Notifications**: Automatic notifications when app states change

Architecture
------------

.. code-block:: text

   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │  External Tool  │    │   ARE MCP       │    │   ARE Apps/     │
   │  (Cursor, etc.) │◄──►│  Server         │◄──►│  Scenarios      │
   │                 │    │                 │    │                 │
   │ • Claude Desktop│    │ • Tool Exposure │    │ • Calendar      │
   │ • Cursor        │    │ • State Monitor │    │ • Email         │
   │ • Custom Client │    │ • Notifications │    │ • File System   │
   └─────────────────┘    └─────────────────┘    └─────────────────┘

Installation
------------

The MCP Server is included with the Meta Agents Research Environments package. Ensure you have the required dependencies:

.. code-block:: bash

   # Install ARE with MCP support
   uv pip install meta-agents-research-environments

   # Install MCP CLI tools (if not already included)
   uv pip install "mcp[cli]"

Usage
-----

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

**Expose Individual Apps**

.. code-block:: bash

   uv run are_simulation_mcp_server.py --apps <app_class1> <app_class2> ... [options]

**Expose Complete Scenarios**

.. code-block:: bash

   uv run are_simulation_mcp_server.py --scenario <scenario_id> [options]

**Options**

* ``--name <server_name>``: Optional name for the MCP server
* ``--transport {stdio|sse}``: Choose transport method (default: stdio)

Examples
~~~~~~~~

**Calendar App Server**

.. code-block:: bash

   uv run are_simulation_mcp_server.py --apps are.simulation.apps.calendar.CalendarApp --name "Calendar MCP Server"

**Multiple Apps Server**

.. code-block:: bash

   uv run are_simulation_mcp_server.py --apps are.simulation.apps.calendar.CalendarApp are.simulation.apps.email_client.EmailClientApp --name "Productivity Suite"

**Complete Scenario Server**

.. code-block:: bash

   uv run are_simulation_mcp_server.py --scenario scenario_tutorial --transport sse

**Using with MCP Inspector**

.. code-block:: bash

   npx @modelcontextprotocol/inspector are_simulation_mcp_server.py --scenario scenario_tutorial

Integration with External Tools
-------------------------------

Claude Desktop Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the Meta Agents Research Environments MCP Server to your Claude Desktop configuration:

.. code-block:: json

   {
     "mcpServers": {
       "are-calendar": {
         "command": "python",
         "args": [
           "/path/to/are_simulation_mcp_server.py",
           "--apps",
           "are.simulation.apps.calendar.CalendarApp",
           "--name",
           "Meta Agents Research Environments Calendar"
         ]
       },
       "are-scenario": {
         "command": "python",
         "args": [
           "/path/to/are_simulation_mcp_server.py",
           "--scenario",
           "scenario_tutorial"
         ]
       }
     }
   }

Cursor Integration
~~~~~~~~~~~~~~~~~~

Configure Cursor to use Meta Agents Research Environments MCP Server for testing scenarios:

.. code-block:: json

   {
     "mcp": {
       "servers": {
         "are.simulation": {
           "command": "python",
           "args": ["/path/to/are_simulation_mcp_server.py", "--scenario", "your_scenario"]
         }
       }
     }
   }

Available Resources
-------------------

The Meta Agents Research Environments MCP Server exposes several MCP resources for monitoring and introspection:

Global Resources
~~~~~~~~~~~~~~~~

* ``app://info``: Information about all exposed apps and their tools

App-Specific Resources
~~~~~~~~~~~~~~~~~~~~~~

For each app, the following resources are available:

* ``app://{app_name}/info``: General information about the specific app
* ``app://{app_name}/state``: Current state of the app (updated in real-time)


Testing Scenarios with External Agents
--------------------------------------

The Meta Agents Research Environments MCP Server enables powerful testing workflows where you can validate your
 scenarios using different AI agents and tools.

Scenario Testing Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Develop Scenario**: Create your Meta Agents Research Environments scenario with apps and validation logic
2. **Expose via MCP**: Use Meta Agents Research Environments MCP Server to expose the scenario
3. **Test with Multiple Agents**: Connect different agentic tools to test the scenario
4. **Compare Results**: Analyze how different agents perform on the same scenario
5. **Iterate and Improve**: Refine scenarios based on multi-agent testing results

Example Testing Setup
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Terminal 1: Start Meta Agents Research Environments MCP Server with your scenario
   uv run are_simulation_mcp_server.py --scenario my_test_scenario --transport sse

   # Terminal 2: Test with Claude Desktop (configured to connect to the server)
   # Use Claude Desktop to interact with the scenario

   # Terminal 3: Test with custom agent
   python my_custom_agent.py --mcp-server http://localhost:8000/sse

Benefits for Scenario Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Multi-Agent Validation**: Test scenarios with different reasoning approaches
* **Real-World Integration**: See how scenarios perform with production AI tools
* **Debugging Support**: Monitor app states and tool calls in real-time
* **Rapid Iteration**: Quickly test scenario changes across multiple agents

API Reference
-------------

.. automodule:: are.simulation.apps.mcp.server.are_simulation_mcp_server
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

ARESimulationMCPServer Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.apps.mcp.server.are_simulation_mcp_server.ARESimulationMCPServer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

ARESimulationToolManager Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.apps.mcp.server.are_simulation_mcp_server.ARESimulationToolManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

   Inherits from ``mcp.server.fastmcp.tools.tool_manager.ToolManager``.

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: are.simulation.apps.mcp.server.are_simulation_mcp_server.load_app_class
   :no-index:

.. autofunction:: are.simulation.apps.mcp.server.are_simulation_mcp_server.load_scenario_class
   :no-index:

Referenced Classes
~~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.apps.app.App
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.scenarios.scenario.Scenario
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
