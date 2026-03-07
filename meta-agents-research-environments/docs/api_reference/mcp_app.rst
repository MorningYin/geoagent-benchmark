..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`tab-external` MCPApp - Model Context Protocol
=======================================================

The MCPApp provides seamless integration with Model Context Protocol (MCP) servers,
allowing Meta Agents Research Environments to connect to and utilize tools from any MCP-compatible server.
This enables modular tool organization and dynamic tool discovery.

Overview
--------

The Model Context Protocol (MCP) is a standardized way for applications to provide
tools and resources to AI systems.
The MCPApp acts as a bridge between Meta Agents Research Environments and MCP servers, automatically discovering
available tools and exposing them as Meta Agents Research Environments app tools.

Key features:

* **Universal Compatibility**: Works with any MCP-compatible server
* **Dynamic Discovery**: Automatically discovers tools, resources, and prompts
* **Flexible Connection**: Supports both stdio and Server-Sent Events (SSE) connections
* **Tool Filtering**: Filter tools by read-only status or exclude specific tools
* **Rich Annotations**: Preserves MCP tool annotations for better tool understanding

When to Use Apps vs MCP
~~~~~~~~~~~~~~~~~~~~~~~

**Choose MCP Tools When:**

* Integrating with existing MCP servers
* Need specific, atomic operations
* Working with external services that provide MCP interfaces
* Rapid prototyping of new functionality
* Leveraging community-developed tools
* No need for scenario reproducibility

**Choose Meta Agents Research Environments Apps When:**

* Modeling complete applications or systems
* Need complex state management
* Require inter-app communication
* Building simulation-specific functionality
* Need fine-grained control over tool behavior and presentation
* Need to reproduce scenarios across multiple runs

**Hybrid Approach:**

The most powerful approach often combines both:

.. code-block:: python

   # Custom Meta Agents Research Environments app for core functionality
   email_app = EmailClientApp()

   # MCP integration for external services
   code_exec = MCPApp(name="CodeExecution", server_command="code-mcp-server")
   wiki_mcp = MCPApp(name="Wikipedia", server_command="wikipedia-mcp-server")

   # All work together in the same simulation
   scenario = Scenario(apps=[email_app, code_exec, wiki_mcp])

This hybrid approach provides the best of both worlds: the rich application modeling of Meta Agents Research Environments Apps combined with the extensive ecosystem and standardization of MCP tools.


Connection Types
----------------

The MCPApp supports two connection methods:

Stdio Connection (Local Servers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For local MCP servers that communicate via standard input/output:

.. code-block:: python

   from are.simulation.apps.mcp.mcp_app import MCPApp

   # Connect to a local Python MCP server
   math_app = MCPApp(
       name="MathTools",
       server_command="python",
       server_args=["path/to/math_server.py"],
       server_env={"PYTHONPATH": "/custom/path"}  # Optional environment variables
   )

SSE Connection (Remote Servers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For remote MCP servers accessible via HTTP Server-Sent Events:

.. code-block:: python

   # Connect to a remote MCP server
   code_app = MCPApp(
       name="CodeExecution",
       server_url="http://localhost:8000/sse"
   )

Authentication
--------------

The MCPApp supports authentication for remote MCP servers through HTTP headers.
Currently, long-lived tokens are supported for authentication. OAuth2 support
is not available at this time.

Long-Lived Token Authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For servers that require authentication via long-lived tokens (such as Home Assistant),
you can provide authentication headers when creating the MCPApp:

.. code-block:: python

   # Example: Home Assistant MCP server with long-lived token
   ha_app = MCPApp(
       name="Home Assistant",
       server_url="http://192.168.0.189:8123/mcp_server/sse",
       sse_headers={
           "Authorization": "Bearer YOURLONGLIVEDTOKENS",
       }
   )

The ``sse_headers`` parameter accepts a dictionary of HTTP headers that will be
sent with each request to the MCP server. This allows for flexible authentication
methods depending on your server's requirements.

.. note::
   OAuth2 authentication is not currently supported. For now, use long-lived
   tokens or API keys provided by your MCP server.

Configuration Options
---------------------

The MCPApp provides extensive configuration options for different use cases:

Basic Configuration
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   app = MCPApp(
       name="MyMCPApp",                    # App name in Meta Agents Research Environments
       server_command="python",            # Command to run the server
       server_args=["server.py"],          # Arguments for the server
       timeout=10.0                        # Timeout for operations
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   app = MCPApp(
       name="FilteredApp",
       server_command="python",
       server_args=["server.py"],
       server_env={"DEBUG": "1"},          # Environment variables
       exclude_tools=["dangerous_tool"],   # Tools to exclude
       only_read_only=True,                # Only include read-only tools
       description_modifier=custom_desc    # Function to modify descriptions
   )

Tool Filtering
--------------

The MCPApp provides powerful filtering capabilities. Excluding some tools or keep only_read_only,
can help use MCPApps in a safer and reproducible way:

Read-Only Filtering
~~~~~~~~~~~~~~~~~~~

Filter tools to only include those marked as read-only:

.. code-block:: python

   # Only expose read-only tools (safe for exploration)
   readonly_app = MCPApp(
       name="SafeTools",
       server_command="python",
       server_args=["server.py"],
       only_read_only=True
   )

Tool Exclusion
~~~~~~~~~~~~~~

Exclude specific tools by name:

.. code-block:: python

   # Exclude potentially dangerous tools
   filtered_app = MCPApp(
       name="FilteredTools",
       server_command="python",
       server_args=["server.py"],
       exclude_tools=["delete_file", "format_disk"]
   )

Description Modification
~~~~~~~~~~~~~~~~~~~~~~~~

Customize tool descriptions dynamically:

.. code-block:: python

   def enhance_description(tool_name: str, original_desc: str) -> str:
       if tool_name == "divide":
           return f"{original_desc} (Note: Does not work with negative numbers)"
       return original_desc

   app = MCPApp(
       name="EnhancedTools",
       server_command="python",
       server_args=["server.py"],
       description_modifier=enhance_description
   )

API Reference
-------------

.. automodule:: are.simulation.apps.mcp.mcp_app
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

MCPApp Class
~~~~~~~~~~~~

.. autoclass:: are.simulation.apps.mcp.mcp_app.MCPApp
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Tool Utility Classes
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: are.simulation.tool_utils.AppTool
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: are.simulation.tool_utils.ToolAttributeName
   :members:
   :undoc-members:
   :no-index:



MCP Tool Annotations
--------------------

The MCPApp preserves and exposes MCP tool annotations, providing rich metadata about tool behavior:

Annotation Types
~~~~~~~~~~~~~~~~

* **readOnlyHint**: Whether the tool modifies its environment
* **destructiveHint**: Whether modifications are destructive or additive
* **idempotentHint**: Whether repeated calls have the same effect
* **openWorldHint**: Whether the tool interacts with external entities
* **title**: Human-readable title for the tool


The MCPApp provides a powerful and flexible way to extend Meta Agents Research Environments with external tools while maintaining safety and control through comprehensive filtering and annotation support.
