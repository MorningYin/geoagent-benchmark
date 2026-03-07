..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`device-desktop` Graphical User Interface
==================================================

Introduction
------------

This guide is for the Meta Agents Research Environments (ARE) UI, explaining the full simulation tool interface, features and components. In the ARE UI, you will be able to run different models on a fixed baseline orchestration and environment, observe agent behavior, and explore Gaia2 environments and scenarios. Reference this guide to help you understand how to get started with ARE UI.

Getting Started
---------------

Upon entering the ARE UI, you will be greeted by an introductory screen that provides an overview explanation of the different workspaces, along with links to documentation. There are two workspaces in the ARE UI: Playground and Scenarios. See below for details on each of these workspaces. The documentation resources will help you learn more about the ARE application itself as well as the Gaia2 benchmark. You can access this screen again anytime by clicking [about] at the bottom of the left navigation pane.

Click **[Got it]** to proceed to the Scenario View

.. thumbnail:: ../_static/ui/getting_started_intro_screen.png
   :alt: Getting started screen
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Welcome Screen - Introduction screen showing workspace overview, documentation links, and navigation to Scenarios and Playground workspaces

Navigation
----------

The ARE Interface is made up of two main navigation panels: a left navigation bar to switch between Scenarios and Playground views; and a top navigation bar with actions and settings related to the workspace. When you land in the ARE UI, you will start in a blank Scenarios workspace view where you can elect to turn on the execution panels from the top navigation bar before running or loading a scenario. See the Scenario workspace section for details on the top navigation functionalities.

.. thumbnail:: ../_static/ui/navigation_interface_overview.png
   :alt: ARE Interface navigation
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Main Interface Layout - Overview of the dual-panel navigation system with left sidebar for workspace switching and top bar for workspace-specific actions

Left Navigation Pane Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the left navigation pane to switch between the Scenario workspace and the Playground workspace.

Click **[Scenarios]** or **[Playground]** to change between views

Within the Playground view you will have additional interface options to engage in agent chat or view agent logs. Use the left navigation pane to select the relevant view for your workflow.

Both the Scenario and Playground views then list the simulated applications you can interface with in the Scenario or Playground workspace.

Click **[Application Name]** to expand a detail view with state, available tools, or calls.

There are also global resources that enable you to report a bug, access documentation or see the overview on ARE.

To create a GitHub Issue click **[Report a Bug]**

To access the Meta Agents Research Environments docs click **[Documentation]**

.. thumbnail:: ../_static/ui/left_navigation_pane_details.png
   :alt: Navigation details
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Left Navigation Panel - Detailed view of workspace switching, application states, tools access, and global resources like bug reporting and documentation

Scenario workspace
------------------

The Scenario workspace allows you to load, visualize and perform actions on ARE scenarios and the events within the scenarios. You can configure the view to your preference by leveraging the top navigation bar.

Toggle **[Execution panels]** if you want to view the scenario run steps and logs.

Note, when you run your first scenario the execution panel will automatically turn on, as you will then be able to pause and track the scenario.

.. thumbnail:: ../_static/ui/scenario_workspace_overview.png
   :alt: Scenario workspace
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Scenario Workspace Overview - Main workspace for loading, visualizing, and executing ARE scenarios with configurable execution panels and navigation controls

How to run a scenario
~~~~~~~~~~~~~~~~~~~~~

You can load scenarios from `Gaia2 dataset on Hugging Face <https://huggingface.co/datasets/meta-agents-research-environments/gaia2>` or load other demo scenarios from the Scenario workspace. Use the buttons in the center frame or select load scenario from the top navigation bar.


For scenarios from the Gaia2 benchmark
**************************************

Click on the **[Hugging Face]** button in the center frame or select **[Hugging Face]** from the drop down menu in the top navigation bar.

You will see a pop up window to make three selections: capability, split (always `validation`) and scenario.

Make your selections then click **[confirm]**

.. thumbnail:: ../_static/ui/gaia2_scenario_selection_popup.png
   :alt: Gaia2 scenario selection
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Gaia2 Scenario Selection Dialog - Interactive popup for selecting capability type, dataset split, and specific scenario from the Gaia2 benchmark dataset

Once you have confirmed your scenario settings, ARE will load the scenario from the Gaia2 benchmark, which will appear in the top half of the frame as a Direct Acyclic Graph (DAG) of the agentic trajectories. Note, some scenarios will not contain scheduled events so will not render a DAG.

For python scenarios from the codebase
**************************************

Select **[Load scenario]** from the drop down menu in the top navigation bar.

You will see a pop up window to confirm source: code and make your scenario selection.

Make your selections then click **[confirm]**. Again once selecting confirm ARE will run the scenario.

.. thumbnail:: ../_static/ui/preloaded_scenario_selection_popup.png
   :alt: Preloaded scenario selection
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Preloaded Scenario Selection - Loading dialog for selecting scenarios from the codebase with source verification and scenario options

For any scenario
****************

You also have the option to upload a json file or direct to a URL from the Load scenario menu.

Click on the **[Upload]** button in the center frame or select **[File]** or **[URL]** from the drop down menu in the top navigation bar.

.. thumbnail:: ../_static/ui/upload_scenario_options.png
   :alt: Upload scenario options
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Upload Scenario Options - File upload interface and URL input options for loading custom scenario JSON files or remote scenario definitions

Scenario Visualization
~~~~~~~~~~~~~~~~~~~~~~

Loaded scenarios display in the top frame of the Scenario workspace as a Direct Acyclic Graph (DAG) of events, and are color-coded by origin:

* Blue: User action
* Pink: Agent action
* Green: Environment event (e.g., incoming email)

The below image is an example of a Scenarios view where all origins and scheduled events are present. Use this view for running and analyzing predefined agent trajectories.

.. thumbnail:: ../_static/ui/scenario_dag_visualization.png
   :alt: Scenario visualization DAG
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Scenario DAG Visualization - Interactive directed acyclic graph showing color-coded event sequences with blue (user), pink (agent), and green (environment) events and actions

Executing and Monitoring Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After reviewing the scenario, ensure your execution panel is on and click on the **[Play]** button. This will start the execution from a clean simulated state.

In the left pane you can monitor the reasoning traces. In the right pane you can monitor the visual timeline of events. At any time you can stop or pause the simulation by using the controls in the left pane. In the control panel you can track total run duration before auto-stop and time increments between simulation ticks.

.. thumbnail:: ../_static/ui/scenario_execution_monitoring.png
   :alt: Scenario execution monitoring
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Scenario Execution Monitoring - Live execution view with reasoning traces in left panel, event timeline in right panel, and playback controls for simulation management

You can also change your monitoring view by clicking on the **[Agent]** toggle, which will show you detailed agent internal logs. Consider readjusting the height of your Scenario view by dragging the horizontal bar between the scenario view and scenario run.

.. thumbnail:: ../_static/ui/agent_logs_detailed_view.png
   :alt: Agent logs view
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Agent Logs Detailed View - Switch to detailed agent internal logs showing reasoning processes, decision trees, and internal state changes during scenario execution

Exporting Scenario
~~~~~~~~~~~~~~~~~~

After completing your run and monitoring the scenario, you have the option to export the DAG as a PNG or export the full scenario trace as a JSON.

Click on the **[menu]** icon in the top navigation bar and select save as PNG or quick save as JSON.

.. thumbnail:: ../_static/ui/export_scenario_menu_options.png
   :alt: Export scenario options
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Export Scenario Options - Menu interface for exporting scenario results as PNG images or complete JSON trace files for analysis and documentation

Playground Workspace
--------------------

The Playground enables you to directly interact with selected agents and experience their capabilities in real time. At this time, you cannot change the default agent or environment for the Playground. The Playground is anchored in a demo, Gaia2-like environment centered around a fictive AI student workspace. When in the Playground Agent Chat Interface, use the bottom composer entry field to prompt the default agent which will operate in the preconfigured, custom simulated environment.

Note: The Agent only has access to the simulated application listed in the left navigation pane. So when interacting with the Agent, it is recommended to ask questions or request actions related to the simulated applications.

Click on the **[menu]** icon in the top navigation bar to save the agentic trajectories for your chat as JSON, markdown or PDF.

.. thumbnail:: ../_static/ui/playground_workspace_overview.png
   :alt: Playground workspace
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Playground Workspace Overview - Interactive chat interface for real-time agent interaction with simulated applications and conversation export options

Interacting with the agent in Playground workspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Click on a predefined prompt or write your own in the composer field. After sending the message, observe the agent interact dynamically with simulated apps via tool calls, and see the agent's reasoning prior to providing a response to your message.

At any time you can stop the scenario by clicking on the **[stop]** icon within the composer.

.. thumbnail:: ../_static/ui/playground_agent_interaction.png
   :alt: Agent interaction in playground
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Agent Interaction in Playground - Live chat session showing predefined prompts, message composer, and real-time agent responses with tool call interactions

Viewing agent logs
~~~~~~~~~~~~~~~~~~

You can also view the agent logs following a conversation with the agent. Use the left navigation pane and select **[agent logs]**. Here you will see detailed logs of the agent's actions and thoughts.

Review the steps and select **[action]** to see more detailed information. In the below example, you can see the tool call details for step 0.

.. thumbnail:: ../_static/ui/agent_logs_action_details.png
   :alt: Agent logs detailed view
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Agent Logs Action Details - Detailed step-by-step breakdown of agent actions showing tool call parameters, responses, and decision-making processes

Applications
------------

The ARE UI currently provides simulated scenarios including data from 10 applications. Note, not all scenarios will include every application. The applications listed below can be found in the left navigation pane. Remember to use these in your simulated chats with the agent.

Leveraging Applications
~~~~~~~~~~~~~~~~~~~~~~~

You can view each application's data by clicking on the application in the left navigation pane. This will open a window that displays the application data type, tools and app state.

See below for each type of information you can view in the UI format.

For viewing application data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After clicking on your selected application, a window will open. Click on the **application name** header and view data in a UI format.

Here is an example of the messaging data.

.. thumbnail:: ../_static/ui/application_data.png
   :alt: Messaging Application Data
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Application Data View - UI-formatted display of messaging application data showing conversations, contacts, and message history in simulated environment

For viewing tool use
~~~~~~~~~~~~~~~~~~~~

In the open window, you can then click on **tools** and in the drop down view the tools available for function calls for the select application.

.. thumbnail:: ../_static/ui/tool_use.png
   :alt: Tool list and manual call
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Tool Usage Interface - Interactive tool list showing available functions for each application with manual call capabilities and parameter specifications

For viewing app state
~~~~~~~~~~~~~~~~~~~~~

In the open window, you can then click on **app state** and this will show you the raw state of your application in code format.

.. thumbnail:: ../_static/ui/app_state.png
   :alt: Raw JSON App State Dump
   :align: center
   :width: 100%
   :group: ui-guide
   :title: Raw JSON App State Dump - Developer view showing the complete JSON representation of application state data for debugging and analysis purposes
