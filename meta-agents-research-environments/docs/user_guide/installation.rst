..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.

    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`download` Installation
================================

The Agents Research Environments (ARE) can be used in two ways: via `uvx` for quick usage without local installation, or installed locally for development and custom scenarios.

:octicon:`zap` Quick Start with uvx (Recommended)
-------------------------------------------------

For most users, we recommend using `uvx` to run the Agents Research Environments commands without installing the package locally:

.. code-block:: bash

   # Install uv (which includes uvx)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Run commands directly
   uvx --from meta-agents-research-environments are-run -s scenario_find_image_file -a default --provider mock
   uvx --from meta-agents-research-environments are-benchmark run --hf meta-agents-research-environments/gaia2 --hf-split validation -l 5
   uvx --from meta-agents-research-environments are-gui -s scenario_find_image_file

This approach:

* **No local installation required**: Commands run in isolated environments
* **Always up-to-date**: Automatically uses the latest version from PyPI
* **Clean system**: No package conflicts or dependency issues
* **Quick setup**: Start using the Agents Research Environments immediately

:octicon:`package` Local Installation from PyPI
-----------------------------------------------

For users who want to dig deeper into the library, develop custom scenarios, or need local access to the codebase, you can install it with pip.

the Agents Research Environments requires python `3.12.*`. We recommend using a virtual environment for installation. Using `uv venv -p 3.12` and `uv pip install` or `uv add` is the recommended approach.

.. code-block:: bash

   pip install meta-agents-research-environments

This will install the Agents Research Environments package and all its dependencies, including the pre-built JavaScript UI components.

Command Line Tools
~~~~~~~~~~~~~~~~~~

After installing ARE, the following command line tools will be available:

* ``are-run``: Run individual scenarios

  .. code-block:: bash

     are-run -s scenario_find_image_file -a default

* ``are-benchmark``: Run benchmarks on scenarios

  .. code-block:: bash

     are-benchmark run -d /path/to/scenarios

* ``are-gui``: Start the ARE GUI server

  .. code-block:: bash

     are-gui -s scenario_find_image_file

Run any command with ``--help`` to see all available options.

:octicon:`code` Installation from Source
----------------------------------------

For contributors and developers who want to modify the codebase, start by cloning the repository:

.. code-block:: bash

   git clone https://github.com/facebookresearch/meta-agents-research-environments.git
   cd meta-agents-research-environments

.. tip::
   For configuration, copy ``example.env`` to ``.env`` and customize as needed.

Now choose your installation based on what you want to develop:

**Scenario and Agent Development**

If you're working on scenarios, agents, or core functionality and just need the command-line tools:

.. code-block:: bash

   uv venv -p 3.12
   source .venv/bin/activate
   uv pip install -e .


**For UI Development**

If you want to start the UI locally or modify the web interface, you'll also need to build the JavaScript components from source.

First, install NodeJS version 23 on your system, then run:

.. code-block:: bash

   # On Linux/MacOS
   BUILD_GUI=1 uv pip install -e .

   # On Windows
   set BUILD_GUI=1 & uv pip install -e .

This will compile the JavaScript UI components during installation, which takes a few additional minutes but gives you full control over the web interface development.

Verification
------------

To verify your installation works correctly:

1. **Test Basic Import** (local installation only)

   .. code-block:: python

      import are
      print(are.__version__)

2. **Run a Simple Scenario**

   .. code-block:: bash

      # With uvx
      uvx --from meta-agents-research-environments are-run --help

      # With local installation
      are-run --help

3. **Check Available Scenarios**

   .. code-block:: bash

      # With uvx
      uvx --from meta-agents-research-environments are-benchmark --help

      # With local installation
      are-benchmark --help

Next Steps
----------

Once you have Meta Agents Research Environments set up, you're ready to:

* :doc:`../quickstart` - Run your first scenario
* :doc:`../foundations/index` - Learn about Meta Agents Research Environments core concepts
* :doc:`../user_guide/benchmarking` - Understand the benchmark and how to run it
