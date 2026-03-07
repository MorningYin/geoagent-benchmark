..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`container` Docker
===========================

This guide explains how to build and run the interactive Graphical User Interface using Docker containers.

Overview
--------

The application uses a multi-stage build process that:

1. Builds the frontend React application
2. Installs Python dependencies
3. Creates a final runtime image with both components

The containerized application exposes a web interface on port 8080 and runs the ``are-gui`` command by default to start the interactive GUI.

Prerequisites
-------------

- Docker installed on your system
- Git (for cloning the repository)
- At least 4GB of available disk space for the build

Building the Docker Image
-------------------------

.. code-block:: bash

   # Build the application
   docker build -t meta-agents-research-environments:latest .

   # Build without cache for clean build
   docker build --no-cache -t meta-agents-research-environments:latest .

   # Build with detailed output
   docker build --progress=plain -t meta-agents-research-environments:latest .

Running the Container
---------------------

Basic Run
~~~~~~~~~

.. code-block:: bash

   # Run with default settings
   docker run -p 8080:8080 meta-agents-research-environments:latest

Run with Environment File
~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``.env`` file based on ``example.env``:

.. code-block:: bash

   # Copy and edit the example file
   cp example.env .env

   # Edit .env with your configuration

   # Run with environment file
   docker run -p 8080:8080 --env-file .env meta-agents-research-environments:latest

Accessing the Application
-------------------------

Once the container is running:

1. Open your browser to ``http://localhost:8080`` or ``https://localhost:8080``
2. The GUI interface should load automatically
