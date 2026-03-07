..
    Copyright (c) Meta Platforms, Inc. and affiliates.
    All rights reserved.
    
    This source code is licensed under the terms described in the LICENSE file in
    the root directory of this source tree.


:octicon:`cpu` LLM Configuration Guide
======================================

This guide covers how to configure and use different Large Language Model (LLM) providers with the Agents Research Environments.

Overview
--------

LLM inference in ARE is powered by `LiteLLM <https://www.litellm.ai/>`_, providing flexible access to various language model providers and local models.
The system supports multiple inference backends to accommodate different deployment scenarios and model preferences.

**Supported Providers**

ARE integrates with multiple LLM providers through `LiteLLM <https://www.litellm.ai/>`_:

* **Llama API**: Meta's hosted Llama models via `Llama API <https://www.llama.com/products/llama-api/>`_
* **Local Models**: Self-hosted models running locally
* **Hugging Face**: Models hosted on `Hugging Face <https://huggingface.co/docs/huggingface_hub/en/guides/inference#supported-providers-and-tasks>`_ Hub
* **Hugging Face Providers**: Various inference providers including:

  * black-forest-labs
  * fal-ai
  * fireworks-ai
  * hf-inference
  * hyperbolic
  * nebius
  * novita
  * replicate
  * sambanova
  * together

Configuration
-------------

LLM engines are configured through the ``LLMEngineConfig`` which specifies:

* ``provider``: The inference provider to use
* ``model_name``: The specific model identifier
* ``endpoint``: Optional custom endpoint URL for local or private deployments

The system automatically creates the appropriate engine based on the provider:

* **LiteLLMEngine**: Used for most providers including llama-api, local, and huggingface
* **HuggingFaceLLMEngine**: Used for Hugging Face inference providers

CLI Usage Examples
------------------

.. note::
   In most CLI examples throughout this documentation, we omit the LLM connection arguments (``-p``, ``--provider``, ``--endpoint``) for brevity.
   You can choose any provider and model combination that suits your needs by adding the appropriate arguments shown below.

**Using Llama API (Recommended)**:

.. code-block:: bash

   # Run with Llama 3.1 70B via Llama API
   are-run -s scenario_find_image_file -a default --provider llama-api -m Llama-4-Maverick-17B-128E-Instruct-FP8

   # Benchmark with Llama API
   are-benchmark -s scenario_find_image_file -a default --provider llama-api -m Llama-4-Maverick-17B-128E-Instruct-FP8

**Using Local Models**:

.. code-block:: bash

   # Run with local model
   are-run -s scenario_find_image_file -a default --provider local -m llama3.1-8b-instruct --endpoint http://localhost:8000

   # Run with Hugging Face local deployment
   are-run -s scenario_find_image_file -a default --provider huggingface -m meta-llama/Llama-3.1-8B-Instruct

**Using Hugging Face Providers**:

.. code-block:: bash

   # Run with Together AI
   are-run -s scenario_find_image_file -a default --provider together -m meta-llama/Llama-3.1-70B-Instruct

   # Run with Fireworks AI
   are-run -s scenario_find_image_file -a default --provider fireworks-ai -m accounts/fireworks/models/llama-v3p1-70b-instruct

Environment Variables
---------------------

Different providers may require specific environment variables:

* **Llama API**: ``LLAMA_API_KEY`` (required), ``LLAMA_API_BASE`` (optional)
* **Hugging Face**: ``HF_TOKEN`` (for private models)
* **Provider-specific**: Each provider may have its own API key requirements

Model Selection
---------------

Choose models based on your requirements:

* **Performance**: Larger models (70B, 405B) for complex reasoning tasks
* **Speed**: Smaller models (8B) for faster inference
* **Cost**: Local models for cost-effective deployment
* **Availability**: Hosted APIs for convenience without infrastructure setup

The default configuration uses Llama API with ``llama3.1-70b-instruct`` for a balance of performance and efficiency.

Next Steps
----------

* :doc:`installation` - Get started with ARE installation
* :doc:`benchmarking` - Run systematic agent evaluations
* :doc:`../foundations/index` - Learn about ARE's core concepts
