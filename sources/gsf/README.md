<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GSF as a data source

This package connects AI-Q to NVIDIA Generative Semantic Fabric (GSF).

## Tools

- `gsf__catalog_search`: finds relevant semantic objects.
- `gsf__text_to_sql`: answers analytical questions with SQL and bounded rows.
- `gsf__text_to_pql`: runs predictive questions through the GSF/Kumo path.

## Configuration

Set `GSF_BASE_URL` to GSF's auth-aware API origin and add the function group to the data-source registry:

By default, the function group owns one shared HTTP connection pool and keeps
authentication request-scoped: each tool invocation obtains the current AI-Q
user token and passes it to GSF without storing it on the client.
`GSF_BASE_URL` must point to GSF's auth-aware API origin.
Connect, pool-acquisition, request-write, and response-read operations use a
300-second timeout by default. Override the connect and read settings only for
a deployment with a measured need for different bounds.

```yaml
function_groups:
  gsf:
    _type: gsf
    base_url: ${GSF_BASE_URL}
    connect_timeout_seconds: 300
    read_timeout_seconds: 300
    include:
      - catalog_search
      - text_to_sql
      - text_to_pql

functions:
  data_sources:
    _type: data_source_registry
    sources:
      - id: gsf
        name: "Enterprise Structured Data"
        description: >-
          Build authorized semantic context and execute bounded structured-data
          queries through GSF.
        default_enabled: true
        requires_auth: true
        tools:
          - gsf
```

When `auth` is omitted, each tool invocation obtains the current AI-Q user token and forwards it to GSF without
storing it on the shared client.

For local development or automated evaluation without an incoming AI-Q user token, explicitly configure password
authentication using environment variables:

```yaml
function_groups:
  gsf:
    _type: gsf
    base_url: ${GSF_BASE_URL}
    auth:
      mode: password
      email: ${GSF_EMAIL}
      password: GSF_PASSWORD
    include:
      - catalog_search
      - text_to_sql
```

When `auth` is omitted, the existing request-scoped AI-Q user-token flow is
used. Password mode carries only the `password` variable name through NAT
configuration and distributed-worker serialization. The worker reads that
variable directly from its process environment immediately before creating the
GSF client. It creates one GSF session, reuses its cookie for local development
or evaluation calls, and signs out when the group closes. Every worker must
receive the named environment variable. The client does not fall back between
authentication methods.

## API mapping

Text-to-SQL uses GSF's `/api/chat/completions` SSE endpoint with
`prediction: false`. Its optional AI-Q `database_name` input is sent to GSF as
`target_db`, selecting an existing GSF connection rather than creating one.
The adapter normalizes GSF's current response fields while preserving optional
semantic and benchmarking fields as they become available.
For text-to-SQL, AI-Q retains GSF's top-level `response` as the provider's
primary analytical result, together with the generated SQL, bounded rows, and
all structured semantic and validation provenance returned by the service.
GSF's optional `thoughts` summary is retained as diagnostic context rather than
authoritative evidence. The normalized response also exposes the bounded row
count and a stable citation key derived from the request identity.

- Catalog search calls `POST /api/question-entity-coverage`.
- Text-to-SQL calls `POST /api/chat/completions` with `prediction: false`.
- Text-to-PQL calls `POST /api/chat/completions` with `prediction: true`.
- Unscoped AI-Q calls rely on GSF's routing and omit database selection.
- GSF-enabled chat requests and automated benchmarks may explicitly set a validated optional `database_name`; AI-Q
  forwards it unchanged to catalog search and to GSF as `target_db` for structured queries.
