## ADDED Requirements

### Requirement: Deploy container exposes REST API for inference
The system SHALL provide a deploy container that exposes REST API endpoints for loading models/workflows and performing inference.

#### Scenario: Load model via API
- **WHEN** a client sends POST to `/load/model` with project_id and model_version
- **THEN** the deploy container downloads the model from xclabel-server
- **AND** loads it into the inference engine pool
- **AND** returns an engine_id for subsequent inference calls

#### Scenario: Load workflow via API
- **WHEN** a client sends POST to `/load/workflow` with project_id and workflow_name
- **THEN** the deploy container downloads the workflow.json from xclabel-server
- **AND** builds a nndeploy Pipeline from the workflow definition
- **AND** returns an engine_id for subsequent inference calls

#### Scenario: Perform inference with loaded engine
- **WHEN** a client sends POST to `/infer` with engine_id and image data
- **THEN** the deploy container acquires the engine's inference lock
- **AND** executes inference serially (queued if concurrent requests arrive)
- **AND** returns inference results including detections and timing

#### Scenario: Perform inference with image URL
- **WHEN** a client sends POST to `/infer` with engine_id and image_url
- **THEN** the deploy container downloads the image from the URL
- **AND** performs inference and returns results

#### Scenario: Unload engine
- **WHEN** a client sends POST to `/unload` with engine_id
- **THEN** the deploy container removes the engine from the pool
- **AND** releases associated memory

### Requirement: Deploy container supports health checks
The system SHALL provide a health check endpoint for monitoring deploy container status.

#### Scenario: Health check returns OK
- **WHEN** a client sends GET to `/health`
- **THEN** the deploy container returns status "ok" and version information

### Requirement: Deploy container lists loaded engines
The system SHALL provide an endpoint to list all currently loaded inference engines.

#### Scenario: List engines
- **WHEN** a client sends GET to `/engines`
- **THEN** the deploy container returns a list of all loaded engines with their type, project_id, and load time
