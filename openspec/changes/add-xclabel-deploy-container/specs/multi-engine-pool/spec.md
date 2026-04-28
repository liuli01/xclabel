## ADDED Requirements

### Requirement: Deploy container maintains inference engine pool
The deploy container SHALL maintain an in-memory pool of loaded inference engines supporting multiple concurrent models and workflows.

#### Scenario: Load multiple models
- **WHEN** a client loads model A and then model B
- **THEN** both models coexist in the engine pool
- **AND** each can be independently used for inference via its engine_id

#### Scenario: Engine pool enforces maximum size
- **WHEN** the number of loaded engines exceeds MAX_ENGINES
- **THEN** the least recently used engine is automatically unloaded
- **AND** the new engine is loaded in its place

#### Scenario: Engine pool tracks usage statistics
- **WHEN** an engine is used for inference
- **THEN** the pool updates the engine's last_used_at timestamp
- **AND** increments the inference request count

### Requirement: Deploy container caches models locally
The deploy container SHALL cache downloaded models and workflows on local disk to avoid re-downloading.

#### Scenario: Cache hit on model load
- **WHEN** a client requests to load a model that exists in local cache
- **THEN** the deploy container skips the HTTP download
- **AND** loads the model directly from cache

#### Scenario: Cache miss on model load
- **WHEN** a client requests to load a model not in local cache
- **THEN** the deploy container downloads it from server
- **AND** saves it to `/app/cache/models/{project_id}_{version}/`

#### Scenario: Cache persists across restarts
- **WHEN** the deploy container restarts
- **THEN** the cache directory contents are preserved
- **AND** previously downloaded models remain available
