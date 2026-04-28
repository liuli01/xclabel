## ADDED Requirements

### Requirement: Server provides model download API
The xclabel-server SHALL expose an API endpoint for downloading trained models by project and version.

#### Scenario: Download specific model version
- **WHEN** deploy container sends GET to `/api/model/download?project=test&version=20260428_104927`
- **THEN** the server locates the model at `projects/test/models/20260428_104927/`
- **AND** packages the directory contents into a zip archive
- **AND** returns the zip file as a binary download

#### Scenario: List model versions
- **WHEN** deploy container sends GET to `/api/model/versions?project=test`
- **THEN** the server scans `projects/test/models/` directory
- **AND** returns a list of version directories with their metadata

#### Scenario: Download latest model
- **WHEN** deploy container sends GET to `/api/model/download?project=test&version=latest`
- **THEN** the server resolves "latest" to the most recent version directory
- **AND** returns the corresponding model package
