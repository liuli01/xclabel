## ADDED Requirements

### Requirement: Server provides workflow export API
The xclabel-server SHALL expose an API endpoint for exporting workflow definitions.

#### Scenario: Export workflow by name
- **WHEN** deploy container sends GET to `/api/workflow/export?project=test&name=detection-pipeline`
- **THEN** the server locates the workflow definition
- **AND** returns the workflow.json content

#### Scenario: List available workflows
- **WHEN** deploy container sends GET to `/api/workflow/list?project=test`
- **THEN** the server returns a list of workflow names and their descriptions

#### Scenario: Export workflow with model references
- **WHEN** deploy container exports a workflow that references model files
- **THEN** the server includes the model file paths in the workflow.json
- **AND** the deploy container resolves these paths to download the actual models
