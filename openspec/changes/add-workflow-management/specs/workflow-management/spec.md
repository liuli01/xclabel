## ADDED Requirements

### Requirement: User can create a workflow
The system SHALL allow users to create a new workflow within a project.

#### Scenario: Successful workflow creation
- **WHEN** user enters a workflow name and clicks "Create"
- **THEN** system creates a new workflow JSON file under `projects/<project>/workflows/`
- **AND** system initializes the workflow with an empty node list and default metadata
- **AND** system returns the new workflow in the list

#### Scenario: Duplicate workflow name
- **WHEN** user attempts to create a workflow with a name that already exists in the same project
- **THEN** system displays an error message "工作流名称已存在"

#### Scenario: Invalid workflow name
- **WHEN** user enters a workflow name containing invalid characters (`/\\:*?"<>|`)
- **THEN** system displays an error message "工作流名称包含非法字符"

### Requirement: User can list all workflows
The system SHALL display all existing workflows for the current project.

#### Scenario: View workflow list
- **WHEN** user navigates to the "Workflows" tab on the project page
- **THEN** system displays a list of all workflows with name, node count, last modified time, and run status

#### Scenario: Empty workflow list
- **WHEN** no workflows exist for the project
- **THEN** system displays a message "暂无工作流，请创建新工作流"

### Requirement: User can edit a workflow
The system SHALL allow users to modify a workflow's node configuration.

#### Scenario: Add a node
- **WHEN** user selects a node type from the node palette and configures its parameters
- **THEN** system appends the node to the workflow's node list

#### Scenario: Reorder nodes
- **WHEN** user clicks "上移" or "下移" on a node
- **THEN** system swaps the node with its adjacent node in the list

#### Scenario: Remove a node
- **WHEN** user clicks "删除" on a node and confirms
- **THEN** system removes the node from the workflow's node list

#### Scenario: Save workflow changes
- **WHEN** user clicks "保存工作流"
- **THEN** system persists the updated node list and configuration to the workflow JSON file

### Requirement: User can delete a workflow
The system SHALL allow users to delete an existing workflow.

#### Scenario: Successful deletion
- **WHEN** user clicks delete and confirms in the confirmation dialog
- **THEN** system removes the workflow JSON file
- **AND** removes the workflow from the list

#### Scenario: Cancel deletion
- **WHEN** user clicks cancel in the confirmation dialog
- **THEN** system does not delete the workflow

### Requirement: User can run a workflow
The system SHALL allow users to execute a workflow sequentially.

#### Scenario: Start workflow execution
- **WHEN** user clicks "运行工作流"
- **THEN** system validates the workflow configuration
- **AND** begins executing nodes in order
- **AND** updates the run status in real-time

#### Scenario: Workflow node execution success
- **WHEN** a node completes successfully
- **THEN** system records the node's output
- **AND** proceeds to the next node

#### Scenario: Workflow node execution failure
- **WHEN** a node fails to execute
- **THEN** system stops the workflow
- **AND** marks the failed node with its error message
- **AND** allows user to retry from the failed node or reset the workflow

#### Scenario: Workflow completion
- **WHEN** all nodes have executed successfully
- **THEN** system marks the workflow as completed
- **AND** displays a summary of all node outputs

### Requirement: User can view workflow execution status
The system SHALL display the real-time status of a running or completed workflow.

#### Scenario: View node status
- **WHEN** user opens a workflow that has been run
- **THEN** system displays each node's execution status: pending, running, success, or failed

#### Scenario: View node logs
- **WHEN** user expands a node in the workflow editor
- **THEN** system displays the execution logs and output for that node
