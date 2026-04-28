## MODIFIED Requirements

### Requirement: User can switch between projects
The system SHALL allow users to switch the active project.

#### Scenario: Switch to another project
- **WHEN** user clicks on a project card or "进入标注" button
- **THEN** system sets the selected project as current
- **AND** system redirects to the annotation page
- **AND** the annotation page loads images and annotations for the current project

#### Scenario: Navigate to training page
- **WHEN** user clicks "模型训练" button on a project card
- **THEN** system navigates to `/train?project=<project_name>`

#### Scenario: Open workflow management tab
- **WHEN** user clicks "工作流" tab on the project page
- **THEN** system displays the workflow list for the selected project
- **AND** provides a button to create a new workflow
