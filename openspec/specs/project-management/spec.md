## ADDED Requirements

### Requirement: User can create a project
The system SHALL allow users to create a new annotation project with a unique name.

#### Scenario: Successful project creation
- **WHEN** user enters a valid project name and clicks "Create"
- **THEN** system creates a new directory under `projects/`
- **AND** system initializes the project with empty annotations and default classes
- **AND** system returns the new project in the list

#### Scenario: Duplicate project name
- **WHEN** user attempts to create a project with a name that already exists
- **THEN** system displays an error message "工程名称已存在"

#### Scenario: Invalid project name
- **WHEN** user enters a project name containing invalid characters (`/\:*?"<>|`)
- **THEN** system displays an error message "工程名称包含非法字符"

### Requirement: User can list all projects
The system SHALL display all existing annotation projects on the project management page.

#### Scenario: View project list
- **WHEN** user navigates to the project management page
- **THEN** system displays a list of all projects with name, image count, and last modified time

#### Scenario: Empty project list
- **WHEN** no projects exist
- **THEN** system displays a message "暂无工程，请创建新工程"

### Requirement: User can switch between projects
The system SHALL allow users to switch the active project.

#### Scenario: Switch to another project
- **WHEN** user clicks on a project card or "进入标注" button
- **THEN** system sets the selected project as current
- **AND** system redirects to the annotation page
- **AND** the annotation page loads images and annotations for the current project

### Requirement: User can rename a project
The system SHALL allow users to rename an existing project.

#### Scenario: Successful rename
- **WHEN** user clicks rename, enters a new valid name, and confirms
- **THEN** system renames the project directory
- **AND** updates the project list

#### Scenario: Rename to existing name
- **WHEN** user attempts to rename a project to a name that already exists
- **THEN** system displays an error message

### Requirement: User can delete a project
The system SHALL allow users to delete an existing project and all its data.

#### Scenario: Successful deletion
- **WHEN** user clicks delete and confirms in the confirmation dialog
- **THEN** system removes the project directory and all its contents
- **AND** removes the project from the list

#### Scenario: Cancel deletion
- **WHEN** user clicks cancel in the confirmation dialog
- **THEN** system does not delete the project
