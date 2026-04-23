## ADDED Requirements

### Requirement: Each project has isolated data storage
The system SHALL store each project's images, annotations, and classes in separate directories.

#### Scenario: Project data isolation
- **WHEN** user imports images into project A
- **THEN** the images are stored in `projects/project-a/` only
- **AND** project B cannot access project A's images

#### Scenario: Annotation isolation
- **WHEN** user creates annotations in project A
- **THEN** the annotations are saved to `projects/project-a/annotations/annotations.json`
- **AND** switching to project B shows different annotations

#### Scenario: Class isolation
- **WHEN** user defines classes in project A
- **THEN** the classes are saved to `projects/project-a/annotations/classes.json`
- **AND** project B has its own independent class list

### Requirement: System tracks current project
The system SHALL maintain the currently active project for each user session.

#### Scenario: Default project on first visit
- **WHEN** user visits the application for the first time
- **THEN** system checks if `projects/` exists
- **AND** if not, migrates existing data to `projects/default/`
- **AND** sets "default" as the current project

#### Scenario: API routes use current project
- **WHEN** any image/annotation API is called
- **THEN** system reads/writes data from/to the current project's directory
- **AND** returns 400 if no current project is set

### Requirement: Project metadata is accessible
The system SHALL provide project metadata including image count and last modified time.

#### Scenario: Get project stats
- **WHEN** user views the project list
- **THEN** system calculates and displays the number of images in each project
- **AND** displays the last modified time of the project's annotation file
