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

#### Scenario: Open model test modal
- **WHEN** user clicks "模型测试" button on a project card
- **THEN** system opens the model test modal for the selected project

#### Scenario: Open trained model list
- **WHEN** user clicks "模型下载" button on a project card
- **THEN** system opens a modal displaying all trained model versions for the project
- **AND** each version shows: task type, YOLO version, base model, epochs, train/val count, class count, mAP50, mAP50-95
- **AND** each version provides "下载 .pt" and "下载 ONNX / 导出 ONNX" buttons

## ADDED Requirements

### Requirement: Project card displays model test and model download buttons
The system SHALL display "模型测试" and "模型下载" action buttons on each project card.

#### Scenario: View project card actions
- **WHEN** user views the project list page
- **THEN** each project card displays the following action buttons: "进入标注", "模型训练", "模型测试", "模型下载", "重命名", "删除"
