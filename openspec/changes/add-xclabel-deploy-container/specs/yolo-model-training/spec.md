## MODIFIED Requirements

### Requirement: User can navigate to training page from project list
The system SHALL provide a training entry point on the project management page for each project.

#### Scenario: Training button visible on project card
- **WHEN** user views the project list page
- **THEN** each project card displays an "进入训练" button next to the "进入标注" button
- **AND** clicking the button navigates to `/train?project=<project_name>`

### Requirement: Training environment is installable on demand
The system SHALL allow users to install the Ultralytics training environment in an isolated venv.

#### Scenario: First-time training environment setup
- **WHEN** user opens the training page and the `plugins/ultralytics` venv is not installed
- **THEN** the system displays an installation prompt with an "安装训练环境" button
- **AND** clicking the button starts installing `ultralytics` in `plugins/ultralytics`
- **AND** installation progress is streamed to the UI in real time

#### Scenario: Training environment already installed
- **WHEN** user opens the training page and the environment is already installed
- **THEN** the system skips the installation prompt and shows the training configuration form directly

### Requirement: User can configure training parameters
The system SHALL allow users to configure training parameters before starting a training job.

#### Scenario: Default training configuration
- **WHEN** user opens the training page for a project
- **THEN** the system pre-fills the form with default values:
  - model: yolo11n.pt
  - epochs: 100
  - batch: 8
  - imgsz: 640
  - device: auto (CPU/GPU auto-detect)

#### Scenario: Custom training configuration
- **WHEN** user modifies any training parameter
- **THEN** the system validates the input (positive integers, supported model names)
- **AND** invalid values display inline error messages

#### Scenario: Train/validation split configuration
- **WHEN** user views the training configuration form
- **THEN** the system displays a "训练/验证比例" slider with default value 80% (train) / 20% (val)
- **AND** the user can adjust the ratio between 50% and 95%

#### Scenario: Base model selection for fine-tuning
- **WHEN** user views the training configuration form
- **THEN** the system displays a "基础模型" selector with two options:
  - "官方预训练权重" (default)
  - "本工程已有模型" (enabled only if `projects/<project>/models/best.pt` exists)

### Requirement: Post-training model export for deploy
The system SHALL automatically export models to a format consumable by the deploy container after successful training.

#### Scenario: Auto-export ONNX after training
- **WHEN** training completes successfully
- **THEN** the system automatically exports the best model to ONNX format
- **AND** saves it to `projects/<project>/models/<version>/best.onnx`
- **AND** the exported model is included in the deploy container download package

#### Scenario: Export metadata for deploy
- **WHEN** training completes successfully
- **THEN** the system generates a `deploy_metadata.json` in the version directory
- **AND** the metadata includes input shape, class names, confidence thresholds, and preprocessing parameters

#### Scenario: Deploy-ready notification
- **WHEN** training and export complete
- **THEN** the system displays a notification with the message "模型已就绪，可部署到推理容器"
- **AND** provides a button to copy the deployment command or API call
