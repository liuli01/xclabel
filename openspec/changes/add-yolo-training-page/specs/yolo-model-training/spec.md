## ADDED Requirements

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
- **AND** selecting "本工程已有模型" uses the project's previous best weights as the starting point

### Requirement: System can start training from project annotations
The system SHALL convert the current project's annotated data to YOLO format and launch training.

#### Scenario: Successful training start
- **WHEN** user clicks "开始训练" with valid parameters
- **THEN** the system exports the current project's images and annotations to a temporary YOLO dataset directory
- **AND** spawns a training subprocess using the `plugins/ultralytics` Python interpreter
- **AND** the training process uses the exported dataset
- **AND** the UI transitions to a progress monitoring view

#### Scenario: Training start with insufficient data
- **WHEN** user attempts to start training but the project has fewer than 10 annotated images
- **THEN** the system displays an error message: "标注数据不足，至少需要 10 张已标注图片"
- **AND** training is not started

### Requirement: System prevents concurrent GPU-intensive tasks
The system SHALL prevent training and AI auto-labeling from running simultaneously to avoid resource conflicts.

#### Scenario: Training blocked by active AI labeling
- **WHEN** user attempts to start training while an AI auto-labeling task is running
- **THEN** the system displays a message: "AI 标注任务正在运行，请先等待完成或取消后再启动训练"
- **AND** training is not started

#### Scenario: AI labeling blocked by active training
- **WHEN** user attempts to start AI auto-labeling while a training job is running
- **THEN** the system displays a message: "模型训练正在进行中，请先等待完成或取消后再启动 AI 标注"
- **AND** AI labeling is not started

### Requirement: Training progress is displayed in real time
The system SHALL display real-time training progress to the user.

#### Scenario: Training progress updates
- **WHEN** a training job is running
- **THEN** the UI displays:
  - current epoch / total epochs
  - box_loss, cls_loss, dfl_loss
  - mAP50, mAP50-95 (if available)
  - estimated remaining time
- **AND** progress updates are pushed via SocketIO at least once per epoch

#### Scenario: Training completes successfully
- **WHEN** training finishes all epochs
- **THEN** the system runs automatic validation on the validation set
- **AND** the system saves the best model weights to `projects/<project>/models/best.pt`
- **AND** the UI displays a completion message with:
  - final mAP50 and mAP50-95
  - precision and recall
  - total training time
- **AND** the trained model file appears in the project's model list

#### Scenario: Training fails
- **WHEN** training encounters a fatal error (e.g., OOM, invalid data)
- **THEN** the system terminates the subprocess gracefully
- **AND** the UI displays the error message
- **AND** partial results are not saved

### Requirement: User can cancel an ongoing training job
The system SHALL allow users to cancel a running training job.

#### Scenario: Cancel training
- **WHEN** user clicks "取消训练" during an active training session
- **THEN** the system sends a termination signal to the training subprocess
- **AND** the UI returns to the training configuration form
- **AND** partial results are not saved
