## ADDED Requirements

### Requirement: User can select a YOLO model for testing
The system SHALL allow users to select an installed YOLO model for inference testing.

#### Scenario: Model selection by version and task type
- **WHEN** user opens the model test modal
- **THEN** the system displays a version selector (YOLOv8 / YOLO11 / YOLO26)
- **AND** a task type selector (detect / segment / pose / obb / classify)
- **AND** a model selector populated with `.pt` files found in the selected version's `models/` directory

#### Scenario: No models installed
- **WHEN** user opens the model test modal and no `.pt` files exist for the selected version
- **THEN** the system displays a message "该版本暂无已安装模型"
- **AND** provides a link to open the settings panel to download models

### Requirement: User can provide test images for inference
The system SHALL allow users to use dataset test images or upload custom images for model inference testing.

#### Scenario: Default test images loaded from dataset
- **WHEN** user opens the model test modal
- **THEN** the system scans `projects/<project>/test/images/` for available test images
- **AND** loads up to 4 test images as default samples
- **AND** displays them as selectable thumbnails

#### Scenario: Successful image upload
- **WHEN** user selects or drops an image file (JPG, PNG, JPEG) into the upload area
- **THEN** the system displays a preview of the uploaded image
- **AND** enables the "开始推理" button

#### Scenario: Invalid file type
- **WHEN** user attempts to upload a non-image file
- **THEN** the system displays an error message "仅支持 JPG/PNG 图片格式"

### Requirement: System can run model inference and return results
The system SHALL execute YOLO inference on the uploaded image using the selected model and return structured detection results.

#### Scenario: Successful inference
- **WHEN** user clicks "开始推理" with a valid model and uploaded image
- **THEN** the system loads the selected model via the corresponding version's virtual environment
- **AND** runs inference on the uploaded image
- **AND** returns detection results in a Roboflow-compatible JSON format with a top-level `predictions` array
- **AND** each prediction MUST contain `x` (center x), `y` (center y), `width`, `height`, `confidence`, `class`, `class_id`, and a unique `detection_id`
- **AND** for segment tasks, each prediction additionally contains `points` as a normalized polygon array
- **AND** for pose tasks, each prediction additionally contains `keypoints` with `x`, `y`, `confidence`, and `name`
- **AND** for obb tasks, each prediction contains `points` as the 4 corner pixel coordinates instead of `x`/`y`/`width`/`height`
- **AND** for classify tasks, the response contains `predictions` with only `class`, `confidence`, and `class_id`

#### Scenario: Inference environment not installed
- **WHEN** user attempts to run inference but the selected version's virtual environment is not installed
- **THEN** the system displays an error message "YOLO 环境未安装，请先安装训练环境"

#### Scenario: GPU resource conflict
- **WHEN** user attempts to run inference while a training job or AI auto-labeling task is active
- **THEN** the system displays a warning message "GPU 资源被占用，请等待其他任务完成"
- **AND** prevents the inference from starting

### Requirement: System displays inference results visually
The system SHALL render the inference results overlaid on the uploaded image in the model test modal.

#### Scenario: Display detection results
- **WHEN** inference completes successfully for a detect task
- **THEN** the system draws bounding boxes on the image preview
- **AND** displays class labels and confidence scores next to each box

#### Scenario: Display segmentation results
- **WHEN** inference completes successfully for a segment task
- **THEN** the system draws bounding boxes and semi-transparent polygon masks on the image preview
- **AND** displays class labels and confidence scores

#### Scenario: Display pose results
- **WHEN** inference completes successfully for a pose task
- **THEN** the system draws bounding boxes and keypoints with skeletal connections on the image preview
- **AND** displays class labels

#### Scenario: Display OBB results
- **WHEN** inference completes successfully for an obb task
- **THEN** the system draws oriented bounding boxes (4-point polygons) on the image preview
- **AND** displays class labels and confidence scores

#### Scenario: Display classification results
- **WHEN** inference completes successfully for a classify task
- **THEN** the system displays the predicted class label and confidence score as text overlay on the image

### Requirement: System cleans up state on modal close
The system SHALL clear all temporary data when the model test modal is closed.

#### Scenario: Close modal and clean up
- **WHEN** user closes the model test modal
- **THEN** the system clears the Canvas, inference result JSON, and any user-uploaded temporary files
- **AND** removes temporary files from `temp/model-test/<project>/`
- **AND** resets the modal to its initial state for the next opening

### Requirement: User can toggle annotation visibility
The system SHALL allow users to show or hide the inference annotations on the result image.

#### Scenario: Toggle annotations
- **WHEN** user clicks the "显示/隐藏标注" toggle
- **THEN** the system redraws the canvas with or without the overlaid annotations
- **AND** preserves the original image preview
