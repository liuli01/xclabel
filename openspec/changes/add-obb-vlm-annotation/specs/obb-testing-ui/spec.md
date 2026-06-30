## ADDED Requirements

### Requirement: ai-config 页面 OBB 模式切换

系统 SHALL 在 `/ai-config` 页面的「图片标注」标签页中提供 OBB 模式切换功能。

#### Scenario: OBB 模式开关
- **WHEN** 用户访问 `/ai-config` 页面
- **THEN** 「图片标注」标签页的提示词区域上方 SHALL 显示一个「OBB 模式」复选框
- **THEN** 该复选框 SHALL 默认不选中

#### Scenario: OBB 模式启用提示词模板
- **WHEN** 用户勾选「OBB 模式」
- **THEN** 提示词文本框 SHALL 切换为 OBB 模板内容
- **THEN** OBB 模板 SHALL 包含 4 角点 JSON 格式说明
- **THEN** OBB 模板 SHALL 包含顺时针点序的要求说明
- **THEN** 用户仍 SHALL 可自由编辑提示词内容

#### Scenario: 切换到非 OBB 模式恢复提示词
- **WHEN** 用户取消勾选「OBB 模式」
- **THEN** 提示词文本框 SHALL 恢复为默认的目标检测提示词模板

### Requirement: OBB 测试结果展示

系统 SHALL 在 ai-config 页面正确展示 OBB 检测结果。

#### Scenario: OBB 渲染结果显示
- **WHEN** 用户在 OBB 模式下完成一次图片标注测试
- **THEN** 渲染结果图 SHALL 显示 OBB 4 点多边形轮廓
- **THEN** 每个多边形 SHALL 带颜色区分
- **THEN** 多边形旁 SHALL 显示标签和置信度

#### Scenario: 非 OBB 模式不受影响
- **WHEN** 用户未勾选 OBB 模式
- **THEN** 标注测试的渲染结果 SHALL 仍为矩形框（与现有行为一致）
