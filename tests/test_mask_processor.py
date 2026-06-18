"""tests/test_mask_processor.py"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ai_manager import MaskProcessor


def test_mask_to_polygons_simple():
    """简单矩形 mask 应生成一个四边形"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:80, 20:80] = 1.0

    polygons = MaskProcessor.mask_to_polygons(mask)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 4
    # 验证所有点在矩形范围内
    for x, y in polygons[0]:
        assert 20 <= x <= 80
        assert 20 <= y <= 80


def test_mask_to_polygons_empty():
    """空 mask 应返回空列表"""
    mask = np.zeros((100, 100), dtype=np.float32)
    polygons = MaskProcessor.mask_to_polygons(mask)
    assert len(polygons) == 0


def test_mask_to_polygons_min_area():
    """小于最小面积的区域应被过滤"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[5:10, 5:10] = 1.0  # 25 像素
    polygons = MaskProcessor.mask_to_polygons(mask, min_area=100)
    assert len(polygons) == 0


def test_mask_to_polygons_multiple():
    """多个独立区域应返回多个多边形"""
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[10:30, 10:30] = 1.0
    mask[60:80, 60:80] = 1.0
    polygons = MaskProcessor.mask_to_polygons(mask, min_area=50)
    assert len(polygons) == 2


def test_mask_to_polygons_circle():
    """圆形 mask 应生成一个近似的多边形"""
    mask = np.zeros((100, 100), dtype=np.float32)
    y, x = np.ogrid[:100, :100]
    center = (50, 50)
    radius = 30
    mask[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = 1.0

    polygons = MaskProcessor.mask_to_polygons(mask, min_area=500)
    assert len(polygons) == 1
    assert len(polygons[0]) >= 8  # 圆近似至少有 8 个顶点


def test_mask_to_polygons_uint8_input():
    """uint8 类型的 mask（值域 0-255）应正确处理"""
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:30, 10:30] = 255

    polygons = MaskProcessor.mask_to_polygons(mask)
    assert len(polygons) == 1


def test_mask_to_polygons_max_vertices():
    """应限制多边形顶点数不超过 max_vertices"""
    mask = np.zeros((100, 100), dtype=np.float32)
    # 创建一个复杂形状（有很多边缘）
    mask[10:90, 10:90] = 1.0
    # 添加锯齿边缘
    for i in range(10, 90, 2):
        mask[i, 10:90] = np.random.random(80) > 0.3

    polygons = MaskProcessor.mask_to_polygons(mask, max_vertices=10, simplify_tolerance=0.5)
    if polygons:
        assert len(polygons[0]) <= 10


def test_mask_to_polygons_batch():
    """批量转换应正确处理多个 mask"""
    masks = np.zeros((3, 100, 100), dtype=np.float32)
    masks[0, 10:30, 10:30] = 1.0
    masks[1, 40:60, 40:60] = 1.0
    masks[2, 70:90, 70:90] = 1.0

    result = MaskProcessor.mask_to_polygons_batch(masks, min_area=50)
    assert len(result) == 3
    for polygons in result:
        assert len(polygons) == 1


def test_mask_to_polygons_three_vertices_minimum():
    """多边形至少有 3 个顶点"""
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[5:8, 5:8] = 1.0  # 非常小的区域

    polygons = MaskProcessor.mask_to_polygons(mask, min_area=1, simplify_tolerance=10)
    # 即使简化过度，也不会产生少于 3 个顶点的多边形
    for poly in polygons:
        assert len(poly) >= 3
