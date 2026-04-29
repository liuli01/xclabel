"""YOLO 模型导出脚本，由主应用通过 subprocess 调用。"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='模型权重路径')
    parser.add_argument('--output', required=True, help='导出文件输出路径')
    args = parser.parse_args()

    # PyTorch 2.6 weights_only 兼容补丁
    try:
        import torch
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load
    except Exception:
        pass

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: 无法导入 ultralytics: {e}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: 模型文件不存在: {args.model}")
        sys.exit(1)

    print(f"加载模型: {args.model}")
    model = YOLO(args.model)

    # 获取输出目录
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"导出 ONNX 格式...")
    try:
        # ultralytics 导出到模型同级目录，先切换工作目录
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = os.path.join(tmpdir, os.path.basename(args.model))
            import shutil
            shutil.copy2(args.model, tmp_model)
            m = YOLO(tmp_model)
            m.export(format='onnx')
            exported = os.path.join(tmpdir, os.path.splitext(os.path.basename(args.model))[0] + '.onnx')
            if os.path.exists(exported):
                shutil.copy2(exported, args.output)
                print(f"ONNX 导出成功: {args.output}")
            else:
                print("ERROR: 导出后未找到 ONNX 文件")
                sys.exit(1)
    except Exception as e:
        print(f"ERROR: ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
