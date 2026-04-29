"""YOLO 模型训练脚本，由主应用通过 subprocess 调用。"""
import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='data.yaml 路径')
    parser.add_argument('--model', default='yolo11n.pt', help='基础模型路径或名称')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0', help='cuda device or cpu')
    parser.add_argument('--project', required=True, help='工程目录，用于保存结果')
    parser.add_argument('--export-dir', required=True, help='临时导出目录，训练后清理')
    parser.add_argument('--yolo-version', default='yolo11', help='YOLO 版本标识')
    parser.add_argument('--version', default=None, help='模型版本号（时间戳格式）')
    parser.add_argument('--dataset-stats', default=None, help='数据集统计信息 JSON 文件路径')
    parser.add_argument('--task', default='detect', help='任务类型: detect, segment, pose, obb, classify')
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

    # numpy 2.0 trapz 兼容补丁
    try:
        import numpy as np
        if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
            np.trapz = np.trapezoid
    except Exception:
        pass

    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: 无法导入 ultralytics: {e}")
        sys.exit(1)

    # 确保模型保存目录存在
    models_dir = os.path.join(args.project, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 确定版本号
    version = args.version
    if not version:
        from datetime import datetime
        version = datetime.now().strftime('%Y%m%d_%H%M%S')

    version_dir = os.path.join(models_dir, version)
    os.makedirs(version_dir, exist_ok=True)

    # 读取数据集统计信息
    dataset_stats = {}
    if args.dataset_stats and os.path.exists(args.dataset_stats):
        try:
            with open(args.dataset_stats, 'r', encoding='utf-8') as f:
                dataset_stats = json.load(f)
        except Exception:
            pass

    # 加载模型（任务类型决定模型类）
    print(f"加载模型: {args.model} (task={args.task})")
    model = YOLO(args.model)

    # 开始训练
    print(f"开始训练: epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz}, device={args.device}, task={args.task}")
    try:
        train_kwargs = {
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch,
            'imgsz': args.imgsz,
            'device': args.device,
            'project': models_dir,
            'name': version,
            'exist_ok': True,
            'verbose': True,
        }
        # 分类任务不需要 data.yaml，直接使用数据集目录
        if args.task == 'classify':
            train_kwargs['data'] = os.path.dirname(args.data)
        model.train(**train_kwargs)
    except Exception as e:
        print(f"ERROR: 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 保存最佳权重到版本目录和固定路径
    best_src = os.path.join(version_dir, 'weights', 'best.pt')
    best_version = os.path.join(version_dir, 'best.pt')
    best_latest = os.path.join(models_dir, 'best.pt')
    if os.path.exists(best_src):
        import shutil
        shutil.copy2(best_src, best_version)
        shutil.copy2(best_src, best_latest)
        print(f"最佳权重已保存: {best_version}")
        print(f"最新权重已更新: {best_latest}")

    # 从 args.yaml -> data.yaml 读取类别名称
    classes = []
    try:
        import yaml
        args_yaml_path = os.path.join(version_dir, 'args.yaml')
        if os.path.exists(args_yaml_path):
            with open(args_yaml_path, 'r', encoding='utf-8') as f:
                args_yaml = yaml.safe_load(f)
            data_yaml_path = args_yaml.get('data')
            if data_yaml_path and os.path.exists(data_yaml_path):
                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                    data_yaml = yaml.safe_load(f)
                names = data_yaml.get('names', {})
                if isinstance(names, dict):
                    classes = [names[i] for i in sorted(names.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else x)]
                elif isinstance(names, list):
                    classes = names
    except Exception as e:
        print(f"WARNING: 读取类别名称失败: {e}")

    # 保存模型元数据
    from datetime import datetime, timezone
    model_info = {
        'version': version,
        'yolo_version': args.yolo_version,
        'task': args.task,
        'base_model': args.model,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'dataset': dataset_stats,
        'classes': classes,
    }
    info_path = os.path.join(version_dir, 'model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"模型信息已保存: {info_path}")

    # 同时保存到最新路径（向后兼容）
    info_latest = os.path.join(models_dir, 'model_info.json')
    with open(info_latest, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    # 运行验证并保存结果（分类任务不运行标准验证）
    if args.task == 'classify':
        val_summary = {'top1_acc': 0, 'top5_acc': 0}
        try:
            val_results = model.val()
            if hasattr(val_results, 'top1'):
                val_summary['top1_acc'] = float(val_results.top1)
            if hasattr(val_results, 'top5'):
                val_summary['top5_acc'] = float(val_results.top5)
            if hasattr(val_results, 'results_dict'):
                rd = val_results.results_dict
                val_summary['top1_acc'] = float(rd.get('metrics/accuracy_top1', val_summary['top1_acc']))
                val_summary['top5_acc'] = float(rd.get('metrics/accuracy_top5', val_summary['top5_acc']))
        except Exception as e:
            print(f"WARNING: 验证失败: {e}")
        result_path = os.path.join(version_dir, 'val_results.json')
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(val_summary, f, indent=2, ensure_ascii=False)
        print(f"验证结果: top1={val_summary['top1_acc']:.4f}")
        result_latest = os.path.join(models_dir, 'val_results.json')
        with open(result_latest, 'w', encoding='utf-8') as f:
            json.dump(val_summary, f, indent=2, ensure_ascii=False)
    else:
        try:
            val_results = model.val()
            print(f"验证结果类型: {type(val_results)}")
            if hasattr(val_results, 'box'):
                box = val_results.box
                print(f"box 类型: {type(box)}, 属性: {dir(box)}")
                val_summary = {
                    'mAP50': float(box.map50) if hasattr(box, 'map50') else 0,
                    'mAP50-95': float(box.map) if hasattr(box, 'map') else 0,
                    'precision': float(box.mp) if hasattr(box, 'mp') else 0,
                    'recall': float(box.mr) if hasattr(box, 'mr') else 0,
                }
            elif hasattr(val_results, 'results_dict'):
                rd = val_results.results_dict
                val_summary = {
                    'mAP50': float(rd.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(rd.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(rd.get('metrics/precision(B)', 0)),
                    'recall': float(rd.get('metrics/recall(B)', 0)),
                }
            else:
                print(f"WARNING: 无法解析验证结果: {val_results}")
                val_summary = {'mAP50': 0, 'mAP50-95': 0, 'precision': 0, 'recall': 0}
            result_path = os.path.join(version_dir, 'val_results.json')
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(val_summary, f, indent=2, ensure_ascii=False)
            print(f"验证结果: mAP50={val_summary['mAP50']:.4f}, mAP50-95={val_summary['mAP50-95']:.4f}")

            # 同时保存到最新路径（向后兼容）
            result_latest = os.path.join(models_dir, 'val_results.json')
            with open(result_latest, 'w', encoding='utf-8') as f:
                json.dump(val_summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            import traceback
            print(f"WARNING: 验证失败: {e}")
            traceback.print_exc()

    print("训练完成")


if __name__ == '__main__':
    main()
