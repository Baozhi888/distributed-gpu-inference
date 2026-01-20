#!/usr/bin/env python3
"""
GPU Worker CLI 安装器和配置向导
提供交互式的安装和配置体验
"""
import os
import sys
import argparse
import subprocess
import platform
import shutil
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

# 尝试导入rich库（用于漂亮的终端输出）
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 简单的控制台输出（无rich时的降级方案）
class SimpleConsole:
    def print(self, *args, **kwargs):
        # 移除style参数
        kwargs.pop('style', None)
        print(*args, **kwargs)

    def rule(self, title=""):
        print(f"\n{'='*50}")
        if title:
            print(f"  {title}")
        print('='*50)

console = Console() if RICH_AVAILABLE else SimpleConsole()


# ==================== 常量定义 ====================

REGIONS = {
    "asia-east": "东亚（中国、日本、韩国）",
    "asia-south": "东南亚（新加坡、泰国）",
    "europe-west": "西欧（德国、法国、英国）",
    "europe-east": "东欧",
    "america-north": "北美（美国、加拿大）",
    "america-south": "南美",
    "oceania": "大洋洲（澳大利亚）"
}

TASK_TYPES = {
    "llm": "大语言模型推理 (LLM)",
    "image_gen": "图像生成 (Stable Diffusion/FLUX)",
    "whisper": "语音识别 (Whisper)",
    "embedding": "文本嵌入 (Embedding)"
}

DEFAULT_SERVER_URL = "https://gpu-inference.example.com"

CONFIG_FILE = "config.yaml"


# ==================== 工具函数 ====================

def clear_screen():
    """清屏"""
    os.system('cls' if platform.system() == 'Windows' else 'clear')


def _probe_nvidia_smi() -> Optional[Dict[str, Any]]:
    """通过 nvidia-smi 获取 GPU 信息（无 PyTorch 时的兜底）"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return None

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return None

        first = [part.strip() for part in lines[0].split(",")]
        model = first[0] if first else "Unknown"
        memory_gb = 0
        if len(first) > 1:
            try:
                memory_gb = round(float(first[1]) / 1024, 1)
            except ValueError:
                memory_gb = 0

        return {
            "count": len(lines),
            "model": model,
            "memory_gb": memory_gb
        }
    except Exception:
        return None


def _detect_cuda_version() -> Optional[Dict[str, int]]:
    """从 nvidia-smi 输出中解析 CUDA 版本"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return None

        match = re.search(r"CUDA Version:\\s*(\\d+)\\.(\\d+)", result.stdout)
        if not match:
            return None

        return {
            "major": int(match.group(1)),
            "minor": int(match.group(2))
        }
    except Exception:
        return None


def _select_torch_index_url(cuda_version: Optional[Dict[str, int]]) -> Optional[str]:
    """根据 CUDA 版本选择 PyTorch 安装源"""
    if not cuda_version:
        return None

    major = cuda_version["major"]
    minor = cuda_version["minor"]
    version_value = major * 100 + minor

    if version_value >= 1204:
        return "https://download.pytorch.org/whl/cu124"
    if version_value >= 1201:
        return "https://download.pytorch.org/whl/cu121"
    if version_value >= 1108:
        return "https://download.pytorch.org/whl/cu118"
    return None


def check_gpu():
    """检测GPU信息"""
    gpu_info = {
        "available": False,
        "count": 0,
        "model": "Unknown",
        "memory_gb": 0,
        "nvidia_detected": False,
        "nvidia_model": None,
        "nvidia_memory_gb": None,
        "nvidia_count": 0
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_info["memory_gb"] = round(props.total_memory / 1024**3, 1)
    except ImportError:
        pass

    if not gpu_info["available"]:
        nvidia_info = _probe_nvidia_smi()
        if nvidia_info:
            gpu_info["nvidia_detected"] = True
            gpu_info["nvidia_count"] = nvidia_info["count"]
            gpu_info["nvidia_model"] = nvidia_info["model"]
            gpu_info["nvidia_memory_gb"] = nvidia_info["memory_gb"]

    return gpu_info


def check_dependencies() -> Dict[str, bool]:
    """检查依赖"""
    deps = {
        "python": sys.version_info >= (3, 9),
        "torch": False,
        "transformers": False,
        "cuda": False
    }

    try:
        import torch
        deps["torch"] = True
        deps["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import transformers
        deps["transformers"] = True
    except ImportError:
        pass

    return deps


def _pip_install(
    requirement: str,
    progress_callback=None,
    index_url: Optional[str] = None,
    extra_args: Optional[List[str]] = None
):
    if progress_callback:
        progress_callback(f"Installing {requirement}...")

    command = [sys.executable, "-m", "pip", "install", requirement]
    if index_url:
        command.extend(["--index-url", index_url])
    if extra_args:
        command.extend(extra_args)

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise Exception(f"Failed to install {requirement}: {result.stderr}")


def install_dependencies(progress_callback=None):
    """安装依赖"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "accelerate>=0.24.0",
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0",
        "httpx>=0.25.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ]

    cuda_version = _detect_cuda_version()
    torch_index_url = _select_torch_index_url(cuda_version)

    if torch_index_url:
        _pip_install(
            "torch",
            progress_callback,
            index_url=torch_index_url,
            extra_args=["--upgrade", "--force-reinstall"]
        )
    else:
        if cuda_version and progress_callback:
            progress_callback(
                f"检测到 CUDA {cuda_version['major']}.{cuda_version['minor']}，"
                "无匹配的 PyTorch 版本，将安装 CPU 版"
            )
        _pip_install("torch>=2.0.0", progress_callback)

    non_torch_requirements = [
        item for item in requirements
        if not re.match(r"^torch([<>=!~].*)?$", item.strip())
    ]

    for req in non_torch_requirements:
        _pip_install(req, progress_callback)


def save_config(config: Dict[str, Any], path: str = CONFIG_FILE):
    """保存配置"""
    import yaml
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def load_config(path: str = CONFIG_FILE) -> Optional[Dict[str, Any]]:
    """加载配置"""
    if not os.path.exists(path):
        return None

    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==================== 交互式配置向导 ====================

class ConfigWizard:
    """交互式配置向导"""

    def __init__(self):
        self.config = {}

    def run(self) -> Dict[str, Any]:
        """运行配置向导"""
        clear_screen()
        self._show_welcome()

        # 步骤1: 服务器配置
        self._configure_server()

        # 步骤2: 区域选择
        self._configure_region()

        # 步骤3: GPU检测
        self._configure_gpu()

        # 步骤4: 任务类型
        self._configure_task_types()

        # 步骤5: 负载控制
        self._configure_load_control()

        # 步骤6: 直连配置
        self._configure_direct_connection()

        # 步骤7: 确认配置
        self._confirm_config()

        return self.config

    def _show_welcome(self):
        """显示欢迎信息"""
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                "[bold cyan]分布式GPU推理 Worker 配置向导[/bold cyan]\n\n"
                "本向导将帮助您配置 GPU Worker 节点\n"
                "您可以随时按 Ctrl+C 退出",
                title="欢迎",
                border_style="cyan"
            ))
        else:
            console.rule("分布式GPU推理 Worker 配置向导")
            print("\n本向导将帮助您配置 GPU Worker 节点")
            print("您可以随时按 Ctrl+C 退出\n")

        input("按 Enter 键继续...")

    def _configure_server(self):
        """配置服务器"""
        console.rule("步骤 1/6: 服务器配置")

        if RICH_AVAILABLE:
            server_url = Prompt.ask(
                "请输入服务器地址",
                default=DEFAULT_SERVER_URL
            )

            use_https = Confirm.ask(
                "是否使用 HTTPS（推荐）",
                default=True
            )
        else:
            server_url = input(f"请输入服务器地址 [{DEFAULT_SERVER_URL}]: ").strip()
            if not server_url:
                server_url = DEFAULT_SERVER_URL

            use_https_input = input("是否使用 HTTPS（推荐）[Y/n]: ").strip().lower()
            use_https = use_https_input != 'n'

        # 确保URL格式正确
        if not server_url.startswith(('http://', 'https://')):
            server_url = ('https://' if use_https else 'http://') + server_url

        self.config['server'] = {
            'url': server_url,
            'timeout': 30,
            'verify_ssl': use_https
        }

        print(f"\n服务器地址: {server_url}")

    def _configure_region(self):
        """配置区域"""
        console.rule("步骤 2/6: 区域选择")

        print("\n可用区域:")
        if RICH_AVAILABLE:
            table = Table(show_header=True)
            table.add_column("序号", style="cyan")
            table.add_column("区域代码", style="green")
            table.add_column("描述")

            for i, (code, desc) in enumerate(REGIONS.items(), 1):
                table.add_row(str(i), code, desc)

            console.print(table)

            choice = IntPrompt.ask(
                "请选择您的区域",
                default=1,
                show_default=True
            )
        else:
            for i, (code, desc) in enumerate(REGIONS.items(), 1):
                print(f"  {i}. {code} - {desc}")

            choice = int(input("\n请选择您的区域 [1]: ") or "1")

        region_codes = list(REGIONS.keys())
        if 1 <= choice <= len(region_codes):
            selected_region = region_codes[choice - 1]
        else:
            selected_region = "asia-east"

        self.config['region'] = selected_region
        print(f"\n已选择区域: {selected_region} ({REGIONS[selected_region]})")

    def _configure_gpu(self):
        """检测和配置GPU"""
        console.rule("步骤 3/6: GPU 检测")

        print("\n正在检测 GPU...")
        gpu_info = check_gpu()

        if gpu_info["available"]:
            if RICH_AVAILABLE:
                console.print(f"[green]检测到 GPU:[/green]")
                console.print(f"  型号: {gpu_info['model']}")
                console.print(f"  显存: {gpu_info['memory_gb']} GB")
                console.print(f"  数量: {gpu_info['count']}")
            else:
                print(f"检测到 GPU:")
                print(f"  型号: {gpu_info['model']}")
                print(f"  显存: {gpu_info['memory_gb']} GB")
                print(f"  数量: {gpu_info['count']}")

            self.config['gpu'] = {
                'model': gpu_info['model'],
                'memory_gb': gpu_info['memory_gb'],
                'count': gpu_info['count'],
                'enable_cpu_offload': gpu_info['memory_gb'] < 16
            }
        else:
            nvidia_memory_gb = gpu_info.get("nvidia_memory_gb")
            if gpu_info.get("nvidia_detected"):
                if RICH_AVAILABLE:
                    console.print("[yellow]检测到 NVIDIA GPU，但 CUDA 版 PyTorch 不可用，将使用 CPU 模式（性能有限）[/yellow]")
                    console.print(f"  型号: {gpu_info.get('nvidia_model', 'Unknown')}")
                    if nvidia_memory_gb:
                        console.print(f"  显存: {nvidia_memory_gb} GB")
                    console.print("[yellow]请安装 CUDA 版 PyTorch（例如：python -m pip install torch --index-url https://download.pytorch.org/whl/cu121）[/yellow]")
                else:
                    print("检测到 NVIDIA GPU，但 CUDA 版 PyTorch 不可用，将使用 CPU 模式（性能有限）")
                    print(f"  型号: {gpu_info.get('nvidia_model', 'Unknown')}")
                    if nvidia_memory_gb:
                        print(f"  显存: {nvidia_memory_gb} GB")
                    print("请安装 CUDA 版 PyTorch（例如：python -m pip install torch --index-url https://download.pytorch.org/whl/cu121）")
            else:
                if RICH_AVAILABLE:
                    console.print("[yellow]未检测到 GPU，将使用 CPU 模式（性能有限）[/yellow]")
                else:
                    print("未检测到 GPU，将使用 CPU 模式（性能有限）")

            self.config['gpu'] = {
                'model': 'CPU',
                'memory_gb': 0,
                'count': 0,
                'enable_cpu_offload': True
            }

    def _configure_task_types(self):
        """配置支持的任务类型"""
        console.rule("步骤 4/6: 任务类型")

        print("\n可用任务类型:")
        if RICH_AVAILABLE:
            table = Table(show_header=True)
            table.add_column("序号", style="cyan")
            table.add_column("类型", style="green")
            table.add_column("描述")
            table.add_column("推荐显存")

            requirements = {
                "llm": "8GB+",
                "image_gen": "12GB+",
                "whisper": "4GB+",
                "embedding": "4GB+"
            }

            for i, (code, desc) in enumerate(TASK_TYPES.items(), 1):
                table.add_row(str(i), code, desc, requirements.get(code, "N/A"))

            console.print(table)
        else:
            requirements = {
                "llm": "8GB+",
                "image_gen": "12GB+",
                "whisper": "4GB+",
                "embedding": "4GB+"
            }
            for i, (code, desc) in enumerate(TASK_TYPES.items(), 1):
                print(f"  {i}. {code} - {desc} (推荐: {requirements.get(code, 'N/A')})")

        print("\n请输入要支持的任务类型序号（用逗号分隔，如: 1,2）")

        if RICH_AVAILABLE:
            choices_str = Prompt.ask("选择", default="1")
        else:
            choices_str = input("选择 [1]: ") or "1"

        type_codes = list(TASK_TYPES.keys())
        selected_types = []

        for choice in choices_str.split(','):
            try:
                idx = int(choice.strip()) - 1
                if 0 <= idx < len(type_codes):
                    selected_types.append(type_codes[idx])
            except ValueError:
                continue

        if not selected_types:
            selected_types = ["llm"]

        self.config['supported_types'] = selected_types
        print(f"\n已选择任务类型: {', '.join(selected_types)}")

    def _configure_load_control(self):
        """配置负载控制"""
        console.rule("步骤 5/6: 负载控制")

        print("\n配置 Worker 的负载控制参数")
        print("这些设置决定了您的 GPU 如何参与任务处理\n")

        if RICH_AVAILABLE:
            acceptance_rate = FloatPrompt.ask(
                "任务接受率 (0.1-1.0, 1.0=接受全部任务)",
                default=1.0
            )

            max_jobs_per_hour = IntPrompt.ask(
                "每小时最大任务数 (0=不限制)",
                default=0
            )

            # 工作时间配置
            set_working_hours = Confirm.ask(
                "是否设置工作时间段（只在特定时间接受任务）",
                default=False
            )
        else:
            acceptance_rate = float(input("任务接受率 (0.1-1.0) [1.0]: ") or "1.0")
            max_jobs_per_hour = int(input("每小时最大任务数 (0=不限制) [0]: ") or "0")
            set_working_hours = input("是否设置工作时间段 [y/N]: ").lower() == 'y'

        self.config['load_control'] = {
            'acceptance_rate': min(1.0, max(0.1, acceptance_rate)),
            'max_jobs_per_hour': max(0, max_jobs_per_hour),
            'max_concurrent_jobs': 1,
            'cooldown_seconds': 0
        }

        if set_working_hours:
            if RICH_AVAILABLE:
                start_hour = IntPrompt.ask("开始时间（24小时制，如 9 表示 9:00）", default=9)
                end_hour = IntPrompt.ask("结束时间（24小时制，如 22 表示 22:00）", default=22)
            else:
                start_hour = int(input("开始时间（24小时制）[9]: ") or "9")
                end_hour = int(input("结束时间（24小时制）[22]: ") or "22")

            self.config['load_control']['working_hours_start'] = start_hour
            self.config['load_control']['working_hours_end'] = end_hour

    def _configure_direct_connection(self):
        """配置直连"""
        console.rule("步骤 6/6: 直连配置")

        print("\n直连模式允许客户端直接与您的 Worker 通信")
        print("这可以降低延迟，但需要您的机器有公网可访问的地址\n")

        if RICH_AVAILABLE:
            enable_direct = Confirm.ask(
                "是否启用直连模式",
                default=False
            )
        else:
            enable_direct = input("是否启用直连模式 [y/N]: ").lower() == 'y'

        self.config['direct'] = {
            'enabled': enable_direct,
            'host': '0.0.0.0',
            'port': 8080,
            'public_url': None
        }

        if enable_direct:
            if RICH_AVAILABLE:
                port = IntPrompt.ask("直连端口", default=8080)
                public_url = Prompt.ask(
                    "公网访问地址（如 http://your-ip:8080，留空自动检测）",
                    default=""
                )
            else:
                port = int(input("直连端口 [8080]: ") or "8080")
                public_url = input("公网访问地址（留空自动检测）: ").strip()

            self.config['direct']['port'] = port
            if public_url:
                self.config['direct']['public_url'] = public_url

    def _confirm_config(self):
        """确认配置"""
        console.rule("配置确认")

        print("\n您的配置如下:\n")

        if RICH_AVAILABLE:
            table = Table(show_header=True)
            table.add_column("配置项", style="cyan")
            table.add_column("值", style="green")

            table.add_row("服务器", self.config['server']['url'])
            table.add_row("区域", f"{self.config['region']} ({REGIONS[self.config['region']]})")
            table.add_row("GPU", self.config['gpu']['model'])
            table.add_row("任务类型", ", ".join(self.config['supported_types']))
            table.add_row("任务接受率", f"{self.config['load_control']['acceptance_rate']*100:.0f}%")
            table.add_row("直连模式", "启用" if self.config['direct']['enabled'] else "禁用")

            console.print(table)

            confirm = Confirm.ask("\n确认保存配置", default=True)
        else:
            print(f"  服务器: {self.config['server']['url']}")
            print(f"  区域: {self.config['region']}")
            print(f"  GPU: {self.config['gpu']['model']}")
            print(f"  任务类型: {', '.join(self.config['supported_types'])}")
            print(f"  任务接受率: {self.config['load_control']['acceptance_rate']*100:.0f}%")
            print(f"  直连模式: {'启用' if self.config['direct']['enabled'] else '禁用'}")

            confirm = input("\n确认保存配置 [Y/n]: ").lower() != 'n'

        if confirm:
            save_config(self.config)
            print(f"\n配置已保存到 {CONFIG_FILE}")
        else:
            print("\n配置已取消")
            sys.exit(0)


# ==================== CLI 命令 ====================

def cmd_install(args):
    """安装依赖"""
    console.rule("安装依赖")

    print("\n正在检查依赖...")
    deps = check_dependencies()

    missing = [k for k, v in deps.items() if not v and k != 'cuda']

    if not missing:
        print("所有依赖已安装!")
        return

    print(f"缺少依赖: {', '.join(missing)}")

    if RICH_AVAILABLE:
        if not Confirm.ask("是否安装缺少的依赖"):
            return
    else:
        if input("是否安装缺少的依赖 [Y/n]: ").lower() == 'n':
            return

    print("\n开始安装...")

    try:
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Installing...", total=None)

                def update_progress(msg):
                    progress.update(task, description=msg)

                install_dependencies(update_progress)
        else:
            install_dependencies(lambda msg: print(f"  {msg}"))

        print("\n依赖安装完成!")

    except Exception as e:
        print(f"\n安装失败: {e}")
        sys.exit(1)


def cmd_configure(args):
    """交互式配置"""
    wizard = ConfigWizard()
    wizard.run()


def cmd_start(args):
    """启动 Worker"""
    config = load_config()

    if not config:
        print("未找到配置文件，请先运行 'gpu-worker configure'")
        sys.exit(1)

    console.rule("启动 Worker")

    print(f"\n服务器: {config['server']['url']}")
    print(f"区域: {config['region']}")
    print(f"任务类型: {', '.join(config['supported_types'])}")
    print()

    # 导入并启动 Worker
    try:
        from main import Worker
        from config import WorkerConfig

        # 转换配置格式
        worker_config = WorkerConfig(**config)
        worker = Worker(worker_config)
        worker.start()

    except ImportError as e:
        print(f"导入失败: {e}")
        print("请确保已安装所有依赖: gpu-worker install")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nWorker 已停止")


def cmd_status(args):
    """查看状态"""
    config = load_config()

    if not config:
        print("未找到配置文件")
        return

    console.rule("Worker 状态")

    # 检查依赖
    deps = check_dependencies()

    if RICH_AVAILABLE:
        table = Table(title="系统状态")
        table.add_column("项目", style="cyan")
        table.add_column("状态", style="green")

        table.add_row("Python", "OK" if deps["python"] else "需要 3.9+")
        table.add_row("PyTorch", "OK" if deps["torch"] else "未安装")
        table.add_row("CUDA", "可用" if deps["cuda"] else "不可用")
        table.add_row("Transformers", "OK" if deps["transformers"] else "未安装")

        console.print(table)
    else:
        print(f"\nPython: {'OK' if deps['python'] else '需要 3.9+'}")
        print(f"PyTorch: {'OK' if deps['torch'] else '未安装'}")
        print(f"CUDA: {'可用' if deps['cuda'] else '不可用'}")
        print(f"Transformers: {'OK' if deps['transformers'] else '未安装'}")

    # GPU信息
    gpu_info = check_gpu()
    if gpu_info['available']:
        print(f"\nGPU: {gpu_info['model']}")
        print(f"显存: {gpu_info['memory_gb']} GB")
    elif gpu_info.get("nvidia_detected"):
        nvidia_memory_gb = gpu_info.get("nvidia_memory_gb")
        print(f"\nGPU: {gpu_info.get('nvidia_model', 'Unknown')} (驱动可用，PyTorch CUDA 不可用)")
        if nvidia_memory_gb:
            print(f"显存: {nvidia_memory_gb} GB")
        print("建议安装 CUDA 版 PyTorch（例如：python -m pip install torch --index-url https://download.pytorch.org/whl/cu121）")
    else:
        print("\nGPU: 未检测到")

    # 配置信息
    print(f"\n配置:")
    print(f"  服务器: {config.get('server', {}).get('url', '未配置')}")
    print(f"  区域: {config.get('region', '未配置')}")
    print(f"  任务类型: {', '.join(config.get('supported_types', []))}")


def cmd_config_set(args):
    """设置单个配置项"""
    config = load_config() or {}

    key = args.key
    value = args.value

    # 解析键路径 (如 load_control.acceptance_rate)
    keys = key.split('.')
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # 尝试解析值类型
    try:
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # 保持字符串

    current[keys[-1]] = value
    save_config(config)

    print(f"已设置 {key} = {value}")


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="分布式GPU推理 Worker CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  gpu-worker install        安装依赖
  gpu-worker configure      交互式配置
  gpu-worker start          启动 Worker
  gpu-worker status         查看状态
  gpu-worker set load_control.acceptance_rate 0.5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # install 命令
    subparsers.add_parser('install', help='安装依赖')

    # configure 命令
    subparsers.add_parser('configure', help='交互式配置向导')

    # start 命令
    subparsers.add_parser('start', help='启动 Worker')

    # status 命令
    subparsers.add_parser('status', help='查看状态')

    # set 命令
    set_parser = subparsers.add_parser('set', help='设置配置项')
    set_parser.add_argument('key', help='配置键 (如 load_control.acceptance_rate)')
    set_parser.add_argument('value', help='配置值')

    args = parser.parse_args()

    if args.command == 'install':
        cmd_install(args)
    elif args.command == 'configure':
        cmd_configure(args)
    elif args.command == 'start':
        cmd_start(args)
    elif args.command == 'status':
        cmd_status(args)
    elif args.command == 'set':
        cmd_config_set(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
