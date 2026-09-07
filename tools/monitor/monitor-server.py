#!/usr/bin/env python3
"""
资源压力测试程序
控制 CPU、内存和 GPU 的占用率
"""

import os
import sys
import time
import threading
import multiprocessing as mp
import numpy as np
import psutil
from typing import List

# 全局控制标志
stop_flag = threading.Event()


class CPUStressor:
    """CPU 压力测试器 - 目标 70~80%"""

    def __init__(self, target_usage_range=(70, 80)):
        self.target_min, self.target_max = target_usage_range
        self.num_cores = mp.cpu_count()
        self.threads: List[threading.Thread] = []
        self.work_ratio = 0.75  # 初始工作比率

    def _cpu_work(self, thread_id: int):
        """单个线程的 CPU 工作负载"""
        while not stop_flag.is_set():
            # 执行计算密集型任务
            start = time.time()
            while time.time() - start < self.work_ratio:
                # 更密集的计算任务
                _ = sum(i * i for i in range(100000))
                _ = [x**2 for x in range(1000)]
            # 休息一段时间
            if self.work_ratio < 1.0:
                time.sleep(1 - self.work_ratio)

    def start(self):
        """启动 CPU 压力测试"""
        print(f"启动 CPU 压力测试，目标: {self.target_min}~{self.target_max}%")
        print(f"CPU 核心数: {self.num_cores}")

        # 为每个核心创建一个线程
        for i in range(self.num_cores):
            t = threading.Thread(target=self._cpu_work, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

        # 动态调整线程
        threading.Thread(target=self._adjust_cpu_usage, daemon=True).start()

    def _adjust_cpu_usage(self):
        """动态调整 CPU 使用率"""
        time.sleep(2)  # 等待稳定
        while not stop_flag.is_set():
            cpu_percent = psutil.cpu_percent(interval=1)

            if cpu_percent < self.target_min:
                self.work_ratio = min(0.95, self.work_ratio + 0.02)
            elif cpu_percent > self.target_max:
                self.work_ratio = max(0.5, self.work_ratio - 0.02)

            time.sleep(2)


class MemoryStressor:
    """内存压力测试器 - 目标 60~80%"""

    def __init__(self, target_usage_range=(60, 80)):
        self.target_min, self.target_max = target_usage_range
        self.memory_blocks: List[np.ndarray] = []
        self.total_memory = psutil.virtual_memory().total

    def start(self):
        """启动内存压力测试"""
        print(f"启动内存压力测试，目标: {self.target_min}~{self.target_max}%")
        print(f"总内存: {self.total_memory / (1024**3):.2f} GB")

        threading.Thread(target=self._allocate_memory, daemon=True).start()

    def _allocate_memory(self):
        """动态分配内存"""
        block_size = 500 * 1024 * 1024  # 每次分配 500 MB

        # 快速分配到目标范围
        mem = psutil.virtual_memory()
        current_usage = mem.percent
        target_bytes = int(self.total_memory * self.target_min / 100)
        used_bytes = mem.used

        if current_usage < self.target_min:
            print(f"快速分配内存到目标范围...")
            try:
                # 计算需要分配多少
                need_bytes = target_bytes - used_bytes
                num_blocks = max(1, need_bytes // block_size)

                for i in range(int(num_blocks)):
                    if stop_flag.is_set():
                        break
                    arr = np.random.rand(block_size // 8)  # 8 bytes per float64
                    arr[0] = 1.0  # 确保实际分配
                    self.memory_blocks.append(arr)
                    mem = psutil.virtual_memory()
                    print(f"内存占用: {mem.percent:.1f}% (已分配 {(i+1)*500} MB)")
                    if mem.percent >= self.target_min:
                        break
            except MemoryError:
                print("内存分配达到系统限制")

        # 动态调整维持在目标范围
        while not stop_flag.is_set():
            mem = psutil.virtual_memory()
            current_usage = mem.percent

            if current_usage < self.target_min - 5:
                # 分配更多内存
                try:
                    arr = np.random.rand(block_size // 8)
                    arr[0] = 1.0
                    self.memory_blocks.append(arr)
                    print(f"内存占用: {current_usage:.1f}% -> 分配了 500 MB")
                except MemoryError:
                    print("内存分配失败")
            elif current_usage > self.target_max:
                # 释放一些内存
                if self.memory_blocks:
                    self.memory_blocks.pop()
                    print(f"内存占用: {current_usage:.1f}% -> 释放了 500 MB")
            else:
                print(f"内存占用: {current_usage:.1f}% (目标范围内)")

            time.sleep(3)


class GPUStressor:
    """GPU 压力测试器 - 显存 85%，GPU 算力 20~90%"""

    def __init__(self, vram_target=85, compute_range=(20, 90)):
        self.vram_target = vram_target
        self.compute_min, self.compute_max = compute_range
        self.gpu_available = False

        try:
            import torch
            self.torch = torch
            if torch.cuda.is_available():
                self.gpu_available = True
                self.device = torch.device('cuda:0')
                print(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("警告: 未检测到 CUDA GPU")
        except ImportError:
            print("警告: PyTorch 未安装，无法进行 GPU 压力测试")
            print("安装命令: pip install torch")

    def start(self):
        """启动 GPU 压力测试"""
        if not self.gpu_available:
            return

        print(f"启动 GPU 压力测试")
        print(f"显存目标: {self.vram_target}%")
        print(f"GPU 算力目标: {self.compute_min}~{self.compute_max}%")

        threading.Thread(target=self._allocate_vram, daemon=True).start()
        threading.Thread(target=self._gpu_compute_work, daemon=True).start()

    def _allocate_vram(self):
        """分配显存到目标值"""
        if not self.gpu_available:
            return

        time.sleep(1)
        total_vram = self.torch.cuda.get_device_properties(0).total_memory
        target_vram = int(total_vram * self.vram_target / 100)

        print(f"总显存: {total_vram / (1024**3):.2f} GB")
        print(f"目标显存占用: {target_vram / (1024**3):.2f} GB")

        vram_blocks = []
        allocated = 0

        # 分配显存
        try:
            while allocated < target_vram and not stop_flag.is_set():
                # 每次分配 100 MB
                block_size = min(100 * 1024 * 1024, target_vram - allocated)
                tensor = self.torch.randn(block_size // 4, device=self.device)  # 4 bytes per float32
                vram_blocks.append(tensor)
                allocated += block_size

                current_vram = self.torch.cuda.memory_allocated(0)
                vram_percent = (current_vram / total_vram) * 100
                print(f"显存占用: {vram_percent:.1f}% ({current_vram / (1024**3):.2f} GB)")

                time.sleep(0.5)
        except RuntimeError as e:
            print(f"显存分配完成或达到限制: {e}")

        # 保持显存占用
        while not stop_flag.is_set():
            time.sleep(5)

    def _gpu_compute_work(self):
        """GPU 计算工作负载"""
        if not self.gpu_available:
            return

        time.sleep(3)  # 等待显存分配完成

        work_intensity = 0.5  # 工作强度

        while not stop_flag.is_set():
            try:
                # 执行矩阵运算
                size = int(2000 * work_intensity)
                if size > 0:
                    start = time.time()
                    a = self.torch.randn(size, size, device=self.device)
                    b = self.torch.randn(size, size, device=self.device)
                    c = self.torch.matmul(a, b)
                    self.torch.cuda.synchronize()
                    elapsed = time.time() - start

                    # 根据计算时间调整工作强度
                    if elapsed < 0.1:
                        work_intensity = min(1.5, work_intensity * 1.1)
                    else:
                        work_intensity = max(0.3, work_intensity * 0.9)

                # 调整工作/休息比率
                time.sleep(0.05)

            except RuntimeError as e:
                print(f"GPU 计算错误: {e}")
                time.sleep(1)


def monitor_resources():
    """监控资源使用情况"""
    print("\n" + "=" * 60)
    print("资源监控 (按 Ctrl+C 停止)")
    print("=" * 60)

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False

    while not stop_flag.is_set():
        try:
            # CPU 和内存
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()

            info = f"\r[CPU: {cpu_percent:5.1f}%] [内存: {mem.percent:5.1f}%]"

            # GPU
            if gpu_available:
                try:
                    vram_used = torch.cuda.memory_allocated(0)
                    vram_total = torch.cuda.get_device_properties(0).total_memory
                    vram_percent = (vram_used / vram_total) * 100
                    info += f" [显存: {vram_percent:5.1f}%]"
                except:
                    pass

            print(info, end='', flush=True)
            time.sleep(1)

        except KeyboardInterrupt:
            break


def main():
    print("=" * 60)
    print("资源压力测试程序")
    print("=" * 60)
    print()

    # 创建压力测试器
    cpu_stressor = CPUStressor(target_usage_range=(70, 80))
    mem_stressor = MemoryStressor(target_usage_range=(60, 80))
    gpu_stressor = GPUStressor(vram_target=85, compute_range=(20, 90))

    # 启动压力测试
    cpu_stressor.start()
    time.sleep(1)
    mem_stressor.start()
    time.sleep(1)
    gpu_stressor.start()

    time.sleep(3)

    # 监控资源
    try:
        monitor_resources()
    except KeyboardInterrupt:
        print("\n\n正在停止压力测试...")
        stop_flag.set()
        time.sleep(2)
        print("已停止")


if __name__ == "__main__":
    main()
