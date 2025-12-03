# 📊 Chapter 5.3: CUDA 内核性能分析 (Profiling)
> **⏱️ 视频时间**：3:13:59 - 3:27:52  
> **💡 核心概要**：盲目优化是万恶之源。本节介绍了现代 CUDA 标准工具链 (**Nsight Systems** & **Nsight Compute**)，演示了如何通过 **NVTX** 标记代码，并建立科学的性能分析工作流，验证优化策略（如 Tiling）的有效性。

---

## 1. ❓ 为什么需要 Profiling？
在动手改代码前，必须知道瓶颈在哪里。

*   **拒绝猜测**：
    *   是卡在 **显存带宽 (Memory Bound)** 还是 **计算能力 (Compute Bound)**？
    *   是 CPU 调度太慢，还是 GPU 执行太慢？
*   **工具演变**：
    *   ❌ `nvprof` / `nvvp`：老旧工具，已被弃用。
    *   ✅ **Nsight Systems (`nsys`)** / **Nsight Compute (`ncu`)**：现代标准。

---

## 2. 🛠️ 核心工具箱 (The Toolset)

### 2.1 🏷️ NVTX (NVIDIA Tools Extension)
**“给代码打标签”**。它允许你在 C++ 代码中手动标记时间范围，让 Timeline 变得可读。

*   **作用**：将晦涩的时间轴翻译成人话。清晰区分哪里是 "Malloc"，哪里是 "MatMul"。
*   **代码示例**：
    ```cpp
    #include <nvtx3/nvToolsExt.h> 
    
    // 开始一个范围
    nvtxRangePush("Matrix Multiplication");  
    
    // ... 执行你的 Kernel ... 
    
    // 结束范围 
    nvtxRangePop();
    ```
*   **编译注意**：必须链接库 `nvcc -o main main.cu -lnvToolsExt`

### 2.2 🌍 Nsight Systems (`nsys`) —— 宏观视角
**“系统级分析器”**。这是性能分析的第一步。

*   **关注点**：**全局时间轴**。
    *   CPU 与 GPU 的交互。
    *   CUDA API 调用耗时。
    *   内存拷贝 (H2D, D2H) 与 Kernel 执行的重叠情况。
*   **常用命令**：
    ```bash
    # 生成报告并打印统计摘要
    nsys profile --stats=true ./your_program
    ```
*   **输出**：`.nsys-rep` (导入 GUI 查看瀑布图) / `.sqlite` (数据库)。

### 2.3 🔬 Nsight Compute (`ncu`) —— 微观视角
**“内核级分析器”**。当你用 `nsys` 抓到最慢的那个 Kernel 后，用 `ncu` 显微镜式地观察它。

*   **关注点**：**单个 Kernel 的硬件利用率**。
*   **核心指标**：**Speed of Light (SOL)**。
    *   **Memory % vs Compute %**：判断瓶颈的金标准。
    *   **高 Memory / 低 Compute** ➡️ **Memory Bound** (需优化数据搬运，如 Tiling)。
    *   **低 Memory / 高 Compute** ➡️ **Compute Bound** (需优化计算指令)。
*   **常用命令**：
    ```bash
    # 针对特定 Kernel 进行分析
    ncu --kernel-name myKernel ./your_program
    ```
> **⚠️ 常见坑 (Linux 权限问题)**：
> 使用 `ncu` 可能会报错。解决方案是修改 `/etc/modprobe.d/nvidia.conf`，添加：
> `options nvidia NVreg_RestrictProfilingToAdminUsers=0`
> 然后重启机器。

---

## 3. 🔄 实战：性能分析工作流 (The Workflow)
通过对比 **Naive MatMul** (朴素版) 和 **Tiled MatMul** (优化版) 展示了标准流程：

### Step 1: 代码插桩与编译
加上 NVTX 标记并编译。
```bash
nvcc -o 00_nvtx 00_nvtx_matmul.cu -lnvToolsExt
nvcc -o 01_naive 01_naive_matmul.cu
nvcc -o 02_tiled 02_tiled_matmul.cu
```

### Step 2: 使用 `nsys` 找瓶颈
```bash
nsys profile --stats=true ./01_naive
nsys profile --stats=true ./02_tiled
```
*   **观察**：终端输出的统计中，`02_tiled` 的 Kernel 耗时显著短于 `01_naive`。

### Step 3: 使用 GUI 深入诊断
打开 GUI 加载报告：
*   **Naive 版**：Global Memory 吞吐低，带宽利用率差。大量时间浪费在从 DRAM 捞数据。
*   **Tiled 版**：Shared Memory 使用率飙升，Global Memory 读取效率提高（合并访问生效），整体吞吐量提升。

---

## 4. 🧰 辅助工具 (CLI Tools)

除了 Nsight，这些轻量级工具也是日常必备：

*   **`nvidia-smi`**：最基础监控。`watch -n 0.1 nvidia-smi` 实时看显存和利用率。
*   **`nvitop`**：`nvidia-smi` 的 Python 增强版。界面丰富，支持交互，强烈推荐。
*   **`compute-sanitizer`**：**Debug 神器** (CUDA 版 Valgrind)。
    *   命令：`compute-sanitizer ./main`
    *   作用：检测 **Memory Leaks** (显存泄漏)、**Out of bounds** (越界访问)、未对齐访问。

---

## 5. 🧑‍🔬 总结：如何科学地优化 CUDA 代码？

不要瞎猜，请遵循 **“科学优化五步法”**：

1.  **Baseline**：先写一个能跑通的 Naive 版本。
2.  **Instrument**：加上 NVTX 标记。
3.  **Identify (nsys)**：
    *   找出系统瓶颈。
    *   是卡在 `cudaMemcpy` (数据传输) 还是 Kernel (计算)？
4.  **Diagnose (ncu)**：
    *   显微镜观察那个最慢的 Kernel。
    *   **Memory Bound?** ➡️ 方案：Shared Memory Tiling, Memory Coalescing, Vectorized Loads (`float4`).
    *   **Compute Bound?** ➡️ 方案：Tensor Cores, Fast Math, Loop Unrolling.
5.  **Optimize & Repeat**：修改代码，重新 Profile，验证 SOL 指标是否提升。

> **🌟 核心格言**
> "Unless you have a specific profiling goal, **start with Nsight Systems** to determine system bottlenecks. Then use **Nsight Compute** to optimize specific kernels."
> (除非目标明确，否则永远先从 nsys 找系统瓶颈，再用 ncu 优化特定内核。)
