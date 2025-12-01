# 📐 Chapter 5: CUDA 编程核心理论基础
> **⏱️ 视频时间**：1:51:40 - 2:31:37  
> **💡 核心概要**：本节解释了 CUDA 编程的“物理定律”。如何将代码从 CPU 迁移到 GPU，如何管理显存，以及 GPU 内部如何通过层级化结构组织成千上万个线程进行并行计算。

---

## 1. 🔍 了解你的硬件：GPU 属性查询
在写代码前，必须了解你手中的“武器”。通常使用 CUDA Samples 中的 `deviceQuery` 工具查看。

*   **Compute Capability (计算能力)**
    *   关键版本号（如 8.0, 8.6, 9.0）。
    *   它决定了硬件支持哪些特性（例如：是否支持 FP8，Tensor Core 的版本，是否支持 Thread Block Clusters）。
*   **显存 (VRAM) 与 带宽**
    *   决定了能跑多大的模型（Batch Size 能开多大）。
*   **L2 Cache 大小**
    *   影响数据复用效率，对性能至关重要。

---

## 2. 🏗️ 编程模型：Host 与 Device
CUDA 采用 **异构计算 (Heterogeneous Computing)** 模型。

### 2.1 角色分工
| 角色 | 硬件位置 | 职责 | 内存 |
| :--- | :--- | :--- | :--- |
| **Host (主机)** | **CPU** | 逻辑控制、串行任务、环境设置 | System RAM |
| **Device (设备)** | **GPU** | 高吞吐量的并行计算 | VRAM (Global Memory) |

### 2.2 执行流程 (Runtime Workflow)
一个标准的 CUDA 程序遵循 **“三步走”** 战略：
1.  **Copy Input** ➡️：将数据从 Host 内存复制到 Device 显存。
2.  **Execute Kernel** ⚙️：CPU 通知 GPU 启动 Kernel，GPU 利用显存数据计算。
3.  **Copy Result** ⬅️：将结果从 Device 显存拷回 Host 内存。

---

## 3. 📝 语法规范：限定符 (Qualifiers)
CUDA 通过特殊的函数限定符来区分代码运行的位置。

| 限定符 | 定义 | 谁调用? | 哪运行? | 特点 |
| :--- | :--- | :--- | :--- | :--- |
| `__global__` | **Kernel** | Host (CPU) | **Device (GPU)** | 必须返回 `void`。它的任务是**副作用**（修改显存中的数据）。 |
| `__device__` | **Helper** | Device (Kernel) | **Device (GPU)** | GPU 内部调用的辅助函数。相当于 `main` 里的子函数。 |
| `__host__` | **Function** | Host | Host | 普通的 C++ 函数。 |

> **📏 变量命名习惯**：
> *   `h_Variable`：表示 Host 端变量（CPU 内存）。
> *   `d_Variable`：表示 Device 端变量（GPU 显存）。
> *   *防止写代码时把 CPU 指针传给 GPU 函数，反之亦然。*

---

## 4. 💾 内存管理 (Memory Management)
C 语言的 `malloc` 无法触及显卡显存。CUDA 有一套专属 API。

*   **分配显存**：`cudaMalloc`
    ```cpp
    float *d_a;
    // 在 GPU Global Memory 上分配 N 个浮点数
    cudaMalloc(&d_a, N * sizeof(float));
    ```
*   **搬运数据**：`cudaMemcpy`
    *   方向参数至关重要（枚举值）：
        *   `cudaMemcpyHostToDevice` (H2D) ➡️
        *   `cudaMemcpyDeviceToHost` (D2H) ⬅️
        *   `cudaMemcpyDeviceToDevice` (D2D) 🔄
*   **释放显存**：`cudaFree`
    *   `cudaFree(d_a);`
    *   ⚠️ **切记**：显存极其宝贵，用完必须释放（避免 Memory Leak）。

---

## 5. 🧱 CUDA 线程层级结构 (The Hierarchy)
> **⚠️ 全课最核心概念**：为了管理数百万个线程，CUDA 采用了三层逻辑结构。

### 5.1 核心层级图解

1.  **Grid (网格)**
    *   代表整个 Kernel 启动涉及的所有线程。
    *   对应硬件资源：**Global Memory** (全局显存)。
2.  **Block (线程块)**
    *   Grid 的组成单元。Grid 包含多个 Block。
    *   对应硬件资源：**SM (Streaming Multiprocessor)**。
    *   **Shared Memory**：Block 内的线程可以共享一块极快的高速缓存（L1 速度）。
    *   **同步**：Block 内线程可用 `__syncthreads()` 同步；**不同 Block 间无法同步**。
3.  **Thread (线程)**
    *   最小执行单元。
    *   对应硬件资源：**CUDA Core**。
    *   拥有私有的 **Registers (寄存器)**。

### 5.2 形象比喻：大楼模型 🏢
*   **Grid** = 一栋大楼。
*   **Block** = 大楼里的房间。
*   **Thread** = 房间里的人。

### 5.3 索引计算 (Indexing)
如何在数百万个线程中知道“我是谁”？我们需要根据坐标计算全局唯一 ID。
*   **内置变量**：
    *   `gridDim`：大楼有多少房间。
    *   `blockDim`：房间能装多少人。
    *   `blockIdx`：当前房间号。
    *   `threadIdx`：当前房间内的排号。

> **🧮 1D 映射公式 (The Magic Formula)**
> 要找到某个人的全局 ID：
> `Global ID` = `(房间号 × 房间容量)` + `房间内排号`
> ```cpp
> int i = blockIdx.x * blockDim.x + threadIdx.x;
> ```

---

## 6. ⚙️ 硬件执行单元：Warp (线程束)
Block 和 Grid 是软件层面的抽象。**Warp 是硬件真正执行调度的单位**。

*   **定义**：一个 Warp 包含 **32 个线程**。
*   **SIMT (Single Instruction, Multiple Threads)**：
    *   Warp 内的 32 个线程在同一时刻执行**完全相同**的指令。
*   **延迟掩盖 (Latency Hiding)**：
    *   **为什么 GPU 快？**
    *   每个 SM 上有多个 Warp 调度器。
    *   当 Warp A 需要去显存取数据（这很慢），调度器会**立即**切换到 Warp B 进行计算。
    *   这种上下文切换（Context Switch）几乎是**零开销**的（因为寄存器已经分配好了）。
    *   *“用计算来掩盖延迟”*。

---

## 总结：GPU 高性能的秘密 🗝️

1.  **海量线程**：通过 `Grid` / `Block` 结构，轻松启动数百万线程。
2.  **延迟掩盖**：利用 `Warp` 极速切换，只要有足够的线程，计算单元就永远不会闲着。
3.  **层次化内存**：
    *   Registers (最快, 私有)
    *   Shared Memory (极快, 块内共享)
    *   Global Memory (慢, 全局共享)
    *   **优化的核心就是把数据尽可能留在 Registers 和 Shared Memory 中。**
