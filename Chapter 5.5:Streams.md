# 🌊 Chapter 5.5: CUDA Streams 与并发执行
> **⏱️ 视频时间**：03:37:12 - 03:55:27  
> **💡 核心概要**：如果说 Kernel 优化是让“工人（GPU核心）干活更快”，那么 Streams（流）就是“流水线管理”。目的是让工人永远不要停下来等待材料。本节介绍了如何通过 **多流 (Multi-Streams)** 和 **页锁定内存 (Pinned Memory)** 实现计算与数据传输的 **重叠 (Overlap)**，从而掩盖延迟。

---

## 1. 🧠 核心直觉 (Intuition)

### 1.1 河流与时间线
将 CUDA Stream 想象成一条条向前流动的 **时间河流**。每一条河流代表一个独立的 **命令队列 (Command Queue)**。

### 1.2 串行 vs. 并行
*   **1️⃣ 默认流 (Default Stream / Stream 0)**
    *   **机制**：如果你不指定 Stream，所有操作都在这里顺序执行。
    *   **弊端**：必须等上一步（如内存拷贝）完全结束，GPU 才能开始下一步（计算）。这会导致 GPU 在等待数据传输时处于 **空闲状态 (Idle)**。
*   **3️⃣ 多流 (Multi-Streams)**
    *   **机制**：我们可以创建多个平行的河流。
    *   **优势**：Stream 1 在搬运数据时，Stream 2 可以在进行计算。

### 1.3 终极目标：掩盖延迟 (Latency Hiding)
在深度学习训练（如 LLM）中，我们希望在计算当前 Batch 的同时，就已经在 **预取 (Prefetching)** 下一个 Batch 的数据。
通过重叠计算和传输，隐藏掉 PCIe 总线缓慢的传输延迟。

---

## 2. 📌 关键前置技术：Pinned Memory (页锁定内存)
> **⚠️ 必须掌握**：要实现真正的“异步数据传输”，Pinned Memory 是必须的。

### 2.1 内存类型对比
| 类型 | 英文 | 描述 | GPU 访问能力 |
| :--- | :--- | :--- | :--- |
| **普通内存** | **Pageable Memory** | 操作系统可以随时将其换出 (Swap out) 到磁盘。虚拟地址不稳定。 | ❌ DMA 引擎无法直接读取。需 CPU 先搬运到临时 Buffer。 |
| **页锁定内存** | **Pinned / Locked Memory** | 告诉 OS：“这块内存我锁定了，不许移动，不许换出”。 | ✅ **DMA 直接访问物理内存**。速度更快，支持异步操作。 |

### 2.2 使用方法
*   **API**：使用 `cudaMallocHost` 分配（而非 C 语言的 `malloc`）。
*   **口诀**：*"We're gonna need this for later, so don't play with it."* (OS 别乱动，这块地盘我有大用)。

---

## 3. 🛠️ 基础操作与 API

### 3.1 Kernel 启动参数
我们在之前的章节见过 `<<<...>>>`，现在终于填上了第 4 个参数：

```cpp
// 完整格式: <<<Grid, Block, SharedMemBytes, Stream>>>
// 如果不指定，默认为 0 (同步流)
myKernel<<<grid, block, 0, stream1>>>(args);
```

### 3.2 异步内存拷贝
这是实现重叠的关键函数：

*   **`cudaMemcpy`** (🛑 Blocking)：
    *   CPU 必须等数据拷完才能往下走。
*   **`cudaMemcpyAsync`** (🟢 Non-blocking)：
    *   CPU 发出指令后 **立即返回**。
    *   拷贝任务在后台的 Stream 中排队执行。
    *   **注意：必须配合 Pinned Memory 使用，否则会自动退化为同步拷贝。**

---

## 4. 🚦 高级协调机制 (Coordination)
当我们有多条河流并行奔跑时，如何防止它们乱套？

### 4.1 Events (事件) —— 时间流上的标记点
*   **⏱️ 用途 1：精确计时**
    *   记录 `start` 和 `stop` 事件，计算 GPU 端的精确耗时（消除了 CPU 启动开销）。
*   **🚥 用途 2：流间同步 (Dependency)**
    *   **场景**：Stream B 需要用到 Stream A 的计算结果。
    *   **方法**：
        1.  Stream A 记录事件 (`cudaEventRecord`)。
        2.  Stream B 等待事件 (`cudaStreamWaitEvent`)。
    *   *类比*：Stream B 看到绿灯前会一直等待，但 **不会阻塞 CPU**（CPU 可以去处理 Stream C）。

### 4.2 Callbacks (回调)
*   **机制**：在 Stream 队列中插入一个 CPU 函数。
*   **场景**：通知 CPU “GPU 活干完了”，触发后续逻辑（如加载下一批数据）。

### 4.3 Priorities (优先级)
*   **场景**：关键推理任务用 **High Priority** 抢占资源，后台预加载用 **Low Priority**。

---

## 5. 🏗️ 代码实战结构 (The Staircase Pattern)

一个典型的 **流水线 (Pipeline)** 代码结构如下：

```cpp
// 1. 分配 Pinned Memory (关键!)
float *h_data;
cudaMallocHost((void**)&h_data, size);

// 2. 创建多个 Streams
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 3. 流水线执行 (Staircase Pattern)

// --- Stream 1 ---
// 放入任务队列：拷贝 -> 计算 -> 拷回
cudaMemcpyAsync(d_in1, h_in1, size, H2D, stream1);
myKernel<<<grid, block, 0, stream1>>>(d_in1, d_out1);
cudaMemcpyAsync(h_out1, d_out1, size, D2H, stream1);

// --- Stream 2 (与 Stream 1 重叠) ---
// 当 Stream 1 在做 Kernel 计算时，
// Stream 2 正在利用空闲的 PCIe 带宽将数据搬入 GPU (H2D)
cudaMemcpyAsync(d_in2, h_in2, size, H2D, stream2);
myKernel<<<grid, block, 0, stream2>>>(d_in2, d_out2);
cudaMemcpyAsync(h_out2, d_out2, size, D2H, stream2);

// 4. 同步
cudaDeviceSynchronize(); // CPU 等待所有流完成
```

---

## 6. 🎓 总结：为什么这很重要？

在深度学习的大规模训练中，GPU 的算力非常昂贵。我们不能容忍 GPU 没事干。

*   **🚫 没有 Streams**：GPU 经常在“发呆”，等待 CPU 把数据从硬盘读到内存，再搬到显存。利用率呈锯齿状。
*   **✅ 有了 Streams**：
    *   **Data Loading** 和 **Model Compute** 完美重叠。
    *   GPU 利用率接近 **100% 直线**。
*   **🔥 PyTorch 启示录**：
    *   这就是为什么你在 PyTorch `DataLoader` 中设置 `num_workers > 0` 和 `pin_memory=True` 能显著加速训练的原因。
    *   `pin_memory=True` 正是调用了 `cudaMallocHost`，为异步传输铺平了道路。
