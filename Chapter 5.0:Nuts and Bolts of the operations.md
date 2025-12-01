这一部分是将理论转化为代码的实操手册，涵盖了你在编写任何 CUDA 程序时都会反复使用的基本语法、内存管理和索引计算逻辑。你可以将以下内容直接追加到你刚才笔记的末尾，作为一个 **“速查手册 (Cheat Sheet)”**。

---

# 🛠️ 实操手册：CUDA 编程核心 (The Nuts and Bolts)

这一部分汇总了 CUDA 编程中出现频率最高的语法、API 和概念。如果说前面的层级结构是“建筑蓝图”，这里就是真正用来盖房子的“砖块和水泥”。

### ⚙️ 核心执行单元 (Execution Units)

*   **Kernel (内核)**
    *   **定义**: 运行在 GPU 上的特殊函数。
    *   **语法**: 使用 `__global__` 关键字标记。
    *   **限制**: 返回值必须是 `void`。
    *   **类比**: 给一大群工人（GPU 线程）下达的统一指令书。
*   **Grid (网格)**
    *   **定义**: 一次 Kernel 启动所产生的所有线程的集合。它是 Block 的集合。
    *   **维度**: 可以是 1D、2D 或 3D。
    *   **用途**: 对应整个计算任务（例如处理整张 4K 图片）。
*   **Block (线程块)**
    *   **定义**: Grid 中的一个子集，包含一组可以相互协作的线程。
    *   **特性**: 块内的线程可以通过 **共享内存 (Shared Memory)** 快速通信并进行同步。
    *   **用途**: 对应任务的一个局部区域（例如图片的一个 16x16 切片）。
*   **Thread (线程)**
    *   **定义**: 最小执行单位。
    *   **特性**: 每个线程拥有唯一的 ID，根据 ID 决定处理数据的哪一部分。

### 🧵 线程索引指南 (Indexing Guide)

如何在数以万计的线程中找到“我是谁”？我们需要组合以下四个内置变量（均为包含 `.x`, `.y`, `.z` 的结构体）：

1.  `threadIdx`: 线程在 **Block 内** 的位置（例如：我是班级里的第几号学生）。
2.  `blockDim`: 一个 **Block** 的大小（例如：一个班级有多少人）。
3.  `blockIdx`: Block 在 **Grid** 中的位置（例如：我是第几班）。
4.  `gridDim`: 整个 **Grid** 的大小（例如：年级里总共有多少个班）。

#### ⚡️ 核心公式：计算全局 ID

这是 CUDA 编程中最通用的公式，用于将层级化的线程映射到线性的数组内存上：

```cpp
// 全局索引 = (当前块的索引 * 块的大小) + 当前线程在块内的索引
int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
```

### 🧱 关键语法与辅助类型

*   **`dim3` 类型**
    *   一个简单的结构体，用于定义 3D 尺寸（x, y, z）。
    *   缺省维度默认为 1。
    *   *示例*: `dim3 blockSize(16, 16, 1);`
*   **`<<< >>>` 启动符**
    *   CUDA 专有的内核启动语法。
    *   **格式**: `KernelName<<<GridDim, BlockDim>>>(args...);`
    *   它告诉 GPU：“启动这个函数，用这种网格结构和这种块结构”。

### 💾 内存管理 API (Host Side)

这些函数在 CPU 代码中调用，用于管理 GPU 上的内存：

1.  **`cudaMalloc`**
    *   **作用**: 在显存（Global Memory）上申请空间。
    *   *对应 C 语言*: `malloc`
2.  **`cudaMemcpy`**
    *   **作用**: 数据搬运工。
    *   **方向标志 (Critical)**:
        *   `cudaMemcpyHostToDevice` (CPU -> GPU)
        *   `cudaMemcpyDeviceToHost` (GPU -> CPU)
    *   *注意*: 这是同步操作，也是主要的性能瓶颈点。
3.  **`cudaFree`**
    *   **作用**: 释放显存。**切记使用，防止显存泄漏。**
4.  **`cudaDeviceSynchronize()`**
    *   **作用**: CPU 等待 GPU 完成所有任务。
    *   **场景**: 因为 Kernel 启动是异步的（CPU 发完指令就接着往下跑了），如果你需要立即读取结果或测量运行时间，必须调用此函数进行“同步”。

### 📊 GPU 内存层级 (Memory Hierarchy)

并非所有显存都是生而平等的，理解它们的区别是性能优化的关键：

| 内存类型 | 可见范围 | 速度 | 用途 |
| :--- | :--- | :--- | :--- |
| **Registers** (寄存器) | 单个 Thread | **最快** | 局部变量，循环计数器。 |
| **Shared Memory** (共享内存) | 单个 Block | **极快** | 块内线程协作数据，手动管理的缓存。 |
| **Global Memory** (全局内存) | 所有 Threads | **慢** (但最大) | 存放大型数组、输入输出数据。 |
| **Constant Memory** (常量内存) | 所有 Threads | 快 (有缓存) | 只读参数，配置常量。 |
| **Local Memory** (本地内存) | 单个 Thread | 慢 | 寄存器不够用时的溢出空间 (Spilling)，应尽量避免。 |

### 📝 代码模板：向量加法

将以上所有概念串联起来的标准模板：

```cpp
// 1. Kernel 定义 (GPU 代码)
__global__ void addArrays(int *a, int *b, int *c, int size) {
    // 计算全局唯一索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 边界检查 (防止线程数多于数据量时越界)
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

// 2. Main 函数 (Host 代码)
int main() {
    // ... (假设 d_a, d_b, d_c 已分配并拷贝数据)
    
    int size = 1000;
    // 定义执行配置
    dim3 blockSize(256); // 每个 Block 256 个线程
    // 向上取整计算所需的 Block 数量，确保覆盖所有数据
    dim3 gridSize((size + blockSize.x - 1) / blockSize.x); 
    
    // 启动 Kernel
    addArrays<<<gridSize, blockSize>>>(d_a, d_b, d_c, size);
    
    // 等待 GPU 完成
    cudaDeviceSynchronize();
    
    // ... (拷回数据 & 清理内存)
}
```
