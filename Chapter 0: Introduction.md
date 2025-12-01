
# 📘 GPU 编程课程笔记：基石与动机
> **⏱️ 视频时间**：00:00 - 16:54  
> **💡 核心概要**：这一部分是整个课程的基石。它不仅明确了课程内容，更深刻阐述了在 AI 时代掌握底层 GPU 编程技术的战略价值。

---

## 🛠️ 学习建议与预备知识 (Prerequisites)

在开始这段旅程前，请做好以下准备：

*   **✅ 必须具备**：
    *   Python 基础。
    *   对深度学习基本概念（神经网络、矩阵乘法）的直观理解。
*   **💡 建议具备**：
    *   C/C++ 基础。*（注：如果缺乏指针和内存管理概念，学习曲线会非常陡峭）*
*   **🧠 核心心态**：
    *   做好面对**陡峭学习曲线**的准备。
    *   这不仅是学语法，更是在学习 **计算机体系结构 (Computer Architecture)**。

---

## 1. 核心论题：为什么你需要学习 CUDA？(The Motivation)

### 1.1 🛡️ 跨越“护城河” (The Moat)
在当前的 AI 浪潮中，掌握 CUDA 意味着拥有一种稀缺的竞争优势。

*   **稀缺性**：会写 PyTorch 的人很多，但能深入底层、编写比默认实现快 **2倍、10倍甚至100倍** Kernel 的人极少。
*   **工程本质**：ML Engineering 不仅仅是画架构图，本质上是 **数据移动 (Data Movement)** 的艺术。
*   **核心技能**：CUDA 是一种思维方式——如何将数据搬运到芯片 -> 并行计算 -> 搬运出去。这是职业生涯的护城河。

### 1.2 📈 摩尔定律终结 vs. 黄氏定律崛起
*   **CPU 的瓶颈 (Moore's Law)**：单核性能提升停滞，无法再依赖 CPU 升级自动获得加速。
*   **GPU 的爆发 (Huang's Law)**：过去十年，GPU 算力呈指数级增长。
*   **范式转移**：程序员必须从 **串行编程 (Sequential)** 转向 **大规模并行编程 (Massively Parallel)**。

---

## 2. 🧱 深度学习的物理瓶颈：内存墙 (The Memory Wall)
> **⚠️ 关键概念**：理解“内存墙”是进行所有 CUDA 优化的前提。

### 2.1 核心指标对比
*   **Compute (FLOPS)**：数学运算能力（加法、乘法）。
*   **Memory Bandwidth (GB/s)**：搬运数据（HBM/VRAM ↔ 计算单元）的能力。

### 2.2 瓶颈的形成：吸管与杯子 (The Straw and The Cup)
过去 20 年，计算能力提升了数万倍，但带宽仅提升了数百倍。

> **🧋 形象比喻**：
> *   **你 (计算单元)**：极度口渴，喝水速度极快。
> *   **杯子里的水 (数据)**：你想处理的内容。
> *   **吸管 (内存带宽)**：**非常细**。
>
> **结论**：无论你喝水能力多强，最终速度取决于**吸管的粗细**。

*   **现状**：目前的 LLM 和深度学习模型主要是 **Memory Bound**（受限带宽），而非 Compute Bound。
*   **优化的本质**：CUDA 优化往往不是为了“算得更快”，而是为了 **“更聪明地移动数据”**，减少对那根细吸管的依赖。

---

## 3. 🗺️ AI 软件栈全景 (The AI Ecosystem Stack)

我们需要明确自己在技术栈中的坐标：

| 层级 | 工具示例 | 描述 |
| :--- | :--- | :--- |
| **User & Frameworks** | `PyTorch`, `TensorFlow` | **易用性高**。Python 编写，开发快，但在极致性能或非标准算子面前存在局限。 |
| **Compilers** | `ONNX Runtime`, `TensorRT` | **图优化**。将 Python 转换为机器表示，进行算子融合 (Operator Fusion)。 |
| **Kernel Languages** | `CUDA`, `Triton` | **🚩 本课关注点**。连接算法与晶体管的桥梁。需手动管理内存与线程。 |
| **Hardware** | `Transistors`, `SASS` | **执行层**。晶体管、逻辑门、汇编指令。 |

---

## 4. 🎓 课程路线图 (Course Curriculum)

课程将分为四个阶段，层层递进：

### Phase 1: 基础 (Foundations)
*   **C/C++ Refresher**：指针、内存分配、Makefiles。
*   **GPU Architecture**：理解 SM、VRAM、L1/L2 Cache。
*   **First Kernels**：编写向量加法 (Vector Add)，掌握 Grid/Block/Thread 结构。

### Phase 2: 性能与工具 (Profiling & Tools)
*   **Nsight Compute**：学会“看”代码。*If you can't measure it, you can't optimize it.*
*   **Roofline Model**：评估性能的理论模型（判断是卡在算力还是带宽）。

### Phase 3: 核心算法与优化 (Algorithms & Optimization)
*   **GEMM (矩阵乘法)**：深度学习的原子操作。从朴素实现优化到接近 `cuBLAS` 性能。
*   **技术点**：
    *   `Coalescing` (内存合并访问)
    *   `Shared Memory Tiling` (共享内存分块)
    *   `Vectorization` (向量化读写)

### Phase 4: 进阶与实战 (Advanced & Practice)
*   **Triton**：学习 OpenAI 的下一代 GPU 语言。
*   **🏆 Final Project**：**从零构建 MNIST 训练流程**
    *   ❌ 不依赖 PyTorch Autograd。
    *   ✅ 自己写 Python 逻辑 -> C 扩展 -> CUDA Kernel。
    *   ✅ 实现前向/反向传播、权重更新、激活函数。
