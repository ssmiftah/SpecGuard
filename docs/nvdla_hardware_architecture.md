# NVDLA Hardware Architectural Specification

> Source: [nvdla/doc on GitHub](https://github.com/nvdla/doc/blob/master/doc/hw/v1/hwarch.rst)  
> License: [NVIDIA Open NVDLA License v1.0](https://nvdla.org/license.html)

---

## Introduction

The NVIDIA® Deep Learning Accelerator (NVDLA) is a configurable fixed function hardware accelerator targeting inference operations in deep learning applications. It provides full hardware acceleration for a convolutional neural network (CNN) by exposing individual building blocks that accelerate operations associated with each CNN layer (e.g., convolution, deconvolution, fully-connected, activation, pooling, local response normalization, etc.). Maintaining separate and independently configurable blocks means that the NVDLA can be sized appropriately for many smaller applications where inferencing was previously not feasible due to cost, area, or power constraints. This modular architecture enables a highly configurable solution that readily scales to meet specific inferencing needs.

**About This Document**

This document focuses on the logical organization and control of the NVIDIA Deep Learning Accelerator. It provides information for those blocks and interfaces that control fundamental operations. The blocks detailed in this document include a functional overview, configuration options, and register listings for that block. All features and functions of all blocks may not be present in every NVDLA implementation.

---

## Functional Description

NVDLA operation begins with the management processor (either a microcontroller or the main CPU) sending down the configuration of one hardware layer, along with an "activate" command. If data dependencies do not preclude this, multiple hardware layers can be sent down to different blocks and activated at the same time (i.e., if there exists another layer whose inputs do not depend on the output from the previous layer). Because every block has a double-buffer for its configuration registers, it can also capture a second layer's configuration to begin immediately processing when the active layer has completed. Once a hardware engine finishes its active task, it will issue an interrupt to the management processor to report the completion, and the management processor will then begin the process again. This command-execute-interrupt flow repeats until inference on the entire network is complete.

NVDLA has two modes of operation: **independent mode** and **fused mode**.

- **Independent mode**: Each individual block is configured for when and what it executes, with each block working on its assigned task. Operations begin and end with memory-to-memory transfers in and out of main system memory or dedicated SRAM.
- **Fused mode**: Some blocks can be assembled as a pipeline, improving performance by bypassing round trips through memory. Blocks communicate through small FIFOs (e.g., the convolution core → Single Data Point Processor → Planar Data Processor → Cross-channel Data Processor, without intermediate memory-to-memory operations).

Each block in the NVDLA architecture supports specific operations integral to inference on deep neural networks. Inference operations are divided into five groups:

- Convolution operations (Convolution core and buffer blocks)
- Single Data Point operations (Activation engine block)
- Planar Data operations (Pooling engine block)
- Multi-Plane operations (Local resp. norm block)
- Data Memory and Reshape operations (Reshape and Bridge DMA blocks)

---

## Convolution Operations

Convolution operations work on two sets of data: offline-trained **weights** (constant between inference runs) and input **feature** data (varies with network input). The NVDLA Convolution Engine supports several modes:

- Direct
- Image-input
- Winograd
- Batching

### Direct Convolution Mode

Direct convolution is the basic mode of operation. NVDLA incorporates a wide multiply–accumulate (MAC) pipeline to support efficient parallel direct convolutional operation.

Two memory bandwidth optimization features are supported:

- **Sparse compression**: The sparser the feature/weight data, the less memory bus traffic. A 60% sparse network can nearly halve memory bandwidth requirements.
- **Second memory interface**: Provides efficient on-chip buffering, increasing memory bandwidth (2x–4x of DRAM) with reduced latency (1/10–1/4 of DRAM).

MAC efficiency depends on alignment of layer channel/kernel numbers with the `Atomic-C` and `Atomic-K` hardware parameters. Misalignment leads to underutilized MACs.

**Hardware Parameters:**
- Atomic-C sizing
- Atomic-K sizing
- Data type support
- Feature support – Compression
- Feature support – Second Memory Bus

### Image-Input Convolution Mode

A special direct convolution mode for the first layer, handling image surface input (typically 1–3 channels). A channel extension feature maintains average MAC utilization close to 50% even when `Atomic-C` is large.

**Hardware Parameters:** All from Direct Convolution mode + Image input support

### Winograd Convolution Mode

An optional algorithm that reduces the number of multiplications at the cost of additional additions. For a 3×3 filter, Winograd reduces MAC operations by a factor of **2.25×**, improving performance and power efficiency.

The equation used:

```
S = A^T [ (GgG^T) ⊙ (C^T dC) ] A
```

Where ⊙ denotes element-wise multiplication. Weight conversion is done offline; weight data size increases, but compute cost is reduced.

**Hardware Parameters:** Feature support – Winograd

### Batching Convolution Mode

Supports processing multiple sets of input activations (from multiple images) simultaneously, reusing weights to save memory bandwidth. Run-time for a single batch is close to that for a single image; overall throughput scales approximately as `batching_size × single-layer performance`.

> **Note:** Maximum batching size is limited by the convolution buffer size and is a hardware design specification constraint.

**Hardware Parameters:**
- Feature batch support
- Max batch number

### Convolution Buffer

An internal RAM reserved for weight and input feature storage, greatly improving memory efficiency by avoiding repeated system memory accesses.

Requires at least 4 ports:
- Read port for feature data
- Read port for weight data
- Write port for feature data
- Write port for weight data

Read bandwidth determines port width: for Atomic-C=16 on INT8, a 128-bit (16 byte) width is required.

**Hardware Parameters:**
- BUFF bank count
- BUFF bank size

---

## Single Data Point Operations (SDP)

The Single Data Point Processor (SDP) applies linear and non-linear functions onto individual data points. Commonly used immediately after convolution. Supports:

- Linear functions: bias, scaling, batch normalization, element-wise operations
- Non-linear functions via lookup tables (LUTs): ReLU, PReLU, Sigmoid, Tanh, etc.

**Hardware Parameters:**
- SDP function support
- SDP throughput

### Linear Functions

Scaling factor/bias can be set at three granularities:
1. **CNN setting** – same value across the whole network (from register config)
2. **Channel setting** – same within a single channel (from memory interface)
3. **Per-pixel setting** – different per feature (from memory interface)

Supported operations:
- **Precision Scaling** – Controls memory bandwidth by scaling feature data to full range before quantization.
- **Batch Normalization** – Per-layer or per-channel linear scaling with trained factors.
- **Bias Addition** – Per-layer, per-channel, or per-feature offset applied to output.
- **Element-Wise Operations** – Operates on two same-shaped (W×H×C) feature cubes: add, subtract, multiply, max.

### Non-Linear Functions

| Function | Definition |
|---|---|
| ReLU | `max(x, 0)` |
| PReLU | `x` if `x > 0`, else `k*x` |
| Sigmoid | `1 / (1 + e^(-x))` |
| Tanh | `(1 - e^(-2x)) / (1 + e^(-2x))` |

More complex non-linear functions are implemented via LUT.

---

## Planar Data Operations (PDP)

The Planar Data Processor (PDP) supports spatial operations common in CNN applications. Configurable at runtime to support different pool group sizes.

Supported pooling functions:
- **Maximum pooling** – returns the maximum value from the pooling window
- **Minimum pooling** – returns the minimum value from the pooling window
- **Average pooling** – returns the average value from the pooling window

**Hardware Parameters:**
- PDP function support
- PDP throughput (number of operations per clock)

---

## Multi-Plane Operations (CDP)

The Cross-channel Data Processor (CDP) is dedicated to performing Local Response Normalization (LRN) functions, a special normalization function operating across channels.

---

## Data Memory and Reshape Operations

### Reshape Engine (Rubik)

The Reshape engine supports data layout transformations. It enables hardware-accelerated data reformatting operations between layers that expect different memory layouts.

### Bridge DMA (BDMA)

The Bridge DMA engine transfers data between CVSRAM (on-chip SRAM) and main memory (MC/DRAM). This supports efficient use of the optional second memory interface.

---

## Hardware Interfaces

NVDLA implements three major connections to the rest of the system:

### Configuration Space Bus (CSB) Interface
- Synchronous, low-bandwidth, low-power, 32-bit control bus
- Used by the CPU to access NVDLA configuration registers
- NVDLA functions as a slave on CSB
- Easily converted to AMBA, OCP, or other system buses via a shim layer

### Interrupt Interface
- 1-bit level-driven interrupt
- Asserted on task completion or error

### Data Backbone (DBB) Interface
- Synchronous, high-speed, highly configurable data bus
- Connects NVDLA to the main system memory subsystem
- Configurable address size, data size, and request sizes

---

## Hardware Parameters Summary

NVDLA is highly parameterizable. Key hardware parameters include:

| Parameter | Description |
|---|---|
| Atomic-C | Number of input channels processed per clock cycle |
| Atomic-K | Number of output kernels processed per clock cycle |
| BUFF bank # | Number of banks in the convolution buffer |
| BUFF bank size | Size of each convolution buffer bank |
| Max batch number | Maximum number of batched inferences |
| Data types | Supported numeric formats (e.g., INT8, FP16) |
| Feature flags | Winograd, sparse compression, second memory bus, etc. |
| SDP throughput | Single data point processor throughput |
| PDP throughput | Planar data processor throughput |

---

*© NVIDIA Corporation. Reproduced under the [NVIDIA Open NVDLA License v1.0](https://nvdla.org/license.html).*
