# Triton & MLIR Compiler Optimization: From TTIR to PTX

**An End-to-End Compiler Analysis and Optimization Project for GPU GEMM Kernels**

## 1. Overview

This project explores the compilation and optimization of GPU kernels through the Triton, MLIR, and LLVM compiler infrastructure.

Using a Triton-based General Matrix Multiplication (GEMM) kernel as a case study, the project investigates how high-level GPU computations are progressively lowered into intermediate representations and ultimately translated into NVIDIA PTX instructions.

The project focuses on three key areas:

1. **Compiler Pipeline Exploration:** Understanding how Triton GEMM is lowered through TTIR, TTGIR, LLVM IR, and PTX.
2. **MLIR-Level Optimization:** Identifying redundant computations and investigating Common Subexpression Elimination (CSE) and Loop-Invariant Code Motion (LICM).
3. **LLVM-Level Optimization:** Analyzing control flow, SSA/PHI nodes, induction variables, and Loop Strength Reduction (LSR) for GPU memory addressing.

The objective is to connect compiler optimization theory with real GPU kernel compilation, IR transformations, and correctness verification.

## 2. Development Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA Tesla T4 (SM75) |
| Operating System | Linux (Google Colab) |
| Python | 3.13.16 |
| PyTorch | 2.11.0+cu130 |
| CUDA | 13.0 |
| Triton | 3.6.0 |
| LLVM | 20 |
| MLIR | 20 |
| Compiler Tools | clang, opt, llc, llvm-as, llvm-dis, mlir-opt |
| Languages | Python, Triton, C++, LLVM IR, PTX |

The experiments were conducted on a single NVIDIA Tesla T4 GPU, with additional LLVM optimization experiments targeting x86-64 for isolated transformation and correctness validation.

## 3. Project Architecture

The project is organized into three experimental modules.

```text
Triton GEMM Kernel
        |
        v
       TTIR
        |
        |  CSE / LICM Analysis
        v
       TTGIR
        |
        |  Layout Conversion
        |  GPU-Specific Lowering
        v
      LLVM IR
        |
        |  CFG / PHI Analysis
        |  Induction Variables
        |  Loop Strength Reduction
        v
       PTX
        |
        v
   GPU Execution
```

The project combines real compiler-generated IR inspection with isolated optimization prototypes.

## 4. Module 1 — Triton Compiler Pipeline Exploration

### Objective

Understand how a Triton GEMM kernel is represented and transformed throughout the compiler pipeline.

### Implementation

Implemented a tiled GEMM kernel using Triton with:

- FP16 input matrices
- FP32 accumulation
- Tile configuration: 32 × 64 × 32
- Four warps per CTA
- Masked global memory accesses
- A K-dimension reduction loop

Extracted and analyzed four compilation stages:

- TTIR
- TTGIR
- LLVM IR
- PTX

### Key Findings

**TTIR Analysis**

The generated TTIR contains:

- 2 `tt.load` operations
- 1 `tt.store` operation
- 1 `tt.dot` operation
- 1 `scf.for` loop

The loop performs tiled matrix multiplication while accumulating partial results.

**TTGIR Analysis**

Investigated GPU-specific layout representations, including:

- Blocked layouts
- Warp and thread mappings
- Shared-memory allocations
- Layout conversions
- Dot-operand layouts

The examined TTGIR showed explicit FP32 conversions and shared-memory-based operand preparation before the dot operation.

**PTX Analysis**

Inspected generated instructions including:

- `ld.global`
- `st.global`
- `add`
- `mul`
- `shl`
- `setp`
- `bra`

These observations helped connect high-level Triton operations with lower-level GPU memory access, address arithmetic, and control flow.

### Outcome

Established an end-to-end understanding of the Triton compilation pipeline and identified optimization opportunities at different IR levels.

## 5. Module 2 — MLIR Optimization: CSE and LICM

### Objective

Identify redundant operations in real Triton GEMM IR and investigate how compiler passes eliminate or move computations.

### Step 1: TTIR Operation Analysis

Analyzed the generated GEMM TTIR and classified operations by type.

The examined optimized TTIR contained 72 operations, including:

| Operation | Count |
|---|---:|
| `tt.broadcast` | 12 |
| `arith.constant` | 11 |
| `arith.muli` | 6 |
| `tt.expand_dims` | 6 |
| `tt.splat` | 6 |
| `tt.addptr` | 6 |
| `tt.load` | 2 |
| `tt.dot` | 1 |
| `scf.for` | 1 |
| `tt.store` | 1 |

Identified repeated tensor-index construction and address calculations associated with GEMM operands.

### Step 2: Compiler Pass Inspection

Captured intermediate representations around compiler optimization passes.

The investigated pass sequence included:

```text
inline
  -> canonicalize
  -> triton-rewrite-tensor-pointer
  -> triton-rewrite-tensor-descriptor-to-pointer
  -> canonicalize
  -> triton-combine
  -> triton-reorder-broadcast
  -> cse
  -> symbol-dce
  -> triton-licm
  -> canonicalize
```

Compared intermediate representations to determine how individual compiler passes modified the IR.

### Step 3: Common Subexpression Elimination

Analyzed the effects of native CSE on GEMM-related IR.

**Observed results:**

| Operation | Before | After | Removed |
|---|---:|---:|---:|
| `tt.expand_dims` | 12 | 6 | 6 |
| `tt.make_range` | 3 | 2 | 1 |
| Total | 79 | 72 | 7 |

Developed a text-based CSE prototype that identified and eliminated the same seven redundant operations in the examined IR.

Validated the resulting SSA references and compared the transformation with native compiler output.

The prototype demonstrates the optimization principle but is not a production MLIR pass.

### Step 4: Loop-Invariant Code Motion

Investigated loop-invariant address calculations within the GEMM reduction loop.

Identified candidate operations including:

- `tt.expand_dims`
- `tt.splat`
- `tt.broadcast`
- `tt.addptr`
- `arith.muli`

For a 16-iteration reduction loop, estimated the dynamic operations that could be avoided by hoisting eligible invariant computations.

| Operation | Hoist Candidates | Estimated Executions Avoided |
|---|---:|---:|
| `tt.expand_dims` | 2 | 30 |
| `tt.splat` | 2 | 30 |
| `tt.broadcast` | 4 | 60 |
| `tt.addptr` | 1 | 15 |
| `arith.muli` | 1 | 15 |

These values represent operation-count estimates, not measured runtime improvements.

### Step 5: MLIR Optimization Prototype

Implemented a small GEMM-related arithmetic optimization prototype using MLIR concepts and C++.

The prototype identifies redundant arithmetic expressions in address calculations.

A toy example demonstrated elimination of a repeated multiplication.

The native MLIR pass was not fully integrated into a standalone executable due to local LLVM/MLIR build configuration issues.

### Outcome

Demonstrated the ability to:

- Inspect real compiler-generated IR
- Identify redundant computations
- Analyze compiler pass effects
- Prototype CSE transformations
- Reason about loop-invariant computations
- Validate structural correctness of transformed IR

## 6. Module 3 — LLVM IR Optimization and Loop Strength Reduction

### Objective

Investigate how LLVM represents GEMM loops and optimizes repeated GPU memory address calculations.

### Step 1: LLVM IR Extraction

Extracted LLVM IR generated from the Triton GEMM kernel.

Analyzed:

- Basic blocks and control-flow structure
- SSA values
- PHI nodes
- Loop induction variables
- Pointer arithmetic
- Memory operations

The inspected LLVM IR contained:

| IR Construct | Count |
|---|---:|
| PHI nodes | 2 |
| Branch instructions | 2 |
| GetElementPtr instructions | 91 |
| Load instructions | 80 |
| Store instructions | 8 |
| Integer comparisons | 6 |

### Step 2: Loop and Induction Variable Analysis

Identified a reduction loop with 16 iterations.

The LLVM IR contains an induction variable represented by a PHI node:

```llvm
%indvars.iv = phi i64
    [ 0, %entry ],
    [ %indvars.iv.next, %loop ]

%offset = shl i64 %indvars.iv, 5

%gep = getelementptr half,
    ptr addrspace(1) %base,
    i64 %offset

%indvars.iv.next = add i64 %indvars.iv, 1
```

The address offset follows:

```text
element_offset = k * 32
byte_offset    = k * 64
```

For 16 iterations, the byte offsets are:

```text
0, 64, 128, 192, ..., 960
```

This reveals a regular memory-address progression that can potentially be represented using pointer increments instead of repeated index-based address calculations.

### Step 3: Loop Strength Reduction

Constructed an isolated LLVM IR example reproducing the GEMM address calculation.

The original form uses:

```text
k = PHI(0, k + 1)
offset = k << 5
address = base + offset
```

Applied LLVM optimization passes and examined the resulting IR.

The optimized form uses a pointer induction variable:

```text
ptr = PHI(base, next_ptr)
next_ptr = ptr + 32 elements
```

Observed LLVM-generated constructs including:

- Pointer PHI nodes
- `getelementptr`
- Scalar Evolution-derived address progression

The original index-shift pattern was eliminated in the isolated optimized example.

### Step 4: Correctness Verification

Compiled both the original and optimized LLVM IR implementations into shared libraries using Clang.

Executed both versions and compared their outputs against expected address offsets.

**Results:**

```text
Original LLVM IR:  PASS
Optimized LLVM IR: PASS
Expected Offsets:  PASS
```

Both implementations produced equivalent address-offset sequences.

### Step 5: PTX Instruction Analysis

Inspected the PTX generated from the real Triton GEMM kernel.

| Instruction Category | Instruction | Count |
|---|---|---:|
| Address arithmetic | `add` | 14 |
| Address arithmetic | `mul` | 1 |
| Address arithmetic | `mad` | 3 |
| Address arithmetic | `shl` | 13 |
| Conversion | `cvt` | 24 |
| Data movement | `mov` | 34 |
| Memory | `ld` | 121 |
| Memory | `st` | 9 |
| Control flow | `bra` | 1 |
| Predicate | `setp` | 3 |

Further inspection identified loop-carried pointer increments and address updates in the generated PTX, including:

```ptx
add.s64 %rd10, %rd10, 64;
add.s64 %rd9, %rd9, 32768;
```

These instructions illustrate different loop-carried address progressions in the generated GPU code.

The remaining shift instructions cannot be attributed exclusively to the original reduction-loop induction variable.

### Outcome

Demonstrated how LLVM optimization can transform index-based addressing into pointer-based induction in an isolated experiment, while connecting these concepts to the address arithmetic observed in real GPU-generated PTX.

The project does not claim a measured speedup from LSR in the original GEMM kernel.

## 7. Key Results

| Area | Result |
|---|---|
| Compiler Pipeline | Extracted and inspected TTIR, TTGIR, LLVM IR, and PTX |
| CSE | Observed elimination of 7 redundant operations |
| LICM | Identified loop-invariant operation candidates and estimated execution reductions |
| MLIR Prototype | Implemented and validated a limited redundant-expression elimination prototype |
| LLVM CFG | Analyzed loop structure, PHI nodes, and induction variables |
| LLVM LSR | Observed index-based addressing transformed into pointer induction |
| Correctness | Verified equivalent address offsets before and after isolated LLVM optimization |
| PTX | Analyzed instruction statistics and loop-carried address updates |

## 8. Technical Concepts Demonstrated

**Compiler Infrastructure**

- Intermediate representations
- Compiler lowering
- SSA and PHI nodes
- Basic blocks and control-flow graphs
- Compiler passes
- Common Subexpression Elimination
- Loop-Invariant Code Motion
- Induction variable analysis
- Loop Strength Reduction

**GPU Compilation**

- Triton kernel compilation
- GPU thread and warp layouts
- Shared-memory operand preparation
- Memory address generation
- PTX inspection
- GPU execution mapping

**Validation and Analysis**

- IR inspection and comparison
- Compiler pass dumps
- Structural SSA validation
- LLVM IR compilation
- Correctness testing
- Instruction-level analysis

## 9. Limitations and Future Work

The current project primarily focuses on compiler analysis, transformation prototypes, and correctness validation.

Several areas remain open for further development:

1. Implement a fully integrated C++ MLIR optimization pass using the MLIR pass infrastructure.
2. Extend optimization analysis to additional Triton kernels.
3. Benchmark generated kernels before and after controlled compiler transformations.
4. Investigate register pressure, instruction count, and occupancy tradeoffs.
5. Explore compiler lowering and optimization strategies for additional accelerator architectures.

## 10. Conclusion

This project provides a practical investigation of the GPU compiler stack, from high-level Triton kernels to low-level LLVM IR and PTX.

By combining real compiler-generated IR analysis, optimization prototypes, and correctness experiments, it demonstrates how compiler transformations can be identified, understood, and validated.

The project also establishes a foundation for further work in GPU compiler development, MLIR/LLVM optimization, and accelerator-specific code generation.

---

**Keywords:** Triton, MLIR, LLVM, GPU Compiler, GEMM, TTIR, TTGIR, PTX, CSE, LICM, Loop Strength Reduction, SSA, Compiler Optimization.