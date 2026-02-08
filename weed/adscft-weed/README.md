# Holographic Ingestion Check

This tool simulates a "Holographic Ingestion" process and subsequent "Bulk Reconstruction" using the Weed tensor library. It demonstrates encoding data into a high-dimensional Hilbert space boundary state and performing inference via tensor contraction.

## Description

The `holographic_ingest_check.cpp` file performs the following steps:

1.  **Phase 1: Holographic Ingestion**
    -   Ingests a data stream (simulated) and maps features to a boundary index in a $2^{24}$ dimensional Hilbert space.
    -   Accumulates these mappings into a raw boundary data vector.
    -   Freezes the state into a `Weed::Tensor` (shared pointer).

2.  **Phase 2: Bulk Reconstruction (Inference)**
    -   Defines a bulk operator targeting a specific region.
    -   Contracts the boundary state tensor with the bulk operator tensor.
    -   Measures the overlap amplitude to verify if data exists in the target region.

## Compilation

To compile the code, use the provided `compile.sh` script:

```bash
./compile.sh
```

**Note:** This code depends on the `Weed` library and `OpenCL`. Ensure that the `Weed` headers (`shared_api.hpp`, `tensors/tensor.hpp`, etc.) are in your include path (e.g., `./include` or `./build/include/common`) and that `libweed` and `libOpenCL` are available.

## Expected Output

When run successfully, the program outputs the progress of ingestion and the result of the inference:

```
--- Phase 1: Holographic Ingestion ---
Ingesting Data Stream...
Mapped first row to index: ...
Freezing State into Weed Tensor...
Ingestion Complete. Boundary Size: 2^24

--- Phase 2: Bulk Reconstruction (Inference) ---
Applying Operator (Contracting Tensor Network)...
Reading Result State...
Inference Result (Overlap Amplitude): ...
>> CONCLUSION: Hypothesis CONFIRMED. (Data found in target region)
```

## Visuals

<img width="685" height="221" alt="Screenshot from 2026-02-08 16-03-27" src="https://github.com/user-attachments/assets/6172e129-0bcd-4e3f-912f-b6a61a9a65d9" />
