# LLM Inference From Scratch to Production 

This repository is my hands-on exploration of how Large Language Models actually perform inference, starting from a basic Transformer forward pass and gradually moving toward efficient, scalable, production-grade LLM serving.

The goal is not only to run a pretrained model and generate text.

I want to understand what happens inside the system when a user sends a prompt, how tokens move through the model, how attention is computed, how the KV cache works, why inference becomes expensive, and what techniques are used to make modern LLM inference faster, cheaper, and more scalable.

The repository moves from the fundamentals of autoregressive generation to modern inference systems involving batching, optimized attention, quantization, speculative decoding, distributed inference, GPU optimization, and production serving.

---

# Why LLM Inference?

Training creates the model, but inference is where the model becomes an actual product.

A trained model still needs to answer real requests:

```text
User Request
     |
     v
API
     |
     v
Tokenizer
     |
     v
LLM
     |
     v
Token Generation
     |
     v
Detokenization
     |
     v
Response
```

At small scale, inference looks simple.

At production scale, the problem becomes much more complicated.

```text
More Users
    |
    v
More Requests
    |
    v
More Tokens
    |
    v
More GPU Memory
    |
    v
More Compute
    |
    v
Higher Latency
    |
    v
Higher Cost
```

The purpose of this repository is to understand the engineering techniques used to solve these problems.

---

# What Happens During LLM Inference?

At a high level:

```text
User Prompt
     |
     v
Tokenizer
     |
     v
Input Token IDs
     |
     v
Embedding
     |
     v
Transformer Blocks
     |
     +------------------+
     |                  |
     v                  v
  Attention             MLP
     |                  |
     +--------+---------+
              |
              v
        Hidden States
              |
              v
           LM Head
              |
              v
           Logits
              |
              v
       Sampling / Decode
              |
              v
        Next Token
              |
              v
       Repeat Generation
```

The model generates tokens autoregressively.

```text
Prompt
  |
  v
Token 1
  |
  v
Token 2
  |
  v
Token 3
  |
  v
Token 4
  |
  v
...
```

Understanding this loop is the foundation of the entire repository.

---

# Prefill and Decode

Modern LLM inference can be understood through two major phases.

```text
                 LLM INFERENCE

                    Prompt
                      |
                      v
                  PREFILL
                      |
                      v
               Initial KV Cache
                      |
                      v
                   DECODE
                      |
          +-----------+-----------+
          |           |           |
          v           v           v
       Token N     Token N+1   Token N+2
          |           |           |
          +-----------+-----------+
                      |
                      v
                  Response
```

## Prefill

The model processes the input prompt.

```text
"Explain machine learning"
            |
            v
       Tokenization
            |
            v
      Transformer
            |
            v
        KV Cache
```

The prompt can contain hundreds, thousands, or more tokens.

Prefill is heavily associated with processing the existing context.

## Decode

The model generates new tokens one at a time.

```text
Existing Context
      +
Previous KV Cache
      +
New Token
      |
      v
Transformer
      |
      v
Next Token
```

This distinction is important because prefill and decode can have very different performance characteristics.

---

# KV Cache

One of the most important concepts in LLM inference is the Key-Value cache.

Without caching, the model would repeatedly recompute information from previous tokens.

Conceptually:

```text
Without KV Cache

Token 1
Token 1 + Token 2
Token 1 + Token 2 + Token 3
Token 1 + Token 2 + Token 3 + Token 4
...
```

With KV caching:

```text
Prompt
  |
  v
Compute K and V
  |
  v
Store in KV Cache
  |
  v
Generate Token
  |
  v
Compute K and V only for new token
  |
  v
Append to Cache
```

The repository will inspect:

* What K and V represent
* KV cache tensor shapes
* Cache memory consumption
* Cache growth with sequence length
* Batch-level KV cache
* KV cache reuse
* Prefix caching
* Paged KV cache

---

# KV Cache Memory

KV cache becomes a major memory consumer during serving.

Conceptually:

```text
More Users
     |
     v
More Sequences
     |
     v
More Tokens
     |
     v
Larger KV Cache
     |
     v
More GPU Memory
```

I will calculate KV cache memory instead of treating it as a hidden implementation detail.

For example:

```text
Batch Size
Sequence Length
Number of Layers
Number of KV Heads
Head Dimension
Data Type
```

will all be connected to the actual memory requirement.

---

# Naive Attention vs Optimized Attention

A basic attention implementation can create large intermediate tensors.

```text
Q
|
v
QK^T
|
v
Attention Matrix
|
v
Softmax
|
v
Attention × V
```

Modern implementations can use fused and memory-efficient kernels.

PyTorch's scaled dot product attention can dispatch to optimized implementations such as FlashAttention-2 and memory-efficient attention when supported.

I will compare:

```text
Naive Attention
        |
        v
Memory Efficient Attention
        |
        v
Fused Attention
        |
        v
FlashAttention
```

and measure the practical differences.

---

# Flash Attention

The repository will explore why optimized attention kernels matter.

The important idea is not simply:

```text
FlashAttention = Faster Attention
```

Instead, I want to understand the underlying problem:

```text
GPU Memory
     |
     v
Memory Reads / Writes
     |
     v
Memory Bandwidth
     |
     v
Attention Performance
```

Modern PyTorch SDPA can automatically select optimized attention implementations depending on the inputs and backend.

---

# Autoregressive Generation

The basic generation loop:

```text
Input Tokens
     |
     v
Model Forward
     |
     v
Logits
     |
     v
Sampling
     |
     v
Next Token
     |
     v
Append Token
     |
     v
Repeat
```

I will implement and compare different generation strategies.

---

# Decoding Strategies

Topics include:

```text
Greedy Decoding
     |
     v
Temperature Sampling
     |
     v
Top-K Sampling
     |
     v
Top-P Sampling
     |
     v
Beam Search
     |
     v
Advanced Sampling
```

The repository will investigate how decoding changes:

* Output quality
* Diversity
* Latency
* Number of tokens generated
* Determinism

---

# Token Generation

For each generated token:

```text
Hidden State
     |
     v
LM Head
     |
     v
Logits
     |
     v
Probability Distribution
     |
     v
Sampling
     |
     v
Token ID
```

The token is then appended to the sequence.

```text
Existing Tokens + New Token
          |
          v
       Next Step
```

This continues until a stopping condition is reached.

---

# Stopping Generation

Generation can stop because of:

```text
EOS Token
Maximum New Tokens
Maximum Context Length
Stop Sequence
Custom Stopping Rule
```

I will explicitly implement these conditions rather than treating generation as an opaque API.

---

# Batch Inference

Single request:

```text
Request
  |
  v
GPU
  |
  v
Response
```

Production serving usually has many requests:

```text
Request 1
Request 2
Request 3
Request 4
Request 5
     |
     v
   Batch
     |
     v
    GPU
```

Batching allows hardware to process multiple requests together.

But batching introduces new problems:

```text
Different Prompt Lengths
Different Generation Lengths
Different Arrival Times
Different KV Cache Sizes
```

I will explore how these problems are handled.

---

# Static Batching

A simple approach:

```text
Request 1
Request 2
Request 3
Request 4
     |
     v
Static Batch
     |
     v
GPU
```

The batch waits for requests to be grouped.

This is simple but can waste compute when requests have different lengths.

---

# Dynamic Batching

Requests can be grouped dynamically:

```text
Incoming Requests
       |
       v
Batch Scheduler
       |
       v
Dynamic Batch
       |
       v
GPU
```

The repository will explore:

* Batch formation
* Queueing
* Maximum batch size
* Maximum waiting time
* Throughput
* Latency

---

# Continuous Batching

Continuous batching changes the way requests are scheduled.

Instead of waiting for an entire batch to finish:

```text
Request A
   |
   +------ generation ------+
                            |
                            v
                         Finished
```

new requests can enter while other sequences are still generating.

Conceptually:

```text
Time
------------------------------------------------>

Request A   [==========]
Request B       [===========]
Request C          [======]
Request D             [===========]
Request E                [======]

             GPU Scheduler
```

Modern inference engines such as vLLM use continuous batching as one of their core serving optimizations.

---

# Paged KV Cache

A major production challenge is efficiently managing KV cache memory.

A simple approach can lead to fragmented or inefficient memory usage.

Paged attention approaches manage KV cache using blocks/pages.

Conceptually:

```text
Sequence
   |
   v
+------+------+------+------+
| Page | Page | Page | Page |
+------+------+------+------+
```

Different requests can use different blocks.

This is one of the important ideas behind high-throughput inference engines such as vLLM.

---

# Prefix Caching

If multiple requests share the same prefix:

```text
Request A:
System Prompt + Question A

Request B:
System Prompt + Question B

Request C:
System Prompt + Question C
```

the common prefix can potentially be reused.

```text
Shared Prefix
     |
     v
Cached KV
     |
 +---+---+
 |   |   |
 A   B   C
```

I will explore when prefix caching helps and what memory tradeoffs it introduces.

---

# Quantization

LLMs can consume large amounts of GPU memory.

Quantization reduces the representation size of model weights and, depending on the method, other inference state.

```text
FP32
  |
  v
FP16 / BF16
  |
  v
INT8
  |
  v
INT4
```

I will explore:

* FP32
* FP16
* BF16
* INT8
* INT4
* Weight-only quantization
* Activation quantization
* GPTQ
* AWQ
* FP8
* Quantization accuracy tradeoffs

Modern serving systems support a broad range of quantization formats and methods.

---

# Why Quantization?

The main questions are:

```text
Can the model fit into GPU memory?

How much memory can be saved?

How much throughput improves?

What accuracy is lost?

Does latency improve?

Does the hardware efficiently support the format?
```

The repository will benchmark these tradeoffs rather than assuming lower precision is always better.

---

# Speculative Decoding

Autoregressive generation normally produces tokens sequentially.

Speculative decoding introduces a draft-and-verify idea:

```text
Large Model
     ^
     |
Verify
     |
Draft Model
     |
Generate Candidates
```

Conceptually:

```text
Draft Model
    |
    v
Token A → Token B → Token C
    |
    v
Large Model Verification
    |
    v
Accept / Reject
```

The goal is to reduce generation latency without changing the target model's intended output distribution under appropriate implementations.

Modern inference engines support multiple speculative decoding approaches, including draft models and other proposer methods.

---

# GPU Memory

Inference performance is strongly connected to memory.

I will inspect:

```text
Model Weights
     +
KV Cache
     +
Activations
     +
Temporary Buffers
     +
CUDA Memory
     =
GPU Memory Usage
```

Experiments will measure:

* Model memory
* KV cache memory
* Peak memory
* Memory fragmentation
* Batch memory
* Sequence-length impact

---

# Latency

Latency is not a single number.

I will measure:

```text
Request Latency
     |
     +--> Queue Time
     |
     +--> Prefill Time
     |
     +--> Decode Time
     |
     +--> Network Time
     |
     +--> Post-processing
```

Important metrics include:

* Time to first token
* Time per output token
* Inter-token latency
* End-to-end latency
* P50
* P95
* P99

---

# Throughput

Throughput answers a different question.

```text
How many requests or tokens can the system process per second?
```

Metrics include:

```text
Requests / second
Tokens / second
Input tokens / second
Output tokens / second
GPU utilization
```

I will compare latency and throughput instead of optimizing one while ignoring the other.

---

# Latency vs Throughput

A production system must balance:

```text
Low Latency
     vs
High Throughput
     vs
Low Cost
```

For example:

```text
Small Batch
    |
    v
Lower Waiting Time
    |
    v
Potentially Lower Throughput
```

while:

```text
Large Batch
    |
    v
Better GPU Utilization
    |
    v
Potentially Higher Latency
```

The repository will use benchmarks to understand this tradeoff.

---

# Parallelism

Large models may not fit on a single GPU.

I will explore:

```text
Data Parallelism
Tensor Parallelism
Pipeline Parallelism
Expert Parallelism
Context Parallelism
```

A distributed inference system can look like:

```text
                LLM
                 |
        +--------+--------+
        |        |        |
       GPU 0    GPU 1    GPU 2
        |        |        |
        +--------+--------+
                 |
                 v
              Output
```

The objective is to understand what is being distributed and why.

---

# Tensor Parallelism

A large model layer can be split across GPUs.

```text
Large Matrix
     |
     +------------+------------+
     |            |            |
     v            v            v
   GPU 0        GPU 1        GPU 2
```

I will explore:

* Weight partitioning
* Communication
* All-reduce
* All-gather
* GPU memory
* Latency
* Scaling efficiency

---

# Pipeline Parallelism

Layers can also be distributed:

```text
GPU 0
Layers 1-10
     |
     v
GPU 1
Layers 11-20
     |
     v
GPU 2
Layers 21-30
```

I will compare pipeline parallelism with tensor parallelism and investigate when each approach makes sense.

---

# Multi-GPU Inference

The complete flow can become:

```text
Client
  |
  v
API Gateway
  |
  v
Scheduler
  |
  v
Inference Engine
  |
  +----------+----------+
  |          |          |
 GPU 0      GPU 1      GPU 2
  |          |          |
  +----------+----------+
             |
             v
          Response
```

---

# Serving Architecture

The final production architecture explored in this repository will look conceptually like:

```text
                         CLIENT
                           |
                           v
                      API Gateway
                           |
                           v
                    Request Queue
                           |
                           v
                     Scheduler
                           |
                           v
                 Inference Engine
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
          GPU 0          GPU 1          GPU 2
             |             |             |
             +-------------+-------------+
                           |
                           v
                     Token Stream
                           |
                           v
                        Client
```

The serving layer will be treated as a system, not simply a Python function around a model.

---

# Streaming

For interactive applications, users usually should not wait for the entire response.

Instead:

```text
Token 1 → Client
Token 2 → Client
Token 3 → Client
Token 4 → Client
...
```

I will explore:

* Streaming APIs
* Server-sent events
* Token buffering
* Backpressure
* Client disconnects
* Cancellation

---

# Request Scheduling

Production inference requires scheduling.

```text
Incoming Requests
       |
       v
+------------------+
| Request Scheduler|
+------------------+
       |
       +------+
       |      |
       v      v
   Prefill   Decode
       |      |
       +------+
          |
          v
         GPU
```

The scheduler must consider:

* Queue length
* Request priority
* Sequence length
* KV cache availability
* GPU utilization
* Batch formation
* Request deadlines

---

# Cancellation

Users may disconnect before generation completes.

A production system should not continue wasting GPU resources unnecessarily.

```text
Request
   |
   v
Generation
   |
   X
Client Disconnect
   |
   v
Cancellation
   |
   v
Free Resources
```

I will explore request lifecycle management.

---

# Fault Handling

Production inference systems must handle:

```text
GPU Failure
Model Loading Failure
Out Of Memory
Timeout
Client Disconnect
Invalid Request
Network Failure
Worker Failure
```

The repository will explore:

```text
Failure
   |
   v
Detection
   |
   v
Recovery
   |
   v
Retry / Failover
   |
   v
Monitoring
```

---

# Benchmarking

Every optimization should be measured.

The benchmark process will be:

```text
Baseline
   |
   v
Optimization
   |
   v
Benchmark
   |
   v
Compare
   |
   v
Analyze
```

Measurements include:

```text
TTFT
Inter-token latency
End-to-end latency
Throughput
GPU utilization
GPU memory
Tokens/sec
Requests/sec
Cost/request
```

---

# Inference Optimization Journey

The repository will progressively move through:

```text
Basic Transformer Inference
          |
          v
KV Cache
          |
          v
Efficient Attention
          |
          v
Batching
          |
          v
Continuous Batching
          |
          v
Paged KV Cache
          |
          v
Prefix Caching
          |
          v
Quantization
          |
          v
Speculative Decoding
          |
          v
GPU Optimization
          |
          v
Distributed Inference
          |
          v
Production Serving
```

---

# From Scratch to Production

The complete learning path is:

```text
01  Transformer Forward Pass
             |
02  Tokenization
             |
03  Autoregressive Generation
             |
04  Sampling
             |
05  KV Cache
             |
06  Prefill
             |
07  Decode
             |
08  Attention Optimization
             |
09  Flash Attention / SDPA
             |
10  Batch Inference
             |
11  Dynamic Batching
             |
12  Continuous Batching
             |
13  Paged KV Cache
             |
14  Prefix Caching
             |
15  Quantization
             |
16  Speculative Decoding
             |
17  GPU Optimization
             |
18  Tensor Parallelism
             |
19  Pipeline Parallelism
             |
20  Multi-GPU Inference
             |
21  Streaming
             |
22  Scheduling
             |
23  Benchmarking
             |
24  Monitoring
             |
25  Production Serving
```

---

# Experiments

This repository will be driven by experiments rather than only implementations.

Examples:

```text
How much memory does KV cache consume?

How does sequence length affect inference memory?

How does batch size affect throughput?

What happens to latency as concurrency increases?

How much faster is optimized attention?

How does FlashAttention compare with naive attention?

How much memory does quantization save?

What accuracy changes after quantization?

When does continuous batching help?

When does batching hurt latency?

How much does prefix caching help?

When is speculative decoding useful?

How does tensor parallelism scale?

Where does communication become the bottleneck?

What is the relationship between TTFT and prompt length?

What is the relationship between decode latency and output length?

How much GPU memory does a production request actually consume?
```

---

# Repository Structure

```text
llm-inference-from-scratch-to-production/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── 01_inference_fundamentals/
│   ├── transformer_forward/
│   ├── tokenizer/
│   ├── logits/
│   ├── sampling/
│   └── generation/
│
├── 02_autoregressive_generation/
│   ├── greedy/
│   ├── temperature/
│   ├── top_k/
│   ├── top_p/
│   ├── beam_search/
│   └── stopping/
│
├── 03_kv_cache/
│   ├── basic_cache/
│   ├── cache_shapes/
│   ├── memory_analysis/
│   ├── batched_cache/
│   └── prefix_cache/
│
├── 04_prefill_decode/
│   ├── prefill/
│   ├── decode/
│   ├── latency/
│   └── benchmarks/
│
├── 05_attention_optimization/
│   ├── naive_attention/
│   ├── sdpa/
│   ├── flash_attention/
│   └── benchmarks/
│
├── 06_batching/
│   ├── single_request/
│   ├── static_batching/
│   ├── dynamic_batching/
│   └── continuous_batching/
│
├── 07_memory_management/
│   ├── gpu_memory/
│   ├── kv_memory/
│   ├── memory_fragmentation/
│   ├── paged_attention/
│   └── prefix_caching/
│
├── 08_quantization/
│   ├── fp16/
│   ├── bf16/
│   ├── fp8/
│   ├── int8/
│   ├── int4/
│   ├── gptq/
│   └── awq/
│
├── 09_speculative_decoding/
│   ├── draft_model/
│   ├── verification/
│   ├── acceptance_rate/
│   └── benchmarks/
│
├── 10_gpu_optimization/
│   ├── cuda/
│   ├── torch_compile/
│   ├── cuda_graphs/
│   ├── memory_optimization/
│   └── profiling/
│
├── 11_distributed_inference/
│   ├── data_parallel/
│   ├── tensor_parallel/
│   ├── pipeline_parallel/
│   ├── expert_parallel/
│   └── multi_gpu/
│
├── 12_serving/
│   ├── api/
│   ├── streaming/
│   ├── scheduling/
│   ├── batching/
│   ├── cancellation/
│   └── health_checks/
│
├── 13_benchmarking/
│   ├── latency/
│   ├── throughput/
│   ├── ttft/
│   ├── itl/
│   ├── memory/
│   └── cost/
│
├── 14_production/
│   ├── deployment/
│   ├── autoscaling/
│   ├── reliability/
│   ├── monitoring/
│   └── observability/
│
├── src/
│   └── inference/
│       ├── model/
│       ├── generation/
│       ├── cache/
│       ├── attention/
│       ├── scheduler/
│       ├── batching/
│       ├── quantization/
│       ├── distributed/
│       └── serving/
│
├── configs/
├── notebooks/
├── scripts/
├── tests/
├── benchmarks/
└── artifacts/
```

---

# Technology

The repository will primarily use:

```text
Python
PyTorch
CUDA
NumPy
Hugging Face Transformers
```

For production-oriented experiments:

```text
FastAPI
Docker
Kubernetes
vLLM
NCCL
Prometheus
Grafana
```

vLLM is particularly relevant for the production comparison section because its current architecture includes PagedAttention, continuous batching, prefix caching, optimized attention, quantization, speculative decoding, and distributed parallelism.

The purpose is not to reproduce every production engine completely. The goal is to understand the ideas behind these systems and then compare my implementations with mature inference engines.

---

# Implementation Philosophy

This repository follows a simple progression:

```text
Understand
    |
    v
Implement
    |
    v
Measure
    |
    v
Optimize
    |
    v
Compare
    |
    v
Productionize
```

I do not want to hide important inference behavior behind a single library call.

For important components, I want to understand:

```text
Input
  ↓
Tensor Shape
  ↓
Operation
  ↓
Memory
  ↓
Computation
  ↓
Output
```

Then I will compare the simple implementation with optimized production approaches.

---

# Final Goal

The final goal is to understand LLM inference at three levels.

## Level 1: Model

```text
Prompt
  ↓
Transformer
  ↓
KV Cache
  ↓
Logits
  ↓
Sampling
  ↓
Next Token
```

## Level 2: Performance

```text
KV Cache
    +
Attention Optimization
    +
Batching
    +
Quantization
    +
Speculative Decoding
    +
GPU Optimization
```

## Level 3: Production

```text
Clients
   ↓
API
   ↓
Scheduler
   ↓
Continuous Batching
   ↓
Inference Engine
   ↓
Multi-GPU
   ↓
Streaming
   ↓
Monitoring
   ↓
Autoscaling
```

The goal is to understand the complete path from:

```text
"How does one token get generated?"
```

to:

```text
"How can millions of tokens be generated efficiently and reliably in a production AI system?"
```

This repository is my exploration of that complete journey, from the fundamentals of LLM inference to modern high-throughput production serving.
