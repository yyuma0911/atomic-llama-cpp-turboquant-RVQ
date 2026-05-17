# atomic-llama-cpp-turboquant-RVQ

A research fork of [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) adding **TQ3_RVQ**, a 3.44 bpw residual vector quantization format for LLM weights.

## What is TQ3_RVQ

TQ3_RVQ extends TurboQuant's WHT-rotated Lloyd-Max quantization stack with a 3-stage residual vector quantization (RVQ) approach over 256-element super-blocks:

- **Stage 1** — 3-bit uniform Lloyd-Max quantization with WHT rotation
- **Stage 2** — 4-bit RVQ correction per 32-element block (16-entry codebook)
- **Stage 3** — 2-bit RVQ correction per 8-element sub-block (4-entry codebook)
- **Joint optimization** — 6-iteration iterative refinement of all stages simultaneously
- **imatrix support** — importance-weighted Wiener filter for scale optimization

Block size: 256 elements · Storage: 110 bytes · Effective bitwidth: **3.4375 bpw**

Target hardware: AMD RX 6900 XT (ROCm/HIP, gfx1030)

Full paper forthcoming.

## License

MIT

This fork is built on:
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (MIT)
- [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) (MIT)
