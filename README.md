# atomic-llama-cpp-turboquant (wht-sign-select)

A research fork of [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) adding **imatrix-weighted WHT sign-pattern selection**, an extension of the **TQ3_1S** weight format that improves perplexity at **+0 bpw**.

## What is WHT sign-pattern selection

TQ3_1S quantizes 32-element weight blocks with a WHT-rotated 8-level Lloyd-Max codebook using one fixed sign pattern (golden-ratio hash). This fork instead selects, **per block**, the best of 4 candidate sign patterns under the imatrix-weighted reconstruction error:

- **Min-over-K selection** — each block picks the pattern minimizing weighted reconstruction error
- **imatrix-aware** — column importance weights steer quantization error away from important weights
- **+0 bpw** — the 2-bit pattern index is stored in the low bits of the existing d0 scale
- **CPU + CUDA** — both decoders read the pattern index back from d0

Block size: 32 elements · Storage: 16 bytes/block (unchanged) · Effective bitwidth: **4.4 bpw**

PPL vs plain TQ3_1S (wikitext-2 / HellaSwag):

- Qwen3.5-4B: **-0.59** / **-0.65**
- Qwen3.5-9B: **-0.69** / **-0.60**
- Phi-3-mini: **-1.28** / Gemma-3-4B: **-1.51** (wikitext-2)

The imatrix is essential: a shuffled-imatrix control removes the gain entirely. Weighted selection also outperforms unweighted selection despite slightly larger total error — the error is placed on unimportant columns.

Target hardware: AMD RX 6900 XT (ROCm/HIP, gfx1030)

Full paper forthcoming.

## License

MIT

This fork is built on:
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)
- [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (MIT)
- [AtomicBot-ai/atomic-llama-cpp-turboquant](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant) (MIT)
