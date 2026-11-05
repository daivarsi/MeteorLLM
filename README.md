# MeteorLLM

An experimental offline Qwen3-0.6B inference app for PlayStation Vita. It uses a C++20 engine, ARM NEON kernels, a rolling 512-token attention window, and a 56 MiB FP16 KV cache. The transformer weights stay resident while the output projection streams its weights from storage.

## Build and install

Install VitaSDK and libvita2d, and set VITASDK to the SDK directory. Run:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

Install the resulting MeteorLLM.vpk on your Vita. The title ID is METEOR001. Prepare model files with Python 3:

    python export_model.py /path/to/Qwen3-0.6B-Q4_K_M.gguf model
    python export_tokenizer.py

Copy weights.bin, embedding.bin, and tokenizer.bin from model/ to ux0:data/VitaDesk/MeteorLLM/. Download MeteorLLM-Q4-output.zip from [v0.2.0](https://github.com/daivarsi/MeteorLLM/releases/tag/v0.2.0), extract output-q4-test.bin, and copy it to the same directory. This file is the optional Q4 output projection for the exact tested model below. Use the original projection for other model variants by omitting the Q4 file. The release does not bundle the resident model, input embeddings, or tokenizer. The exact tested Q4_K_M artifact was requantized from Qwen3-0.6B-Q8_0.gguf (Qwen revision 23749fefcc72300e3a2ad315e1317431b06b590a) using llama.cpp b12078 (19e8a86fe). Tested Q4_K_M SHA-256: 15e434d8ddf0af38fe3f7003af6ce6eac4edada6df34324aa45979f9cc79f02e.

The host validation tools build separately with cmake -S host -B build-host and cmake --build build-host --config Release.

## Measured configuration

The release uses these defaults. Existing configuration files in the app data directory can override them:

| File | Value |
|---|---:|
| workers.txt | 3 |
| batch.txt | 8 |
| kernel.txt | 2 |
| attention.txt | 1 |
| output-rows.txt | 256 |

On a physical Vita, these runs used the same 67-token formatted prompt and generated token IDs 51, 6190, and 13 (Tavi.), followed by EOS. Timing starts at prompt submission and ends at EOS. Model load is excluded.

| Configuration | First token | Through EOS |
| --- | ---: | ---: |
| Earlier optimized build | 92.58 s | 136.38 s |
| Rolling context, original kernels | 92.26 s | 134.30 s |
| Int8 Q4 prefill and decode | 74.08 s | 112.06 s |
| Int8 Q4/Q6 with Q4 output projection | 66.26 s | 92.03 s |
| Vocabulary shortlist | 60.63 s | 69.57 s |
| Tiled int8, template cache disabled | 57.98 s | 66.95 s |
| Tiled int8, template ready | 32.71 s | 41.69 s |
| v0.2.0 eight-token Q4 SIMD kernel, template ready | 30.07 s | 39.00 s |
| v0.2.1, Vectorized Q4 unpacking and Q6 lane kernel, template ready | 28.18 s | 37.14 s |

The benchmark prompt was “Mira is older than Soren. Soren is older than Tavi. Who is youngest? Answer with only the name.” The warm run cached only the 32 fixed tokens before the question. The release benchmark prepared the cache in 22.72 seconds at startup and used about 3.5 MiB for it. Warm submission plus template preparation took 61.72 seconds, excluding model load. The previous tiled build took 66.95 seconds with caching disabled. The newest kernel was benchmarked with the template ready. The external overclock was configured for 500 MHz and the app's clock query reported 333 MHz. Effective frequency was not independently verified. The earlier completed single-worker fused build took 718.26 seconds on the same prompt. This benchmark covers one prompt, and the model can still fail simple arithmetic.

The Q4 projection and int8 activation paths are approximate. With the Q4 file present, the fast output mode scores about 25,000 tokens and checks the full vocabulary when the leading scores are close. The shortlist can omit better rare or multilingual tokens even when it looks confident. Create output-full.txt in the app data directory and restart to score the full vocabulary. Create prefix-cold.txt and restart to disable shared-template caching.

Cross opens the keyboard, Triangle starts a new chat, and Circle cancels. Square cycles the response limit through 128, 256 (default), 512, 1024, and Until EOS. Drag the front touchscreen in the conversation area, or hold Up or Down, to scroll through the full chat. At the bottom, the view follows new text as it arrives.

## Third-party notices

The quantization layouts and decoding derive from ggml. See licenses/ for the upstream ggml and Qwen license texts.

## Rolling context

MeteorLLM carries its KV cache forward between turns. The last 512 tokens are available to attention, and RoPE continues at the absolute token position when the cache wraps. Evicted tokens leave the active window. Retained states may still reflect earlier attention to them, so rolling eviction and rebuilding a fresh 512-token prompt can produce different results. The app leaves CPU frequency control to the system or external overclock plugin.

Until EOS stops at the model's end token or the safety ceiling of 4,096 generated tokens or 64 KiB of response text. Circle can stop either prefill or decode. After an interrupted inference pass, the next message can replay the most recent cached token IDs from position zero to recover cleanly. Triangle clears the conversation and the cache. The UI keeps up to 128 completed turns and 128 KiB of completed text, plus the current turn.
