# MeteorLLM

An experimental offline Qwen3-0.6B inference app for PlayStation Vita. It uses a C++20 engine, ARM NEON kernels, a 512-token context, and a 56 MiB FP16 KV cache. The transformer weights stay resident while the output projection streams its weights from storage.

## Build and install

Install VitaSDK and libvita2d, and set VITASDK to the SDK directory. Run:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build

Install the resulting MeteorLLM.vpk on your Vita. The title ID is METEOR001. Prepare model files with Python 3:

    python export_model.py /path/to/Qwen3-0.6B-Q4_K_M.gguf model
    python export_tokenizer.py

Copy weights.bin, embedding.bin, and tokenizer.bin from model/ to ux0:data/VitaDesk/MeteorLLM/. Model weights are distributed separately by their publisher. The exact tested Q4_K_M artifact was requantized from Qwen3-0.6B-Q8_0.gguf (Qwen revision 23749fefcc72300e3a2ad315e1317431b06b590a) using llama.cpp b12078 (19e8a86fe). Tested Q4_K_M SHA-256: 15e434d8ddf0af38fe3f7003af6ce6eac4edada6df34324aa45979f9cc79f02e.

The host validation tools build separately with cmake -S host -B build-host and cmake --build build-host --config Release.

## Measured configuration

To reproduce the measured configuration, place these text files in the app data directory:

| File | Value |
|---|---:|
| workers.txt | 3 |
| batch.txt | 8 |
| kernel.txt | 2 |
| attention.txt | 1 |
| output-rows.txt | 256 |

On a physical Vita at 333 MHz, the measured configuration returned Tavi. and reached EOS in 136.381 seconds for a 67-token formatted ordering prompt. First token arrived at 92.575 seconds. The previous completed single-worker fused build took 718.256 seconds for the same tokens. The model can fail simple arithmetic and the benchmark covers one prompt.

Cross opens the keyboard, Triangle starts a new chat, and Circle cancels. Up and Down scroll the answer.

## Third-party notices

The quantization layouts and decoding derive from ggml. See licenses/ for the upstream ggml and Qwen license texts.
