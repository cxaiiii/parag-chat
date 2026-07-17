# 🪷 Parag Chat (v0)

**Try it: https://cxaiiii.github.io/parag-chat/**

Parag is a **0.5 B-parameter chat model** built by [Chaitanya (cxaiiii)](https://github.com/cxaiiii),
distilled from a custom dataset onto Qwen2.5-0.5B and quantized to Q4_K_M GGUF (~380 MB).

There is **no server**. The model weights live in this repo (split into <100 MB
chunks), download to your browser once, and run locally via
[llama.cpp](https://github.com/ggml-org/llama.cpp) compiled to WebAssembly
([wllama](https://github.com/ngxson/wllama)). Chats never leave your device.

## How it works

- `model/` — the GGUF split into 80 MB chunks + `manifest.json`; the page
  fetches them same-origin, stitches them into one Blob, and caches them
  with the Cache API so repeat visits skip the download.
- `coi-serviceworker.min.js` — injects COOP/COEP headers on GitHub Pages so
  `SharedArrayBuffer` is available and inference runs multi-threaded.
- `assets/app.js` — formats the conversation as ChatML (the GGUF carries no
  chat template), streams tokens, and stops on `<|im_end|>`.

## Running locally

Any static file server works:

```
npx http-server . -p 8137
```

## Notes

- One-time ~380 MB download; Wi-Fi and a laptop/desktop recommended.
- It's a tiny model trained on a small distilled dataset — it follows chat
  style well for its size, but don't expect frontier-model knowledge.
