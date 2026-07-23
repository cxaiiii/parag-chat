import os
import json
import glob

# Config - Bumping to v4.1 to bypass GitHub Pages CDN cache
model_id = "parag-v4.1-0.6B"
gguf_path = r"C:\Users\saxen\Downloads\parag-v4-0.6b-Q4_K_M.gguf"
chunk_size = 10 * 1024 * 1024  # 10MB chunks
out_dir = r"C:\Dev\parag-chat-web\model"

# Clean up all old chunks in the directory
old_chunks = glob.glob(os.path.join(out_dir, "*.bin"))
for old in old_chunks:
    os.remove(old)
print(f"Deleted {len(old_chunks)} old chunks.")

print(f"Splitting {gguf_path} into 10MB chunks for {model_id}...")
total_size = os.path.getsize(gguf_path)
chunks = []

with open(gguf_path, "rb") as f:
    chunk_idx = 0
    while True:
        data = f.read(chunk_size)
        if not data:
            break
            
        chunk_name = f"{model_id}-{chunk_idx:03d}.bin"
        chunk_path = os.path.join(out_dir, chunk_name)
        
        with open(chunk_path, "wb") as chunk_file:
            chunk_file.write(data)
            
        chunks.append({
            "file": chunk_name,
            "size": len(data)
        })
        chunk_idx += 1

manifest = {
    "totalSize": total_size,
    "chunks": chunks
}

manifest_path = os.path.join(out_dir, f"manifest-{model_id}.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Created {len(chunks)} chunks and updated manifest at {manifest_path}.")
