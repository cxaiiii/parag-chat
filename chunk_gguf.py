"""Split a GGUF into browser-sized chunks and write the manifest wllama loads.

    python chunk_gguf.py --gguf <path> --model-id parag-v5-0.8B

Was hardcoded to one model and one absolute path on one machine, which meant
every release needed the file edited before it could run. Same behaviour, now
parameterised.

--keep-old is off by default, matching the original: the app serves one model
at a time and stale chunks are dead weight in a GitHub Pages repo. Pass it to
leave earlier chunk sets in place when you want the selector to offer both.

NOTE ON REPO SIZE. These chunks are committed to git, and git keeps every
version forever -- deleting the old .bin files frees the working tree but not
the history. This repo's .git is already 2.2 GB for that reason. Serving the
model from a HuggingFace repo instead of from GitHub Pages would stop the
growth and remove the need for the cache-busting model renames in the log.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

CHUNK_SIZE = 10 * 1024 * 1024  # 10MB: wllama fetches these in parallel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-id", required=True,
                    help="e.g. parag-v5-0.8B; becomes the chunk prefix and the "
                         "data-model value the button in index.html uses")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "model"))
    ap.add_argument("--keep-old", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.keep_old:
        old = glob.glob(os.path.join(args.out_dir, "*.bin"))
        for p in old:
            os.remove(p)
        stale = [p for p in glob.glob(os.path.join(args.out_dir, "manifest-*.json"))
                 if args.model_id not in os.path.basename(p)]
        for p in stale:
            os.remove(p)
        print(f"removed {len(old)} old chunks and {len(stale)} stale manifests")

    total = os.path.getsize(args.gguf)
    chunks = []
    with open(args.gguf, "rb") as fh:
        idx = 0
        while True:
            data = fh.read(CHUNK_SIZE)
            if not data:
                break
            name = f"{args.model_id}-{idx:03d}.bin"
            with open(os.path.join(args.out_dir, name), "wb") as out:
                out.write(data)
            chunks.append({"file": name, "size": len(data)})
            idx += 1

    manifest = {"totalSize": total, "chunks": chunks}
    mpath = os.path.join(args.out_dir, f"manifest-{args.model_id}.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"{args.gguf}")
    print(f"  {total / 1048576:.1f} MB -> {len(chunks)} chunks")
    print(f"  manifest: {mpath}")
    print(f"  wire it up with: <button class=\"ms-btn\" "
          f"data-model=\"{args.model_id}\">")


if __name__ == "__main__":
    main()
