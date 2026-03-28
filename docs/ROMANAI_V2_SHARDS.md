# RomanAI v2 native `.4dai` shards (Cl(4,0) carriers)

## Expected layout (no third-party base weights)

RomanAI research payloads use **only** custom shards; there is **no** Falcon or other vendor base model in this path.

| Role | Filename (convention) |
|------|------------------------|
| Primary | `romanai_v2_part1.4dai` |
| Additional | `romanai_v2_part2.4dai`, `romanai_v2_part3.4dai`, … |

Shards are **not** committed to git by default (large binaries). Place them in the **same directory as your `Modelfile`** (or use absolute `FROM` paths), then run **`4dollama create`** from that directory so relative `FROM` resolves correctly.

## Ingest into 4DOllama

```powershell
cd <folder_containing_Modelfile_and_shards>
4dollama create romanai-v2 -f .\Modelfile
```

Multiple JSON `romanai.4dai` layers are merged; binary/safetensors shards follow the `create` rules in **`docs/4DOLLAMA_REFERENCE_FOR_LLMS.md`**.

## Repo root `Modelfile`

The root **`Modelfile`** points at **`FROM ./romanai_v2_part1.4dai`**. Until you copy or forge **`romanai_v2_part1.4dai`** next to it, `4dollama create` will fail with **`[4DOLLAMA FATAL] Shard not found`** and the absolute path searched—by design.
