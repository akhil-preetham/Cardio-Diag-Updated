# Environment record reconciliation

## Authoritative hashes

| Artifact | SHA-256 |
|---|---|
| Original extracted notebook | `990f24622ca3aefe2f62e37f7d9e1097fe14b118a49ee398ff3ee67568bfbe4d` |
| Prepared working copy | `666ab88c049dba064057e79d55fb413bca37a1c7de3f3bb8c24cad070844b646` |
| `heart5.csv` | `e4068802a044e8a1e3ae9fa81deae7025711bf4f13e3057f96e4f1cf3b0180c2` |

Independent read-only verification matched the patch manifest, `SHA256SUMS.txt`, and the first-run input hash manifest.

## Record reconciliation

- `base_notebook_environment.txt` is preserved unchanged and contains the historical transcription error in the working-copy hash.
- `base_notebook_environment_corrected.txt` records the correct working-copy hash and is authoritative for future documentation.
- `base_notebook_patch_manifest.json` agrees with both notebook hashes.
- `SHA256SUMS.txt` agrees with both notebook hashes and the dataset hash.
- `runs/author_faithful/base_five_states_run_001/input_hash_manifest.json` agrees with all three authoritative hashes.

No notebook, dataset, package, or environment was modified to resolve this documentation inconsistency.
