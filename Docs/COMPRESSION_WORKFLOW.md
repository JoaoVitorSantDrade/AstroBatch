# AstroAlign and AstroStack compression workflow

## Choosing the AstroAlign output format

AstroAlign now exposes **Comprimir saída FITS (RICE_1)** in its storage
settings. It is enabled by default, preserving the existing output format:

- science pixels use RICE_1 tiled compression;
- `VALID_MASK` uses PLIO_1 tiled compression.

Disable the option to write ordinary, uncompressed `ImageHDU` extensions for
both the science image and `VALID_MASK`. The empty primary HDU and the file
name remain unchanged, so aligned files remain compatible with the rest of
the workflow.

For scripted calls to `process_all_alignments`, pass a native boolean:

```python
config = {"compress_output": False}
```

Omitting `compress_output` keeps compression enabled.

## AstroStack cache behavior

AstroStack's `.astrostack_fits_cache` is exclusively a decompression cache.
Only a science `CompImageHDU` is eligible for it. Uncompressed aligned files
are read directly and do not get hashed, reopened for cache generation, or
written into the cache directory.

Standard unsigned FITS images often contain `BZERO=32768`. This is FITS value
scaling, not tile compression, so AstroStack deliberately reads those files
directly. If memory mapping cannot be used with their scaling cards, it falls
back to a normal FITS read while retaining the same pixel values.

The cache status line reports the number of compressed candidates, cache hits,
rebuilds, direct fallbacks, uncompressed frames bypassed, and skipped frames.
On a repeated stack of unchanged compressed input, rebuilds should be zero and
hits should match the compressed candidate count.

## Cache maintenance and verification

Cache entries are keyed by source path, size, and modification time, and are
written atomically. A changed or corrupt source entry rebuilds automatically.
Old entries are harmless and are not deleted automatically; remove the input
folder's `.astrostack_fits_cache` directory manually when disk space is needed.

Run the regression checks with:

```powershell
python -m unittest tests.test_astroalign_output tests.test_stacking_cache tests.test_compression_pipeline
```
