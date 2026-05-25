# Data

This folder is where you place the raw dataset files required to run the pipeline.

## Required files

Download the following two files from the Google Maps Reviews dataset hosted by the McAuley Lab at UC San Diego:

**Download link:** https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/

| File | Description |
|------|-------------|
| `meta-Alaska.json` | Business metadata (name, category, location, etc.) |
| `review-Alaska.json` | Customer reviews (text, star rating, timestamp, etc.) |

Both `.json` and `.json.gz` (compressed) formats are supported — the pipeline handles both automatically.

## Why are these files not included?

The raw dataset files are not included in this repository for two reasons:

1. **Size** — the Alaska review file contains over 1 million records and is too large for GitHub
2. **Attribution** — the dataset is published and maintained by the McAuley Lab; please download it directly from their source and cite their work if you use it

## Citation

Yan, S., et al. (2023). *Personalized Showcases: Generating Multi-Modal Explanations for Recommendations.*
Proceedings of the 46th International ACM SIGIR Conference. ACM.
https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/

## Once downloaded

Place both files in this folder, then open `final_code_version.py` and update the three path variables:

```python
META_PATH    = "data/meta-Alaska.json"
REVIEWS_PATH = "data/review-Alaska.json"
OUTPUT_DIR   = "results/"
```

Then run:

```bash
python final_code_version.py
```
