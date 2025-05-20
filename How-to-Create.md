🗣️ Piper TTS Training: Optimus Prime Voice
This project demonstrates how to prepare and train a custom TTS voice using Piper, starting from a Hugging Face .parquet dataset.

I hope I remembered everything I have done.

---
## 💸 Donations Appreciated!
If you find this plugin useful, please consider donating. Your support is greatly appreciated!

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=TWRQVYJWC77E6)
---

## 🙌 Acknowledgements

- [Piper TTS](https://github.com/rhasspy/piper) for the amazing TTS engine.
- [ifansnek/piper-train-docker](https://hub.docker.com/r/ifansnek/piper-train-docker) for great working Docker image to train a voice (glad I didnt have to make this)
- [Optimus Prime Voicelines Dataset](https://huggingface.co/datasets/srinivasbilla/optimus_prime_voicelines_hf) by [@srinivasbilla](https://huggingface.co/srinivasbilla)

📁 Project Structure
```
.
├── Dockerfile
├── docker-compose.yml
├── prepare_dataset.py
├── requirements.txt
├── output/
│   ├── train.parquet         # Downloaded from Hugging Face
│   ├── wavs/                 # Extracted WAV files
│   └── metadata.csv          # Piper-compatible metadata
├── checkpoints/
│   └── lessac.ckpt           # Base checkpoint for transfer learning
└── cache/                    # (Optional) Torch cache directory

```
📥 Step 1: Download Dataset & Checkpoint
```
# Create directories
mkdir -p output checkpoints

# Download Optimus Prime dataset
wget https://huggingface.co/datasets/srinivasbilla/optimus_prime_voicelines_hf/resolve/main/data/train-00000-of-00001.parquet \
     -O output/train.parquet

# Download base checkpoint
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/high/epoch%3D2218-step%3D838782.ckpt \
     -O checkpoints/lessac.ckpt

```
🧪 Step 2: Create files
🐳 docker-compose.yml
```
version: '3.8'
services:
  optimus-prime-prep:
    build:
      context: .
    volumes:
      - ./output:/app/output
```
📦 Dockerfile
```
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prepare_dataset.py .

CMD ["python3", "prepare_dataset.py"]
```
📜 requirements.txt
```
pandas
pyarrow
wget
soundfile
```
🧰 prepare_dataset.py
```

import os
import pandas as pd

# Paths
output_dir = "/app/output"
wav_dir = os.path.join(output_dir, "wavs")
metadata_file = os.path.join(output_dir, "metadata.csv")
parquet_file = os.path.join(output_dir, "train.parquet")

# Ensure output dirs
os.makedirs(wav_dir, exist_ok=True)

print("Reading parquet...")
df = pd.read_parquet(parquet_file)

with open(metadata_file, "w") as f:
    for idx, row in df.iterrows():
        audio_bytes = row["audio"]["bytes"]
        text = row["text"].strip()

        wav_filename = f"{idx}.wav"
        wav_path = os.path.join(wav_dir, wav_filename)

        with open(wav_path, "wb") as wf:
            wf.write(audio_bytes)

        f.write(f"{wav_filename}|{text}\n")

print("✅ Dataset prepared.")
```
🧪 Step 3: Prepare Dataset
This uses Docker to convert the .parquet file to .wav + metadata.csv:
```
docker-compose run --rm optimus-prime-prep
```

🏋️ Step 4: Train the Model


🗣️ Run the following Docker command to train:

```
docker run --gpus all --rm -it \
  -v $(pwd)/output:/dataset \
  -v $(pwd)/checkpoints:/base_checkpoints \
  -v $(pwd)/cache:/cache \
  -e CHECKPOINT=lessac.ckpt \
  -e PYTORCH_JIT=0 \
  -e PYTORCH_NO_CUDA_MEMORY_STATS=1 \
  -e TORCH_LOAD_DEPRECATION_WARNING=0 \
  --shm-size=32g \
  ifansnek/piper-train-docker \
  bash -c "python3 -m piper_train.train \
    --precision 16 \
    --accumulate-grad-batches 2 \
    --learning-rate 1e-4 \
    --batch-size 32 \
    --gradient-clip-val 1.0"
```

📊 View TensorBoard
Once training starts, open:
```
http://localhost:6006
```

You'll see graphs under the "Scalars" tab for:

Metric	Meaning
- train/loss	Training loss (should gradually decrease)
- val/loss	Validation loss (ideally lower and smoother than training loss)
- train/lr	Learning rate (if using a scheduler)

⏱️ Estimating Training Time
Each step or epoch's duration will be printed in the terminal.

To estimate total time:

Note the time for one step in the logs, e.g.:
```
Global step 500: train/loss = 0.3462 (time: 2.1s/step)
Multiply by the total steps (e.g., 200,000 steps):
2.1s × 200,000 ≈ 116 hours (or ~5 days)
```

Training can be stopped early once validation loss plateaus or starts rising (overfitting). The model usually gets good results within 30k–100k steps, depending on data size and quality.
