# Inference

Run commands from the repository root after completing [setup](../setup/README.md).

```bash
bash run_scripts/inference/run_example.sh
```

The example uses `images/example_question.png` and prints the generated response, replacing the unreadable latent span with `<latent>`.

Common overrides:

```bash
MODEL_PATH="$PWD/models/Monet-7B" \
LATENT_SIZE=10 \
GPU_MEMORY_UTILIZATION=0.9 \
bash run_scripts/inference/run_example.sh
```

`MODEL_PATH` must contain the downloaded checkpoint, and `LATENT_SIZE` must match the intended checkpoint configuration. Lower `GPU_MEMORY_UTILIZATION` if the GPU is shared or vLLM runs out of memory.
