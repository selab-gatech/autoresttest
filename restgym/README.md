# AutoRestTest for RESTgym

This folder packages AutoRestTest as a [RESTgym](https://github.com/restgym/restgym) tool for REST League 2027, where LLMs must run locally.
The image serves Google's 4-bit QAT build of Gemma 4 26B-A4B ([`google/gemma-4-26B-A4B-it-qat-q4_0-gguf`](https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf)) with llama.cpp's `llama-server`, and AutoRestTest calls it through its usual OpenAI-compatible client.
Everything is downloaded at build time, so a run needs no network access except to the API under test.

## Files

- `Dockerfile` builds on llama.cpp's CUDA 12 server image and adds the model, GloVe vectors, Python 3.10, and AutoRestTest at a pinned commit (`AUTORESTTEST_REF`).
- `entrypoint.sh` starts `llama-server` on a free local port, writes `configurations.toml` from RESTgym's `API`, `HOST`, `PORT`, and `TIME_BUDGET` variables, waits for the model, then runs AutoRestTest in a restart loop. Graph and Q-table caching is on, so a restart reuses the setup from the same container instead of repeating it.
- `restgym-tool-config.yml` enables the tool in RESTgym.

## Install into RESTgym

Copy the three files into a plain `tools/autoresttest/` folder in RESTgym, then build with RESTgym's build script:

```bash
mkdir -p ../RESTgym/tools/autoresttest
cp restgym/Dockerfile restgym/entrypoint.sh restgym/restgym-tool-config.yml ../RESTgym/tools/autoresttest/
```

The image includes the 14.4 GB model file.

## GPU

The model is meant to run on one NVIDIA GPU with 24 GB of memory (an RTX 4090 in the competition).
RESTgym's `run.py` starts tool containers without a GPU, so the GPU has to be passed to the tool container, for example with `device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]` in `run.py`, or `--gpus all` with `docker run`.
If llama.cpp finds no CUDA device, the entrypoint does not start the model, since generating values on the CPU would take most of the hour. AutoRestTest then runs without LLM-generated values.
The tool's log starts with the devices llama.cpp found, then shows `No CUDA device found`, `Model server ready.`, or, if the server failed to load, `Model server not ready`.

To run the image by hand against an API already listening on `localhost:9090`:

```bash
docker run --rm --gpus all --network host --memory 16g --cpus 8 \
  -e API=pet-clinic -e HOST=localhost -e PORT=9090 -e TIME_BUDGET=60 \
  -v "$PWD/../RESTgym/apis/pet-clinic/specifications/pet-clinic-openapi.json:/specifications/pet-clinic-openapi.json:ro" \
  restgym-autoresttest
```

## Changing versions

The model, llama.cpp image, and AutoRestTest commit are pinned in the `Dockerfile`.
To try another GGUF model, change the `MODEL_*` build arguments, including `MODEL_SHA256`, which the build checks.
To ship newer AutoRestTest code, set `AUTORESTTEST_REF` to a pushed commit or tag.
