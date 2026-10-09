# AutoRestTest for RESTgym, with its LLM served locally by llama.cpp.
# RESTgym builds this file from its root, with this folder copied to tools/autoresttest/.
# The evaluation host is x86-64 with an NVIDIA GPU, so the platform is fixed for builds on other machines.
FROM --platform=linux/amd64 ghcr.io/ggml-org/llama.cpp:server-cuda12-b11434@sha256:c6fd6053d3b6a1a54849ab5468971cf3ae03fbf3d72fdfb25a1f862e320cace3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Google's 4-bit QAT build of Gemma 4 26B-A4B. The vision projector is skipped since prompts are text only.
ARG MODEL_REPO=google/gemma-4-26B-A4B-it-qat-q4_0-gguf
ARG MODEL_REVISION=d1c082be9cf3c8a514acf63b8761f4b41935842e
ARG MODEL_FILE=gemma-4-26B_q4_0-it.gguf
ARG MODEL_SHA256=3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d
ENV MODEL_PATH=/models/${MODEL_FILE}
RUN mkdir -p /models \
    && curl -fL --retry 5 --retry-all-errors -o "$MODEL_PATH" "https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REVISION}/${MODEL_FILE}" \
    && echo "${MODEL_SHA256}  ${MODEL_PATH}" | sha256sum -c -

# GloVe vectors for the operation graph, cached where gensim looks so no download happens at runtime.
ENV GENSIM_DATA_DIR=/opt/gensim-data
RUN mkdir -p "$GENSIM_DATA_DIR/glove-wiki-gigaword-50" \
    && curl -fL --retry 5 --retry-all-errors -o "$GENSIM_DATA_DIR/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz" \
        https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz \
    && echo "5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24  $GENSIM_DATA_DIR/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz" | sha256sum -c -

# Python 3.10, which AutoRestTest's pinned numpy and scipy require. OpenBLAS would otherwise
# start a thread per host core, far more than the container's 8-CPU quota.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=4
RUN curl -fL --retry 5 --retry-all-errors -o /tmp/uv.tar.gz https://github.com/astral-sh/uv/releases/download/0.12.23/uv-x86_64-unknown-linux-gnu.tar.gz \
    && echo "9167d72b3319674b6303c4cbe071854bba13ebdf3d76b1a7cbdc175471fb66d6  /tmp/uv.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/uv.tar.gz -C /usr/local/bin --strip-components=1 --no-same-owner \
    && rm /tmp/uv.tar.gz \
    && uv venv --python 3.10 "$VIRTUAL_ENV"

# AutoRestTest at a pinned commit, installed from poetry.lock.
ARG AUTORESTTEST_REF=19e69456f37177547b5748a4d90bd6f18783ace2
WORKDIR /tool
RUN curl -fL --retry 5 --retry-all-errors "https://github.com/selab-gatech/AutoRestTest/archive/${AUTORESTTEST_REF}.tar.gz" \
        | tar -xz --strip-components=1 --wildcards --no-wildcards-match-slash '*/src' '*/pyproject.toml' '*/poetry.lock' '*/README.md' \
    && uvx --from poetry==2.2.1 poetry install --only main --no-interaction --no-ansi \
    && rm -rf /root/.cache

COPY ./tools/autoresttest/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The base image's health check probes port 8080, which on the host network belongs to someone else.
HEALTHCHECK NONE
ENTRYPOINT ["/entrypoint.sh"]
