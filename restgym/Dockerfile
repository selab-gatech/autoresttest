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
# The AUTORESTTEST_REF below uses the 50-dimensional vectors and later commits the 300-dimensional
# ones, so both are included until the ref moves past that change.
ENV GENSIM_DATA_DIR=/opt/gensim-data
RUN for vectors in \
        glove-wiki-gigaword-50:5c55f98957aa9fed8d2ac5fb1dcff57af3b23c5a3ee7af3f7945f8d49198eb24 \
        glove-wiki-gigaword-300:0a7aebbe49097dc6e5ffff7a25e9aa20181a6e862050ca68bd2f36e056739e00; do \
        name="${vectors%%:*}" && file="$GENSIM_DATA_DIR/$name/$name.gz" \
        && mkdir -p "$GENSIM_DATA_DIR/$name" \
        && curl -fL --retry 5 --retry-all-errors -o "$file" \
            "https://github.com/RaRe-Technologies/gensim-data/releases/download/$name/$name.gz" \
        && echo "${vectors#*:}  $file" | sha256sum -c - || exit 1; \
    done

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

# Save the 300-dimensional vectors in gensim's format, where AutoRestTest looks for them
# (keyed-vectors/ beside the download). They then load in under a second instead of being
# parsed for about 20 s at every start.
RUN name=glove-wiki-gigaword-300 && dir="$GENSIM_DATA_DIR/$name" \
    && mkdir -p "$dir/keyed-vectors" \
    && python -c "import sys; from gensim.models import KeyedVectors; KeyedVectors.load_word2vec_format(sys.argv[1]).save(sys.argv[2], separately=['vectors'])" \
        "$dir/$name.gz" "$dir/keyed-vectors/$name.kv"

COPY ./tools/autoresttest/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# The base image's health check probes port 8080, which on the host network belongs to someone else.
HEALTHCHECK NONE
ENTRYPOINT ["/entrypoint.sh"]
