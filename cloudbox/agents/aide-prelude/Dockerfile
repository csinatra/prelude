FROM mlebench-env

# where to put submission.csv, will be extracted
ARG SUBMISSION_DIR
ENV SUBMISSION_DIR=${SUBMISSION_DIR}
# where to put any logs, will be extracted
ARG LOGS_DIR
ENV LOGS_DIR=${LOGS_DIR}
# where to put any code, will be extracted
ARG CODE_DIR
ENV CODE_DIR=${CODE_DIR}
# where to put any other agent-specific files, will not be necessarily extracted
ARG AGENT_DIR
ENV AGENT_DIR=${AGENT_DIR}

RUN mkdir ${LOGS_DIR} ${CODE_DIR} ${AGENT_DIR}

ARG CONDA_ENV_NAME=agent
ARG REQUIREMENTS=${AGENT_DIR}/requirements.txt

# copy just the requirements file, so that we can cache conda separately from the agent files
COPY requirements.txt ${AGENT_DIR}/requirements.txt

# Requirements for opencv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# create conda environment and install the requirements to it
RUN conda run -n ${CONDA_ENV_NAME} pip install -r ${AGENT_DIR}/requirements.txt && \
    conda clean -afy

# put all the agent files in the expected location
COPY . ${AGENT_DIR}

# Overlay our patched aideml Anthropic backend onto the pip-installed copy:
# implements the func_spec (function-calling) path upstream left as
# NotImplementedError, so the AIDE feedback step runs single-provider on Claude.
# Finds the installed file by import so there's no hardcoded site-packages path.
# See patches/backend_anthropic.py for the full file + what changed vs upstream.
RUN conda run -n ${CONDA_ENV_NAME} python -c \
    "import aide.backend.backend_anthropic as m, shutil; \
     shutil.copyfile('${AGENT_DIR}/patches/backend_anthropic.py', m.__file__)"
