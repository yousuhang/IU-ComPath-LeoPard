FROM --platform=linux/amd64 pytorch/pytorch

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

#FROM --platform=linux/amd64 gcc
#gcc/gcc:4.9
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
#ENV PYTHONUNBUFFERED 1


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libopenslide0 \
#    curl \
#    ca-certificates \
#    sudo \
#    git \
#    bzip2 \
#    libx11-6 \
    gcc \
 && rm -rf /var/lib/apt/lists/*
#RUN mkdir /app
WORKDIR /opt/app

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

#RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
# && chown -R user:user /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#
#USER user

ENV PYTHONUNBUFFERED 1


COPY --chown=user:user requirements.txt /opt/app/


# You can add any Python dependencies to requirements.txt
# RUN apt update && apt install gcc
RUN python -m pip install wheel
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt



COPY --chown=user:user resources/ /opt/app/resources
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
