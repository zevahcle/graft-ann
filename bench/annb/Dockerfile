FROM ann-benchmarks

RUN apt-get update && apt-get install -y --no-install-recommends g++ git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "git+https://github.com/zevahcle/graft-ann@v0.2"
RUN python3 -c "import graft; print('graft', graft.__version__)"
