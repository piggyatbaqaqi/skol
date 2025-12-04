FROM quay.io/jupyter/pyspark-notebook:spark-3.5.3

# Elevate to root
USER root

RUN apt-get update -y && apt-get install -y \
    curl \ 
    wget \
    iputils-ping \
    lsb-release \
    unzip \
    iproute2 

# minio 
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc -o /usr/local/bin/mc && \ 
    chmod a+x /usr/local/bin/mc

# # back to jovyan 
USER jovyan    

# pip installs
RUN pip install --upgrade --user rise folium selenium plotly chart-studio seaborn \
    minio import-ipynb markdown xlrd pycode-similar polars duckdb ipynb_path \
    beautifulsoup4 zss lxml geocoder openai redis confluent-kafka
    
# pip install jupyterlab extensions and server extensions
RUN pip install --user jupyterlab_rise jupyterlab-lsp ipylab \ 
    python-lsp-server[rope] ipywidgets ipympl jupyterlab_executor jupyterlab-git
    # jupyterlab_code_formatter 
    # jedi-language-server 

 # create ipython 
RUN ipython profile create default


