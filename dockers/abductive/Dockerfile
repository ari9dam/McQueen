FROM pytorch/pytorch

ENV ES_PKG_NAME elasticsearch-1.7.3
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update

# Install system level packages needed to install elastic search or dependancies.
RUN apt-get -y install software-properties-common

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

# Install Elasticsearch.
RUN wget https://download.elastic.co/elasticsearch/release/org/elasticsearch/distribution/deb/elasticsearch/2.3.1/elasticsearch-2.3.1.deb

RUN dpkg -i elasticsearch-2.3.1.deb

# Mount elasticsearch.yml config
# ADD config/elasticsearch.yml /etc/elasticsearch/elasticsearch.yml

# Expose ports.
#   - 9200: HTTP
#   - 9300: transport
EXPOSE 9200
EXPOSE 9300

RUN pip install pytorch_pretrained_bert

RUN mkdir /results

# Copy remaining code.
COPY . .

CMD ["/bin/bash"]