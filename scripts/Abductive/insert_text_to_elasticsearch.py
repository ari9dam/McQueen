#!/usr/bin/env python
from __future__ import print_function

import json
import re
import sys

try:
  # for Python 3.0 and later
  from urllib.request import urlopen
except ImportError:
  # fallback to Python 2
  from urllib2 import urlopen

# Reads text input on STDIN, splits it into sentences, gathers groups of
# sentences and issues bulk insert commands to an Elasticsearch server running
# on localhost.
#ELASTIC_SEARCH_URL = 'http://10.218.110.98:9200/rocn/sentence/_bulk' this one with id, rocn1 is without id, rocn2 is the new pattern with sentences, #rocnew is for entire story as arindam asked, roccombined is mix of all Kbs, rocarindam for 3 sentence pairs of arindam data, rocarida new as last one got messed up by repeatation.

ELASTIC_SEARCH_URL = 'http://localhost:9200/roccombined/sentence/_bulk'
DOCUMENTS_PER_POST = 100000

def sentences_to_elasticsearch_payload(sentences):
  payload_lines = []
  for sentence in sentences:
    payload_lines += [ json.dumps({"index":{}}) ]
    payload_lines += [ json.dumps({"body":sentence}) ]
  return "\n".join(payload_lines)

def bulk_load_elasticsearch(sentences, url):
  payload = sentences_to_elasticsearch_payload(sentences)
  response_file = urlopen(url, payload.encode('utf8'))
  response = json.loads(response_file.read().decode('utf8'))
  print("Posted {0} documents ({1} bytes) to {2}. Elasticsearch errors = {3}".format(
    len(sentences),
    len(payload),
    url,
    str(response.get("errors", "?"))
  ))

def lines_to_sentences(line_stream):
  for line in line_stream:
    line_cleaned = re.sub(r'([^a-zA-Z0-9\.])', " ", line).strip()
    # for sentence in line_cleaned.split("."):
    #   if len(sentence) == 0:
    #     continue
      # yield sentence
    yield line_cleaned

def groups(stream, size):
  batch = []
  for item in stream:
    batch += [item]
    if len(batch) % size == 0:
      yield batch
      batch = []
  if len(batch) > 0:
    yield batch

def main():
  sentence_count = 0

  for sentences in groups(lines_to_sentences(sys.stdin), DOCUMENTS_PER_POST):
    bulk_load_elasticsearch(sentences, ELASTIC_SEARCH_URL)
    sentence_count += len(sentences)

  print("Documents posted:", sentence_count)

if __name__ == "__main__":
  main()
