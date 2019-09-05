import sys,requests

from tqdm import tqdm
import collections
import re
import jsonlines
from multiprocessing import Pool,Manager,Queue,cpu_count
import uuid
import json

Record = collections.namedtuple("Record","idx name birthplace parents")


def create_query(resource):
    return """
        SELECT ?property ?hasValue ?isValueOf
        WHERE {
          { RESOURCE ?property ?hasValue }
        }""".replace("RESOURCE",resource)

def query_dbpedia(q,epr= "http://dbpedia.org/sparql", f='application/json'):
    try:
        params = {'query': q}
        resp = requests.get(epr, params=params, headers={'Accept': f})
        return resp.json()
    except Exception as e:
        print(e, file=sys.stdout)
        raise
        
def is_birthplace(x):
    if x['property'].get('value','notfound') == 'http://dbpedia.org/ontology/birthPlace':
        return True
    return False

def is_parent(x):
    if x['property'].get('value','notfound') == 'http://dbpedia.org/ontology/parent':
        return True
    if x['property'].get('value','notfound') == 'http://dbpedia.org/property/parents':
        return True
    return False

def get_entity_name(resource):
    return resource.split("/")[-1].split(">")[0]
    
def extract_info(resource,resp):
    birthplace = list(filter(lambda x: is_birthplace(x), resp['results']['bindings']))
    parents = list(filter(lambda x: is_parent(x), resp['results']['bindings']))
    name = get_entity_name(resource)
    birthplace = [get_entity_name(x['hasValue']['value']) for x in birthplace]
    parents = [get_entity_name(x['hasValue']['value']) for x in parents]
    record = Record(idx=uuid.uuid4(),name=name,birthplace=birthplace,parents=parents)
    return record

def worker(resource_list,queue):
    for resource in resource_list:
        resp = query_dbpedia(create_query(resource))
        record= extract_info(resource,resp)
        record={"id":str(record.idx),"name":record.name,"parents":record.parents,"birth_place":record.birthplace}
        queue.put(json.dumps(record))
    return "response_list"

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def listener(queue):
    '''listens for messages on the q, writes to file. '''
    input_file = "/scratch/pbanerj6/datasets/only_persons.txt"
    with open(input_file+".jsonl",'w') as writer:
        while 1:
            m = queue.get()
            print(m)
            if m == 'kill':
                writer.write('killed')
                break
            writer.write(m+'\n')
            writer.flush()
            
def download_dbpedia():
    input_file = "/scratch/pbanerj6/datasets/xab"
    outputfile = input_file+ ".out" 
    name_count = 0

    with open(input_file,"r") as ifd:
        resourcelist = [line.strip().split(' ')[0] for line in tqdm(ifd.readlines())]
        
    manager = Manager()
    q = manager.Queue()    
    pool = Pool(cpu_count() + 2)
    watcher = pool.apply_async(listener, (q,))
    
    #fire off workers
    jobs = []
    for chunk in chunks(resourcelist,10000):
        job = pool.apply_async(worker, (chunk, q))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()


if __name__ == "__main__":
    download_dbpedia()