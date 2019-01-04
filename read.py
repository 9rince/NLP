import json
from pprint import pprint

with open('train-v2.0.json') as json_data:
    d = json.load(json_data)
pprint(d['data'][0]['paragraphs'].shape())


