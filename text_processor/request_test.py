import requests
import logging
logging.basicConfig(level=logging.INFO, file='info.log')
logger = logging.getLogger()


with open('/GIT/lda_pipeline/text_processor/data/sample.txt') as f:
        sample = f.read()

request = {'id_': 'sample_id', 'text': sample}       
url = 'http://127.0.0.1:8000/process'
r = requests.post(url,json=request)

logger.log(r.json())