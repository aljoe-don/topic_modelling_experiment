import logging
logging.basicConfig(level=logging.INFO, file='info.log')

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import json

from nlp_utilities import TextCleaner


app = FastAPI()
logger = logging.getLogger()

# Initialize model
lda = gensim.models.LdaModel.load('data/lda_model')
dictionary = gensim.corpora.dictionary.Dictionary.load_from_text('data/lda_dict.txt')
logger.log('Model Ready')


class Data(BaseModel):
    id_: str
    text: str


@app.post("/process")
def process(data: Data):

    data = data.dict()
    text = data['text']

    # Clean incoming text data to prep for model
    tc = TextCleaner()
    text = tc.remove_artifacts(text)
    text = tc.prepare_text_data(text)

    # Get LDA embedding for text sample
    bow = dictionary.doc2bow(text)
    lda_embedding = lda[bow]

    logger.log('Model prediction complete')


    return {'id': data['id_'], 'embedding': json.dumps(lda_embedding)}


if __name__=="__main__":
    uvicorn.run('main:app')


