import regex as re
import gensim
from gensim.parsing.preprocessing import STOPWORDS
import nltk

from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
# nltk.download('all')
from nltk.corpus import wordnet


class TextCleaner:
    def __init__(self):
        pass

    # Remove artifacts that arise from source system
    def remove_artifacts(self, text):
            header     = "^[\w-]+:" # All post headers
            response   = "^>|^#|^:|^\|" # Text demarked as source of response post
            reply_ref  = "^In article <|\w+\s(writes|wrote):$|\)\s(writes|wrote):$" # Other artifacts
            
            output = ''
            # If any line in text sample contains an artifact, remove it
            for line in text.split('\n'):
                if any((re.findall(header, line), re.findall(response, line), re.findall(reply_ref, line))):
                    continue 
                else:
                    output = '{} {}\n'.format(output, line)
            
            return output

    # Feed a word (with POS tag), return the lemmatization and the stem of the word
    def lemmatize_and_stem(self, word, pos_tag=wordnet.NOUN):
        return PorterStemmer().stem(WordNetLemmatizer().lemmatize(word, pos=pos_tag))


    # Feed a phrase, return list of tuples, each containing a word from phrase and POS tag
    def tag_POS(self, phrase):
        phrase = nltk.word_tokenize(phrase)
        phrase = nltk.pos_tag(phrase)
        return phrase


    # Separate contractions so that each word gets its own tag (NLTK can be clumsy if you
    # don't do this!)
    def remove_contraction(self, phrase):
        expansions = {r"he's"  : "he is"   , r"she's": "she is", r"it's" : "it is" ,
                    r"let's"  : "let us"  , r"n\'t" : " not"  , r"\'re" : " are"  ,
                    r"\'d"    : " would"  , r"\'ll" : " will" , r"\'t"  : " not"  ,
                    r"\'ve"   : " have"   , r"\'m"  : " am"}
        for k in expansions:
            phrase = re.sub(k, expansions[k], phrase)
        return(phrase)


    # Feed POS tag from NLTK, convert it to be usable by wordnet
    def get_wordnet_pos(self, treebank_tag):
        initial = treebank_tag[0]
        convert = {'J': wordnet.ADJ, 'V': wordnet.VERB, 'N': wordnet.NOUN, 'R' : wordnet.ADV}
        try:
            return convert[initial]
        except:
            return wordnet.NOUN # Default of lemmatizer argument


    # Feed text, return a list of of lemmatized, stemmed words from that text
    def prepare_text_data(self, text):
        result = []
        tokens = []
        
        text = self.tag_POS(self.remove_contraction(text.lower()))
        
        for token in text:
            tokens.append((token[0], self.get_wordnet_pos(token[1])))


        for token in tokens:
            # Dependency Issues: can't use Gensim
            if token[0] not in gensim.parsing.preprocessing.STOPWORDS and len(token[0]) >= 3:
                result.append(self.lemmatize_and_stem(token[0], token[1]))
        
        return(result)


if __name__=='__main__':

    tc = TextCleaner()

    with open('data/sample.txt') as f:
        sample = f.read()

    print(sample)
    print(tc.remove_artifacts(sample))