import pandas
import spacy
import plac
from pathlib import Path
from ner_book_titles_v6 import trim_entity_spans, retokenize_docs
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.gold import GoldParse

@plac.annotations(
    key_file=("Key dataset to compare to", "option", "nm", str),
    input_dir=("Input directory for model", "option", "o", Path)
)
def main(key_file='mturk-results-v6.csv', input_dir='./output_v6'):
    nlp = spacy.load(input_dir)
    Doc.set_extension("ents", default=[])
    scorer = Scorer()

    # read in key file
    key = pandas.read_csv(key_file).tail(300).reset_index(drop=True)

    # turn Answer passages into docs, then retokenize to act as an 'Answer Key' set
    docs = list(nlp.pipe(key['Answer']))
    keys = retokenize_docs(key, docs)

    # for each doc, score the book model for accuracy compared to key
    for i in range(len(keys)):
        doc, ents = keys[i]
        doc = nlp(doc)
        print(doc)
        print(ents['entities'])
        scorer.score(doc, GoldParse(doc=doc, entities=ents['entities']))

    # print score -- pay attention to 'BOOK' type
    print(scorer.ents_per_type)

if __name__ == '__main__':
    plac.call(main)
