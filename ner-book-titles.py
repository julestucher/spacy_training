from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
from spacy.tokens import Doc
import pandas
import json
import re


LABEL = "BOOK"

nlp = spacy.load('en_core_web_md')
data1 = pandas.read_csv('title-id-mturk-results-7-12.csv')
data = data1.head(7000)
test = pandas.read_csv('title-id-mturk-results-7-12.csv').tail(len(data1)-7000)


@plac.annotations(
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(output_dir='./output', n_iter=100):
    random.seed(0)

    ner = nlp.get_pipe('ner')
    ner.add_label(LABEL)

    #define ents extentsion for Docs
    Doc.set_extension("ents", default=[])

    docs = list(nlp.pipe(data['Answer']))
    TRAIN_DATA = retokenize_docs(docs)

    TRAIN_DATA = trim_entity_spans(TRAIN_DATA)

    optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

     # test the trained model
    test_text = "Harry Potter is a good book"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

def retokenize_docs(docs):
    TRAIN_DATA = []
    for i in range(len(docs)):
        doc = docs[i]
        start = data['Start'][i][1:len(data['Start'][i])-1]
        starts = start.split(',')
        end = data['End'][i][1:len(data['End'][i])-1]
        ends = end.split(',')
        if starts != ['']:
            for s in range(len(starts)):
                starts[s] = int(starts[s])
                ends[s] = int(ends[s])
            titles = data['Titles'][i][1:len(data['Titles'][i])-1]
            titles = titles.split(',')
            for t in range(len(titles)):
                while titles[t][0:1].isalpha() == False:
                    if titles[t][0:1] not in ('"', "'"):
                        starts[t] = starts[s] + 1
                    titles[t] = titles[t][1:]
                while titles[t][len(titles[t])-1:].isalpha() == False:
                    titles[t] = titles[t][:len(titles[t])-1]
            k = 0
            while k < len(doc):
                token = doc[k]
                if token.idx in starts:
                    ind = starts.index(token.idx)
                    title = nlp(titles[ind])
                    k = token.i + len(title)
                    with doc.retokenize() as retokenizer:
                        retokenizer.merge(doc[token.i:token.i+len(title)], attrs={"ENT_TYPE": LABEL})
                    temp = (starts[ind], ends[ind], LABEL)
                    doc._.ents.append(temp)
                else:
                    if token.ent_type_ != '':
                        temp = (int(token.idx), int(token.idx) + len(token.text), token.ent_type_)
                        doc._.ents.append(temp)
                    k += 1
            print(doc._.ents)
            ent_dict = {'entities': doc._.ents}
            inst = (doc.text, ent_dict)
            TRAIN_DATA.append(inst)
    return TRAIN_DATA


def trim_entity_spans(data):
    """Removes leading and trailing white spaces from entity spans.

    Args:
        data (list): The data to be cleaned in spaCy JSON format.

    Returns:
        list: The cleaned data.
    """

    invalid_span_tokens = re.compile(r'\s')

    cleaned_data = []
    for text, annotations in data:
        entities = annotations['entities']
        valid_entities = []
        for start, end, label in entities:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and invalid_span_tokens.match(
                    text[valid_start]):
                valid_start += 1
            while valid_end > 1 and invalid_span_tokens.match(
                    text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])

    return(cleaned_data)

if __name__ == '__main__':
    plac.call(main)
