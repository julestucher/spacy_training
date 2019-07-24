import spacy
from ner_book_titles_v6 import trim_entity_spans, retokenize_docs
import pandas
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.gold import GoldParse


def main():
# test the trained model
    nlp = spacy.load('./output_v6')
    Doc.set_extension("ents", default=[])

    data1 = pandas.read_csv('mturk-blanks.csv').reset_index(drop=True)
    keys = list(nlp.pipe(data1['Answer']))
    #keys = retokenize_docs(data1, keys)

    print(keys)
    scorer = Scorer()

    test_ans = {'Answer': [], 'TextTitle': [], 'Start': [], 'End': [], 'Titles': []}
    for i in range(len(keys)):
        print(i)
        print(keys[i])
        doc = keys[i]
        #doc = nlp(doc)
        #scorer.score(doc, GoldParse(doc=doc, entities=ents['entities']))
        print("Entities in '%s'" % doc.text)
        test_ans['Answer'].append(doc.text)
        test_ans['TextTitle'].append(data1['TextTitle'][i])
        test_ans['Start'].append([])
        test_ans['End'].append([])
        test_ans['Titles'].append([])
        for ent in doc.ents:
            if ent.label_ == 'BOOK':
                print(ent.label_, ent.text)
                test_ans['Start'][len(test_ans['Answer'])-1].append(ent.start_char)
                test_ans['End'][len(test_ans['Answer'])-1].append(ent.end_char)
                test_ans['Titles'][len(test_ans['Answer'])-1].append(ent.text)
    test_ans = pandas.DataFrame(data=test_ans)
    test_ans.to_csv('test-blanks-v6.csv')

    print(scorer.ents_per_type)

if __name__ == '__main__':
    main()
