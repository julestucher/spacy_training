import spacy
import plac
from pathlib import Path
from ner_book_titles_v6 import trim_entity_spans, retokenize_docs
import pandas
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.gold import GoldParse

@plac.annotations(
    file_to_nlp=("List of passages for the NER", "option", "nm", str),
    input_dir=("Input directory for model", "option", "o", Path),
    output_str=("Filename for the output file", "option", "n", str),
)

def main(file_to_nlp='mturk-results-v6.csv', input_dir='./output_v6', output_str='test-mturk-v6.csv'):
    # load the trained model
    nlp = spacy.load(input_dir)

    # read the data set and pipe through nlp
    data1 = pandas.read_csv(file_to_nlp).reset_index(drop=True)
    keys = list(nlp.pipe(data1['Answer']))

    test_ans = {'Answer': [], 'TextTitle': [], 'Start': [], 'End': [], 'Titles': []}
    for i in range(len(keys)):
        doc = keys[i]
        print("Books in '%s'" % doc.text)
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
    test_ans.to_csv(output_str)

if __name__ == '__main__':
    plac.call(main)
