from spacy.tokens import Doc
import spacy
import pandas
import json


LABEL = "BOOK"

nlp = spacy.load('en_core_web_md')
data1 = pandas.read_csv('title-id-mturk-results-7-12.csv')
data = data1.head(50)
test = pandas.read_csv('title-id-mturk-results-7-12.csv').tail(len(data1)-5000)

TRAIN_DATA = []

def main():
    ner = nlp.get_pipe('ner')
    ner.add_label(LABEL)

    #define ents extentsion for Docs
    Doc.set_extension("ents", default=[])

    docs = list(nlp.pipe(data['Answer']))
    docs = retokenize_docs(docs)

    with open('data.json', 'w') as outfile:
        json.dump(TRAIN_DATA, outfile)


def retokenize_docs(docs):
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
            ent_dict = {'entities': doc._.ents}
            inst = (doc.text, ent_dict)
            TRAIN_DATA.append(inst)
    print(TRAIN_DATA)
    return docs


if __name__ == '__main__':
    main()
