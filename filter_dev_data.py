import pandas
import spacy
from ner_book_titles_v6 import trim_entity_spans, retokenize_docs
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.gold import GoldParse

nlp = spacy.load('./output_v5')
Doc.set_extension("ents", default=[])

STOP_WORDS = set("""
a
about
above
after
again
against
all
am
an
and
any
are
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can't
cannot
could
couldn't
did
didn't
do
does
doesn't
doing
don't
down
during
each
few
for
from
further
had
hadn't
has
hasn't
have
haven't
having
he
he'd
he'll
he's
her
here
here's
hers
herself
him
himself
his
how
how's
i
i'd
i'll
i'm
i've
if
in
into
is
isn't
it
it's
its
itself
let's
me
more
most
mustn't
my
myself
no
nor
not
of
off
on
once
only
or
other
ought
our
ours	ourselves
out
over
own
same
shan't
she
she'd
she'll
she's
should
shouldn't
so
some
such
than
that
that's
the
their
theirs
them
themselves
then
there
there's
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
very
was
wasn't
we
we'd
we'll
we're
we've
were
weren't
what
what's
when
when's
where
where's
which
while
who
who's
whom
why
why's
with
won't
would
wouldn't
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
""".split())

def test_candidate(title, text):
    cand_doc = nlp(title)
    title_doc = nlp(text)
    tokens = []
    for token in title_doc:
        tokens.append(token.lower_)
        tokens.append(token.text)
        tokens.append(token.lemma_)
    for token in cand_doc:
        if token.lower_ not in tokens and token.lower_ not in STOP_WORDS:
            return False
    return True

def make_titles(data, i):
    start = data['Start'][i][1:len(data['Start'][i])-1]
    starts = start.split(',')
    end = data['End'][i][1:len(data['End'][i])-1]
    ends = end.split(',')
    if starts != ['']:
        titles = []
        for s in range(len(starts)):
            titles.append(data['Answer'][i][int(starts[s]):int(ends[s])])
        return titles
    else:
        return []

def main():
    data1 = pandas.read_csv('mturk-blanks.csv').reset_index(drop=True)
    print(data1)
    print(len(data1))
    scorer = Scorer()
    key = pandas.read_csv('title-id-mturk-results-7-12.csv').reset_index(drop=True)
    test = pandas.read_csv('model-full-test-v5.csv').reset_index(drop=True)

    print(key.head())
    print(test.head())
    title_col = []
    if len(key) != len(test):
        print("ERROR")
        return
    test['TextTitle'] = key['TextTitle']
    countE = 0
    for i in range(len(test)):
        new_titles = []
        titles = make_titles(test, i)
        for title in titles:
            if test_candidate(title, test['TextTitle'][i]):
                new_titles.append(title)
        title_col.append(new_titles)
    test['Titles'] = title_col
    title_col = []
    for i in range(len(data1)):
        print(i)
        new_titles = []
        titles = make_titles(data1, i)
        for title in titles:
            if test_candidate(title, data1['TextTitle'][i]):
                new_titles.append(title)
        title_col.append(new_titles)
    keys = list(nlp.pipe(data1['Answer']))
    keys = retokenize_docs(data1, keys)
    for i in range(len(keys)):
        doc, ents = keys[i]
        doc = nlp(doc)
        scorer.score(doc, GoldParse(doc=doc, entities=ents['entities']))
    print(scorer.ents_per_type)
    #  data1['Titles'] = title_col
    data1.to_csv('test-blanks.csv')

if __name__ == '__main__':
    main()
