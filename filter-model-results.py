import pandas
import spacy
import plac
from pathlib import Path

nlp = spacy.load('./output_v6')


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

# Takes a book title and a candidate title, returns true if it's a viable title
# false if otherwise
def test_candidate(cand, title):
    cand_doc = nlp(cand)
    title_doc = nlp(title)
    tokens = []
    for token in title_doc:
        tokens.append(token.lower_)
        tokens.append(token.text)
        tokens.append(token.lemma_)
    for token in cand_doc:
        if token.lower_ not in tokens and token.lower_ not in STOP_WORDS:
            return False
    return True


@plac.annotations(
    input_file=("Input file for filtering", "option", "o", str)
)
# Main method filters the results of the Book-trained NER (from test-book-model.py)
def main(input_file='test-blanks-v6.csv'):
    data1 = pandas.read_csv(input_file)
    title_col = []
    for i in range(len(data1)):
        new_titles = []
        titles = []
        start = data1['Start'][i][1:len(data1['Start'][i])-1]
        starts = start.split(',')
        end = data1['End'][i][1:len(data1['End'][i])-1]
        ends = end.split(',')
        if starts != ['']:

            # list of title candidates
            for s in range(len(starts)):
                titles.append(data1['Answer'][i][int(starts[s]):int(ends[s])])
        for title in titles:
            if test_candidate(title, data1['TextTitle'][i]):
                new_titles.append(title)
        title_col.append(new_titles)
    data1['Titles'] = title_col
    data1.to_csv(input_file+'-filtered.csv')

if __name__ == '__main__':
    plac.call(main)
