"""Find species descriptions."""

import argparse
import csv
import itertools
import numpy  # type: ignore
import regex as re  # type: ignore
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from sklearn.naive_bayes import BernoulliNB  # type: ignore
from sklearn.dummy import DummyClassifier  # type: ignore
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
from sklearn.calibration import CalibratedClassifierCV  # type: ignore
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, RidgeClassifierCV, SGDClassifier, LogisticRegression  # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score  # type: ignore
from sklearn.multiclass import OneVsRestClassifier  # type: ignore
from sklearn.svm import *  # type: ignore
import pandas  # type: ignore

import paragraph
from paragraph import Paragraph
from line import Line
from file import read_files
from label import Label
from taxon import Taxon, group_paragraphs

SEED=12345

def parse_annotated(contents: Iterable[Line]) -> Iterator[Paragraph]:
    """Return paragraphs in annotated block form.

    Do not apply heuristic methods to divide paragraphs."""
    pp = Paragraph()
    for line in contents:
        pp.append_ahead(line)

        if line.contains_start():
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue

        if pp.last_line and pp.last_line.end_label() is not None:
            (retval, pp) = pp.next_paragraph()
            yield retval
            continue


def parse_paragraphs(contents: Iterable[Line]) -> Iterator[Paragraph]:
    pp = Paragraph()
    for line in contents:
        pp.append_ahead(line)

        next_pp = pp.split_at_nomenclature()
        if next_pp:
            if not pp.is_empty():
                yield pp
            (retval, pp) = next_pp.next_paragraph()
            yield retval
            continue

        # New document triggers a new paragraph.
        if pp.last_line and pp.last_line.filename != line.filename:
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Page break triggers a new paragraph.
        if line.startswith(''):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Page break is a whole paragraph.
        if pp.is_page_header():
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Leading tab triggers a new paragraph.
        if line.startswith('\t'):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Tables start with a few long lines and
        # continue to grow as long as we have short lines.
        if pp.is_table():
            if line.is_short(pp.short_line):
                continue
            else:
                if pp.is_all_long():
                    continue
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Blocks of blank lines are a paragraph.
        if pp.is_blank():
            if line.is_blank():
                continue
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Figures end with a blank line, or period or colon at the end
        # of a line.
        if pp.is_figure():
            if (not line.is_blank() and
                not pp.detect_period() and
                not pp.endswith(':')):
                continue
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Leading hyphen triggers a new paragraph.
        if line.startswith('-'):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # A table starts a new paragraph.
        if pp.next_line.is_table():
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # Synonymy reference ends a taxon.
        if pp.last_line and pp.last_line.search(r'\([Ss]yn.*\)$'):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # A taxon ends in nov., nov. comb., nov. sp., ined.,
        # emend. (followed by emender), or nom. sanct.
        if pp.last_line and pp.last_line.search(
                r'(nov\.|nov\.\s?(comb\.|sp\.)|[(]?in\.?\s?ed\.[)]?|'
                r'[(]?nom\.\s?sanct\.[)]?|emend\..*)$'
        ):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # A short line ends a paragraph.
        if pp.last_line and pp.last_line.is_short(pp.short_line):
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

        # A blank line ends a paragraph.
        if line.is_blank():
            (retval, pp) = pp.next_paragraph()
            if not retval.is_empty():
                yield retval
            continue

    pp.close()
    if not pp.is_empty():
        yield pp


def remove_interstitials(paragraphs: Iterable[Paragraph]) -> Iterator[Paragraph]:
    for pp in paragraphs:
        if (pp.is_blank() or
            pp.is_figure() or
            pp.is_page_header() or
            pp.is_table()
        ):
            continue
        yield(pp)


def target_classes(paragraphs: Iterable[Paragraph],
                   default: Label,
                   keep: List[Label]) -> Iterator[Paragraph]:
    for pp in paragraphs:
        if pp.top_label() in keep:
            yield pp
            continue
        yield pp.replace_labels([default])


def perform(classifiers, vectorizers, train_data, test_data):
    for classifier in classifiers:
      for vectorizer in vectorizers:
        string = ''
        string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

        numpy.random.seed(SEED)

        start = time.time()
        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # score
        vectorize_text = vectorizer.transform(test_data.v2)
        score = classifier.score(vectorize_text, test_data.v1)
        string += '. Has score: ' + str(score)
        end = time.time()
        string += ' elapsed time ' + str(end - start)
        print(string)


def perform_confusion_matrix(classifiers, vectorizers, train_data, test_data, emit_csv: bool):
    if emit_csv:
        print('classifier,vectorizer,time,label,precision,recall,f1,support')

    for classifier in classifiers:
      for vectorizer in vectorizers:
        numpy.random.seed(SEED)

        start = time.time()
        # train
        vectorize_text = vectorizer.fit_transform(train_data.v2)
        classifier.fit(vectorize_text, train_data.v1)

        # Build the confusion matrix.
        transformed_text = vectorizer.transform(test_data.v2)
        predicted_labels = classifier.predict(transformed_text)
        end = time.time()
        elapsed_time = end - start
        cm = confusion_matrix(test_data.v1, predicted_labels)
        if emit_csv:
            print(csv_report(test_data, elapsed_time, predicted_labels, cm,
                             classifier.__class__.__name__,
                             vectorizer.__class__.__name__)
            )
        else:
            print(human_report(test_data, elapsed_time, predicted_labels, cm,
                               classifier.__class__.__name__,
                               vectorizer.__class__.__name__))


def human_report(test_data: pandas.core.frame.DataFrame,
                 elapsed_time: float,
                 predicted_labels: numpy.ndarray,
                 cm: numpy.ndarray,
                 classifier_name: str,
                 vectorizer_name: str) -> str:
    string = ''
    string += classifier_name + ' with ' + vectorizer_name

    cr = classification_report(test_data.v1, predicted_labels)
    string += ' elapsed time ' + str(elapsed_time)
    string += '\n' + str(cr)
    string += '\nConfusion matrix\n' + str(cm)
    return string


def csv_report(test_data: pandas.core.frame.DataFrame,
               elapsed_time: float,
               predicted_labels: numpy.ndarray,
               cm: numpy.ndarray,
               classifier_name: str,
               vectorizer_name: str) -> str:
    result = []
    u = numpy.unique(test_data.v1,  return_counts=True)
    labels = u[0]
    support = u[1]
    precision = precision_score(test_data.v1, predicted_labels, average = None)
    recall = recall_score(test_data.v1, predicted_labels, average = None)
    f1 = f1_score(test_data.v1, predicted_labels, average = None)

    for i in range(len(labels)):
        result.append(
            '{classifier},{vectorizer},{time:f},'
            '{label},{precision},{recall},{f1},{support}'.format(
            classifier=classifier_name,
            vectorizer=vectorizer_name,
            time=elapsed_time,
            label=labels[i],
            precision=precision[i],
            recall=recall[i],
            f1=f1[i],
            support=support[i]
        ))
    return '\n'.join(result)


def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, nargs='+', help='the file to search for descriptions')
    # Actions
    parser.add_argument(
        '--dump_phase',
        help='Dump the output of these phases and exit.',
        default=[], type=int, action='append')
    parser.add_argument(
        '--dump_files',
        help='Dump lists of files to process.',
        action='store_true')
    parser.add_argument(
        '--test_classifiers',
        help='Test a set of classifiers against the input files.',
        action='store_true')
    parser.add_argument(
        '--test_classifiers_by_label',
        help='Test a set of classifiers against the input files, reporting by label.',
        action='store_true')
    parser.add_argument(
        '--output_annotated',
        help='Output YEDDA-annotated file.',
        action='store_true')
    parser.add_argument(
        '--group_paragraphs',
        help='Group Nomenclature paragraphs with matching Description paragraphs.',
        action='store_true')
    # Control options
    parser.add_argument(
        '--reinterpret',
        help='Append reinterpretations of various elements. Values={suffix, latinate, punctuation, year, abbrev}.',
        default=[], type=str, action='append')
    parser.add_argument(
        '--classifier',
        help='Which classifier should we use for actual runs?',
        type=str, default='RandomForestClassifier')
    parser.add_argument(
        '--vectorizer',
        help='Which vectorizer should we use for actual runs?',
        type=str, default='TfidfVectorizer')
    parser.add_argument(
        '--keep_interstitials',
        help='Keep figures, tables, and blanks.',
        action='store_true')
    parser.add_argument(
        '--fast',
        help='Skip slower vectorizers and classifiers.',
        action='store_true')
    parser.add_argument(
        '--suppress_text',
        help='Suppress raw text. Evaluate only reinterpreted text.',
        action='store_true')
    parser.add_argument(
        '--label',
        default=[],
        help='Labels to retain for training purposes.',
        type=str,
        action='append')
    parser.add_argument(
        '--output_label',
        default=[],
        help='Labels to output.',
        type=str,
        action='append')
    parser.add_argument(
        '--annotated_paragraphs',
        help='Use paragraph boundaries as annotated, not the heuristic boundaries.',
        action='store_true')
    parser.add_argument(
        '--insert_nomenclature',
        help='Use regex to convert some paragraphs to Nomenclature.',
        action='store_true')
    parser.add_argument(
        '--csv',
        help='In test_classifiers_by_label, emit a csv.',
        action='store_true')

    args = parser.parse_args()

    if not args.label:
        args.labels = ['Nomenclature', 'Description']

    args.output_labels = [Label(l) for l in args.output_label]

    # This makes for a significant increase in Nomenclature scores.
    # Specifically, with RandomForestClassifier and TfidfVectorizer,
    # we see +3% over the best previous classifier.
    args.reinterpret.append('punctuation')

    try:
        i = args.file.index('evaluate')
        args.training_files = args.file[:i]
        args.evaluate_files = args.file[i+1:]
    except ValueError:
        args.training_files = args.file
        args.evaluate_files = []

    return args


def main():
    args = define_args()

    Paragraph.set_reinterpretations(args.reinterpret)

    if args.dump_files:
        print('\ntraining_files:', args.training_files)
        print('\nevaluate_files:', args.evaluate_files)

    classifiers = [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        KNeighborsClassifier()
    ]
    vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        HashingVectorizer()
    ]

    fast_classifiers = [
        BernoulliNB(),
        RandomForestClassifier(n_estimators=100, n_jobs=-1),
        AdaBoostClassifier(),
        # BaggingClassifier(),
        ExtraTreesClassifier(),
        GradientBoostingClassifier(),
        DecisionTreeClassifier(),
        CalibratedClassifierCV(),
        DummyClassifier(),
        PassiveAggressiveClassifier(),
        RidgeClassifier(),
        # RidgeClassifierCV(),
        SGDClassifier(),
        OneVsRestClassifier(SVC(kernel='linear')),
        OneVsRestClassifier(LogisticRegression()),
        # KNeighborsClassifier()  # Actually not slow, but we run out of memory.
    ]
    fast_vectorizers = [
        CountVectorizer(),
        TfidfVectorizer(),
        # HashingVectorizer()
    ]

    if args.fast:
        classifiers = fast_classifiers
        vectorizers = fast_vectorizers
    try:
        i = [c.__class__.__name__ for c in classifiers].index(args.classifier)
    except ValueError:
        raise ValueError('Unknown classifier %s' % args.classifier)
    classifier = classifiers[i]

    try:
        i = [v.__class__.__name__ for v in vectorizers].index(args.vectorizer)
    except ValueError:
        raise ValueError('Unknown vectorizer %s' % args.vectorizer)
    vectorizer = vectorizers[i]

    contents = read_files(args.training_files)

    if args.annotated_paragraphs:
        phase1 = parse_annotated(contents)
    else:
        phase1 = parse_paragraphs(contents)

    if 1 in args.dump_phase:
        print('Phase 1')
        print('=======')
        phase1 = list(phase1)
        print(repr(phase1))
        if 1 == max(args.dump_phase):
            sys.exit(0)

    if args.keep_interstitials:
        phase2 = phase1
    else:
        phase2 = remove_interstitials(phase1)
    phase1 = None  # Potentially recover memory.

    if 2 in args.dump_phase:
        print('Phase 2')
        print('=======')
        phase2 = list(phase2)
        print(repr(phase2))
        if 2 == max(args.dump_phase):
            sys.exit(0)

    # All labels need to be resolved for this phase. The easiest way
    # to assure this is to convert to list.
    phase3 = target_classes(
        list(phase2),
        default=Label('Misc-exposition'),
        keep=[Label(l) for l in args.labels]
    )

    phase2 = None

    if 3 in args.dump_phase:
        print('Phase 3')
        print('=======')
        phase3 = list(phase3)
        print(repr(phase3))
        if 3 == max(args.dump_phase):
            sys.exit(0)

    phase3 = list(phase3)
    sample_size = len(phase3)

    if args.group_paragraphs:
        writer = csv.DictWriter(sys.stdout, fieldnames=Taxon.FIELDNAMES)
        writer.writeheader()
        for taxon in group_paragraphs(phase3):
            for d in taxon.dictionaries():
                writer.writerow(d)
        sys.exit(0)

    numpy.random.seed(SEED)
    cutoff = int(sample_size * 0.70)
    permutation = numpy.random.permutation(phase3)
    phase3 = None
    learn = paragraph.to_dataframe(permutation[:cutoff], args.suppress_text)
    test = paragraph.to_dataframe(permutation[cutoff:], args.suppress_text)

    if args.test_classifiers:
        perform(
            classifiers,
            vectorizers,
            learn,
            test
        )
        sys.exit(0)

    if args.test_classifiers_by_label:
        perform_confusion_matrix(
            classifiers,
            vectorizers,
            learn,
            test,
            emit_csv=args.csv
        )
        sys.exit(0)

    phase4 = []
    if args.evaluate_files:
        # train
        vectorize_text = vectorizer.fit_transform(learn.v2)
        classifier.fit(vectorize_text, learn.v1)

        # predict
        if args.keep_interstitials:
            evaluated = (
                parse_paragraphs(read_files(args.evaluate_files)))
        else:
            evaluated = remove_interstitials(
                parse_paragraphs(read_files(args.evaluate_files)))
        for pp in evaluated:
            text = str(pp)
            vectorize_text = vectorizer.transform([text])
            predict = classifier.predict(vectorize_text)[0]
            if args.insert_nomenclature and pp.contains_nomenclature():
                predict = 'Nomenclature'
            phase4.append(pp.replace_labels(labels=[Label(predict)]))


        if args.output_annotated:
            if not args.output_labels:
                print('\n'.join([pp.as_annotated() for pp in phase4]))
            else:
                print('\n'.join([pp.as_annotated()
                                 for pp in phase4
                                 if pp.top_label() in args.output_labels]))

    if 4 in args.dump_phase:
        print('Phase 4')
        print('=======')
        print(repr(phase4))
        if 4 == max(args.dump_phase):
            sys.exit(0)



if __name__ == '__main__':
    main()
