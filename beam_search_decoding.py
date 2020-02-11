# -*- coding: utf-8 -*-

import argparse
import csv
import os.path

import numpy as np


class ArgumentError(Exception):
    """Error in start argument."""


class FormatDataError(Exception):
    """Error in format data."""


class Alphabet:
    """Getting from file and handling letters alphabet."""
    def __init__(self, file_name):
        self.file_name = file_name
        self.letters = str()

    def get_letters(self):
        """Read from file and save letters alphabet."""
        with open(self.file_name, mode='r', encoding='UTF8') as alphabet_file:
            for line in alphabet_file:
                self.letters += line.strip('\n')


class Probabilities:
    """Getting from file and handling matrix probabilities."""
    def __init__(self, file_name, required_accuracy=0.03):
        self.required_accuracy = required_accuracy
        self.file_name = file_name
        self.raw_array = [[]]
        self.array = np.array(self.raw_array)

    def create_array(self):
        """Create from csv file array with matrix probabilities."""
        self.csv_file_reader()
        self.__check_correct_format()
        self.array = np.array(self.raw_array)

    def csv_file_reader(self):
        """Read from csv file and save matrix probabilities."""
        with open(self.file_name, mode='r', newline='', encoding='UTF8') as csv_file:
            self.raw_array = [[float(cell) for cell in row] for row in csv.reader(csv_file)]

    def __check_correct_format(self):
        """Check correct format source data"""
        min_accuracy = 1 - self.required_accuracy
        max_accuracy = 1 + self.required_accuracy
        for num_line, row in enumerate(self.raw_array, start=1):
            # Sum of all numbers must be 1
            if not min_accuracy < sum(row) < max_accuracy:
                raise FormatDataError(f'The sum of the probability numbers is not 1 in row number {num_line}')

            # Number of elements in all rows should be equal
            # For first row don't compare
            if num_line != 1:
                if len(row) != reference_len_line:
                    raise FormatDataError(f'Different number of probabilities in line {num_line} and first line.')
            else:
                reference_len_line = len(row)


class Decoder:
    """Beam search as described by the paper of Hwang et al. and the paper of Graves et al."""
    def __init__(self, mat, classes, beam_width=25):
        self.result = str()
        self.mat = mat
        self.classes = classes
        self.beam_width = beam_width

    class __BeamEntry:
        """information about one single beam at specific time-step"""

        def __init__(self):
            self.pr_total = 0  # blank and non-blank
            self.pr_non_blank = 0  # non-blank
            self.pr_blank = 0  # blank
            self.pr_text = 1  # LM score
            self.labeling = ()  # beam-labeling

    class __BeamState:
        """information about the beams at specific time-step"""

        def __init__(self):
            self.entries = {}

        def norm(self):
            """length-normalise LM score"""
            for (k, _) in self.entries.items():
                labeling_len = len(self.entries[k].labeling)
                self.entries[k].pr_text = self.entries[k].pr_text ** (1.0 / (labeling_len if labeling_len else 1.0))

        def sort(self):
            """return beam-labelings, sorted by probability"""
            beams = [v for (_, v) in self.entries.items()]
            sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total * x.pr_text)
            return [x.labeling for x in sorted_beams]

    def __call__(self):
        """Beam search as described by the paper of Hwang et al. and the paper of Graves et al."""

        mat = self.mat
        beam_width = self.beam_width
        blank_idx = len(self.classes)
        max_t, max_c = mat.shape

        # initialise beam state
        last = self.__BeamState()
        labeling = ()
        last.entries[labeling] = self.__BeamEntry()
        last.entries[labeling].pr_blank = 1
        last.entries[labeling].pr_total = 1

        # go over all time-steps
        for t in range(max_t):
            curr = self.__BeamState()

            # get beam-labelings of best beams
            best_labelings = last.sort()[0:beam_width]

            # go over best beams
            for labeling in best_labelings:

                # probability of paths ending with a non-blank
                pr_non_blank = 0
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    pr_non_blank = last.entries[labeling].pr_non_blank * mat[t, labeling[-1]]

                # probability of paths ending with a blank
                pr_blank = last.entries[labeling].pr_total * mat[t, blank_idx]

                # add beam at current time-step if needed
                self.__add_beam(curr, labeling)

                # fill in data
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].pr_non_blank += pr_non_blank
                curr.entries[labeling].pr_blank += pr_blank
                curr.entries[labeling].pr_total += pr_blank + pr_non_blank
                curr.entries[labeling].pr_text = last.entries[
                    labeling].pr_text  # beam-labeling not changed, therefore also LM score unchanged from

                # extend current beam-labeling
                for c in range(max_c - 1):
                    # add new char to current beam-labeling
                    new_labeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        pr_non_blank = mat[t, c] * last.entries[labeling].pr_blank
                    else:
                        pr_non_blank = mat[t, c] * last.entries[labeling].pr_total

                    # add beam at current time-step if needed
                    self.__add_beam(curr, new_labeling)

                    # fill in data
                    curr.entries[new_labeling].labeling = new_labeling
                    curr.entries[new_labeling].pr_non_blank += pr_non_blank
                    curr.entries[new_labeling].pr_total += pr_non_blank

            # set new beam state 9
            last = curr

        # normalise LM scores according to beam-labeling-length
        last.norm()

        # sort by probability
        best_labeling = last.sort()[0]  # get most probable labeling

        # map labels to chars
        for label in best_labeling:
            self.result += self.classes[label]

    def __add_beam(self, beam_state, labeling):
        """add beam if it does not yet exist"""
        if labeling not in beam_state.entries:
            beam_state.entries[labeling] = self.__BeamEntry()

    def faster_search(self):
        """More faster beam search as described by the paper of Hwang et al. and the paper of Graves et al.

        It's variant work about 20% faster.
        """
        mat = self.mat
        beam_width = self.beam_width
        blank_idx = len(self.classes)
        max_t, max_c = mat.shape

        # initialise beam state
        labeling = ()
        last = {
            labeling: {
                'pr_total': 1,
                'pr_non_blank': 0,
                'pr_blank': 1,
                'pr_text': 1,
                'labeling': (),
            }
        }

        # go over all time-steps
        for t in range(max_t):
            curr = dict()
            best_labelings = self.__sort(last)[0:beam_width]

            # go over best beams
            for labeling in best_labelings:

                # probability of paths ending with a non-blank
                pr_non_blank = 0
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    pr_non_blank = last[labeling]['pr_non_blank'] * mat[t, labeling[-1]]

                # probability of paths ending with a blank
                pr_blank = last[labeling]['pr_total'] * mat[t, blank_idx]

                # add beam at current time-step if needed
                self.__add_beam_faster(curr, labeling)

                # fill in data
                curr[labeling]['labeling'] = labeling
                curr[labeling]['pr_non_blank'] += pr_non_blank
                curr[labeling]['pr_blank'] += pr_blank
                curr[labeling]['pr_total'] += pr_blank + pr_non_blank
                curr[labeling]['pr_text'] = last[labeling]['pr_text']

                # extend current beam-labeling
                for c in range(max_c - 1):
                    # add new char to current beam-labeling
                    new_labeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        pr_non_blank = mat[t, c] * last[labeling]['pr_blank']
                    else:
                        pr_non_blank = mat[t, c] * last[labeling]['pr_total']

                    # add beam at current time-step if needed
                    self.__add_beam_faster(curr, new_labeling)

                    # fill in data
                    curr[new_labeling]['labeling'] = new_labeling
                    curr[new_labeling]['pr_non_blank'] += pr_non_blank
                    curr[new_labeling]['pr_total'] += pr_non_blank

            # set new beam state 9
            last = curr

        # normalise LM scores according to beam-labeling-length
        for (k, _) in last.items():
            labeling_len = len(last[k]['labeling'])
            last[k]['pr_text'] = last[k]['pr_text'] ** (1.0 / (labeling_len if labeling_len else 1.0))

        self.result = ''.join(map(
            lambda label: self.classes[label], self.__sort(last)[0]
        ))

    @staticmethod
    def __add_beam_faster(beam_state, labeling):
        """add beam if it does not yet exist"""
        if labeling not in beam_state:
            beam_state[labeling] = {
                'pr_total': 0,
                'pr_non_blank': 0,
                'pr_blank': 0,
                'pr_text': 1,
                'labeling': (),
            }

    @staticmethod
    def __sort(entries):
        """return beam-labelings, sorted by probability"""
        sorted_beams = sorted(entries.values(), reverse=True, key=lambda x: x['pr_total'] * x['pr_text'])
        return [x['labeling'] for x in sorted_beams]


class StartupArgs:
    """Startup key arguments handling. Get command line arguments, check it and return dict with startup parameters."""
    def __init__(self):
        self.__kwargs = {}
        self.alphabet_file_name = ''
        self.matrix_file_name = ''

    def fetch(self):
        """Fetch arguments and check it."""
        self.__parse_arg_params()
        self.__check_args()
        self.alphabet_file_name = self.__kwargs.get('alphabet_file_name', '')
        self.matrix_file_name = self.__kwargs.get('matrix_file_name', '')

    def __parse_arg_params(self):
        """Startup key getting. Get command line arguments and return dict with startup parameters."""
        cmd_parser = argparse.ArgumentParser(
            description='Script implements the algorithm basic version of beam search decoding.'
        )
        cmd_parser.add_argument('-a',
                                dest='alphabet_file_name',
                                required=True,
                                help='File with alphabet. Need write path (optional) and file name. '
                                     'Example: "alphabet.txt"'
                                )
        cmd_parser.add_argument('-m',
                                dest='matrix_file_name',
                                required=True,
                                help='File name matrix of probabilities. Need write path (optional) and file name. '
                                     'Example: "probabilities.csv"'
                                )
        self.__kwargs = vars(cmd_parser.parse_args())

    def __check_args(self):
        """Check startup key arguments."""
        for file_name, file_path in self.__kwargs.items():
            if not os.path.exists(file_path) or os.path.getsize(file_path) < 2:
                raise ArgumentError(f'Argument "{file_name}" don\'t exists or empty')


def main():
    args = StartupArgs()
    args.fetch()

    alphabet = Alphabet(args.alphabet_file_name)
    alphabet.get_letters()

    matrix_probabilities = Probabilities(args.matrix_file_name)
    matrix_probabilities.create_array()

    beam_search = Decoder(matrix_probabilities.array, alphabet.letters)
    beam_search()
    # For fast search use:
    # beam_search.faster_search()
    print(beam_search.result)


if __name__ == "__main__":
    main()
