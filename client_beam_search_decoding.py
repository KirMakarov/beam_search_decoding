# -*- coding: utf-8 -*-

"""Beam search decoding: Client for send to server alphabet and matrix probabilities"""

import json
import socket

import beam_search_decoding


def main():
    """Send to server alphabet and matrix probabilities and receive decoded text.

    return: decoded text
    """
    args = beam_search_decoding.StartupArgs()
    args.fetch()

    # Setup socket
    address_to_server = ('localhost', 11011)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(address_to_server)

    # Prepare data
    alphabet = beam_search_decoding.Alphabet(args.alphabet_file_name)
    alphabet.get_letters()

    matrix_probabilities = beam_search_decoding.Probabilities(args.matrix_file_name)
    matrix_probabilities.csv_file_reader()

    data_string = json.dumps({
        'alphabet': alphabet.letters,
        'matrix': matrix_probabilities.raw_array,
    })

    client.send(bytes(data_string, encoding='UTF-8'))

    data = client.recv(1024)
    print(str(data))


if __name__ == "__main__":
    main()
