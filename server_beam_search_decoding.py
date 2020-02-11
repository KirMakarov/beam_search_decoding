# -*- coding: utf-8 -*-

"""Beam search decoding: server for receive alphabet and matrix probabilities"""

import json
import socket

import numpy as np

import beam_search_decoding


def main():
    """Receive alphabet and matrix probabilities from clients. Return: decoded text"""
    server_address = ('localhost', 11011)

    # Setup socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(server_address)
    server_socket.listen(1)
    print('Server is running, please, press "ctrl+c" to stop.')

    # Requests listen
    while True:
        connection, address = server_socket.accept()
        print("New connection from {address}".format(address=address))

        data_buff = connection.recv(1024)

        # data_buff = bytes()
        # connection.setblocking(False)
        # while True:
        #     data = connection.recv(1024)
        #     if not data:
        #         break
        #     data_buff += data

        data_loaded = json.loads(data_buff.decode('utf-8'))
        print(f'Received data: {data_loaded}')

        probabilities = np.array(data_loaded['matrix'])
        alphabet = data_loaded['alphabet']

        beam_search = beam_search_decoding.Decoder()
        beam_search(probabilities, alphabet)
        print(f'Result beam search: "{beam_search.result}"')

        connection.send(bytes(beam_search.result, encoding='UTF-8'))
        del beam_search

        connection.close()


if __name__ == "__main__":
    main()
