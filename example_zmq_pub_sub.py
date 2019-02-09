#!/usr/bin/env python

import zmq
import argparse
from multiprocessing import Process
import random
import datetime as dt
import time
import logging

DEFAULT_PORT = 5556

logging.basicConfig(level=logging.INFO)

# TODO: allow CTRL-C to shutdown client and server gracefully


def parse_args():
    """parse arguments:
    port = port on which to set up pub/sub client/server (default: 5556)
    """
    _parser = argparse.ArgumentParser(description='example script demonstrating zmq pub/sub')
    _parser.add_argument(
        '-p', '-port', type=int,
        dest='port', nargs='?',
        default=DEFAULT_PORT, help='port to use for pub/sub communication'
        )
    _parsed_args = _parser.parse_args()
    return _parsed_args


def subscriber(port):
    """simple zmq subscriber"""
    _context = zmq.Context()
    _socket = _context.socket(zmq.SUB)

    _socket.connect("tcp://localhost:{}".format(port))
    _socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        _msg = _socket.recv()
        logging.info("subscriber received msg: {}".format(_msg))


def publisher(port):
    """simple zmq publisher"""
    _context = zmq.Context()
    _socket = _context.socket(zmq.PUB)
    _socket.bind("tcp://*:{}".format(port))

    while True:
        _topic = random.randrange(9999, 10005)
        _message = dt.datetime.now()
        _socket.send_string("{} {}".format(_topic, _message.strftime("%y-%m-%d %H:%M:%S")))
        time.sleep(1)


def main():
    """main entry point for example_zmq_pub_sub.py"""

    _args = parse_args()

    Process(target=publisher, args=(_args.port,)).start()

    Process(target=subscriber, args=(_args.port,)).start()


if __name__ == '__main__':
    main()
