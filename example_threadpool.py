#!/usr/bin/env python
# -*- enconding: utf-8 -*-

import threading
import queue
import logging
import time
from numpy.random import randint

logging.basicConfig(level=logging.INFO)


class Worker(threading.Thread):
    """simple worker class"""
    def __init__(self, _queue, _id):
        self._queue = _queue
        self._id = _id
        self._timeout = 10
        super().__init__()

    def run(self):
        while True:
            try:
                func, args, kwargs = self._queue.get(timeout=self._timeout)
                res = func(*args, **kwargs)
                logging.info('thread id: {:02.0f} task complete. value:'
                             '{}'.format(self._id, res))
                self._queue.task_done()
            except queue.Empty:
                logging.info('worker queue has been empty for %0.0f seconds,'
                             ' shutting down', self._timeout)
                break
            except Exception as e:
                logging.warn('thread id: %02.0f raised the following error: %s',
                             self._id, e)
                self._queue.task_done()

class ThreadPool:
    """basic threadpool class"""
    def __init__(self, num_threads):
        self.queue = queue.Queue()
        self.workers = tuple(Worker(self.queue, i) for i in range(num_threads))
        for worker in self.workers:
            worker.start()

    def worker_threads(self):
        return len(self.workers)

    def put(self, task):
        self.queue.put(task)

    def join(self):
        self.queue.join()
        logging.info('queue joined, all work completed')

if __name__ == '__main__':
    pool = ThreadPool(2)
    logging.info('number of threads %f', pool.worker_threads())
    def f(x):
        sleep_time = randint(5)
        logging.info('sleeping %f seconds', sleep_time)
        time.sleep(sleep_time)
        return x*x
    for val in range(10):
        pool.put([f, [val], {}])
    pool.join()
