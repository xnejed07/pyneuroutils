import numpy as np
import zmq
import time
import sys
import os
import threading
from pyneuroutils.transforms import *
def timer_connect_error():
    raise Exception("Connection Error")


def timer_recv_error():
    raise Exception("Recv Error")


class MEFD_client(object):
    def __init__(self, ip, wait=60, verbose=False):
        self.ip = ip
        self.wait = wait
        self.verbose = verbose
        self.context = None
        if self.verbose:
            print("client-init", os.getpid())

    def connect(self, port,test_connection=True):
        """
        This should be called in a new spawned process, not in main process -> where __init__ is called
        :param port:
        :return:
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.poll = zmq.Poller()
        self.socket.connect("tcp://{}:{}".format(self.ip, port))
        self.poll.register(self.socket, zmq.POLLIN)
        if self.verbose:
            print('client-connect', os.getpid(), port)
        if test_connection:
            self.socket.send_pyobj({'task': "GET_STATUS"})
            q = self.socket.recv_json()
            print(q)

    def request(self,**kwargs):
        self.socket.send_pyobj(kwargs)
        socks = dict(self.poll.poll(self.wait * 1000))
        if socks.get(self.socket) == zmq.POLLIN:
            response = self.socket.recv_json()
            return response
        else:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            self.poll.unregister(self.socket)
            raise Exception("Server not responding")


    def request_data(self, path, channel, password, start=None, stop=None, **kwargs):
        WORK = {'path': path,
               'channel': channel,
               'start': start,
               'stop': stop,
               'password': password}
        for key,value in kwargs.items():
            WORK[key] = value
        #self.socket.send_json(WORK)
        self.socket.send_pyobj(WORK)

        socks = dict(self.poll.poll(self.wait * 1000))
        if socks.get(self.socket) == zmq.POLLIN:
            response = self.recv_array()
            return response
        else:
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close()
            self.poll.unregister(self.socket)
            raise Exception("Server not responding")

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array"""
        md = self.socket.recv_json(flags=flags, )
        if 'exception' in md:
            err = "SERVER: "+md['ip']+":"+md['port']+" ERROR: "+md['exception']
            raise Exception(err)
        msg = self.socket.recv(flags=flags, copy=copy, track=track)
        buf = memoryview(msg)
        A = np.frombuffer(buf, dtype=md['dtype'])
        return A.reshape(md['shape'])



if __name__ == "__main__":
    for i in range(10):
        try:
            t0 = time.time()
            client = MEFD_client(ip='10.144.10.73', wait=10)
            client.connect(port=54321)
            response = client.request(task="GET_STATUS")

            transform =compose([filtfilt(n=3,wn=0.1,btype='low'),
                                sample_1d(n=10,window=5000),
                                zscore(axis=-1)])

            data = client.request_data(path='/mnt/Helium/data/fnusa/seeg-040/Easrec_sciexp-seeg040_151001-1120.mefd',
                                    channel='A1',
                                    password='bemena',
                                    start=None,
                                    stop=None,
                                    transform=transform,
                                    task="GET_DATA")
            stop = 1
        except Exception as exc:
            print(exc)
        finally:
            print(time.time() - t0)
    print("OK")
