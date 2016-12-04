# -*- coding: utf-8 -*-

"""emulating a client/server environment"""

from . import assert_eq

import asyncio

class Socket:
    _out_queue = None
    _in_queue = None

    _send_ftrs = None

    def __init__(self):
        self._out_queue = asyncio.Queue()
        self._send_ftrs = []

    def _setup(self, peer):
        assert isinstance(peer, Socket)
        self._in_queue = peer._out_queue

    def send(self, msg):
        self._send_ftrs.append(self._out_queue.put(msg))
        return self

    async def recv(self):
        await self.wait()
        return await self._in_queue.get()

    async def wait(self):
        """wait all unfinished send"""
        while self._send_ftrs:
            await self._send_ftrs.pop()

    @classmethod
    def make_pair(cls):
        """make a socket pair that are connected"""
        s0 = cls()
        s1 = cls()
        s0._setup(s1)
        s1._setup(s0)
        return s0, s1


def run_cs_session(server, client):
    """run a client/server session

    :param server: async callable that takes a :class:`Socket` object to act as
        the server
    :param client: async callable that takes a :class:`Socket` object to act as
        the client
    """
    s0, s1 = Socket.make_pair()
    ft0 = server(s0)
    ft1 = client(s1)

    task = asyncio.wait([ft0, ft1])
    loop = asyncio.get_event_loop()
    loop.run_until_complete(task)


def main():
    async def echo_srv(socket):
        socket.send(await socket.recv())
        await socket.wait()

    async def echo_client(socket):
        socket.send('hello')
        assert_eq(await socket.recv(), 'hello')

    run_cs_session(echo_srv, echo_client)

if __name__ == '__main__':
    main()
