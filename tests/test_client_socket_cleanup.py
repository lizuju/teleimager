import unittest
from unittest.mock import patch

import zmq


class SocketCleanupTests(unittest.TestCase):
    def setUp(self):
        self.context = zmq.Context()
        self.sockets = []
        self.socket_types = []
        real_socket = self.context.socket

        def socket(*args, **kwargs):
            value = real_socket(*args, **kwargs)
            self.sockets.append(value)
            self.socket_types.append(args[0])
            return value

        self.patcher = patch.object(self.context, 'socket', side_effect=socket)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.context.destroy(linger=0)

    def test_publisher_stop_closes_its_socket_without_clock_variable(self):
        from teleimager.client import ZMQ_PublisherThread
        publisher = ZMQ_PublisherThread(0, host='127.0.0.1', context=self.context)
        publisher._queue.put_nowait(None)
        publisher.run()
        self.assertTrue(publisher._started.is_set())
        self.assertEqual(len(self.sockets), 1)
        self.assertTrue(self.sockets[0].closed)
        self.assertIsNone(publisher._socket)

    def test_subscriber_stop_closes_pending_clock_request_and_image_socket(self):
        from teleimager.client import ZMQ_SubscriberThread
        subscriber = ZMQ_SubscriberThread('127.0.0.1', 65001, context=self.context)

        def stop_on_poll(timeout):
            subscriber._running = False
            return []

        with patch('teleimager.client.zmq.Poller') as poller:
            poller.return_value.poll.side_effect = stop_on_poll
            subscriber.run()
        self.assertEqual(self.socket_types, [zmq.SUB, zmq.REQ])
        self.assertTrue(all(sock.closed for sock in self.sockets))
        self.assertIsNone(subscriber._socket)


if __name__ == '__main__':
    unittest.main(verbosity=2)
