import torch
import torch.distributed as dist

def built_in_allreduce(send):
    dist.all_reduce(send, op=dist.ReduceOp.SUM)
    send /= float(dist.get_world_size())

def ring_allreduce(send):
    """
    Custom ring all-reduce fucntion that averages the gradients.

    :param send: The gradients computed at an individual node. The
        function overwrites them with the shared averages.
    """

    rank = dist.get_rank()
    size = dist.get_world_size()
    if size <= 1:
        return

    chunks = torch.chunk(send, size)
    maxsize = max((chunk.size() for chunk in chunks))
    recv_buffer = torch.empty(maxsize, dtype=send.dtype)

    right = (rank + 1) % size
    left = (rank-1) % size

    send_reqs = []

    # First pass
    for i in range(size - 1):
        to_send = (rank-i) % size
        to_recv = (to_send - 1) % size
        req = dist.isend(chunks[to_send], right)
        send_reqs.append(req)
        dist.recv(recv_buffer, left)        # Receiving needs to be blocking
        chunks[to_recv][:] += recv_buffer[:len(chunks[to_recv])]

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished
    send_reqs = []

    # We now have result[r+1] on node with rank r

    # Second pass
    for i in range(size-1):
        to_send = (rank - i + 1) % size
        to_recv = (to_send - 1) % size
        req = dist.isend(chunks[to_send], right)
        send_reqs.append(req)
        dist.recv(recv_buffer, left)         # Receiving needs to be blocking
        chunks[to_recv][:] = recv_buffer[:len(
            chunks[to_recv])]  # [:] indicates deepcopy

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished

    # Dividing result by the number of devices
    send /= float(size)

def ring_allreduce_gpu(send, groups):
    """
    Custom ring all-reduce fucntion that averages the gradients.

    :param send: The gradients computed at an individual node. The
        function overwrites them with the shared averages.
    """

    rank = dist.get_rank()
    size = dist.get_world_size()
    if size <= 1:
        return

    chunks = torch.chunk(send, size)
    maxsize = max((chunk.size() for chunk in chunks))
    recv_buffer = torch.empty(maxsize, dtype=send.dtype, device=send.device)

    right = (rank + 1) % size
    left = (rank-1) % size
    right_group = groups[rank]
    left_group = groups[left]

    send_reqs = []

    # First pass
    for i in range(size - 1):
        to_send = (rank-i) % size
        to_recv = (to_send - 1) % size
        req = dist.broadcast(chunks[to_send], rank, group=right_group, async_op=True)
        #req = dist.isend(chunks[to_send], right)
        send_reqs.append(req)
        req2 = dist.broadcast(recv_buffer[:len(chunks[to_recv])], left, group=left_group, async_op=True)
        #dist.recv(recv_buffer, left)        # Receiving needs to be blocking
        req2.wait()
        chunks[to_recv][:] += recv_buffer[:len(chunks[to_recv])]

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished
    send_reqs = []

    # We now have result[r+1] on node with rank r

    # Second pass
    for i in range(size-1):
        to_send = (rank - i + 1) % size
        to_recv = (to_send - 1) % size
        req = dist.broadcast(chunks[to_send], rank, group=right_group, async_op=True)
        #req = dist.isend(chunks[to_send], right)
        send_reqs.append(req)
        req2 = dist.broadcast(recv_buffer[:len(chunks[to_recv])], left, group=left_group, async_op=True)
        #dist.recv(recv_buffer, left)        # Receiving needs to be blocking
        req2.wait()
        chunks[to_recv][:] = recv_buffer[:len(
            chunks[to_recv])]  # [:] indicates deepcopy

    for send_req in send_reqs:
        send_req.wait()     # Need to wait till sending is finished

    # Dividing result by the number of devices
    send /= float(size)
