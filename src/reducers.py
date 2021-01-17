import torch.multiprocessing as mp


class ReduceImmediatelly:
    def __init__(self, reduce_fn):
        self.reduce_fn = reduce_fn
        self.queue = Queue(maxsize=128)

    def put(self, buff):
        self.reduce_fn(buff)
        self.queue.put(buff)

    def get(self):
        return self.queue.get()

    def cleanup(self):
        pass

class NodeAggregateReducer:
    def __init__(self, to_cpu_queue, from_cpu_queue, should_cleanup):
        self.to_cpu_queue = to_cpu_queue
        self.from_cpu_queue = from_cpu_queue
        self.should_cleanup = should_cleanup

    def put(self, buff):
        self.to_cpu_queue.put(buff)

    def get(self):
        return self.from_cpu_queue.get()

    def cleanup(self):
        if self.should_cleanup:
            self.to_cpu_queue.put(None)

class NodeAgreggateReducerCPU:
    def __init__(self, reduce_fn, ndevs):
        self.reduce_fn = reduce_fn
        self.to_cpu_queues = [mp.Queue() for _ in range(ndevs)]
        self.from_cpu_queues = [mp.Queue() for _ in range(ndevs)]
        self.reducers = [NodeAggregateReducer(to_cpu, from_cpu, i==0) for (i, (to_cpu, from_cpu)) 
                                        in enumerate(zip(self.to_cpu_queues, self.from_cpu_queues))]

    def pump(self):
        while True:
            agg = None
            for que in self.to_cpu_queues:
                buff = que.get()
                if buff == None:
                    return
                if agg == None:
                    agg = buff
                else:
                    agg[:] += buff
            self.reduce_fn(agg)
            for que in self.from_cpu_queues:
                que.put(agg)
