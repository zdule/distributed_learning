import torch
import torch.distributed as dist

from utils import Level, print_d
import traceback
import sys

#from timing import start_timing_experiment, start_timer, end_timer, writeout_timer

from threading import Event, Thread
from queue import Queue

class Group:
    def __init__(self, group_id, tensors, get_hook, grad_buff_device="cpu"):
        self.group_id = group_id
        self.ready = 0
        self.tensors = tensors
        self.event = Event()

        total_size = sum((t.numel() for t in tensors))
        self.grad_buffer = torch.empty(total_size, dtype=tensors[0].dtype, device=grad_buff_device)

        if get_hook != None:
            self.hooks = []
            for t in tensors:
                self.hooks.append(t.register_hook(get_hook(self)))

    def fuse(self, wait_none=True):
        curr = 0
        for t in self.tensors:
            if wait_none:
                while t.grad == None:
                    pass
            if t.grad != None:
                self.grad_buffer[curr:curr+t.numel()] = t.grad.view(-1)[:]
            curr += t.numel()

    def unfuse(self, tensor):
        curr = 0
        for t in self.tensors:
            if t.grad != None:
                t.grad.view(-1)[:] = tensor[curr:curr+t.numel()]
            curr += t.numel()
    
    def add_grad(self):
        self.ready += 1
        assert(self.ready <= len(self.tensors))
        if self.ready == len(self.tensors):
            self.event.set()
            return True
        return False

class OurDist:
    def _fusion_grouping_gen(model, grouping_size=0):
        params = reversed(list(model.parameters()))
        to_fuse = []
        running_size = 0
        for param in params:
            size = param.element_size() * param.numel()
            if len(to_fuse) == 0 or running_size + size <= grouping_size:
                to_fuse.append(param)
                running_size += size
            else:
                yield to_fuse
                to_fuse = [param]
                running_size = size
        if to_fuse != []:
            yield to_fuse

    def __init__(self, model, reducer, grouping_size=0, grad_buff_device="cpu"):
        self.model = model
        self.reducer = reducer

        self.started_threads = False
        self.groups_processed = 0
        self.done_processing_event = Event()
        self.start_processing_event1 = Event()
        self.start_processing_event2 = Event()
        self.shutting_down = False

        self.groups = [Group(i, tensors, self.get_hook_for_group, grad_buff_device) for i, tensors in enumerate(OurDist._fusion_grouping_gen(self.model, grouping_size))]
        self.threads = [Thread(target=self.send_gradients_to_center_thread),
                        Thread(target=self.recieve_gradients_from_center_thread)]
        for thread in self.threads:
            thread.start()

    def cleanup(self):
        self.shutting_down = True
        self.start_processing_event1.set()
        self.start_processing_event2.set()

    def get_hook_for_group(self, group):
        def hook(grad):
            if group.add_grad():
                self.groups_processed += 1
            if not self.started_threads:
                self.started_threads = True
                self.start_processing_event1.set()
                self.start_processing_event2.set()
        return hook 

    def send_gradients_to_center_thread(self):
        try:
            while True:
                self.start_processing_event1.wait()
                self.start_processing_event1.clear()

                if self.shutting_down: return

                for group in self.groups:
                    group.event.wait()
                    group.fuse()
                    self.reducer.put(group.grad_buffer)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            

    def recieve_gradients_from_center_thread(self):
        try:
            while True:
                self.start_processing_event2.wait()
                self.start_processing_event2.clear()

                if self.shutting_down: return

                for group in self.groups:
                    group.unfuse(self.reducer.get())
                self.done_processing_event.set()
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()

    def sync_gradients(self):
        self.done_processing_event.wait()

    def _find_unused(self, output):
        unused = dict()
        seen = set()
        for g in self.groups:
            for t in g.tensors:
                unused[t] = g
        st = [output.grad_fn]
        seen.add(output.grad_fn)
        while len(st) > 0:
            x = st.pop() 
            if len(x.next_functions) == 0:
                del unused[x.variable]
            else:
                for y in x.next_functions:
                    if y[0] is None:
                        continue
                    if not y[0] in seen:
                        seen.add(y[0])
                        st.append(y[0])
        return unused

    def forward(self, data):
        for group in self.groups:
            group.ready = 0
            group.event.clear()
        self.groups_processed = 0
        self.started_threads = False
        self.done_processing_event.clear()
        
        output = self.model.forward(data)
        unused = self._find_unused(output)
        for t, group in unused.items():
            t.grad = torch.empty_like(t)
            if group.add_grad():
                self.groups_processed += 1
        return output

    def __call__(self, data):
        return self.forward(data)

    def __getattr__(self, attr):
        return getattr(self.model,attr)

class SeqMergeDist:
    def __init__(self, model, reducer, grouping_size, _grad_buffer_device="cpu"):
        self.model = model
        self.reducer = reducer

        self.groups = [Group(i, tensors, None) for i, tensors in enumerate(OurDist._fusion_grouping_gen(self.model, grouping_size))]

    def cleanup(self):
        pass

    def sync_gradients(self):
        for group in self.groups:
            group.fuse(wait_none=False)
            self.reducer.put(group.grad_buffer)
            group.unfuse(self.reducer.get())

    def forward(self, data):
        return self.model.forward(data)

    def __call__(self, data):
        return self.model.forward(data)

    def __getattr__(self, attr):
        return getattr(self.model,attr)

class SeqDist:
    def __init__(self, model, reducer, _grouping_size, _grad_buffer_device="cpu"):
        self.model = model
        self.reducer = reducer

    def cleanup(self):
        pass

    def sync_gradients(self):
        for param in self.model.parameters():
            self.reducer.put(param.grad.view(-1))
            param.grad.view(-1)[:] = self.reducer.get()

    def forward(self, data):
        return self.model.forward(data)

    def __call__(self, data):
        return self.model.forward(data)

    def __getattr__(self, attr):
        return getattr(self.model,attr)

class WarmupDist:
    def __init__(self, model, reducer, _grouping_size, _grad_buffer_device="cpu"):
        self.model = model
        self.reducer = reducer
        self.out = None

    def cleanup(self):
        pass

    def sync_gradients(self):
        pass

    def forward(self, data):
        if self.out == None:
            self.out = torch.zeros_like(self.model.forward(data), requires_grad=True)
        return self.out

    def __call__(self, data):
        return self.forward(data)

    def __getattr__(self, attr):
        return getattr(self.model,attr)

