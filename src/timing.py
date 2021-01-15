import time

collected_data = []
starts = {}
sums = {}
counts = {}
experiment_name = None

def start_timing_experiment(exp_n):
    global experiment_name
    if experiment_name != None:
        print("adding data")
        collected_data.append((experiment_name, {k : v/counts[k] for (k,v) in sums.items()}))
    experiment_name = exp_n

def start_timer(name):
    starts[name] = time.monotonic_ns()

def end_timer(name):
    sums[name] = time.monotonic_ns() - starts[name]
    if name not in counts:
        counts[name] = 1
    else:
        counts[name] += 1

def writeout_timer(filename):
    start_timing_experiment("done")    
    print(f"Writing out a log file ({filename})")
    if len(collected_data) == 0:
        return
    f = open(filename, 'w')
    f.write("experiment_name, " + ", ".join(collected_data[0][1].keys()) + "\n")
    for d in collected_data:
        f.write(d[0] + ", " + ", ".join((str(d[1][k]) for k in collected_data[0][1].keys())) + "\n")
    f.flush()
    f.close()
    print("Done writing out log file")
    
