import time

collected_data = []
starts = {}
sums = {}
counts = {}

def end_timing_experiment(experiment_name, extra_fields={}):
    global sums, counts, starts
    collected_data.append((experiment_name, dict({k : v/counts[k] for (k,v) in sums.items()}, **extra_fields)))
    sums = counts = starts = {}
    sums = {}
    counts = {}
    stars = {}

def start_timer(name):
    starts[name] = time.time()*1000

def end_timer(name):
    if name not in sums:
        sums[name] = 0
    sums[name] += time.time()*1000 - starts[name]
    if name not in counts:
        counts[name] = 1
    else:
        counts[name] += 1

def writeout_timer(filename):
    global collected_data
    print(f"Writing out a log file ({filename})")
    if len(collected_data) == 0:
        return
    f = open(filename, 'w')
    f.write("experiment_name, " + ", ".join(collected_data[0][1].keys()) + "\n")
    for d in collected_data:
        f.write(d[0] + ", " + ", ".join((str(d[1][k]) for k in collected_data[0][1].keys())) + "\n")
    f.flush()
    f.close()
    collected_data = []
    print("Done writing out log file")
    
