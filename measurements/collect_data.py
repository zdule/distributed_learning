import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

category_single = ["single"]
categories1 = ["central_node_reduce", "overlap", "seq_merge", "ourdist"]
categories2 = ["ddp", "onestep_reduce"]
categories3 = ["ddp", "onestep_overlap", "onestep_reduce", "onestep_seq_merge"]
config1 = [(1,1),(1,2), (1,4), (2,4), (4,4), (8,4)]
config2 = [(1,1),(2,1), (4,1), (8,1), (16,1), (32,1)]

def load_experiment_fusion(folder, fusion_sizes, categories, config):
    all_data = None 
    for fusion_size in fusion_sizes:
        for category in categories:
            for nodeid in range(config[3][0]):
                for deviceid in range(config[3][1]):
                    data = pd.read_csv(f"{folder}/{fusion_size}/{category}_{nodeid}_{deviceid}_times.csv", sep=" *, *")
                    data["fusion_size"] = fusion_size
                    if all_data is None:
                        all_data = data
                    else:
                        all_data = all_data.append(data)
    return all_data
 
def load_experiment(prefix,folders, categories, config, extrapolate=[]):
    all_data = None 
    for folder, devices in folders:
        for category in categories:
            for nodeid in range(config[devices][0]):
                for deviceid in range(config[devices][1]):
                    data = pd.read_csv(f"{prefix}/{folder}/{category}_{nodeid}_{deviceid}_times.csv", sep=" *, *")
                    data["devices"] = config[devices][0] * config[devices][1]
                    orig = data.copy()
                    for fac in extrapolate:
                        data1 = orig.copy()
                        data1['devices'] *= fac
                        data = data.append(data1)
                    if all_data is None:
                        all_data = data
                    else:
                        all_data = all_data.append(data)
    return all_data

def reduce_data(data):
    data = data.groupby(["devices","experiment_name"]).mean().reset_index("devices")
    data['throughput'] = 1000*data['devices']*data['data_len']/data['batch']
    return data

def reduce_data_fusion(data):
    return data.groupby(["fusion_size","experiment_name"]).mean().reset_index("fusion_size")

translate = {
    "single" : "Ideal",
    "overlap" : "2-step pipelining",
    "seq_merge" : "2-step fusion",
    "ourdist" : "2-step pipelining+fusion",
    "ddp" : "PyTorch DDP",
    "central_node_reduce" : "2-step central reduce",
    "onestep_reduce" : "1-step pipelining+fusion",
    "onestep_overlap" : "onestep_overlap",
    "onestep_seq_merge" : "onestep_seq_merge"
}

def plot(data, categories, name='throughput'):
    plt.figure(figsize=(6,4))
    for cat in categories:
        catd = data.loc[(cat),["devices", "throughput"]]
        plt.plot(catd['devices'], catd['throughput'], label=translate[cat], marker='+', markersize=6)
    plt.xlabel("Number of workers")
    plt.xticks([1,2,4,8,16])
    plt.ylabel("Throughput [images/second]")
    plt.tight_layout()
    plt.legend() 
    plt.savefig(f"{name}.png")
    plt.show()

def plot_fusion(data, categories):
    data['fusion_size'] = data['fusion_size'].astype(str)
#sns.set()
    data['other'] = data['data2dev'] + data['zero_grad'] + data['optimizer_step']
    data.set_index(['fusion_size'])[['other','get_data','forward','backprop','sync']].plot(kind='bar', stacked=True, rot=0)
    plt.xlabel("Fussion tensor size [KiB]")
    plt.ylabel("One iteration latency [ms]")
    plt.tight_layout()
    plt.legend() 
    plt.savefig("fusion.png")
    plt.show()

def plot_randint_data():
    experiment_single = [("results/experiment_single_1_33846316",0)]
    experiments = [ ("results/experiment1_1_33847002",0),
                    ("results/experiment1_2_33846550",1), ("results/experiment1_4_33846297", 2), 
                    ("results/experiment1_8_33846299",3), ("results/experiment1_16_33846301", 4),]
#("results/experiment1_32_33846965",5)]
    experiments2 = [("results/experiment2_1_33846552",0),
                    ("results/experiment2_2_33846303",1), ("results/experiment2_4_33846305", 2), 
                    ("results/experiment2_8_33846308",3), ("results/experiment2_16_33895822", 4)]
    data = load_experiment("gpu2", experiment_single, category_single, config1, [2,4,8,16])
    data = data.append(load_experiment("gpu2", experiments, categories1, config1))
    data = data.append(load_experiment("gpu2", experiments2, categories2, config2))
    data = reduce_data(data)

    print(data)
    plot(data, category_single + categories1+ categories2, "randint_gpu_throughput")
    plot_abc(data,"ourdist","gpu_ourdist_latency")
    plot_abc(data,"ddp","gpu_ddp_latency")

def plot_randint_cpu_data():
#experiment_single = [("results/experiment_single_2_33910790",0)]
#experiment_single = [("results/experiment_single_1_33912337",0)]
    experiment_single = [("results/experiment_single_1_33912371",0)]
    experiments3 = [("results/experiment3_2_33910203", 1), ("results/experiment3_4_33910204",2),
                    ("results/experiment3_8_33910753", 3), ("results/experiment3_16_33910206",4),
                    ("results/experiment3_32_33910207", 5)]
    data = load_experiment("cpu", experiment_single, category_single, config1, [2,4,8,16, 32])
    data = data.append(load_experiment("cpu", experiments3, categories3, config2))
    data = reduce_data(data)
    print(data)
    plot(data, ["single"] + categories3)
    
    fusion_ourdist_gpu_data = load_experiment_fusion("cpu/results/fusion_experiment_ourdist_16_33910429", [256,1024, 4096, 16384, 65536],["ourdist"], config2)
    fusion_reduced = reduce_data_fusion(fusion_ourdist_gpu_data)
    print(fusion_reduced)
    plot_fusion(fusion_reduced, ["ourdist"])

def plot_fusion_cpu():
    fusion_ourdist_gpu_data = load_experiment_fusion("cpu/results/fusion_experiment_ddp_16_33762651", [64,256,1024, 4096, 16384, 65536],["ddp"], config2)
    fusion_reduced = reduce_data_fusion(fusion_ourdist_gpu_data)
    plot_fusion(fusion_reduced, ["ourdist"])

def plot_abc(data,cat, name):
    data = data.loc[(cat)]
    data['other'] = data['data2dev'] + data['zero_grad'] + data['optimizer_step']
    data.set_index(['devices'])[['other','get_data','forward','backprop','sync']].plot(kind='area', stacked=True, rot=0)
    plt.xticks([1,2,4,8,16])
    plt.xlabel("Number of workers")
    plt.ylabel("Single iteration latency [ms]")
    plt.tight_layout()
    plt.legend() 
    plt.savefig(f"{name}.png")
    plt.show()


#plot_randint_cpu_data()
plot_randint_data()
#plot_fusion_cpu()
experiments_single = [("results/experiment_single_1_33773136",0)]
experiments1 = [("results/experiment1_2_33771233",1), ("results/experiment1_4_33723601", 2), 
                ("results/experiment1_8_33723621",3), ("results/experiment1_16_33723627", 4)]

experiments2 = [("results/experiment2_2_33723651", 1), ("results/experiment2_4_33723662",2),
                ("results/experiment2_8_33723719", 3), ("results/experiment2_16_33723723",4)]
experiments3 = [("results/experiment3_2_33742377", 1), ("results/experiment3_4_33742415",2),
                ("results/experiment3_8_33742483", 3), ("results/experiment3_16_33751512",4)]

gpu2_experiments1 = [("results/experiment1_2_33774647",1), ("results/experiment1_4_33774648", 2),
                  ("results/experiment1_8_33774649",3), ("results/experiment1_16_33774650", 4),]
#("results/experiment1_32_33789660",5)]

#data = load_experiment("gpu1",experiments1, categories1, config1)
data = load_experiment("gpu1", experiments_single, category_single, config1, [2,4,8,16])
data = data.append(load_experiment("gpu2",gpu2_experiments1, categories1, config1))
data = data.append(load_experiment("gpu1",experiments2, categories2, config2))
data = reduce_data(data)
plot(data, category_single + categories1+ categories2)
plot_abc(data,'ddp', "ddp_latency_breakdown")
plot_abc(data,'ourdist', "ourdist_latency_breakdown")

data_cpu = load_experiment("cpu", experiments3, categories3, config2)
data_cpu = reduce_data(data_cpu)
#plot(data_cpu, categories3)

fusion_ourdist_gpu_data = load_experiment_fusion("gpu1/results/fusion_experiment_ourdist_16_33723740", [1024, 4096, 16384, 65536],["ourdist"], config1)
fusion_reduced = reduce_data_fusion(fusion_ourdist_gpu_data)
#print(fusion_reduced)
plot_fusion(fusion_reduced, ["ourdist"])
