import os,sys,time

#############CONFIG#############
DEVICE = "cuda:0"
python_path = "/opt/anaconda3/envs/dgl/bin/python"
python_path = python_path if os.path.exists(python_path) else "python"
assert(__name__=="__main__")
root_path=os.path.abspath(os.path.dirname(__file__))
os.chdir(root_path)
assert(sys.platform=="linux")
pyf_feature_embedding1000=os.path.abspath(os.path.join(root_path,"../..","feature_embedding1000.py"))
assert(os.path.exists(pyf_feature_embedding1000))

###########################################
EPOCH = 1000 # You Should Change The Value
LR = 0.0005 # You Should Change The Value
# MOREOVER, PARAMS BELOW (eg. various file path) YOU SHOULD CHANGE TO FIT YOUR OWN PROJECT
#############END###########################


#==============================================================================
# 1. embedding train&test
dataset_path=os.path.join(root_path,"dataset")
print("\nfeature embedding")
## train
embedding_path=os.path.join(dataset_path,"embed")
test_model=os.path.join(embedding_path,"embed_nets_epoch_XX.pt") # SHOULD MATCH WITH EPOCH OR YOUR CUSTOM METHODS
embeded_dataset=os.path.join(dataset_path,"embeded.bin")
params=[
    "--train",
    f"--dataset='{os.path.join(dataset_path,'graphs.bin')}'",
    f"--epoch={EPOCH}",
    f"--lr={LR}",
    f"--train_output='{embedding_path}'",
]
cmd=f"{python_path} {pyf_feature_embedding1000} {' '.join(params)}"
if os.path.exists(test_model):
    print(f"exist {test_model}")
else:
    print(f"train embed model: {test_model}")
    print(cmd)
    os.system(cmd)
## test
params=[
    "--test",
    f"--dataset='{os.path.join(dataset_path,'graphs.bin')}'",
    f"--test_model='{test_model}'",
    f"--test_output='{embeded_dataset}'"
]
cmd=f"{python_path} {pyf_feature_embedding1000} {' '.join(params)}"
if os.path.exists(embeded_dataset):
    print(f"exist {embeded_dataset}")
else:
    print(f"generate {embeded_dataset}")
    print(cmd)
    os.system(cmd)
print()

# 2. 
print("train:")
for fold in range(5):
    os.system(f"{python_path} run.py --fold={fold} --device={DEVICE}")