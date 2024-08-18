from datasets.data_loader import get_loader_segment




index = 143
datapath = './datasets/'

dataset_name = 'MSL' ##  SMD, MSL, SMAP, PSM, SWAT, NIPS_TS_Swan, UCR

data_path = datapath + dataset_name + '/'
batch_size = 128

data_loader = get_loader_segment(index, data_path, batch_size, win_size=100, step=100, mode='train', dataset=dataset_name)
data_loader = get_loader_segment(index, data_path, batch_size, win_size=100, step=100, mode='val', dataset=dataset_name)
data_loader = get_loader_segment(index, data_path, batch_size, win_size=100, step=100, mode='test', dataset=dataset_name)
print("Read Success!!!")