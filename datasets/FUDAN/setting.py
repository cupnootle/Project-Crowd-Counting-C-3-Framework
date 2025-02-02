from easydict import EasyDict as edict

# init
__C_FUDAN = edict()

cfg_data = __C_FUDAN

__C_FUDAN.STD_SIZE = (768,1024)
__C_FUDAN.TRAIN_SIZE = (576,768)
__C_FUDAN.DATA_PATH = '../ProcessedData/Fudan-UCC'               

__C_FUDAN.MEAN_STD = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

__C_FUDAN.LABEL_FACTOR = 1
__C_FUDAN.LOG_PARA = 100.

__C_FUDAN.RESUME_MODEL = ''#model path
__C_FUDAN.TRAIN_BATCH_SIZE = 6 #imgs

__C_FUDAN.VAL_BATCH_SIZE = 6 # 


