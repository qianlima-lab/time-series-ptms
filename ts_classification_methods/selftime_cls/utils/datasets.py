def nb_dims(dataset):
    if dataset in ["unipen1a", "unipen1b", "unipen1c"]:
        return 2
    return 1

def nb_classes(dataset):
    if dataset=='MFPT':
        return 15
    if dataset == 'XJTU':
        return 15
    if dataset == "CricketX":
        return 12 #300
    if dataset == "UWaveGestureLibraryAll":
        return 8 # 945
    if dataset == "DodgerLoopDay":
        return 7
    if dataset == "InsectWingbeatSound":
        return 11
