def get_MF_config(org_config):
    config = {
        "device": org_config["device"],
        "dim": int(org_config["out_dim"])
        }
    return config


def get_LightGCN_config(org_config):
    config = {
        "device": org_config["device"],
        "dim": int(org_config["out_dim"]),
        "n_layers": 3,
        }
    return config

def get_NeuMF_config(org_config):
    config = {
        "device": org_config["device"],
        "dim": int(org_config["out_dim"]),
        "layer_sizes": [128, 64, 32],
        }
    return config



def get_Origin_config(org_config):
    config = {
        "device": org_config["device"],
        }
    return config

def get_SelfDenoise_config(org_config):
    config = {
        "begin_adv": 5,
        "item_num": org_config["item_num"],
        "temperature": org_config["temp"],
        }
    return config