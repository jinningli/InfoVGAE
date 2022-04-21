import time

def update_arg_with_config_name(args, config_name, phrase):
    """
    :param args: argparse dict
    :param config_name: name of config
    :param phrase: train / evaluate
    :return: argparse dict
    """
    if config_name == "InfoVGAE_election_1D":
        setattr(args, "data_path", "dataset/election/data.csv")
        setattr(args, "stopword_path", "dataset/election/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 1)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "epochs", 80)

    elif config_name == "InfoVGAE_election_2D":
        setattr(args, "data_path", "dataset/election/data.csv")
        setattr(args, "stopword_path", "dataset/election/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 2)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 80)

    elif config_name == "InfoVGAE_election_3D":
        setattr(args, "data_path", "dataset/election/data.csv")
        setattr(args, "stopword_path", "dataset/election/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.01)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 80)

    elif config_name == "InfoVGAE_eurovision_3D":
        setattr(args, "data_path", "dataset/eurovision/data.csv")
        setattr(args, "stopword_path", "dataset/eurovision/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 118)

    elif config_name == "InfoVGAE_timme_3D":
        setattr(args, "data_path", "dataset/timme/data.csv")
        setattr(args, "stopword_path", "dataset/timme/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "pos_weight_lambda", 19.0)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 97)

    elif config_name == "InfoVGAE_timme_follow_3D":
        setattr(args, "data_path", "dataset/timme/data.csv")
        setattr(args, "stopword_path", "dataset/timme/stopwords_en.txt")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.1)
        setattr(args, "pos_weight_lambda", 14.0)
        setattr(args, "use_follow", True)
        setattr(args, "follow_path", "dataset/timme/friend_list.csv")
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 102)

    elif config_name == "InfoVGAE_bill_3D":
        setattr(args, "data_path", "")
        setattr(args, "stopword_path", "")
        setattr(args, "model", "VGAE")
        setattr(args, "hidden2_dim", 3)
        setattr(args, "learning_rate", 0.01)
        setattr(args, "lr_D", 1e-3)
        setattr(args, "gamma", 1e-3)
        setattr(args, "seed", 0)
        setattr(args, "epochs", 132)

    else:
        raise NotImplementedError("Unknown config name")

    setattr(args, "output_path", "./output/{}_{}".format(config_name, time.strftime("%Y%m%d%H%M%S", time.localtime())))
    return args
