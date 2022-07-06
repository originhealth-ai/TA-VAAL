import transforms.classification_transforms as clftrf


def transforms_composer(transforms_dict):
    transforms_list = []

    for key in transforms_dict.keys():
        trf = getattr(clftrf,key)(**transforms_dict[key])
        transforms_list.append(trf)

    return clftrf.Compose(transforms = transforms_list)
