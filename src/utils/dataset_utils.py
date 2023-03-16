

def get_color_dict(dataset_type, n_classes=10):
    color_dict = {}
    if dataset_type == '2_Spurious_MNIST':
        for i in range(n_classes):
            if i < n_classes // 2:
                color_dict[i] = 0
            else:
                color_dict[i] = 1
    else:
        raise ValueError("Dataset type {} not supported for get_color_dict()".format(dataset_type))

    return color_dict