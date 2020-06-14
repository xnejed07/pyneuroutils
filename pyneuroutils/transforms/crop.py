from pyneuroutils.transforms.transform import transform


class crop(transform):
    def __init__(self, dim_x=None, dim_y=None, **kwargs):
        super().__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y

    def __call__(self,x):
        if x.ndim != 3:
            raise Exception("InvalidInputDimensins: {}".format(x.shape))
        if self.dim_x is not None:
            x = x[:,:,:self.dim_x]

        if self.dim_y is not None:
            x = x[:,:self.dim_y,:]

        return x