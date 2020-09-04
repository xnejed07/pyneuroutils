from pyneuroutils.transforms.transform import transform


class crop(transform):
    def __init__(self, dim_0=None, dim_1=None,dim_2=None):
        super().__init__()
        self.dim_0 = dim_0
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def __call__(self,x):

        if self.dim_0 is not None:
            x = x[self.dim_0[0]:self.dim_0[1],...]

        if self.dim_1 is not None:
            x = x[:,self.dim_1[0]:self.dim_1[1],...]

        if self.dim_2 is not None:
            x = x[:,:,self.dim_2[0]:self.dim_2[1],...]


        return x