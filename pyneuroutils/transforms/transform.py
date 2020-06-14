

class transform(object):
    def __init__(self):
        pass

    def __call__(self,x):
        raise NotImplementedError

    def __repr__(self):
        x = "{}(".format(self.__class__.__name__)

        for key,value in self.__dict__.items():
            if key[0] is not "_":
                x+= "{} = {}, ".format(key,value)
        x=x[:-2]
        x+=')'
        return x

