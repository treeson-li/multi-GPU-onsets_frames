class Const:
    class ConstError(TypeError) : pass
    class ConstCaseError(ConstError):pass

    def __setattr__(self, name, value):
            if name in self.__dict__:
                raise self.ConstError("Can't change const value!")
            if not name.isupper():
                raise self.ConstCaseError('const "%s" is not all letters are capitalized' %name)
            self.__dict__[name] = value

import sys
sys.modules[__name__] = Const()

import real_time_constant
real_time_constant.ONSET=0
real_time_constant.FRAME=1
real_time_constant.VELOCITY=2
real_time_constant.SPEC=3
real_time_constant.MERGE=4
real_time_constant.PLUS=5
real_time_constant.MULTI=6

real_time_constant.PADDING_ERROR = 4