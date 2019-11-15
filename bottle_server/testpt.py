from rectangle import Point
import json
from functools import singledispatch

a = Point(9, 10)

print(str(a))

@singledispatch 
def to_serializable(val):
    return str(val)

class CustomEncoder(json.JSONEncoder):

     def default(self, o):

         return {'__{}__'.format(o.__class__.__name__): o.__dict__}

print(json.dumps(a, cls=CustomEncoder) )

