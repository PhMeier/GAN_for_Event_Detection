"""
This class represents an Entity. Saves important data like id, start, end and type.
"""

class Entity:

    def __init__(self, id, start, end, type):
        self.id = id
        self.start = start
        self.end = end
        self.type = type

    def __repr__(self):
        return "[%r, %r, %r, %r]" % (self.id, self.start, self.end, self.type)

    def __str__(self):
        return "[%r, %r, %r, %r]" % (self.id, self.start, self.end, self.type)
