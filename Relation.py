"""
Similar to Event and Entitiy, these classes represent the event relations.
"""

class Bridging:
    def __init__(self, iD, typ, arg, related_to):
        self.iD = iD
        self.typ = typ
        self.arg = arg
        self.related_to = related_to

    def __repr__(self):
        return "[%r, %r, %r, %r]" % (self.iD, self.typ, self.arg, self.related_to)

    def __str__(self):
        return "[%r, %r, %r, %r]" % (self.iD, self.typ, self.arg, self.related_to)


class Whole_Member_Relation():
    def __init__(self, iD, typ, ent1, ent2):
        self.iD = iD
        self.typ = typ
        self.ent1 = ent1
        self.ent2 = ent2

    def __repr__(self):
        return "[%r, %r, %r, %r]" % (self.iD, self.typ, self.ent1, self.ent2)

    def __str__(self):
        return "[%r, %r, %r, %r]" % (self.iD, self.typ, self.ent1, self.ent2)


class Set_Member_Relation():
    def __init__(self, iD, typ, set_, mem1, mem2):
        self.iD = iD
        self.typ = typ
        self.set = set_
        self.mem1 = mem1
        self.mem2 = mem2

    def __repr__(self):
        return "[%r, %r, %r, %r, %r]" % (self.iD, self.typ, self.set, self.mem1, self.mem2)

    def __str__(self):
        return "[%r, %r, %r, %r, %r]" % (self.iD, self.typ, self.set, self.mem1, self.mem2)