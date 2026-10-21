class Block:
    def __init__(self, data):
        self.l = data[0]
        self.w = data[1]
        self.h = data[2]
        self.vol = data[3]
        self.n = data[4]

    def volume(self):
        return self.vol
    
    def lw(self):
        return self.l * self.w
    
    def lh(self):
        return self.l * self.h
    
    def wh(self):
        return self.w * self.h