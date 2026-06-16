class Block:
    def __init__(self, data):
        self.l = round(data[0] * 587) / 587
        self.w = round(data[1] * 587) / 587
        self.h = round(data[2] * 587) / 587
        self.n = round(data[3] * 587) / 587

    def volume(self):
        return self.l * self.w * self.h
    
    def lw(self):
        return self.l * self.w
    
    def lh(self):
        return self.l * self.h
    
    def wh(self):
        return self.w * self.h