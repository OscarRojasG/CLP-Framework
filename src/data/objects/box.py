class Box:
    def __init__(self, data):
        self.l = data[0]
        self.w = data[1]
        self.h = data[2]
        self.n = data[3]
        
    def volume(self):
        return self.l * self.w * self.h