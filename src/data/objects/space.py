class Space:
    def __init__(self, data):
        self.x = round(data[0] * 587) / 587
        self.y = round(data[1] * 587) / 587
        self.z = round(data[2] * 587) / 587
        self.l = round(data[3] * 587) / 587
        self.w = round(data[4] * 587) / 587
        self.h = round(data[5] * 587) / 587