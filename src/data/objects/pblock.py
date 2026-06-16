class PBlock:
    def __init__(self, data):
        self.id = int(data[0])
        self.x = round(data[1] * 587) / 587
        self.y = round(data[2] * 587) / 587
        self.z = round(data[3] * 587) / 587