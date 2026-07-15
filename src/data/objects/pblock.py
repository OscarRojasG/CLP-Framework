class PBlock:
    def __init__(self, data):
        self.id = int(data[0])
        self.x = data[1]
        self.y = data[2]
        self.z = data[3]