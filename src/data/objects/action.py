class Action:
    def __init__(self, data):
        self.block_id = int(data[0])
        self.vcs = data[1]
        self.loss = data[2]
        self.cs = data[3]