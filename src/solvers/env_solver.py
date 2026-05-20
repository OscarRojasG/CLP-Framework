from data.objects import *

class EnvSolver():
    def process_block_data(self, data):
        return [Block(data[i:i+4]) for i in range(0, len(data), 4)]
    
    def process_pblock_data(self, data):
        return [PBlock(data[i:i+4]) for i in range(0, len(data), 4)]
    
    def process_action_data(self, data):
        return [Action(data[i:i+4]) for i in range(0, len(data), 4)]

    def process_space_data(self, data):
        return Space(data)
    
    def process_pblock_data_batch(self, data_batch):
        data = []
        for state in data_batch:
            data.append([PBlock(state[i:i+4]) for i in range(0, len(state), 4)])
        return data

    def process_action_data_batch(self, data_batch):
        data = []
        for state in data_batch:
            data.append([Action(state[i:i+4]) for i in range(0, len(state), 4)])
        return data
    
    def process_space_data_batch(self, data_batch):
        data = []
        for state in data_batch:
            data.append(Space(state))
        return data