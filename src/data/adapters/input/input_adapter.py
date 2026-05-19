from data.adapters.data_adapter import DataAdapter
from data.objects import *
from abc import abstractmethod

class InputAdapter(DataAdapter):
    def __init__(self, data_keys, max_blocks: int, max_pblocks: int, max_actions: int):
        super().__init__(data_keys)
        self.max_blocks = max_blocks
        self.max_pblocks = max_pblocks
        self.max_actions = max_actions

    def input_2_vec(self, blocks: list[Block], space: Space, pblocks: list[PBlock], actions: list[Action]):
        enc_data = self.enc_2_vec(blocks)
        dec_data = self.dec_2_vec(space, pblocks, actions)

        return (*enc_data, *dec_data)
    
    @abstractmethod
    def enc_2_vec(self, blocks: list[Block]):
        pass
    
    @abstractmethod
    def dec_2_vec(self, space: Space, pblocks: list[PBlock], actions: list[Action]):
        pass