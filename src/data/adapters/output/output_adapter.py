from data.adapters.data_adapter import DataAdapter
from data.objects import Action
from abc import abstractmethod

class OutputAdapter(DataAdapter):
    def __init__(self, data_keys):
        super().__init__(data_keys)

    @abstractmethod
    def output_2_vec(self, actions: list[Action], selected_block: int, greedy: float):
        pass