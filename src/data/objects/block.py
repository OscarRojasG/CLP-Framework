class Block:
    def __init__(self, data):
        self.l = data[0]
        self.w = data[1]
        self.h = data[2]
        self.n = data[3]

        self.boxes = {}
        
        # Procesamos desde el índice 4 en adelante
        # Usamos range con paso 2 para ir tomando (ID, cantidad)
        raw_items = data[4:]
        for i in range(0, len(raw_items), 2):
            box_id = int(raw_items[i])
            quantity = int(raw_items[i+1])
            self.boxes[box_id] = quantity

    def volume(self):
        return self.l * self.w * self.h
    
    def lw(self):
        return self.l * self.w
    
    def lh(self):
        return self.l * self.h
    
    def wh(self):
        return self.w * self.h