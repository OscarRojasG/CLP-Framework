class BoxType:
    def __init__(self, box_id, dims, rots, qty):
        self.id = box_id        # identificador de la caja
        self.dims = dims        # (L, W, H)
        self.rots = rots        # (rotX, rotY, rotZ)
        self.qty = qty          # cantidad de cajas


class CLPInstance:
    def __init__(self, container, boxes : list[BoxType]):
        self.container = container  # (L, W, H)
        self.n_types = len(boxes)
        self.boxes = boxes          # lista de BoxType