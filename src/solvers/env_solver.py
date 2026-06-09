from data.objects import *
import plotly.graph_objects as go

class EnvSolver():
    def process_block_data(self, data):
        return [Block(data[i:i+4]) for i in range(0, len(data), 4)]
    
    def process_box_data(self, data):
        return [Box(data[i:i+4]) for i in range(0, len(data), 4)]
    
    def process_boxes_per_block(self, data, block_data):
        for i, boxes in enumerate(data):
            for box, amount in boxes:
                block_data[i].boxes[box] = amount
    
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
    
    def _get_cube_mesh(self, x, y, z, l, w, h, color, name, opacity=1.0, is_wireframe=False):
        x_verts = [x,   x+l, x+l, x,   x,   x+l, x+l, x  ]
        y_verts = [y,   y,   y+w, y+w, y,   y,   y+w, y+w]
        z_verts = [z,   z,   z,   z,   z+h, z+h, z+h, z+h]

        if is_wireframe:
            lines_x, lines_y, lines_z = [], [], []
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for e in edges:
                lines_x.extend([x_verts[e[0]], x_verts[e[1]], None])
                lines_y.extend([y_verts[e[0]], y_verts[e[1]], None])
                lines_z.extend([z_verts[e[0]], z_verts[e[1]], None])
            return go.Scatter3d(
                x=lines_x, y=lines_y, z=lines_z,
                mode='lines',
                line=dict(color=color, width=2),
                name=name,
                hoverinfo='skip'
            )
        else:
            # 2 triángulos por cara, 6 caras = 12 triángulos
            i_idx = [0, 0,  4, 4,  0, 0,  2, 2,  0, 0,  1, 1]
            j_idx = [1, 2,  5, 6,  1, 5,  3, 7,  3, 7,  2, 6]
            k_idx = [2, 3,  6, 7,  5, 4,  7, 6,  7, 4,  6, 5]

            return go.Mesh3d(
                x=x_verts, y=y_verts, z=z_verts,
                i=i_idx, j=j_idx, k=k_idx,
                color=color,
                opacity=opacity,
                name=name,
                flatshading=True
            )
        
    def plot(self, blocks, env):
        placed_data_batch, space_data_batch = env.get_path()

        pblocks_history = self.process_pblock_data_batch(placed_data_batch)
        spaces_history = self.process_space_data_batch(space_data_batch)

        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
        
        frames = []
        num_steps = len(pblocks_history)
        max_blocks = max(len(pblocks_history[t]) for t in range(num_steps))
        max_traces = 1 + max_blocks + 1  # wireframe + bloques + espacio

        placeholder = lambda: go.Mesh3d(
            x=[0,0,0], y=[0,0,0], z=[0,0,0],
            i=[0], j=[0], k=[0],
            opacity=0, showlegend=False, hoverinfo='skip'
        )

        for t in range(num_steps):
            frame_data = []

            # Contenedor base (1 traza)
            frame_data.append(self._get_cube_mesh(
                0, 0, 0, 1.0, 1.0, 1.0,
                color='rgba(128, 128, 128, 0.8)', name="Contenedor Base",
                is_wireframe=True
            ))

            # Bloques acumulados (1 traza por bloque)
            for pb in pblocks_history[t]:
                orig_block = blocks[pb.id]
                color = colors[pb.id % len(colors)]
                frame_data.append(self._get_cube_mesh(
                    pb.x, pb.y, pb.z, orig_block.l, orig_block.w, orig_block.h,
                    color=color, name=f"Bloque {pb.id}"
                ))

            # Placeholders para bloques faltantes
            missing_blocks = max_blocks - len(pblocks_history[t])
            for _ in range(missing_blocks):
                frame_data.append(placeholder())

            # Espacio disponible (1 traza) o placeholder
            if t < len(spaces_history):
                sp = spaces_history[t]
                frame_data.append(self._get_cube_mesh(
                    sp.x, sp.y, sp.z, sp.l, sp.w, sp.h,
                    color='rgba(255, 165, 0, 0.3)', name="Espacio Disponible",
                    opacity=0.2
                ))
            else:
                frame_data.append(placeholder())

            frames.append(go.Frame(
                data=frame_data,
                traces=list(range(max_traces)),
                name=str(t)
            ))

        initial_data = list(frames[0].data)

        layout = go.Layout(
            title="Simulación de Empaquetado 3D (Espacio Normalizado)",
            scene=dict(
                xaxis=dict(title='X', range=[0, 1.1]),
                yaxis=dict(title='Y', range=[0, 1.1]),
                zaxis=dict(title='Z', range=[0, 1.1]),
                aspectmode='cube'
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="▶ Reproducir", method="animate", args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True)]),
                    dict(label="⏸ Pausar", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(t)], dict(mode="immediate", frame=dict(duration=200, redraw=True))], label=f"Paso {t}") for t in range(num_steps)],
                transition=dict(duration=0),
                x=0, y=0, currentvalue=dict(font=dict(size=12), prefix="Estado: ", visible=True, xanchor="right")
            )]
        )

        fig = go.Figure(data=initial_data, layout=layout, frames=frames)
        fig.show()