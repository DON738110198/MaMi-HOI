import viser
import trimesh
import time
import glob
import viser.transforms as tf
import numpy as np
import natsort
from pathlib import Path
import tyro
import os
import uuid

# DEFAULT_ROOT_DIR = r"/data2/wh/hoi_diffusion_model/gama_50"
DEFAULT_ROOT_DIR = r"/data2/wh/hoi_diffusion_model/control_GAPA_chois_long_seq_in_scene_results/objs_single_window_cmp_settings/chois"
def load_ply_mesh(filepath):

    if filepath is None:
        return np.zeros((1, 3)), np.zeros((0, 3), dtype=int)

    try:
        mesh = trimesh.load_mesh(str(filepath), process=False)
        if hasattr(mesh, 'faces'):
             if len(mesh.faces) > 0 and not (mesh.faces.shape[1] == 3 or mesh.faces.shape[1] == 4):
                return np.array(mesh.vertices), np.zeros((0, 3), dtype=int)
             return np.array(mesh.vertices), np.array(mesh.faces)
        elif hasattr(mesh, 'vertices'):
             return np.array(mesh.vertices), np.zeros((0, 3), dtype=int)
        return np.zeros((1, 3)), np.zeros((0, 3), dtype=int)
    except Exception as e:
        print(f"加载 {filepath} 失败: {e}")
        return np.zeros((1, 3)), np.zeros((0, 3), dtype=int)


def get_floor_vertices(half_size, center_pos=(0, 0, 0)):



    z_height = -0.02
    cx, cy = center_pos[0], center_pos[1]
    return np.array([
        [cx - half_size, cy - half_size, z_height],
        [cx + half_size, cy - half_size, z_height],
        [cx + half_size, cy + half_size, z_height],
        [cx - half_size, cy + half_size, z_height],
    ])

def main(root_dir: Path = Path(DEFAULT_ROOT_DIR)):

    if not root_dir.exists():
        print(f"错误: 找不到根路径 '{root_dir}'")
        return

    sample_folders = natsort.natsorted([f for f in root_dir.iterdir() if f.is_dir()], key=lambda x: x.name)
    if not sample_folders:
        print(f"在 {root_dir} 下没有找到样本文件夹。")
        return
    sample_names = [f.name for f in sample_folders]
    print(f"扫描到 {len(sample_names)} 个样本。")


    server = viser.ViserServer()


    state = {
        "human_files": [],
        "object_files": [],
        "waypoint_files": [],
        "marked_handles": [],
        "current_frame_idx": 0,
        "floor_center": (0, 0, 0)
    }


    sequence_dropdown = server.gui.add_dropdown("选择样本 (Sample)", options=sample_names, initial_value=sample_names[0])
    frame_slider = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)


    with server.gui.add_folder("场景环境 (Environment)", viser.Icon.SUN):
        floor_color_picker = server.gui.add_rgb("地板颜色", initial_value=(0.5, 0.5, 0.5))
        floor_visible_checkbox = server.gui.add_checkbox("显示地板", initial_value=True)
        floor_size_slider = server.gui.add_slider("地板大小 (半边长)", min=0.1, max=100.0, step=0.1, initial_value=5.0)


    with server.gui.add_folder("标记工具 (Markers)", viser.Icon.PIN):
        mark_btn = server.gui.add_button("标记当前帧 (Mark)", icon=viser.Icon.CAMERA)
        clear_marks_btn = server.gui.add_button("清除所有标记 (Clear)", icon=viser.Icon.TRASH)
        mark_opacity = server.gui.add_slider("标记透明度", min=0.1, max=1.0, step=0.1, initial_value=0.5)


    with server.gui.add_folder("相机控制 (Camera)", viser.Icon.DEVICE_GAMEPAD):
        camera_speed_slider = server.gui.add_slider("移动速度 (Speed)", min=0.01, max=1.0, step=0.01, initial_value=0.1)
        server.gui.add_markdown(
            "**快捷键控制:**\n\n"
            "* **W / S**: 前进 / 后退\n"
            "* **A / D**: 向左 / 向右\n"
            "* **Q / E**: 向上 / 向下\n"
        )


    with server.gui.add_folder("人体 (Human)", viser.Icon.USER):
        show_person_checkbox = server.gui.add_checkbox("显示人体", initial_value=True)
        person_color_picker = server.gui.add_rgb("人体颜色", initial_value=(59, 159, 179))


    with server.gui.add_folder("物体 (Object)", viser.Icon.BOX):
        show_object_checkbox = server.gui.add_checkbox("显示物体", initial_value=True)
        object_color_picker = server.gui.add_rgb("物体颜色", initial_value=(220, 161, 237))

    # Waypoints
    with server.gui.add_folder("路径点 (Waypoints)", viser.Icon.MAP_PIN):
        show_waypoint_checkbox = server.gui.add_checkbox("显示 Waypoints", initial_value=True)
        waypoint_color_picker = server.gui.add_rgb("Waypoints 颜色", initial_value=(0.2, 0.8, 0.2))




    initial_floor_verts = get_floor_vertices(floor_size_slider.value, state["floor_center"])

    floor_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    floor_handle = server.scene.add_mesh_simple(
        name="/floor",
        vertices=initial_floor_verts,
        faces=floor_faces,
        color=floor_color_picker.value,
        visible=True
    )

    human_mesh_handle = server.scene.add_mesh_simple(
        name="/human_mesh", vertices=np.zeros((1, 3)), faces=np.zeros((0, 3)),
        color=person_color_picker.value, visible=True
    )

    object_mesh_handle = server.scene.add_mesh_simple(
        name="/object_mesh", vertices=np.zeros((1, 3)), faces=np.zeros((0, 3)),
        color=object_color_picker.value, visible=True
    )

    waypoint_mesh_handle = server.scene.add_mesh_simple(
        name="/waypoint_mesh", vertices=np.zeros((1, 3)), faces=np.zeros((0, 3)),
        color=waypoint_color_picker.value, visible=True
    )



    @floor_color_picker.on_update
    def _(_): floor_handle.color = floor_color_picker.value

    @floor_visible_checkbox.on_update
    def _(_): floor_handle.visible = floor_visible_checkbox.value

    @floor_size_slider.on_update
    def _(_):
        new_size = floor_size_slider.value
        new_verts = get_floor_vertices(new_size, state["floor_center"])
        floor_handle.vertices = new_verts

    def handle_camera_move(event: viser.GuiEvent):

        if tf is None: return

        cam = server.scene.camera
        current_pos = np.array(cam.position)
        current_wxyz = np.array(cam.wxyz)

        R = tf.SO3(wxyz=current_wxyz).as_matrix()
        forward_vec = -R[:, 2]
        right_vec = R[:, 0]
        up_vec = R[:, 1]

        step_size = camera_speed_slider.value
        new_pos = current_pos.copy()

        key_val = getattr(event, 'key', None)
        if key_val is None and hasattr(event, 'client'):
             key_val = getattr(event.client, 'key', None)

        if key_val:
            k = key_val.lower()
            if k == "w": new_pos += forward_vec * step_size
            elif k == "s": new_pos -= forward_vec * step_size
            elif k == "a": new_pos -= right_vec * step_size
            elif k == "d": new_pos += right_vec * step_size
            elif k == "q": new_pos += up_vec * step_size
            elif k == "e": new_pos -= up_vec * step_size
            cam.position = new_pos

    try:
        keys_to_bind = ["w", "a", "s", "d", "q", "e", "W", "A", "S", "D", "Q", "E"]
        for k in keys_to_bind:
            server.gui.add_key_event_handler(key=k, handler=handle_camera_move)
    except AttributeError:
        pass


    def clear_all_marks():
        for handle in state["marked_handles"]:
            handle.remove()
        state["marked_handles"] = []


    def update_marked_visibility():

        for handle in state["marked_handles"]:

            if "human" in handle.name:
                handle.visible = show_person_checkbox.value
            elif "obj" in handle.name:
                handle.visible = show_object_checkbox.value

    def parse_sample_data(sample_name, root_dir):
        sample_path = root_dir / sample_name
        if not sample_path.exists(): return [], [], []

        sub_dirs = [x for x in sample_path.iterdir() if x.is_dir()]
        hoi_dir, waypoint_dir = None, None

        for d in sub_dirs:
            d_name = d.name.lower()
            if "ball" in d_name:
                waypoint_dir = d
            elif "objs" in d_name and "ball" not in d_name:
                hoi_dir = d

        human_plys, object_plys, waypoint_plys = [], [], []

        if hoi_dir:
            all_hoi_files = natsort.natsorted(glob.glob(str(hoi_dir / "*.ply")))
            for p_str in all_hoi_files:
                p = Path(p_str)
                if p.stem.endswith("_object"): object_plys.append(p)
                elif p.stem.isdigit(): human_plys.append(p)

        if waypoint_dir:
            waypoint_plys = natsort.natsorted(list(waypoint_dir.glob("*.ply")))

        return human_plys, object_plys, waypoint_plys

    def load_sequence_data(sample_name):
        clear_all_marks()

        human_files, object_files, waypoint_files = parse_sample_data(sample_name, root_dir)
        state.update({"human_files": human_files, "object_files": object_files, "waypoint_files": waypoint_files})

        num_frames = min(len(human_files), len(object_files))
        print(f"\n--- 加载样本: {sample_name} ---")


        if num_frames > 0:
            h_v, _ = load_ply_mesh(human_files[0])
            if len(h_v) > 0:
                start_center = np.mean(h_v, axis=0)
                state["floor_center"] = start_center
                new_floor_verts = get_floor_vertices(floor_size_slider.value, start_center)
                floor_handle.vertices = new_floor_verts


        # Waypoints
        if waypoint_files:
            all_w_verts, all_w_faces = [], []
            vertex_offset = 0
            for wp_path in waypoint_files:
                v, f = load_ply_mesh(wp_path)
                if len(v) > 0:
                    all_w_verts.append(v)
                    all_w_faces.append(f + vertex_offset)
                    vertex_offset += len(v)
            if all_w_verts:
                waypoint_mesh_handle.vertices = np.vstack(all_w_verts)
                if any(len(x) > 0 for x in all_w_faces):
                    waypoint_mesh_handle.faces = np.vstack(all_w_faces)
                else:
                    waypoint_mesh_handle.faces = np.zeros((0, 3), dtype=int)
                waypoint_mesh_handle.visible = show_waypoint_checkbox.value
            else:
                waypoint_mesh_handle.visible = False
        else:
            waypoint_mesh_handle.visible = False

        if num_frames > 0:
            frame_slider.max = max(1, num_frames - 1)
            frame_slider.value = 0
            update_scene(0)
        else:
            human_mesh_handle.visible = False
            object_mesh_handle.visible = False

    def update_scene(frame_idx: int):
        if not state["human_files"]: return
        if frame_idx >= len(state["human_files"]): return


        if show_person_checkbox.value:
            h_v, h_f = load_ply_mesh(state["human_files"][frame_idx])
            human_mesh_handle.vertices, human_mesh_handle.faces = h_v, h_f
            human_mesh_handle.visible = True
        else: human_mesh_handle.visible = False


        if show_object_checkbox.value:
            o_v, o_f = load_ply_mesh(state["object_files"][frame_idx])
            object_mesh_handle.vertices, object_mesh_handle.faces = o_v, o_f
            object_mesh_handle.visible = True
        else: object_mesh_handle.visible = False

        state["current_frame_idx"] = frame_idx


    @mark_btn.on_click
    def _(_):
        idx = state["current_frame_idx"]
        if not state["human_files"]: return

        unique_id = str(uuid.uuid4())[:8]

        if show_person_checkbox.value:
            h_v, h_f = load_ply_mesh(state["human_files"][idx])

            mark_h = server.scene.add_mesh_simple(
                name=f"/marked/human_{unique_id}", vertices=h_v, faces=h_f,
                color=person_color_picker.value, opacity=mark_opacity.value
            )
            state["marked_handles"].append(mark_h)

        if show_object_checkbox.value:
            o_v, o_f = load_ply_mesh(state["object_files"][idx])

            mark_o = server.scene.add_mesh_simple(
                name=f"/marked/obj_{unique_id}", vertices=o_v, faces=o_f,
                color=object_color_picker.value, opacity=mark_opacity.value
            )
            state["marked_handles"].append(mark_o)

    @clear_marks_btn.on_click
    def _(_): clear_all_marks()

    @sequence_dropdown.on_update
    def _(_): load_sequence_data(sequence_dropdown.value)

    @frame_slider.on_update
    def _(_): update_scene(int(frame_slider.value))


    @show_person_checkbox.on_update
    @show_object_checkbox.on_update
    def _(_):

        update_scene(int(frame_slider.value))

        update_marked_visibility()

    @person_color_picker.on_update
    def _(_): human_mesh_handle.color = person_color_picker.value
    @object_color_picker.on_update
    def _(_): object_mesh_handle.color = object_color_picker.value
    @waypoint_color_picker.on_update
    def _(_): waypoint_mesh_handle.color = waypoint_color_picker.value
    @show_waypoint_checkbox.on_update
    def _(_):
        if state["waypoint_files"]: waypoint_mesh_handle.visible = show_waypoint_checkbox.value


    if sample_names: load_sequence_data(sample_names[0])
    print(f"\nViser 服务已启动。标记对象的可见性现在与主复选框同步。")
    while True: time.sleep(0.1)

if __name__ == "__main__":
    tyro.cli(main)