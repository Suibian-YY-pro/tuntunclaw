import os.path
import sys

import cv2
import glfw
import mujoco
import mujoco.viewer
import numpy as np
import spatialmath as sm
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MANIPULATOR_ROOT = ROOT_DIR / "manipulator_grasp"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(MANIPULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(MANIPULATOR_ROOT))

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.arm.robot import Robot, UR5e
from manipulator_grasp.utils import mj
from camera_view import load_view_config


class UR5GraspEnv:
    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None

        render_size = int(os.getenv("OPENCLAW_RENDER_SIZE", "960"))
        self.height = render_size
        self.width = render_size
        self.fovy = np.pi / 4
        self.camera_fovy_deg = 45.0

        self.camera_name = "cam"
        self.camera_id = -1
        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_option = None
        self.offscreen_perturb = None
        self.offscreen_viewport = None
        self.glfw_window = None
        self.render_backend = None
        self.scene_option = None
        self.scene_path = None
        self.render_camera = None

    def _make_render_camera(self):
        config = load_view_config(self.scene_path) if self.scene_path else None
        if config is None:
            return None
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.mj_model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = config["lookat"]
        camera.azimuth = config["azimuth"]
        camera.elevation = config["elevation"]
        camera.distance = config["distance"]
        camera.trackbodyid = -1
        camera.fixedcamid = -1
        return camera

    def _default_hold_action(self):
        action = np.zeros(7, dtype=np.float64)
        action[:6] = self.robot_q
        return action

    def _settle_scene(self, steps: int = 300):
        action = self._default_hold_action()
        for _ in range(steps):
            self.mj_data.ctrl[:] = action
            mujoco.mj_step(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _apply_saved_view(self, viewer_handle):
        config = load_view_config(self.scene_path) if self.scene_path else None
        if config is None:
            viewer_handle.cam.lookat[:] = [4.55, -3.60, 1.0]
            viewer_handle.cam.azimuth = 145
            viewer_handle.cam.elevation = -22
            viewer_handle.cam.distance = 5.4
            return
        viewer_handle.cam.lookat[:] = config["lookat"]
        viewer_handle.cam.azimuth = config["azimuth"]
        viewer_handle.cam.elevation = config["elevation"]
        viewer_handle.cam.distance = config["distance"]

    def _make_scene_option(self):
        option = mujoco.MjvOption()
        # Match RoboCasa / robosuite practice: hide collision geoms (group 0).
        option.geomgroup[0] = 0
        return option

    def _safe_close_renderer(self, renderer):
        if renderer is None:
            return
        try:
            renderer.close()
        except Exception:
            pass

    def _close_render_resources(self):
        self._safe_close_renderer(self.mj_renderer)
        self._safe_close_renderer(self.mj_depth_renderer)
        self.mj_renderer = None
        self.mj_depth_renderer = None

        if self.glfw_window is not None:
            try:
                glfw.make_context_current(None)
            except Exception:
                pass
            try:
                glfw.destroy_window(self.glfw_window)
            except Exception:
                pass
            self.glfw_window = None

        self.offscreen_context = None
        self.offscreen_scene = None
        self.offscreen_camera = None
        self.offscreen_option = None
        self.offscreen_perturb = None
        self.offscreen_viewport = None
        self.render_backend = None

    def refresh_render_backend(self):
        self._close_render_resources()
        self._init_rendering()
        if self.mj_viewer is not None:
            try:
                self._apply_saved_view(self.mj_viewer)
                self.mj_viewer.sync()
            except Exception:
                self.mj_viewer = None

    def _init_standard_renderers(self):
        self.scene_option = self._make_scene_option()
        self.mj_renderer = mujoco.renderer.Renderer(
            self.mj_model, height=self.height, width=self.width
        )
        self.mj_depth_renderer = mujoco.renderer.Renderer(
            self.mj_model, height=self.height, width=self.width
        )
        camera = self.render_camera if self.render_camera is not None else (self.camera_id if self.camera_id >= 0 else 0)
        self.mj_renderer.update_scene(
            self.mj_data, camera, scene_option=self.scene_option
        )
        self.mj_depth_renderer.update_scene(
            self.mj_data, camera, scene_option=self.scene_option
        )
        self.mj_depth_renderer.enable_depth_rendering()
        self.render_backend = "renderer"

    def _init_glfw_offscreen(self):
        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.glfw_window = glfw.create_window(
            self.width, self.height, "MuJoCoOffscreen", None, None
        )
        if self.glfw_window is None:
            glfw.terminate()
            raise RuntimeError("GLFW offscreen window creation failed")

        glfw.make_context_current(self.glfw_window)

        self.camera_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        self.offscreen_camera = mujoco.MjvCamera()
        if self.render_camera is not None:
            self.offscreen_camera = self.render_camera
        elif self.camera_id != -1:
            mujoco.mjv_defaultCamera(self.offscreen_camera)
            self.offscreen_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.offscreen_camera.fixedcamid = self.camera_id
        else:
            mujoco.mjv_defaultFreeCamera(self.mj_model, self.offscreen_camera)

        self.offscreen_option = self._make_scene_option()
        self.offscreen_perturb = mujoco.MjvPerturb()
        self.offscreen_scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)
        self.offscreen_context = mujoco.MjrContext(
            self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value
        )
        self.offscreen_viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.offscreen_context
        )
        self.render_backend = "glfw_offscreen"

    def _init_rendering(self):
        self.mj_renderer = None
        self.mj_depth_renderer = None
        self.render_backend = None
        preferred = os.getenv("OPENCLAW_RENDER_BACKEND", "glfw").strip().lower()
        if preferred in {"glfw", "glfw_offscreen"}:
            self._init_glfw_offscreen()
            print("[env] render backend=glfw_offscreen")
            return
        if preferred == "renderer":
            self._init_standard_renderers()
            print("[env] render backend=renderer")
            return
        try:
            self._init_standard_renderers()
            print("[env] render backend=renderer")
        except Exception as exc:
            print(
                f"[env] standard mujoco.Renderer failed, fallback to GLFW offscreen: {exc}"
            )
            self._safe_close_renderer(self.mj_renderer)
            self._safe_close_renderer(self.mj_depth_renderer)
            self.mj_renderer = None
            self.mj_depth_renderer = None
            self._init_glfw_offscreen()
            print("[env] render backend=glfw_offscreen")

    def _try_launch_viewer(self):
        if os.getenv("OPENCLAW_VIEWER", "1").strip().lower() in {"0", "false", "no"}:
            self.mj_viewer = None
            return
        try:
            self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            self._apply_saved_view(self.mj_viewer)
            try:
                self.mj_viewer.opt.geomgroup[0] = 0
            except Exception:
                pass
            self.mj_viewer.sync()
        except Exception as exc:
            self.mj_viewer = None
            print(f"[env] passive viewer unavailable: {exc}")

    def _render_with_glfw_offscreen(self):
        if self.glfw_window is None or self.offscreen_context is None:
            raise RuntimeError("GLFW offscreen renderer not initialized")

        glfw.make_context_current(self.glfw_window)
        mujoco.mjv_updateScene(
            self.mj_model,
            self.mj_data,
            self.offscreen_option,
            self.offscreen_perturb,
            self.offscreen_camera,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.offscreen_scene,
        )
        mujoco.mjr_render(
            self.offscreen_viewport, self.offscreen_scene, self.offscreen_context
        )

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.zeros((self.height, self.width), dtype=np.float32)
        mujoco.mjr_readPixels(
            rgb, depth, self.offscreen_viewport, self.offscreen_context
        )

        rgb = np.flipud(rgb)
        depth = np.flipud(depth)

        extent = self.mj_model.stat.extent
        near = self.mj_model.vis.map.znear * extent
        far = self.mj_model.vis.map.zfar * extent
        metric_depth = near / (1.0 - depth * (1.0 - near / far))

        return {"img": rgb, "depth": metric_depth}

    def reset(self):
        scenes_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets",
            "scenes",
        )
        default_scene = os.path.join(scenes_dir, "scene_robocasa_layout51_style34.xml")
        filename = os.getenv("OPENCLAW_SCENE_XML", default_scene)
        if not os.path.isabs(filename):
            filename = os.path.join(scenes_dir, filename)
        if not os.path.exists(filename):
            fallback_scene = os.path.join(scenes_dir, "scene.xml")
            print(f"[env] scene not found, fallback to {fallback_scene}")
            filename = fallback_scene
        self.scene_path = Path(filename)
        self.mj_model = mujoco.MjModel.from_xml_path(filename)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.camera_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name
        )
        if self.camera_id >= 0:
            self.camera_fovy_deg = float(self.mj_model.cam_fovy[self.camera_id])
            self.fovy = np.deg2rad(self.camera_fovy_deg)
        else:
            self.camera_fovy_deg = 45.0
            self.fovy = np.deg2rad(self.camera_fovy_deg)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        view_config = load_view_config(self.scene_path)
        self.render_camera = self._make_render_camera()
        if view_config is not None:
            print(
                "[env] saved view locked: "
                f"lookat={view_config['lookat']}, "
                f"azimuth={view_config['azimuth']:.6f}, "
                f"elevation={view_config['elevation']:.6f}, "
                f"distance={view_config['distance']:.6f}"
            )
        else:
            print(f"[env] no saved view found for {self.scene_path.name}; using scene camera")

        self.robot = UR5e()
        self.robot.set_base(
            mj.get_body_pose(self.mj_model, self.mj_data, "ur5e_base").t
        )
        self.robot_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mj.attach(
            self.mj_model,
            self.mj_data,
            "attach",
            "2f85",
            self.robot.fkine(self.robot_q),
        )
        robot_tool = sm.SE3.Trans(0.0, 0.0, 0.13) * sm.SE3.RPY(
            -np.pi / 2, -np.pi / 2, 0.0
        )
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()
        self._settle_scene()

        self._init_rendering()
        self._try_launch_viewer()

    def close(self):
        if self.mj_viewer is not None:
            try:
                self.mj_viewer.close()
            except Exception:
                pass
            self.mj_viewer = None

        self._close_render_resources()

        cv2.destroyAllWindows()
        try:
            glfw.terminate()
        except Exception:
            pass

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        if self.mj_viewer is not None:
            try:
                self.mj_viewer.sync()
            except Exception:
                self.mj_viewer = None

    def render(self):
        for attempt in range(2):
            if self.render_backend == "renderer":
                camera = self.render_camera if self.render_camera is not None else (self.camera_id if self.camera_id >= 0 else 0)
                self.mj_renderer.update_scene(
                    self.mj_data, camera, scene_option=self.scene_option
                )
                self.mj_depth_renderer.update_scene(
                    self.mj_data, camera, scene_option=self.scene_option
                )
                result = {
                    "img": self.mj_renderer.render(),
                    "depth": self.mj_depth_renderer.render(),
                }
            elif self.render_backend == "glfw_offscreen":
                result = self._render_with_glfw_offscreen()
            else:
                raise RuntimeError("No rendering backend is available")

            img = np.asarray(result["img"])
            depth = np.asarray(result["depth"])
            depth_valid = np.isfinite(depth)
            invalid_depth = (
                depth.size == 0
                or not np.any(depth_valid)
                or (np.all(depth_valid) and np.max(depth) > 100.0 and np.min(depth) == np.max(depth))
            )
            invalid_rgb = img.size == 0 or (np.max(img) == 0 and invalid_depth)
            if not invalid_depth and not invalid_rgb:
                return result

            if attempt == 0:
                print("[env] render output invalid, rebuilding render backend without resetting scene state")
                self.refresh_render_backend()
                continue
            raise RuntimeError("Render backend returned invalid RGB-D frame")

        raise RuntimeError("No rendering backend is available")


if __name__ == "__main__":
    env = UR5GraspEnv()
    env.reset()
    for _ in range(1000):
        env.step()
    imgs = env.render()
    print(imgs["img"].shape, imgs["depth"].shape)
    env.close()
