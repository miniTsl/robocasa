"""
    Replay saved actions until error_step, then let the remote model take over with full_reasoning.
"""
import argparse
import csv
import h5py
import json
import os
import robosuite
import imageio
import cv2
import numpy as np
from robocasa.utils.dataset_registry import get_ds_path
from termcolor import colored
from robocasa.eval.simulate_server import SimulateServer

# python -m robocasa.eval.simulator_replay_rollout --port 8011 --save_actions --recover_json /home/zhangxinyue/robocasa/robocasa/eval/ep.json --replay_id 0
single_stage_tasks=[
    "PnPCounterToCab",
    "PnPCabToCounter",
    "PnPCounterToSink",
    "PnPSinkToCounter",
    "PnPCounterToMicrowave",
    "PnPMicrowaveToCounter",
    "PnPCounterToStove",
    "PnPStoveToCounter",
    "OpenSingleDoor",
    # "CloseSingleDoor",   # File damaged
    "OpenDoubleDoor",
    "CloseDoubleDoor",
    "OpenDrawer",
    "CloseDrawer",
    "TurnOnSinkFaucet",
    "TurnOffSinkFaucet",
    "TurnSinkSpout",
    "TurnOnStove",
    "TurnOffStove",
    "CoffeeSetupMug",
    "CoffeeServeMug",
    "CoffeePressButton",
    "TurnOnMicrowave",
    "TurnOffMicrowave",
    "NavigateKitchen",
]
multi_stage_tasks=[
    "ArrangeVegetables",
    "MicrowaveThawing",
    "RestockPantry",
    "PreSoakPan",
    "PrepareCoffee",
]
class Simulator:
    def __init__(
        self,
        task,
        save_dir,
        port,
        episode_length_factor,
        chunk_length,
        camera_height,
        camera_width,
        save_images=True,
        save_actions=False,
    ):
        self.task = task
        self.save_dir = save_dir
        self.port = port
        self.dataset_path = get_ds_path(task, ds_type="human_im")
        self.episode_length_factor = episode_length_factor
        self.chunk_length = chunk_length
        self.camera_names = ["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.save_images = save_images
        self.save_actions = save_actions
        self._init_server()
    
    def _init_server(self):
        self.server = SimulateServer(self.port)
        metadata = self.server.metadata
        # save the metadata from model policy to a json file (for debug and identification)
        with open(os.path.join(self.save_dir, "model_policy_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
    def _get_demo_list(self):
        self.dataset_path = os.path.expanduser(self.dataset_path)
        f = h5py.File(self.dataset_path, "r")
        demos = list(f["data"].keys())
        demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))
        f.close()
        return demos
    
    def _init_env(self, episode_index):
        self.dataset_path = os.path.expanduser(self.dataset_path)
        f = h5py.File(self.dataset_path, "r")
        demos = self._get_demo_list()
        
        assert episode_index < len(demos), f"episode_index({episode_index}) must be less than demo_nums({len(demos)})"
        ep = demos[episode_index]
        print(colored(f"Running with initial state from {ep}", "yellow"))
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]  # reference actions from the dataset
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)

        # get_env_metadata_from_dataset
        env_meta = json.loads(f["data"].attrs["env_args"])
        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"] = self.camera_names
        env_kwargs["ignore_done"] = False
        env_kwargs["horizon"] = int(actions.shape[0] * self.episode_length_factor)
        # set the random option for texture and camera initialization
        env_kwargs["generative_textures"] = None #"100p"
        env_kwargs["randomize_cameras"] = False #True
        print(
            colored(
                "Initializing environment for {}...".format(env_kwargs["env_name"]),
                "light_blue",
            )
        )
        print(colored(f"Setting task horizon {actions.shape[0]} * {self.episode_length_factor} = {env_kwargs['horizon']}", "light_blue"))
        # create the simulation environment
        
        env = robosuite.make(**env_kwargs)
        f.close()
        return ep, env, states, initial_state, actions

    def _reset_to(self, env, initial_state):
        if "model" in initial_state:
            if initial_state.get("ep_meta", None) is not None:
                # set relevant episode information from episode metadata
                ep_meta = json.loads(initial_state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
                env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(env, "set_ep_meta"):  # newer versions
                env.set_ep_meta(ep_meta)
                
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            env.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(initial_state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = env.edit_model_xml(initial_state["model"])

            env.reset_from_xml_string(xml)
            env.sim.reset()
            # hide teleop visualization after restoring from model
            # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
            # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in initial_state:
            env.sim.set_state_from_flattened(initial_state["states"])
            env.sim.forward()

        # update state as needed
        if hasattr(env, "update_sites"):
            # older versions of environment had update_sites function
            env.update_sites()
        if hasattr(env, "update_state"):
            # later versions renamed this to update_state
            env.update_state()
    
    def _get_sim_render_image(self, env, camera_name, height, width):
        return env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
    
    def _run_single_trial(
        self,
        env,
        states_ref,
        initial_state,
        actions_ref,
        episode_index,
        trial_index,
        video_path,
        save_images=True,
        saved_actions_np=None,
        error_step=None,
        full_reasoning=None,
        last_reasoning_step=None,
    ):
        """each trial; if saved_actions_np and error_step set, replay that many steps first"""

        # reference trajectory length from the dataset
        traj_len = actions_ref.shape[0]
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "light_blue"))
            instruction = lang
        else:
            instruction = ""

        # create video writer
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(colored(f"Trial video save path: {video_path}", "light_blue"))
        video_writer = imageio.get_writer(video_path, fps=20)

        # create trial folder for saving images
        if save_images:
            trial_images_dir = os.path.join(os.path.dirname(video_path), f"trial_{trial_index}_images")
            os.makedirs(trial_images_dir, exist_ok=True)
            # create subdirectories for each camera view
            for cam_name in self.camera_names:
                cam_dir = os.path.join(trial_images_dir, cam_name)
                os.makedirs(cam_dir, exist_ok=True)
            print(colored(f"Trial images save path: {trial_images_dir}", "light_blue"))

        # prepare txt path for writing inference info
        txt_path = os.path.splitext(video_path)[0] + ".txt"
        inference_info_lines = []

        # set the environment to initial state. Notice there is another env.sim reset after env.reset
        self._reset_to(env, initial_state)
        # print the shape of the images from the eye-in-hand camera
        # print(colored(f"Observation shape: {init_obs['robot0_eye_in_hand_image'].shape}", "light_blue")) # [128, 128, 3]
        
        action_deviation = 0.0
        state_deviation = 0.0
        pred_actions = []  # actions actually passed to env.step
        success = False
        step_action_mse_list = []
        step_state_mse_list = []

        i = 0
        sampled_history_images = []
        reasoning_start_step = int(last_reasoning_step) if last_reasoning_step is not None else None
        # obs = {}
        # Get the real initial observation
        obs = env._get_observations(force_update=True)   # the images in _get_observations() is already [128, 128, 3]

        if saved_actions_np is not None and error_step is not None:
            for k in range(int(error_step)):
                if reasoning_start_step is not None and i >= reasoning_start_step and (i - reasoning_start_step) % 20 == 0:
                    sampled_images = self.server._process_images_for_model_inference(obs)
                    sampled_history_images.append(sampled_images)

                video_img = []
                for cam_name in self.camera_names:
                    im = self._get_sim_render_image(env, cam_name, self.camera_height, self.camera_width)
                    video_img.append(im)
                    if save_images:
                        img_path = os.path.join(trial_images_dir, cam_name, f"step_{i:05d}.jpg")
                        cv2.imwrite(img_path, im[:, :, ::-1])  # convert RGB to BGR for cv2
                video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                video_writer.append_data(video_img)

                action = np.array(saved_actions_np[k], dtype=np.float64, copy=True).reshape(-1)
                if action.shape[0] >= 12:
                    action[7:] = [0, 0, 0, 0, -1]
                pred_actions.append(np.array(action, copy=True))
                obs, reward, done, info = env.step(action)

                if i < traj_len - 1:
                    step_action_mse = np.mean((action - actions_ref[i]) ** 2)
                    action_deviation += step_action_mse
                    step_action_mse_list.append((i, step_action_mse))  # (step, action_mse)
                    state_playback = np.array(env.sim.get_state().flatten())
                    step_state_mse = np.mean((states_ref[i + 1] - state_playback) ** 2)
                    state_deviation += step_state_mse
                    step_state_mse_list.append((i, step_state_mse))  # (step, state_mse)

                if i % 100 == 0:
                    print(f"Executing step {i} of episode {episode_index + 1}, trial {trial_index + 1}")
                if not success and env._check_success():
                    success = True
                    print(colored(f"Task succeeds at step {i}", "green"))

                if success:
                    break

                i += 1


        first = True
        while i + self.chunk_length <= env.horizon:
            # get the response from the server
            # if first:
            #     model_response = self.server.get_response_from_server(obs, instruction, i, full_reasoning=full_reasoning, sampled_history_images=sampled_history_images)
            #     first = False
            # else:
            #     model_response = self.server.get_response_from_server(obs, instruction, i)
            model_response = self.server.get_response_from_server(obs, instruction, i)
            predict_action = model_response["predict_action"]

            inference_idx = i
            inference_mode = model_response.get("inference_mode", "")
            reasoning = model_response.get("reasoning", "")
            infer_ms = model_response["server_timing"]["infer_ms"]
            line = f"step：{inference_idx}，inference_mode：{inference_mode}，reasoning：{reasoning}，infer_ms：{infer_ms}"
            inference_info_lines.append(line)

            # fix the action dimension from 7 to 12, so the base will not move randomly (not needed anymore since we train the model with the base action)
            #action_queue = predict_action[0, :, :]
            action_queue = predict_action
            # execute the action chunk
            for j in range(min(self.chunk_length, action_queue.shape[0])):
                # render video and save images
                video_img = []
                for cam_name in self.camera_names:
                    im = self._get_sim_render_image(env, cam_name, self.camera_height, self.camera_width)
                    video_img.append(im)
                    if save_images:
                        img_path = os.path.join(trial_images_dir, cam_name, f"step_{i:05d}.jpg")
                        cv2.imwrite(img_path, im[:, :, ::-1])  # convert RGB to BGR for cv2
                video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                video_writer.append_data(video_img)
                
                action = action_queue[j, :]
                action[7:] = [0, 0, 0, 0, -1] # not needed anymore since we train the model with the base action
                # record the exact action sent to simulator (make a copy to avoid aliasing)
                pred_actions.append(np.array(action, copy=True))
                # step the environment with the action(12,)
                obs, reward, done, info = env.step(action)

                # compute norms when the predicted action length < groundtruth action length
                if i < traj_len - 1:
                    step_action_mse = np.mean((action - actions_ref[i]) ** 2)
                    action_deviation += step_action_mse
                    step_action_mse_list.append((i, step_action_mse))  # (step, action_mse)
                    state_playback = np.array(env.sim.get_state().flatten())
                    step_state_mse = np.mean((states_ref[i + 1] - state_playback) ** 2)
                    state_deviation += step_state_mse
                    step_state_mse_list.append((i, step_state_mse))  # (step, state_mse)

                if i % 100 == 0:
                    print(f"Executing step {i} of episode {episode_index + 1}, trial {trial_index + 1}")
                # check if succeeds
                if not success and env._check_success():
                    success = True
                    print(colored(f"Task succeeds at step {i}", "green"))
                
                if success:
                    break
                
                i += 1

            if success:
                # render video and save images for the last step
                video_img = []
                for cam_name in self.camera_names:
                    im = env.sim.render(
                        height=self.camera_height, width=self.camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                    # save individual camera images
                    if save_images:
                        img_path = os.path.join(trial_images_dir, cam_name, f"step_{i:05d}.jpg")
                        cv2.imwrite(img_path, im[:, :, ::-1])  # convert RGB to BGR for cv2
                video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                video_writer.append_data(video_img)
                print(colored(f"Saved video and images for the successful state at step {i}", "light_blue"))
                break
        
        # close video writer
        video_writer.close()

        # save predicted / executed actions for this trial
        if self.save_actions:
            actions_path = os.path.splitext(video_path)[0] + "_actions.npy"
            # TODO: check data type of saved actions
            np.save(actions_path, np.array(pred_actions, dtype=np.float64))
            print(colored(f"Saved executed actions to: {actions_path}", "light_blue"))
        
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            for line in inference_info_lines:
                f_txt.write(line + "\n")

        action_mse_txt_path = os.path.splitext(video_path)[0] + "_action_mse.txt"
        with open(action_mse_txt_path, "w", encoding="utf-8") as f_action_mse:
            f_action_mse.write("step, action_mse\n")
            for step, mse in step_action_mse_list:
                f_action_mse.write(f"{step},{mse}\n")
        print(colored(f"Step-wise action MSE saved to: {action_mse_txt_path}", "light_blue"))

        state_mse_txt_path = os.path.splitext(video_path)[0] + "_state_mse.txt"
        with open(state_mse_txt_path, "w", encoding="utf-8") as f_state_mse:
            f_state_mse.write("step, state_mse\n")
            for step, mse in step_state_mse_list:
                f_state_mse.write(f"{step},{mse}\n")
        print(colored(f"Step-wise state MSE saved to: {state_mse_txt_path}", "light_blue"))

        # summary
        actual_length = i + 1 if success else i
        if success:
            print(colored(f"Task succeeds", "green"))
        else:
            print(colored(f"Exceeds max steps {env.horizon}, task fails", "red"))

        action_mse = action_deviation / min(actual_length, traj_len)
        state_mse = state_deviation / min(actual_length, traj_len)
        print(colored(f"Action MSE: {action_mse}", "yellow"))
        print(colored(f"State MSE: {state_mse}", "yellow"))

        return {
            "trial_index": trial_index,
            "gt_length": traj_len,
            "actual_length": actual_length,
            "action_mse": action_mse,
            "state_mse": state_mse,
            "success": 1 if success else 0,
            "instruction": instruction
        }

    @staticmethod
    def _resolve_action_path(recover_json_path, action_path_field):
        p = action_path_field
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(recover_json_path)), p))
        else:
            p = os.path.expanduser(p)
        return p

    def run_recovery_simulation(self, recover_json_path, replay_id):
        """Load ep_w_recover.json; replay actions from action_path until error_step; then model with full_reasoning.

        Outputs (video, txt, mse, optional actions) go under dirname(action_path)/replay_rollout/.
        Video name includes replay_id.
        """
        recover_json_path = os.path.abspath(recover_json_path)
        with open(recover_json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["task_type"] == self.task
        ep_id = int(cfg["ep_id"])
        trial_id = int(cfg["trial_id"])
        error_step = int(cfg["error_step"])
        full_reasoning = cfg.get("last_reasoning")
        last_reasoning_step = cfg.get("last_reasoning_step", None)

        action_path_abs = self._resolve_action_path(recover_json_path, cfg["action_path"])
        saved = np.load(action_path_abs)
        if saved.ndim == 1:
            saved = np.expand_dims(saved, 0)

        replay_out_dir = os.path.join(os.path.dirname(action_path_abs), "replay_rollout")
        os.makedirs(replay_out_dir, exist_ok=True)
        rid = str(replay_id).replace(os.sep, "_").replace("/", "_")
        video_path = os.path.join(replay_out_dir, f"recover_replay_{rid}.mp4")

        _, env, states_ref, initial_state, actions_ref = self._init_env(ep_id)
        try:
            return self._run_single_trial(
                env,
                states_ref,
                initial_state,
                actions_ref,
                ep_id,
                trial_id,
                video_path,
                save_images=self.save_images,
                saved_actions_np=saved,
                error_step=error_step,
                full_reasoning=full_reasoning,
                last_reasoning_step=last_reasoning_step,
            )
        finally:
            env.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Replay from ep_w_recover.json then model recovery.")
    parser.add_argument("--recover_json", type=str, required=True, help="ep_w_recover.json (must contain action_path, etc.)")
    parser.add_argument(
        "--replay_id",
        type=str,
        required=True,
        help="Label for this replay run; used in output video name (e.g. 0, 1, run2)",
    )
    parser.add_argument("--episode_length_factor", type=float, default=2)
    parser.add_argument("--chunk_length", type=int, default=20)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save_images", action="store_true", help="Save images for each step in trial folders")
    parser.add_argument("--save_actions", action="store_true", help="Save executed actions for this recovery trial")
    parser.add_argument(
        "--recover_attempts",
        type=int,
        default=50,
        help="Number of recovery attempts on the same trajectory",
    )

    args = parser.parse_args()

    with open(args.recover_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    task = cfg["task_type"]
    action_path_abs = Simulator._resolve_action_path(args.recover_json, cfg["action_path"])
    replay_out_dir = os.path.join(os.path.dirname(action_path_abs), "replay_rollout")
    os.makedirs(replay_out_dir, exist_ok=True)

    camera_height = 512
    camera_width = 768

    server = Simulator(
        task,
        replay_out_dir,
        args.port,
        args.episode_length_factor,
        args.chunk_length,
        camera_height,
        camera_width,
        save_images=args.save_images,
        save_actions=args.save_actions,
    )

    attempt_results = []
    total_success = 0
    assert args.recover_attempts >= 1, "recover_attempts must be >= 1"
    if args.recover_attempts == 1:
        result = server.run_recovery_simulation(args.recover_json, args.replay_id)
        attempt_results.append(
            {
                "attempt_idx": 0,
                "replay_id": str(args.replay_id),
                "success": int(result.get("success", 0)),
                "actual_length": result.get("actual_length", ""),
                "action_mse": result.get("action_mse", ""),
                "state_mse": result.get("state_mse", ""),
            }
        )
    else:
        for attempt_idx in range(args.recover_attempts):
            replay_id_i = f"{args.replay_id}_attempt{attempt_idx}"
            print(colored(f"Running recovery attempt {attempt_idx + 1}/{args.recover_attempts}, replay_id={replay_id_i}", "yellow"))
            result = server.run_recovery_simulation(args.recover_json, replay_id_i)
            if result.get("success", 0):
                total_success += 1
            print("now success count", total_success)
            attempt_results.append(
                {
                    "attempt_idx": attempt_idx,
                    "replay_id": replay_id_i,
                    "success": int(result.get("success", 0)),
                    "actual_length": result.get("actual_length", ""),
                    "action_mse": result.get("action_mse", ""),
                    "state_mse": result.get("state_mse", ""),
                }
            )

    summary_csv_path = os.path.join(replay_out_dir, f"recover_summary_{str(args.replay_id)}.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["attempt_idx", "replay_id", "success", "actual_length", "action_mse", "state_mse"],
        )
        writer.writeheader()
        writer.writerows(attempt_results)
    print(colored(f"Saved recovery summary to: {summary_csv_path}", "light_blue"))
    print(colored(f"Total successful recoveries: {total_success}/{args.recover_attempts}", "light_blue"))

    server.server.close()