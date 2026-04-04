"""
    Simulator class for robocasa environment.
"""
import argparse
import h5py
import json
import os 
import csv
import robosuite
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
from robocasa.utils.dataset_registry import get_ds_path
from termcolor import colored
from collections import OrderedDict
from robocasa.eval.simulate_server import SimulateServer
import pdb
from robocasa.utils.eval_utils import create_eval_env

single_stage_tasks=[
    # "PnPCounterToCab",
    # "PnPCabToCounter",
    # "PnPCounterToSink",
    # "PnPSinkToCounter",
    # "PnPCounterToMicrowave",
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
    def __init__(self, task, num_trials, save_dir, port, episode_length_factor, chunk_length, camera_height, camera_width, save_images=True, save_actions=True):
        self.task=task
        self.num_trials=num_trials
        self.save_dir=save_dir
        self.port=port
        self.dataset_path=get_ds_path(task, ds_type="human_im")
        self.episode_length_factor=episode_length_factor
        self.chunk_length=chunk_length
        self.camera_names=["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]
        self.camera_height=camera_height
        self.camera_width=camera_width
        self.save_images=save_images
        self.save_actions=save_actions
        self.plot_comparison=False
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
    
    def _run_single_trial(self, env, trial_index, video_path, save_images=True):
        """each trial"""

        # reference trajectory length from the dataset
        ep_meta = env.get_ep_meta()
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

        # print the shape of the images from the eye-in-hand camera
        # print(colored(f"Observation shape: {init_obs['robot0_eye_in_hand_image'].shape}", "light_blue")) # [128, 128, 3]
        pred_actions = []  # actions actually passed to env.step
        success = False

        i = 0
        # obs = {}
        # Get the real initial observation
        obs= env.reset()   # the images in _get_observations() is already [128, 128, 3]
        while i + self.chunk_length <= env.horizon:
            # get the response from the server
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

                if i % 100 == 0:
                    print(f"Executing step {i} of trial {trial_index + 1}")
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

        # summary
        actual_length = i + 1 if success else i
        if success:
            print(colored(f"Task succeeds", "green"))
        else:
            print(colored(f"Exceeds max steps {env.horizon}, task fails", "red"))

        return {
            "trial_index": trial_index,
            "actual_length": actual_length,
            "success": 1 if success else 0,
            "instruction": instruction
        }

    def run_simulation(self):
        print(colored(f"\nEvaluating task {self.task}", "light_blue"))
        
        task_dir = os.path.join(self.save_dir, self.task)
        os.makedirs(task_dir, exist_ok=True)
        
        # task level
        all_results = []
        
        # create task level CSV file
        task_csv_path = os.path.join(task_dir, "avg.csv")
        task_csv_headers = [
            "trial_index",
            "actual_length",
            "success",
            "instruction",
        ]
        
        with open(task_csv_path, "w", newline="") as task_csv_file:
            task_csv_writer = csv.DictWriter(task_csv_file, fieldnames=task_csv_headers)
            task_csv_writer.writeheader()
            
            env = create_eval_env(env_name=self.task, seed=0)
            for trial_index in range(self.num_trials):
                print(colored(f"\n--- Trial {trial_index + 1}/{self.num_trials} ---", "light_blue"))
                # video path
                video_path = os.path.join(task_dir, f"trial_{trial_index}.mp4")
                # run single trial
                result = self._run_single_trial(
                    env, trial_index, video_path,
                    save_images=self.save_images
                )

                # save to task CSV
                task_row = {k: result[k] for k in task_csv_headers}
                task_csv_writer.writerow(task_row)
                task_csv_file.flush()

                all_results.append(result)

            env.close()

        # calculate task level statistics
        total_trials = len(all_results)
        avg_actual_length = sum(r["actual_length"] for r in all_results) / total_trials
        avg_success_rate = sum(r["success"] for r in all_results) / total_trials
        
        # save task level JSON summary file
        task_summary = {
            "task": self.task,
            "num_trials": self.num_trials,
            "total_trials": total_trials,
            "episode_length_factor": self.episode_length_factor,
            "avg_actual_length": avg_actual_length,
            "avg_success_rate": avg_success_rate
        }
        
        task_json_path = os.path.join(task_dir, f"{self.task}_summary.json")
        with open(task_json_path, "w") as f:
            json.dump(task_summary, f, indent=4)
        
        print(f"\n{'='*60}")
        print(colored(f"Task {self.task} Evaluation Complete!", "green"))
        print((f"{'='*60}"))
        print(f"Total Trials: {total_trials}")
        print(f"Average Trial Length: {avg_actual_length:.2f}")
        print(f"Results saved to: {task_dir}")
        print(colored(f"Average Success Rate: {avg_success_rate:.2%}", "green"))
        

    def plot_action_comparison(self, pred_traj_np, gt_traj_np):
        episode=self.episodes[0] if isinstance(self.episodes,list) else self.episodes

        timesteps = pred_traj_np.shape[0]
        dim = pred_traj_np.shape[1]

        fig, axs = plt.subplots(dim, 1, figsize=(15, 5 * dim), sharex=True)
        fig.suptitle(f"Action Comparison for {self.task} episode {episode}", fontsize=16)

        if dim == 1:
            axs = [axs]

        for k in range(dim):
            axs[k].plot(range(timesteps), gt_traj_np[:, k], label="Ground Truth")
            axs[k].plot(range(timesteps), pred_traj_np[:, k], label="Prediction")
            axs[k].set_ylabel(f"Action Dim {k+1}")
            axs[k].legend()
            axs[k].grid(True)

        axs[-1].set_xlabel("Timestep")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        save_path = os.path.join(self.save_dir, f"{self.task}_action_comparison.png")
        plt.savefig(save_path)
        print(f"Saved plot to {save_path}")
        plt.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # number of test times for each episode
    parser.add_argument("--num_trials",type=int, default=50)
    # eposide legth factor, default to x times the original eposide length
    parser.add_argument("--episode_length_factor", type=float, default=2)
    # chunk length for action choice, default to 20
    parser.add_argument("--chunk_length", type=int, default=20)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save_dir", type=str, default="/home/zhangxinyue/robocasa/eval_trials/0303_debug/pi05_baseline_simple_eval")
    parser.add_argument("--save_images", action="store_true", help="Save images for each step in trial folders")
    parser.add_argument("--save_actions", action="store_true", help="Save executed actions passed to env.step for each trial")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    camera_height=512
    camera_width=768

    
    for task in single_stage_tasks:
        server = Simulator(task, args.num_trials, args.save_dir, args.port, args.episode_length_factor, args.chunk_length, camera_height, camera_width, save_images=args.save_images, save_actions=args.save_actions)
        server.run_simulation()
        server.server.close()