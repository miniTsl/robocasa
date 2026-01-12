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
import pdb
from robocasa.utils.dataset_registry import get_ds_path
from termcolor import colored
from collections import OrderedDict
from robocasa.scripts.simulate_server import SimulateServer



class Simulator:
    def __init__(self, task, num_episodes, num_trials, save_dir, port, episode_length_factor, camera_height, camera_width, is_baseline=False):
        self.task=task
        self.num_episodes=num_episodes
        self.num_trials=num_trials
        self.save_dir=save_dir
        self.port=port
        self.server=SimulateServer(self.port)
        self.dataset_path=get_ds_path(task, ds_type="human_raw")
        self.episode_length_factor=episode_length_factor
        self.camera_names=["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"]
        self.camera_height=camera_height
        self.camera_width=camera_width
        self.is_baseline=is_baseline

        self.plot_comparison=False
        
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
        demos = list(f["data"].keys())
        # sort demos by episode index
        demos = sorted(demos, key=lambda x: int(x.split("_")[-1]))
        
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
                # set relevant episode information
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
            # reset() can return the initial observation
            obs=env.reset()
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

        # if should_ret:
        #     # only return obs if we've done a forward call - otherwise the observations will be garbage
        #     return get_observation()
        return obs
    
    
    def _run_single_trial(self, env, states, initial_state, actions_ref, episode_index, trial_index, video_path):
        """each trial"""

        # reference trajectory length from the dataset
        traj_len = actions_ref.shape[0]
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "light_blue"))
            instruction = [lang]
        else:
            instruction = [""]

        # create video writer
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        print(colored(f"Trial video save path: {video_path}", "light_blue"))
        video_writer = imageio.get_writer(video_path, fps=20)

        # prepare txt path for writing inference info
        txt_path = os.path.splitext(video_path)[0] + ".txt"
        inference_info_lines = []

        # set the environment to initial state and obtain the initial observation
        obs = self._reset_to(env, initial_state)

        action_deviation = 0.0
        state_deviation = 0.0
        pred_actions = []
        success = False

        i = 0
        # adopted action length for each chunk from the server
        chunk_length = 20

        while i + chunk_length < env.horizon:
            # get the response from the server
            model_response = self.server.get_response_from_server(obs, instruction, i)
            predict_action = model_response["predict_action"]

            step_idx = i
            inference_mode = model_response.get("inference_mode", "")
            reasoning = model_response.get("reasoning", "")
            infer_ms = model_response["server_timing"]["infer_ms"]
            line = f"step：{step_idx}，inference_mode：{inference_mode}，reasoning：{reasoning}，infer_ms：{infer_ms}"
            inference_info_lines.append(line)

            # fix the action dimension from 7 to 12, so the base will not move randomly
            action_queue = predict_action[0, :, :]
            # execute the action chunk
            for j in range(min(chunk_length, action_queue.shape[0])):
                action = action_queue[j, :]
                action[7:] = [0, 0, 0, 0, -1]
                pred_actions.append(action)
                # step the environment with the action(12,)
                obs, reward, done, info = env.step(action)
                i += 1

                # compute if the predicted action length < groundtruth action length
                if i < traj_len - 1:
                    action_deviation += np.linalg.norm(action - actions_ref[i])
                    state_playback = np.array(env.sim.get_state().flatten())
                    state_deviation += np.linalg.norm(states[i + 1] - state_playback)

                if i % 100 == 0:
                    print(f"Executing step {i} of episode {episode_index + 1}, trial {trial_index + 1}")
                # check if succeeds
                if not success and env._check_success():
                    success = True
                    print(colored(f"Task succeeds at step {i}", "green"))
                # render video
                video_img = []
                for cam_name in self.camera_names:
                    im = env.sim.render(
                        height=self.camera_height, width=self.camera_width, camera_name=cam_name
                    )[::-1]
                    video_img.append(im)
                video_img = np.concatenate(video_img, axis=1)  # concatenate horizontally
                video_writer.append_data(video_img)

        # close video writer
        video_writer.close()

        # 写inference信息到txt文件
        with open(txt_path, "w", encoding="utf-8") as f_txt:
            for line in inference_info_lines:
                f_txt.write(line + "\n")

        # summary
        actual_length = i
        if success:
            print(colored(f"Task succeeds", "green"))
        else:
            print(colored(f"Exceeds max steps {env.horizon}, task fails", "red"))

        action_mse = action_deviation / min(actual_length, traj_len)
        state_mse = state_deviation / min(actual_length, traj_len)
        print(colored(f"Action MSE: {action_mse}", "yellow"))
        print(colored(f"State MSE: {state_mse}", "yellow"))

        if self.plot_comparison:
            pred_traj_np = np.array(pred_actions[:min(len(pred_actions), actions_ref.shape[0])])
            gt_traj_np = actions_ref[:min(len(pred_actions), actions_ref.shape[0]), :]
            self.plot_action_comparison(pred_traj_np, gt_traj_np)

        return {
            "trial_index": trial_index,
            "gt_length": traj_len,
            "actual_length": actual_length,
            "action_mse": action_mse,
            "state_mse": state_mse,
            "success": 1 if success else 0,
            "instruction": instruction
        }

    def run_simulation(self):
        demos = self._get_demo_list()
        total_demos = len(demos)
        
        # determine the number of episodes to evaluate
        if self.num_episodes <= 0 or self.num_episodes > total_demos:
            num_episodes = total_demos
        else:
            num_episodes = self.num_episodes
        
        print(colored(f"\nEvaluating task {self.task}, total demos in dataset: {total_demos}, evaluating {num_episodes} episodes", "yellow"))
        
        task_dir = os.path.join(self.save_dir, self.task)
        os.makedirs(task_dir, exist_ok=True)
        
        # task level
        all_results = []
        task_instruction = None
        
        # create task level CSV file
        task_csv_path = os.path.join(task_dir, "avg.csv")
        task_csv_headers = ["episode_index", "trial_index", "gt_length", "actual_length", "action_mse", "state_mse", "success"]
        
        with open(task_csv_path, "w", newline="") as task_csv_file:
            task_csv_writer = csv.DictWriter(task_csv_file, fieldnames=task_csv_headers)
            task_csv_writer.writeheader()
            
            # outer loop: iterate over episodes
            for episode_index in range(num_episodes):
                print(colored(f"\n{'='*60}", "light_blue"))
                print(colored(f"Starting Episode {episode_index + 1}/{num_episodes}", "light_blue"))
                print(colored(f"{'='*60}", "light_blue"))
                
                # create episode directory
                episode_dir = os.path.join(task_dir, f"episode_{episode_index}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # episode level CSV file
                episode_csv_path = os.path.join(episode_dir, f"episode_{episode_index}_trials.csv")
                episode_csv_headers = ["trial_index", "gt_length", "actual_length", "action_mse", "state_mse", "success"]
                
                episode_results = []
                
                with open(episode_csv_path, "w", newline="") as episode_csv_file:
                    episode_csv_writer = csv.DictWriter(episode_csv_file, fieldnames=episode_csv_headers)
                    episode_csv_writer.writeheader()
                    
                    # inner loop: iterate over trials
                    for trial_index in range(self.num_trials):
                        print(colored(f"\n--- Episode {episode_index + 1}, Trial {trial_index + 1}/{self.num_trials} ---", "light_blue"))
                        
                        # video path
                        video_path = os.path.join(episode_dir, f"trial_{trial_index}.mp4")
                        
                        # run single trial
                        # initialize the environment for each trial
                        episode_name, env, states, initial_state, actions_ref = self._init_env(episode_index)
                        result = self._run_single_trial(
                            env, states, initial_state, actions_ref,
                            episode_index, trial_index, video_path
                        )
                        
                        # record instruction (should be the same for all episodes)
                        if task_instruction is None and result["instruction"]:
                            task_instruction = result["instruction"]
                        
                        # save to episode CSV
                        episode_row = {k: result[k] for k in episode_csv_headers}
                        episode_csv_writer.writerow(episode_row)
                        episode_csv_file.flush()
                        
                        # save to task CSV
                        task_row = {"episode_index": episode_index, **episode_row}
                        task_csv_writer.writerow(task_row)
                        task_csv_file.flush()
                        
                        episode_results.append(result)
                        all_results.append({**result, "episode_index": episode_index})
                        
                        # close the environment
                        env.close()
                
                # print episode summary
                episode_success_rate = sum(r["success"] for r in episode_results) / len(episode_results)
                print(colored(f"\nEpisode {episode_index} Summary: Success Rate = {episode_success_rate:.2%}", "green"))
                

        # calculate task level statistics
        total_trials = len(all_results)
        avg_gt_length = sum(r["gt_length"] for r in all_results) / total_trials
        avg_actual_length = sum(r["actual_length"] for r in all_results) / total_trials
        avg_success_rate = sum(r["success"] for r in all_results) / total_trials
        avg_action_mse = sum(r["action_mse"] for r in all_results) / total_trials
        avg_state_mse = sum(r["state_mse"] for r in all_results) / total_trials
        
        # save task level JSON summary file
        task_summary = {
            "task": self.task,
            "instruction": task_instruction if task_instruction else "",
            "num_episodes": num_episodes,
            "num_trials_per_episode": self.num_trials,
            "total_trials": total_trials,
            "avg_gt_length": avg_gt_length,
            "avg_actual_length": avg_actual_length,
            "avg_success_rate": avg_success_rate,
            "avg_action_mse": avg_action_mse,
            "avg_state_mse": avg_state_mse
        }
        
        task_json_path = os.path.join(task_dir, f"{self.task}_summary.json")
        with open(task_json_path, "w") as f:
            json.dump(task_summary, f, indent=4)
        
        print("\n{'='*60}")
        print(colored(f"Task {self.task} Evaluation Complete!", "green"))
        print(("{'='*60}"))
        print(f"Total Episodes: {num_episodes}")
        print(f"Trials per Episode: {self.num_trials}")
        print(f"Total Trials: {total_trials}")
        print(f"Average Action MSE: {avg_action_mse:.4f}")
        print(f"Average State MSE: {avg_state_mse:.4f}")
        print(f"Results saved to: {task_dir}")
        print(colored(f"Average Success Rate: {avg_success_rate:.2f}", "green"))
        

    def save_visual_obs(self,agent_view_center,agent_view_left,agent_view_right,eye_in_hand,output_root,num):
        agentview_path=os.path.join(output_root,"agent_view")
        os.makedirs(agentview_path,exist_ok=True)
        agentview_path=os.path.join(agentview_path,f"center_{num}.jpg")
        eye_path=os.path.join(output_root,"eye_in_hand")
        os.makedirs(eye_path,exist_ok=True)
        eye_path=os.path.join(eye_path,f"{num}.jpg")
        #convert BGR to RGB before saving
        cv2.imwrite(agentview_path,agent_view_center[:,:,::-1])
        cv2.imwrite(agentview_path.replace("center","left"),agent_view_left[:,:,::-1])
        cv2.imwrite(agentview_path.replace("center","right"),agent_view_right[:,:,::-1])
        cv2.imwrite(eye_path,eye_in_hand[:,:,::-1])

    def plot_action_comparison(self,pred_traj_np,gt_traj_np):
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
    # number of episodes to evaluate for each task, default to evaluate all episodes
    parser.add_argument("--num_episodes",type=int, default=10)
    # number of test times for each episode
    parser.add_argument("--num_trials",type=int, default=5)
    # eposide legth factor, default to 3 times the original eposide length
    parser.add_argument("--episode_length_factor", type=float, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--save_dir", type=str, default="/home/sunyi/robocasa/eval_trials/0109_test_ours_always_bor_action_chunk20")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # # tasks to evaluate
    # tasks = OrderedDict(
    #     [
    #         ("ArrangeVegetables", "arrange vegetables on a cutting board"),
    #         ("MicrowaveThawing", "place frozen food in microwave for thawing"),
    #         ("RestockPantry", "restock cans in pantry"),
    #         ("PreSoakPan", "prepare pan for washing"),
    #         ("PrepareCoffee", "make coffee"),
    #     ]
    # )
    
    tasks=["PrepareCoffee"]
    
    for task in tasks:
        camera_height=512
        camera_width=768
        server = Simulator(task, args.num_episodes, args.num_trials, args.save_dir, args.port, args.episode_length_factor, camera_height, camera_width, is_baseline=False)
        server.run_simulation()