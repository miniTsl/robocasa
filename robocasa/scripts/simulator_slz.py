"""
    Simulator class for robocasa environment.
"""
import argparse
import h5py
import json
import os 
import robosuite
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pdb
from robocasa.utils.dataset_registry import get_ds_path
from termcolor import colored
from collections import OrderedDict
from robocasa.scripts.simulate_server_slz import SimulateServer



class Simulator:
    def __init__(self,port,dataset_path,episodes,video_path,task,test_times,camera_height,camera_width):
        self.port=port
        #initialize the server object here
        self.server=SimulateServer(self.port)
        #get the task configuration and environment
        self.dataset_path=dataset_path
        self.episodes=episodes
        self.save_dir=video_path
        self.task=task
        self.test_times=test_times
        #这个不要动，是配置环境的
        self.camera_names=["robot0_eye_in_hand","robot0_agentview_left","robot0_agentview_right"]#"robot0_agentview_center",
        self.camera_height=camera_height
        self.camera_width=camera_width
        #模型版本(实验组/对照组)
        self.action_only=True
        #保存每个相机的历史帧
        self.history_frames={}
        self.history_frames["wrist_view"]={}
        self.history_frames["left_view"]={}
        self.history_frames["right_view"]={}
        #reason内容
        self.thought=[]

        self.plot_comparison=False
    def _init_env(self,test_time):
        self.dataset_path=os.path.expanduser(self.dataset_path)
        f=h5py.File(self.dataset_path,"r")
        #初始状态initial_state有多少种应该依赖于原本的数据集里有多少种demo，还是需要加载一下原数据集里的demo
        #如果单个任务评测的epiosde数>原数据集中的demo数，再考虑重复评测
        demos = list(f["data"].keys())
        demo_nums=len(demos)
        assert demo_nums>=self.test_times,f"demo_nums({demo_nums}) must be greater than or equal to test_times({self.test_times})"
        idx=test_time*(demo_nums//self.test_times)
        #demos里面是ascii序，重排一下
        demos=sorted(demos,key=lambda x:int(x.split("_")[-1]))
        ep=demos[idx]
        print(f"Running with initial state from {ep}")
        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()] #reference actions from the dataset
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        pdb.set_trace()
        #get_env_metadata_from_dataset
        env_meta=json.loads(f["data"].attrs["env_args"])
        env_kwargs=env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = True
        env_kwargs["camera_names"]=self.camera_names
        env_kwargs["ignore_done"]=False
        env_kwargs["horizon"]=int(actions.shape[0]*1.8) #设定轨迹长度上限为标准答案的2倍
        print(
                colored(
                    "Initializing environment for {}...".format(env_kwargs["env_name"]),
                    "yellow",
                )
            )
        print(f"Setting task horizon {env_kwargs['horizon']}")
        #create the simulation environment
        env = robosuite.make(**env_kwargs)
        
        return ep,env,states,initial_state,actions

    def _reset_to(self,env,initial_state):
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

    def _save_frames(self,obs,iter_step):
        self.history_frames["wrist_view"][iter_step]=obs["robot0_eye_in_hand_image"]
        self.history_frames["left_view"][iter_step]=obs["robot0_agentview_left_image"]
        self.history_frames["right_view"][iter_step]=obs["robot0_agentview_right_image"]
        if self.action_only:
            #最多保存当前step的前100帧
            if iter_step>100:
                del self.history_frames["wrist_view"][iter_step-101]
                del self.history_frames["left_view"][iter_step-101]
                del self.history_frames["right_view"][iter_step-101]


    def _sample_images(self,iter_step):
         # 采样时对图片进行处理，将图片翻转并转换为uint8格式
        def _process_image(img):
            #Flip image up-down 
            img=np.flipud(img)
            # send uint8 format
            if img.dtype!=np.uint8:
                img=(img*255).astype(np.uint8)
            return img

        sampled_images={}
        sampled_images["wrist_view"]=[]
        sampled_images["left_view"]=[]
        sampled_images["right_view"]=[]
        #只输出action的模型
        if self.action_only:
            #从当前帧往前每隔 20 帧采样，最多 5 帧（包括当前帧）
            frame_idx=[iter_step-20*k for k in range(4,-1,-1) if iter_step-20*k>=0]
            for idx in frame_idx:
                sampled_images["wrist_view"].append(_process_image(self.history_frames["wrist_view"][idx]))
                sampled_images["left_view"].append(_process_image(self.history_frames["left_view"][idx]))
                sampled_images["right_view"].append(_process_image(self.history_frames["right_view"][idx]))
            return sampled_images
        else:
            raise NotImplementedError("Not implemented for non-action-only models")
    
    def _clear_history_frames(self):
        self.history_frames["wrist_view"]={}
        self.history_frames["left_view"]={}
        self.history_frames["right_view"]={}
    
    def run_simulation(self):
        count_success=0
        action_mse=[]
        state_mse=[]
        for k in range(self.test_times):
            #clear
            self.thought=[]
            self._clear_history_frames()
            #call init_env first
            episode_name,env,states,initial_state,actions_ref=self._init_env(k)
            #重新设置视频保存路径
            #must specify an .mp4 file to initialize video_writer
            video_path=os.path.join(self.save_dir,"video",self.task+f"_{episode_name}.mp4")
            os.makedirs(os.path.dirname(video_path),exist_ok=True)
            print(f"Video path:{video_path}")
            self.video_writer=imageio.get_writer(video_path,fps=20)
            # reference trajectory length from the dataset
            traj_len=actions_ref.shape[0]
            ep_meta=json.loads(initial_state["ep_meta"])
            lang=ep_meta.get("lang",None)
            if lang is not None:
                print(colored(f"Instruction:{lang}","green"))
                # in order to prepare the observation for wall-x server
                self.thought.append(lang)
                # self.server.set_instruction(lang)
            #set the environment to initial state and obtain the initial observation
            obs=self._reset_to(env,initial_state)
            # 保存初始帧
            self._save_frames(obs,0)
            #if success, break the loop after 10 more steps
           
            action_deviation=0.0
            state_deviation=0.0
            pred_actions=[]
            action_hist=[]
            
            #仿真步数
            i=0
            #每次执行动作的步长
            chunk_length=3
            print(colored("Running simulation...","yellow"))
            while i+chunk_length<env.horizon:
                #obtain the action from the server
                #按规则采样图片
                sampled_images=self._sample_images(i)
                pred_action=self.server.get_actions_from_server(obs,sampled_images,self.thought,i)
                #extract the action from the response dictionary
                #固定7~12维，底盘确实不会乱动了
                action_queue=pred_action[0,:,:]
                #执行ACTION_CHUNK(不超过horizon的前提下)
                for j in range(min(chunk_length,action_queue.shape[0])):
                    action=action_queue[j,:]
                    action[7:] = [0, 0, 0, 0, -1]
                    pred_actions.append(action)
                    #step the environment with the action(12,)
                    obs, reward, done, info = env.step(action)
                    i+=1
                    self._save_frames(obs,i)
                    # compute if the predicted action length<groundtruth action length
                    if i < traj_len - 1:
                        # count deviation from the groundtruth state
                        # print(f"action:{action}; actions_ref:{actions_ref[i]}")
                        action_deviation+=np.linalg.norm(action-actions_ref[i])
                        state_playback = np.array(env.sim.get_state().flatten())
                        state_deviation+=np.linalg.norm(states[i + 1] - state_playback)

                    if i%50==0:
                        print(f"Excuating step {i} of test {k}")
                    #check if succeeds
                    success=env._check_success()
                    if success:
                        if count_success==0:
                            print(colored(f"task succeeds at step {i}, end after 10 more steps","green"))
                        count_success+=1
                        if count_success>10:
                            break
                    #render video
                    if self.video_writer is not None:
                        video_img=[]
                        for cam_name in self.camera_names:
                            im = env.sim.render(
                                height=self.camera_height, width=self.camera_width, camera_name=cam_name
                            )[::-1]
                            video_img.append(im)
                        video_img = np.concatenate(
                            video_img, axis=1
                        )  # concatenate horizontally
                        self.video_writer.append_data(video_img)
            #summary
            if success:
                print(colored(f"task succeeds","green"))
                count_success+=1
            else:
                print(colored(f"Exceeds max steps {env.horizon}, task fails","red"))

            print(colored(f"Action MSE: {action_deviation/traj_len}","yellow"))
            action_mse.append(action_deviation/traj_len)
            print(colored(f"State MSE: {state_deviation/traj_len}","yellow"))
            state_mse.append(state_deviation/traj_len)

            if self.plot_comparison:
                pred_traj_np = np.array(pred_actions[:min(len(pred_actions),actions_ref.shape[0])])
                gt_traj_np = actions_ref[:min(len(pred_actions),actions_ref.shape[0]),:]
                self.plot_action_comparison(pred_traj_np,gt_traj_np)
        
        with open(os.path.join(self.save_dir,f"{self.task}_summary.json"),"w") as f:
            json.dump({"success":count_success,"total":self.test_times,"action_mse":action_mse,"state_mse":state_mse},f,indent=4)
        

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
    # number of episodes to evaluate for each task
    parser.add_argument("--episodes",type=int,default=20)
    # maximum number of steps per episode
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--port", type=int, default=8003)
    parser.add_argument("--video_path", type=str, default="/data/songzelin/robocasa/demo_tasks")
    
    args = parser.parse_args()
    assert(args.video_path is not None)

    if not os.path.exists(args.video_path):
        os.makedirs(args.video_path, exist_ok=True)
    # tasks to evaluate
    tasks = OrderedDict(
        [
            ("ArrangeVegetables", "arrange vegetables on a cutting board"),
            ("MicrowaveThawing", "place frozen food in microwave for thawing"),
            ("RestockPantry", "restock cans in pantry"),
            ("PreSoakPan", "prepare pan for washing"),
            ("PrepareCoffee", "make coffee"),
        ]
    )
    
    #evaluate all tasks
    task="PreSoakPan"
    dataset=get_ds_path(task,ds_type="human_raw")
    camera_height=512
    camera_width=768
    test_times=10
    # initialize Simulator
    server=Simulator(args.port,dataset,args.episodes,args.video_path,task,test_times,camera_height,camera_width)
    server.run_simulation()



        
    parser=argparse.Namespace()
    