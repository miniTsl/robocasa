"""
    SimulateServer class for robocasa environment. Use WebSocket to communicate with the wall-x server, act as a client to receive actions from the wall-x server and send observations back to the wall-x server.
"""
import time
import threading
import asyncio
import websockets
import msgpack
import msgpack_numpy as m
m.patch()#enable msgpack to serialize numpy arrays
import numpy as np
import math
import logging
from PIL import Image
import os
import pdb
from PIL import Image

from typing import Dict,Any,List
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

class SimulateServer:
    """
    Client for running simulations and communicating with the Wall-X WebSocket inference server.
    """

    def __init__(self, port=8003):
        """Initialize the simulation client with server connection details."""
        self.uri = f"ws://localhost:{port}"
        self.env = None
        self.websocket = None
        self.metadata = None
        
        # Asyncio loop management for sync-async bridge
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to start
        while not self.loop.is_running():
            time.sleep(0.01)
            
        # Connect immediately
        self.connect()

    def set_instruction(self,instruction):
        self.instruction=instruction

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def connect(self):
        """Synchronously connect to server."""
        future = asyncio.run_coroutine_threadsafe(self._connect_async(), self.loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            print(f"Failed to connect to Wall-X server: {e}")
            raise

    async def _connect_async(self):
        logger.info(f"Connecting to Wall-X server at {self.uri}...")
        self.websocket = await websockets.connect(
            self.uri, 
            max_size=None,
            ping_interval=None,
            ping_timeout=None
        )
        # Receive metadata
        self.metadata = msgpack.unpackb(await self.websocket.recv())
        print(f"Connected! Server metadata: {self.metadata}")

    def _process_images_for_model_inference(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        def _process_image(img):
            #Flip image up-down 
            img=np.flipud(img)
            # send uint8 format
            if img.dtype!=np.uint8:
                img=(img*255).astype(np.uint8)
            return img
        images_for_model = {
            "robot0_eye_in_hand": _process_image(obs["robot0_eye_in_hand_image"]),
            "robot0_agentview_left": _process_image(obs["robot0_agentview_left_image"]),
            "robot0_agentview_right": _process_image(obs["robot0_agentview_right_image"])
        }
        return images_for_model
    
    def _get_response(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action from the Wall-X inference server.
        This is a synchronous method that calls the async websocket.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._get_response_async(observations), self.loop
        )
        return future.result()

    async def _get_response_async(self, obs: Dict) -> Dict:
        if self.websocket is None:
            raise RuntimeError("Not connected to server")
        
        await self.websocket.send(msgpack.packb(obs))
        response = msgpack.unpackb(await self.websocket.recv())
        return response


    def get_response_from_server(self, observations: Dict[str, Any], instruction: List[str], iter_step: int) -> Dict:
        """Process observations and get response from the inference server."""
        def _combine_images(left_img,right_img,eye_in_hand_img):
            #已弃用
            #2*2拼接图片，右下角留白
            combined_img=np.zeros((2*left_img.shape[0],2*left_img.shape[1],3),dtype=np.uint8)
            combined_img[:left_img.shape[0],:left_img.shape[1]]=left_img
            combined_img[:left_img.shape[0],left_img.shape[1]:]=right_img
            combined_img[left_img.shape[0]:,:left_img.shape[1]]=eye_in_hand_img
            # 填充右下角空白
            combined_img[left_img.shape[0]:,left_img.shape[1]:]=np.ones_like(combined_img[left_img.shape[0],left_img.shape[1]])*255
            # 内容不变，缩小到原图大小
            combined_img=Image.fromarray(combined_img)
            combined_img=combined_img.resize((left_img.shape[0],left_img.shape[1]),resample=Image.Resampling.LANCZOS)
            return np.array(combined_img)
            
        

        def _quat2axisangle(quat):
            """
            Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
            """
            # clip quaternion
            if quat[3] > 1.0:
                quat[3] = 1.0
            elif quat[3] < -1.0:
                quat[3] = -1.0

            den = np.sqrt(1.0 - quat[3] * quat[3])
            if math.isclose(den, 0.0):
                # This is (close to) a zero degree rotation, immediately return
                return np.zeros(3)

            return (quat[:3] * 2.0 * math.acos(quat[3])) / den

        def _normalize_state(state):
            #参照wall-x中的client示例，新增对state的归一化过程(已弃用)
            state_min=[0.14009679853916168, -4.078073024749756, 0.7538334727287292, -2.4598827362060547, -3.416140079498291, -1.9598420858383179, 0.0005838712677359581]
            state_std=[5.863479137420654, 3.9139790534973145, 0.8156288266181946, 5.811497211456299, 7.361121654510498, 4.683920383453369, 0.039888788014650345]
            state=(state-state_min)/state_std
            # normalize to [-1,1]
            state=state*2-1
            state=np.clip(state,-1,1)
            return state

        # --- Data Mapping Logic: Env Obs -> Env Message ---
        env_message = {
            "instruction": instruction,
            "step_index": iter_step,
            "dataset_names": ["umi1/robocasa"],
        }

        env_message.update(self._process_images_for_model_inference(observations))

        # add state
        # robot0_eef_pos (3,) robot0_eef_quat (4,) robot0_gripper_qpos (2,)
        if "robot0_eef_pos" in observations and "robot0_eef_quat" in observations and "robot0_gripper_qpos" in observations:
            eef_pos = observations["robot0_eef_pos"]
            eef_quat = _quat2axisangle(observations["robot0_eef_quat"])
            gripper_qpos = observations["robot0_gripper_qpos"][:1]#keep dim
            state = np.concatenate([eef_pos, eef_quat, gripper_qpos])
            env_message["state"]=state # normalize放到wall-x端进行
        else:
            print("ERROR: state not found in observations")
            
        
        # --- Request Action ---
        model_response = self._get_response(env_message) # predict_action shape: [pred_horizon, action_dim] = [32, 12]
        model_response["predict_action"] = model_response["predict_action"].copy() #make actions writable
        return model_response

    def close(self):
        if self.loop.is_running():
            if self.websocket:
                asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)