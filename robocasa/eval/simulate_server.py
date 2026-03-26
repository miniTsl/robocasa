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
import pdb
from typing import Dict,Any,List
from dataclasses import dataclass, field
import h5py
path = "/home/sunyi/robocasa/datasets/v0.1/multi_stage/brewing/PrepareCoffee/2024-05-07/demo_im128.hdf5"

# add proprioceptive state
# the normalization is done in the wall-x server side
# List of all required proprioceptive states
STATE_KEYS = [
"robot0_eef_pos",          # 3
"robot0_eef_quat",         # 4
"robot0_gripper_qpos",     # 2
"robot0_gripper_qvel",     # 2
"robot0_joint_pos",        # 7
"robot0_base_to_eef_pos"   # 3
]

IMAGE_KEYS = [
"robot0_eye_in_hand_image", # 3, 128, 128
"robot0_agentview_left_image", # 3, 128, 128
"robot0_agentview_right_image" # 3, 128, 128
]

def replace_obs_with_gt(obs, index):
    f = h5py.File(path, "r")
    for key in STATE_KEYS:
        obs[key] = f["data"]["demo_1"]["obs"][key][index]
    for key in IMAGE_KEYS:
        obs[key] = f["data"]["demo_1"]["obs"][key][index]

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
            "wrist view": _process_image(obs["robot0_eye_in_hand_image"]),
            "left view": _process_image(obs["robot0_agentview_left_image"]),
            "right view": _process_image(obs["robot0_agentview_right_image"])
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


    def get_response_from_server(self, observations: Dict[str, Any], instruction: str, iter_step: int) -> Dict:
        """Process observations and get response from the inference server."""
        # replace_obs_with_gt(observations, iter_step) # for debugging, replace the observations with the ground truth from the dataset, to test if the model can predict the correct action based on the perfect observations
        def _quat2axisangle(quat):
            """
            Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
            
            change the quaternion to axis-angle
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

        # --- Data Mapping Logic: Env Obs -> Env Message ---
        env_message = {
            "instruction": instruction,
            "step_index": iter_step,
            "dataset_names": ["umi1/robocasa"],
        }
        # pdb.set_trace()
        env_message.update(self._process_images_for_model_inference(observations))

        missing_keys = [k for k in STATE_KEYS if k not in observations]
        if len(missing_keys) == 0:
            eef_pos = observations["robot0_eef_pos"]
            eef_quat = _quat2axisangle(observations["robot0_eef_quat"])
            gripper_qpos = observations["robot0_gripper_qpos"]
            gripper_qvel = observations["robot0_gripper_qvel"]
            joint_pos = observations["robot0_joint_pos"]
            base_to_eef_pos = observations["robot0_base_to_eef_pos"]

            # Concatenate all the states into a 1D tensor
            state = np.concatenate([
                eef_pos,
                eef_quat,
                gripper_qpos,
                gripper_qvel,
                joint_pos,
                base_to_eef_pos
            ])
            env_message["state"] = state
        else:
            print(f"ERROR: missing state(s) in observations: {missing_keys}")
            
        
        # --- Request Action ---
        model_response = self._get_response(env_message) # predict_action shape: [pred_horizon, action_dim] = [32, 12]
        model_response["predict_action"] = model_response["predict_action"].copy() #make actions writable
        return model_response

    def close(self):
        if self.loop.is_running():
            if self.websocket:
                asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)
            self.loop.call_soon_threadsafe(self.loop.stop)