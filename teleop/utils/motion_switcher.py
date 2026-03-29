# for motion switcher
import json

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
# for loco client
from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_GET_FSM_ID
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
import time

# MotionSwitcher used to switch mode between debug mode and ai mode
class MotionSwitcher:
    def __init__(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(1.0)
        self.msc.Init()

    def Enter_Debug_Mode(self):
        try:
            status, result = self.msc.CheckMode()
            while result['name']:
                self.msc.ReleaseMode()
                status, result = self.msc.CheckMode()
                time.sleep(1)
            return status, result
        except Exception as e:
            return None, None

    def Exit_Debug_Mode(self):
        try:
            status, result = self.msc.SelectMode(nameOrAlias='ai')
            return status, result
        except Exception as e:
            return None, None

class LocoClientWrapper:
    def __init__(self):
        self.client = LocoClient()
        self.client.SetTimeout(1.0)
        self.client.Init()

    def Enter_Damp_Mode(self):
        return self.client.SetFsmId(1)

    def Exit_Damp_Mode(self):
        return self.client.SetFsmId(200)

    def Move(self, vx, vy, vyaw):
        return self.client.SetVelocity(vx, vy, vyaw, duration=1.0)

    def Get_Fsm_Id(self):
        try:
            code, data = self.client._Call(ROBOT_API_ID_LOCO_GET_FSM_ID, json.dumps({}))
            if code != 0:
                return code, None
            payload = json.loads(data) if isinstance(data, str) else data
            if isinstance(payload, dict):
                return code, payload.get("data")
            return code, None
        except Exception:
            return None, None

if __name__ == '__main__':
    ChannelFactoryInitialize(0, networkInterface="enx6c1ff76c623a") # 0 for real robot, 1 for simulation
    ms = MotionSwitcher()
    status, result = ms.Enter_Debug_Mode()
    print("Enter debug mode:", status, result)
    time.sleep(5)
    status, result = ms.Exit_Debug_Mode()
    print("Exit debug mode:", status, result)
    time.sleep(2)
