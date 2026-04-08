from fastapi import FastAPI
from pydantic import BaseModel
from env import GrievanceEnv, GrievanceAction

app = FastAPI()
# Initialize with easy mode by default for the ping test
env = GrievanceEnv("easy_triage") 

@app.get("/")
def health_check():
    """The automated ping check"""
    return {"status": "ok", "message": "Environment is live"}

@app.post("/reset")
def reset_env():
    """The automated reset check"""
    obs = env.reset()
    return {"observation": obs.dict()}

@app.get("/state")
def get_state():
    return env.state()

@app.post("/step")
def step_env(action: GrievanceAction):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done
    }
