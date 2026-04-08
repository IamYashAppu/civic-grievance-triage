from pydantic import BaseModel, Field
from typing import Literal, Dict, Any, Tuple, List
import uuid

# --- SPACES ---
class GrievanceObservation(BaseModel):
    grievance_id: str
    text: str = Field(description="The citizen's complaint text.")
    is_done: bool = Field(default=False, description="True if episode is complete.")

class GrievanceAction(BaseModel):
    category: Literal["infrastructure", "sanitation", "water", "power", "emergency"]
    severity_level: Literal[1, 2, 3, 4, 5] = Field(description="1: Nuisance, 5: Life-threatening.")
    target_departments: List[Literal["public_works", "waste_mgmt", "water_board", "grid_ops", "police", "fire"]]

class GrievanceReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str

# --- ENVIRONMENT ---
class GrievanceEnv:
    def __init__(self, task_level: str = "easy_triage"):
        self.task_level = task_level
        self.current_state = None
        self.step_count = 0
        self.max_steps = 1 # One-shot triage task

        # The 3 Tasks (Difficulty Progression)
        self.datasets = {
            "easy_triage": {
                "text": "There is a deep pothole on 5th avenue damaging car tires.",
                "ans_category": "infrastructure",
                "ans_severity": 2,
                "ans_depts": ["public_works"]
            },
            "medium_triage": {
                "text": "Street lights are out on Main St, and someone is illegally dumping toxic waste in the dark alley.",
                "ans_category": "sanitation",
                "ans_severity": 4,
                "ans_depts": ["waste_mgmt", "police"]
            },
            "hard_triage": {
                "text": "URGENT! A car crashed into a fire hydrant. Water is flooding the street, getting close to a downed power line, and the driver is trapped!",
                "ans_category": "emergency",
                "ans_severity": 5,
                "ans_depts": ["fire", "water_board", "grid_ops", "police"]
            }
        }

    def reset(self) -> GrievanceObservation:
        self.step_count = 0
        data = self.datasets.get(self.task_level, self.datasets["easy_triage"])
        self.current_state = {
            "id": str(uuid.uuid4())[:8],
            "text": data["text"],
            "expected": data
        }
        return GrievanceObservation(grievance_id=self.current_state["id"], text=self.current_state["text"])

    def state(self) -> Dict[str, Any]:
        return {"current_grievance_id": self.current_state["id"] if self.current_state else None}

    def step(self, action: GrievanceAction) -> Tuple[GrievanceObservation, GrievanceReward, bool, Dict]:
        self.step_count += 1
        expected = self.current_state["expected"]
        
        score = 0.0
        feedback_msgs = []

        # Grader Logic (Calculates up to 1.0)
        # 1. Category check (Worth 0.2)
        if action.category == expected["ans_category"]:
            score += 0.2
            feedback_msgs.append("Category correct.")
            
        # 2. Severity check (Worth 0.3 - partial credit if close)
        sev_diff = abs(action.severity_level - expected["ans_severity"])
        if sev_diff == 0:
            score += 0.3
            feedback_msgs.append("Exact severity.")
        elif sev_diff == 1:
            score += 0.15 # Partial reward for being off by 1
            feedback_msgs.append("Severity slightly off.")

        # 3. Department check (Worth 0.5)
        correct_depts = set(expected["ans_depts"])
        agent_depts = set(action.target_departments)
        
        # Calculate intersection over union for departments
        intersection = correct_depts.intersection(agent_depts)
        if len(correct_depts) > 0:
            dept_score = (len(intersection) / len(correct_depts)) * 0.5
            score += dept_score
            if dept_score == 0.5:
                feedback_msgs.append("Perfect department routing.")
            else:
                feedback_msgs.append("Partial/incorrect departments.")

        # Cap score at 1.0 just in case
        score = min(max(score, 0.0), 1.0)
        
        done = True # One-shot task
        obs = GrievanceObservation(grievance_id=self.current_state["id"], text=self.current_state["text"], is_done=True)
        reward = GrievanceReward(score=score, feedback=" ".join(feedback_msgs))
        
        return obs, reward, done, {}