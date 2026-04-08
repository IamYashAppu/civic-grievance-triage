import os
import json
from openai import OpenAI
from env import GrievanceEnv, GrievanceAction

# 1. Setup API Client (Uses hackathon standard variables)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Initialize the OpenAI client as required by the rules
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key-for-local-testing" 
)

BENCHMARK = "civic-grievance-triage"
TASKS = ["easy_triage", "medium_triage", "hard_triage"]

def run_inference():
    for task_name in TASKS:
        # Initialize your environment for the current difficulty level
        env = GrievanceEnv(task_level=task_name)
        obs = env.reset()
        
        # [START] MANDATORY LOG
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")
        
        step_num = 1
        action_str = "failed"
        reward_val = 0.00
        is_done = False
        error_msg = "null"
        rewards_history = []
        
        # System prompt instructing the LLM on how to play our environment
        prompt = f"""You are a city triage AI. 
        Analyze this complaint: "{obs.text}"
        Return a JSON object with exactly these keys:
        - "category" (one of: infrastructure, sanitation, water, power, emergency)
        - "severity_level" (integer 1 to 5)
        - "target_departments" (list containing one or more of: public_works, waste_mgmt, water_board, grid_ops, police, fire)
        Only output valid JSON, nothing else."""

        try:
            # Call the LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            # Parse the LLM output
            llm_output = response.choices[0].message.content
            
            # Clean up markdown if the LLM adds it
            if "```json" in llm_output:
                llm_output = llm_output.split("```json")[1].split("```")[0].strip()
            elif "```" in llm_output:
                llm_output = llm_output.split("```")[1].split("```")[0].strip()
                
            action_dict = json.loads(llm_output)
            
            # Convert dictionary to our Pydantic Action model
            action = GrievanceAction(**action_dict)
            action_str = f"{action.category}-{action.severity_level}"
            
            # Take the step in the environment to get the score!
            new_obs, reward, is_done, info = env.step(action)
            reward_val = reward.score
            rewards_history.append(f"{reward_val:.2f}")
            success = is_done
            
        except Exception as e:
            # If the LLM hallucinates bad JSON or fails, we catch it safely
            error_msg = '"' + str(e).replace('"', '') + '"'
            success = False
            rewards_history.append("0.00")
        
        # [STEP] MANDATORY LOG
        print(f"[STEP] step={step_num} action={action_str} reward={reward_val:.2f} done={str(is_done).lower()} error={error_msg}")
        
        # [END] MANDATORY LOG
        score_val = reward_val 
        rewards_str = ",".join(rewards_history) if rewards_history else "0.00"
        print(f"[END] success={str(success).lower()} steps={step_num} score={score_val:.2f} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()