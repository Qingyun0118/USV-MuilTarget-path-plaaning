import sys
sys.path.insert(0, '.')
from my_USV_Env import MyUsvEnv
from usv_evaluation import MultiObjectiveEvaluator
import numpy as np

print("Creating environment...")
env = MyUsvEnv(record_flag=False)
print("Environment created.")

# Create evaluator with environment
evaluator = MultiObjectiveEvaluator(env=env, usv_length=10.0, usv_speed=6.0, fuel_consumption=1.0)

# Test path (some points)
points = [(5.0, 5.0), (50.0, 50.0), (80.0, 80.0)]
print("Evaluating path:", points)
result = evaluator.evaluate(points)
print("Result:", result)

# Additional test: compute each objective separately
print("\nDetailed breakdown:")
print("Path length:", evaluator.compute_path_length(points))
print("Smoothness:", evaluator.compute_smoothness(points))
print("Economic cost:", evaluator.compute_economic_cost(points))
print("Safety:", evaluator.compute_safety(points))

# Test with more points (zigzag)
points2 = [(5.0, 5.0), (30.0, 60.0), (70.0, 30.0), (90.0, 90.0)]
print("\nEvaluating zigzag path:", points2)
result2 = evaluator.evaluate(points2)
print("Result2:", result2)