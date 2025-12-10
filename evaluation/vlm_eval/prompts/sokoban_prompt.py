# 占位符格式: {player}, {goal}, {box}, {wall}, {floor}
SOKOBAN_SYSTEM_PROMPT_TEMPLATE = """You are given an image of a grid-based Sokoban puzzle.
{wall} tiles represent walls and cannot be crossed.
{floor} tiles represent open floor tiles that can be moved through.
The {player} represents the player or agent.
The {box} represents the box that needs to be pushed.
The {goal} represents the goal destination for the box.
Task:
Infer the complete movement sequence required for the {player} to push the {box} onto the {goal} goal.
The {player} moves in four directions: up, down, left, right.
When the {player} moves into a box, it automatically pushes the box if there is space behind it.
The box and the {player} cannot cross or overlap any walls.
Diagonal movement is not allowed, and the camera remains fixed from a top-down view.
Output Format:
Return the entire movement sequence as a JSON array of directional actions, where each element is one of "up", "down", "left", or "right".
Do not include any explanations or additional text.
Example of expected output:
{{
  "actions": ["right", "right", "down", "left", "down"]
}}
"""

SOKOBAN_USER_PROMPT_TEMPLATE = """Infer the complete movement sequence required for the {player} to push the {box} onto the {goal} goal.
"""


