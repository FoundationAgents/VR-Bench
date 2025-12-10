# 占位符格式: {player}, {goal}, {wall}, {floor}
MAZE_SYSTEM_PROMPT_TEMPLATE = """You are given an image of a grid-based maze.
{wall} tiles represent walls and cannot be crossed.
{floor} tiles represent open paths that can be moved through.
The {player} represents the starting point of the path.
The {goal} represents the goal or destination.


Task:
Infer the shortest valid path from the {player} starting point to the {goal} goal.
Movement can only occur between adjacent open tiles — up, down, left, or right.
Diagonal movement is not allowed, and the path must not cross or touch any walls.


Output Format:
Return the entire movement sequence of the {player} as a JSON array of directions, where each element is one of "up", "down", "left", or "right".
Do not include any explanations or additional text.


Example of expected output:
{{
  "path": ["up", "up", "left", "down", "right", "right"]
}}
"""

MAZE_USER_PROMPT_TEMPLATE = """Infer the shortest valid path from the {player} starting point to the {goal} goal.
"""



