# 占位符格式: {player}, {goal}, {trap}, {floor}
TRAPFIELD_SYSTEM_PROMPT_TEMPLATE = """You are given an image of a grid-based maze.
{trap} tiles represent trap zones that must be avoided.
{floor} tiles represent open paths that can be moved through.
The {player} represents the starting point of the path.
The {goal} represents the goal or destination.
Task:
Infer the shortest valid path for the {player} to reach the {goal}.
Movement can only occur between adjacent open tiles — up, down, left, or right.
Diagonal movement is not allowed.
The path must not cross or touch any trap tiles.
Output Format:
Return the full movement sequence of the {player} as a JSON array of directions, where each element is one of "up", "down", "left", or "right".
Do not include any explanations, reasoning, or extra text.
Example of expected output:
{{
  "path": ["left", "left", "down", "down"]
}}
"""

TRAPFIELD_USER_PROMPT_TEMPLATE = """Infer the shortest valid path for the {player} to reach the {goal}.
"""



