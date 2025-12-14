"""
Maze Mode Skin Generation Prompts
"""

# Style consistency prompt for image-to-image generation
STYLE_CONSISTENT_PROMPT = """Above is the style reference image. Generate a new asset matching this exact visual style.

{base_prompt}

CRITICAL REQUIREMENTS:
1. Match the art style, color palette, and rendering technique of the reference image
2. The new asset MUST look like it comes from the SAME GAME as the reference
3. PIXEL ART STYLE: Use retro pixel art aesthetic with appropriate detail:
   - Moderate pixel granularity (16x16 to 32x32 pixel level)
   - Include texture details, shading, and depth layers
   - Clear pixel borders and defined edges
   - Vintage game visual style with appropriate level of detail
   - NOT overly simplified - maintain texture richness
4. STRONG VISUAL DISTINCTION: This asset must be HIGHLY DISTINGUISHABLE from other game elements:
   - Use CONTRASTING colors (different hue, saturation, or brightness)
   - Use DISTINCT shapes and visual patterns
   - Ensure HIGH CONTRAST and CLEAR VISUAL IDENTITY
   - Make it instantly recognizable at a glance
5. WALL TILE REQUIREMENTS (for wall assets only):
   - Wall tiles MUST be COMPLETELY FILLED squares with NO empty or transparent areas
   - Wall MUST cover the ENTIRE tile area from edge to edge
   - NO irregular shapes, peaks, or protrusions extending beyond the square boundary
   - NO gaps, holes, or partial coverage in wall tiles
   - Wall and floor MUST have DISTINCTLY DIFFERENT visual appearance (different colors, textures, or patterns)
   - Wall should be clearly recognizable as an impassable barrier
6. Balance: Maintain thematic coherence with the reference while ensuring strong visual differentiation and pixel art aesthetics
"""