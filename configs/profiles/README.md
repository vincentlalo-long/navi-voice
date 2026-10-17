# Speaker & Character Profiles

This directory stores character-specific configuration overlays.

## How to Create a Custom Character Profile

1. Duplicate `profile.example.json` to `<your_character>.json`:
   ```bash
   cp voice/configs/profiles/profile.example.json voice/configs/profiles/my_character.json
   ```
2. Edit `my_character.json`:
   - Set `speaker.name` (e.g. for GPT-SoVITS speaker label).
   - Set `file_prefix` for output WAV naming.
   - Customize `emotion_rules` with keywords and volume thresholds matching your character's personality archetype (e.g., Tsundere, Kuudere, Genki, Serious/Analytical, etc.).
3. Execute the pipeline with your profile:
   ```bash
   ./voice/run.sh --profile my_character
   ```

> **Note**: Custom `*.json` profiles (except `profile.example.json`) are automatically ignored by Git to preserve privacy and prevent copyright leaks.
