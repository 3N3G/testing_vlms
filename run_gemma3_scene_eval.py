import argparse
import io
import os
import time
from typing import Dict, List, Tuple

from PIL import Image
from gemma import gm

# Embedded preamble content from scenefutureunderstanding.txt
BACKGROUND_TEXT = (
    "Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. "
    "The player can move in the four cardinal directions using WASD and can interact with the tile directly in front of them in the direction they are facing using SPACE. "
    "Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest. "
    "The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). "
    "Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. "
    "Mana is used for casting spells or enchanting items and will naturally recover. "
    "Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. "
    "If the players health falls beneath 0 they will die and the game will restart. "
    "To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. "
    "Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. "
    "The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). "
    "There are 9 levels in total."
)

CONTROLS_LINES = [
    "q: noop",
    "w: up",
    "d: right",
    "s: down",
    "a: left",
    "space: do",
    "1: make_wood_pickaxe",
    "2: make_stone_pickaxe",
    "3: make_iron_pickaxe",
    "4: make_diamond_pickaxe",
    "5: make_wood_sword",
    "6: make_stone_sword",
    "7: make_iron_sword",
    "8: make_diamond_sword",
    "t: place_table",
    "tab: sleep",
    "r: place_stone",
    "f: place_furnace",
    "p: place_plant",
    "e: rest",
    ",: ascend",
    ".: descend",
    "y: make_iron_armour",
    "u: make_diamond_armour",
    "i: shoot_arrow",
    "o: make_arrow",
    "g: cast_fireball",
    "h: cast_iceball",
    "j: place_torch",
    "z: drink_potion_red",
    "x: drink_potion_green",
    "c: drink_potion_blue",
    "v: drink_potion_pink",
    "b: drink_potion_cyan",
    "n: drink_potion_yellow",
    "m: read_book",
    "k: enchant_sword",
    "l: enchant_armour",
    "[: make_torch",
    "]: level_up_dexterity",
    "-: level_up_strength",
    "=: level_up_intelligence",
    ";: enchant_bow",
]


def parse_prompts_by_image(definition_file_path: str) -> Dict[int, List[str]]:
    """Parse prompts grouped under 'Image N:' sections in the definition file.

    Returns a mapping from image index (1..4) to a list of prompt strings.
    """
    with open(definition_file_path, "r", encoding="utf-8") as file_handle:
        lines = [line.rstrip("\n") for line in file_handle]

    current_image_index: int = -1
    prompts_by_image: Dict[int, List[str]] = {}

    for line in lines:
        stripped = line.strip()
        # Detect headers like "Image 1:" (case-insensitive), without regex
        if stripped.endswith(":"):
            header = stripped[:-1]
            parts = header.split()
            if len(parts) == 2 and parts[0].lower() == "image" and parts[1].isdigit():
                current_image_index = int(parts[1])
                prompts_by_image[current_image_index] = []
                continue

        if current_image_index != -1:
            if stripped.startswith("* "):
                prompt = stripped[2:].strip()
                if prompt:
                    prompts_by_image[current_image_index].append(prompt)

            # Stop conditions are implicit when next Image section is encountered

    return prompts_by_image


def load_image_as_gemma_jpeg(image_path: str) -> Image.Image:
    """Load an image and re-encode to JPEG as Gemma is trained on JPEG format."""
    image = Image.open(image_path)
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return Image.open(buffer)


def build_preamble(background_text: str, controls_lines: List[str]) -> str:
    controls_bulleted = "\n".join(f"- {line}" for line in controls_lines)
    preamble = (
        "You are playing a Minecraft-inspired 2D game called Craftax. Your character is in the center of the frame, and is facing the direction shown by the yellow triangle. \n"
        "Use the following rules and available actions to reason about the current frame.\n\n"
        f"Game instructions:\n{background_text}\n\n"
        f"Possible actions (controls):\n{controls_bulleted}\n\n"
        "Answer the question about the current frame based only on the provided image."
    )
    return preamble


def run_gemma3_on_prompts(image: Image.Image, prompts: List[str], preamble: str) -> List[Tuple[str, str, float]]:
    """Run Gemma 3 on each prompt for a single image and return (prompt, reply, elapsed_s)."""
    model = gm.nn.Gemma3_4B()
    params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

    sampler = gm.text.ChatSampler(
        model=model,
        params=params,
        multi_turn=False,
    )

    responses: List[Tuple[str, str, float]] = []
    for prompt in prompts:
        full_prompt = f"{preamble}\n\nQuestion: {prompt} \n<start_of_image>"
        start_time = time.perf_counter()
        reply = sampler.chat(full_prompt, images=image)
        elapsed = time.perf_counter() - start_time
        print(f"Prompt: {prompt}")
        print(f"Response: {reply}")
        print(f"Elapsed: {elapsed:.3f} s\n")
        responses.append((prompt, reply, elapsed))
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemma 3 on scene-understanding prompts for a selected image.")
    parser.add_argument(
        "--image",
        type=int,
        choices=[1, 2, 3, 4],
        required=True,
        help="Which image ID to evaluate (1-4). Maps to images/<ID>.png",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="/nfs/aidm_nfs/gene/testing_vlms/images",
        help="Directory containing 1.png..4.png",
    )
    parser.add_argument(
        "--definition_file",
        type=str,
        default="/nfs/aidm_nfs/gene/testing_vlms/scenefutureunderstanding.txt",
        help="Path to scenefutureunderstanding.txt containing prompts.",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Path to write prompts and responses. Defaults to results_image{N}.txt in the project dir.",
    )
    args = parser.parse_args()

    image_path = os.path.join(args.images_dir, f"{args.image}.png")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    prompts_by_image = parse_prompts_by_image(args.definition_file)
    if args.image not in prompts_by_image or not prompts_by_image[args.image]:
        raise ValueError(f"No prompts found for Image {args.image} in {args.definition_file}")

    print(f"Loading image: {image_path}")
    image = load_image_as_gemma_jpeg(image_path)

    preamble = build_preamble(BACKGROUND_TEXT, CONTROLS_LINES)

    print(f"Running Gemma 3 on Image {args.image} with {len(prompts_by_image[args.image])} prompts...\n")
    results = run_gemma3_on_prompts(image, prompts_by_image[args.image], preamble)

    # Write to output file
    default_out = f"/nfs/aidm_nfs/gene/testing_vlms/results_image{args.image}.txt"
    out_path = args.output_txt or default_out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Num prompts: {len(results)}\n\n")
        for idx, (prompt, reply, elapsed) in enumerate(results, start=1):
            f.write(f"=== Q{idx} ===\n")
            f.write(f"Question: {prompt}\n")
            f.write(f"Response: {reply}\n")
            f.write(f"Elapsed: {elapsed:.3f} s\n\n")
    print(f"Wrote results to: {out_path}")


if __name__ == "__main__":
    main()


