import argparse
import time
from typing import List, Dict
import os
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# Embedded preamble content
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


def build_preamble(background_text: str, controls_lines: List[str]) -> str:
    """Build a preamble with game rules and controls."""
    controls_bulleted = "\n".join(f"- {line}" for line in controls_lines)
    preamble = (
        "You are playing a Minecraft-inspired 2D game called Craftax. Your character is in the center of the frame, and is facing the direction shown by the yellow triangle. \n"
        "Use the following rules and available actions to reason about the current frame.\n\n"
        f"Game instructions:\n{background_text}\n\n"
        f"Possible actions (controls):\n{controls_bulleted}\n\n"
        "Answer the question about the current frame based only on the provided image."
    )
    return preamble


def load_model(model_name: str, use_flash_attention: bool = False):
    """Load the Qwen3-VL model and processor."""
    if use_flash_attention:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype="auto", 
            device_map="auto"
        )
    
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def generate_response(
    model,
    processor,
    image_path: str,
    prompt: str,
    preamble: str,
    max_new_tokens: int,
    temperature: float,
    seed: int = 42,
) -> str:
    """Generate a response for a single prompt with an image."""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Combine preamble with prompt
    full_prompt = f"{preamble}\n\nQuestion: {prompt}"
    
    # Prepare messages in the format expected by Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": full_prompt},
            ],
        }
    ]
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    # Generate
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )
    
    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    return output_text[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-4B-Instruct with Hugging Face Transformers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model identifier from Hugging Face",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images",
        help="Directory containing the images",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--flash_attention",
        action="store_true",
        help="Use flash attention 2 for better performance",
    )
    args = parser.parse_args()
    
    # Define prompts for each image
    image_prompts: Dict[int, List[str]] = {
        1: [
            "What tile is the character directly facing?",
            "How many trees are visible right now?",
            "Describe the scene. What is on the screen right now?",
            "What should the character do right now to progress?",
            "What should the character do in the future?",
        ],
        2: [
            "What tile is the character directly facing?",
            "Describe the scene. What is on the screen right now?",
            "What should the character do right now to progress?",
            "What should the character do in the future?",
            "If the character is low on hunger, what should they do?",
        ],
        3: [
            "What tile is the character directly facing?",
            "Describe the scene. What is on the screen right now?",
            "What should the character do right now to progress?",
            "What should the character do in the future?",
            "If the character is low on hunger, what should they do?",
            "If the character has stone and wood, what should they do?",
        ],
        4: [
            "What tile is the character directly facing?",
            "Describe the scene. What is on the screen right now?",
            "What should the character do right now to progress?",
            "What should the character do in the future?",
        ],
    }
    
    # Validate images directory
    images_dir = os.path.expanduser(args.images_dir)
    if not os.path.isabs(images_dir):
        images_dir = os.path.abspath(images_dir)
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Validate all image files exist
    image_paths = {}
    for i in range(1, 5):
        image_path = os.path.join(images_dir, f"{i}.png")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        try:
            Image.open(image_path).verify()
            image_paths[i] = image_path
        except Exception as e:
            raise ValueError(f"Invalid image file {image_path}: {e}")
    
    print(f"Loading model: {args.model}")
    print(f"Flash Attention 2: {args.flash_attention}")
    model, processor = load_model(args.model, args.flash_attention)
    print("Model loaded successfully!\n")
    
    # Build preamble with game context
    preamble = build_preamble(BACKGROUND_TEXT, CONTROLS_LINES)
    
    # Run inferences
    total_inferences = 0
    total_time = 0.0
    
    for image_num in range(1, 5):
        image_path = image_paths[image_num]
        prompts = image_prompts[image_num]
        
        print(f"\n{'='*80}")
        print(f"IMAGE {image_num}: {image_path}")
        print(f"{'='*80}\n")
        
        for prompt_idx, prompt in enumerate(prompts, start=1):
            print(f"Question {prompt_idx}/{len(prompts)}: {prompt}")
            print('-' * 80)
            
            start_time = time.perf_counter()
            output = generate_response(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=prompt,
                preamble=preamble,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
            )
            elapsed = time.perf_counter() - start_time
            
            print(f"Response: {output}")
            print(f"Time: {elapsed:.3f} s\n")
            
            total_inferences += 1
            total_time += elapsed
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total images processed: 4")
    print(f"Total inferences: {total_inferences}")
    print(f"Total time: {total_time:.3f} s")
    print(f"Average time per inference: {total_time/total_inferences:.3f} s")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()