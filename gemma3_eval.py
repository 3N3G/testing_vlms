from PIL import Image
from gemma import gm
import time
import numpy as np
import io

# Model and parameters
model = gm.nn.Gemma3_4B()
params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_4B_IT)

image = Image.open("/nfs/aidm_nfs/gene/testing_vlms/im2.png")
image = image.convert('RGB')

buffer = io.BytesIO()
image.save(buffer, format='JPEG', quality=95)  # Gemma is trained on JPEG format
buffer.seek(0)
image = Image.open(buffer)

# Example of multi-turn conversation
sampler = gm.text.ChatSampler(
    model=model,
    params=params,
    multi_turn=False,
)

prompt1 = "What type of tile is the character facing in the image? E.g. grass, dirt, stone, etc. <start_of_image>"
prompt2 = "How much health points does the character have? <start_of_image>"
prompt3 = "Are there any animals in the image? <start_of_image>"
prompt4 = "Are there any plants in the image? <start_of_image>"
prompt5 = "Are there any trees in the image? <start_of_image>"
prompt6 = "Is there any water in the image? <start_of_image>"

# First inference
start_time = time.perf_counter()
out0 = sampler.chat(prompt1, images=image)
print(out0)
elapsed_s = time.perf_counter() - start_time
print(f"First inference: {elapsed_s:.3f} s")

# Second inference
start_time = time.perf_counter()
out1 = sampler.chat(prompt2, images=image)
print(out1)
elapsed_s = time.perf_counter() - start_time
print(f"Second inference: {elapsed_s:.3f} s")

# Third inference
start_time = time.perf_counter()
out2 = sampler.chat(prompt3, images=image)
print(out2)
elapsed_s = time.perf_counter() - start_time
print(f"Third inference: {elapsed_s:.3f} s")

# Fourth inference
start_time = time.perf_counter()
out3 = sampler.chat(prompt4, images=image)
print(out3)
elapsed_s = time.perf_counter() - start_time
print(f"Fourth inference: {elapsed_s:.3f} s")

# Fifth inference
start_time = time.perf_counter()
out4 = sampler.chat(prompt5, images=image)
print(out4)
elapsed_s = time.perf_counter() - start_time
print(f"Fifth inference: {elapsed_s:.3f} s")

# Sixth inference
start_time = time.perf_counter()
out5 = sampler.chat(prompt6, images=image)
print(out5)
elapsed_s = time.perf_counter() - start_time
print(f"Sixth inference: {elapsed_s:.3f} s")