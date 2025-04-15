import os
import ray
from ray.exceptions import RayActorError
import pefile
import asyncio
from tqdm.asyncio import tqdm_asyncio

from pe2image import PE2Image

MAX_CONCURRENT = 16  # number of actor

async def process_file(file_path, output_path, semaphore):
    async with semaphore:
        actor = None
        try:
            # PE file's name without extenstion 
            file_base = os.path.splitext(os.path.basename(file_path))[0]

            # check target's RGB existence
            rgb_output_path = os.path.join(output_path, "RGB", f"{file_base}.png")
            if os.path.exists(rgb_output_path):
                return file_path, f"[SKIPPED] {rgb_output_path} already exists"

            use_gpu = False

            actor = PE2Image.options().remote(
                file_path=file_path,
                pe_image_width=256,
                window_size=32,
                verbose=False,
                use_gpu=use_gpu
            )

            result = await actor.make_image.remote(output_path, step_size=1)
            await ray.kill(actor)

            return file_path, result

        except (pefile.PEFormatError, ValueError, RayActorError) as e:
            if actor is not None:
                try:
                    await ray.kill(actor)
                except Exception:
                    pass
            return file_path, f"[SKIPPED] {e}"

        except Exception as e:
            if actor is not None:
                try:
                    await ray.kill(actor)
                except Exception:
                    pass
            return file_path, f"[ERROR] Unexpected: {e}"


async def main():
    ray.init(_temp_dir="/data/malware2024/ray_tmp")
    input_path = "/data/malware2024/pe2image/input/testR"
    output_path = "output"
    
    file_list = [
        os.path.join(input_path, f)
        for f in os.listdir(input_path)
        if os.path.isfile(os.path.join(input_path, f))
    ]

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [process_file(file_path, output_path, semaphore) for file_path in file_list]

    pbar = tqdm_asyncio(total=len(tasks), desc="Process")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        file_path, result = await coro
        if isinstance(result, str) and result.startswith("[SKIPPED]"):
            print(f"{file_path}: {result}")
        completed += 1
        pbar.update(1)

    pbar.close()

if __name__ == '__main__':
    asyncio.run(main())
