from radon import segment_lines_with_radon
from os import listdir
import sys
import pathlib


def sort_func(file_name: str):
    name = file_name.split(sep=".")[0]
    return int(name)


def run_radon_segment_for_files(dir, output_prefix):
    files = listdir(dir)
    files.sort(key=sort_func)
    files_with_errors = []
    for file_name in files:
        try:
            segment_lines_with_radon(f"{dir}/{file_name}", output_prefix)
        except:
            files_with_errors.append(file_name)

    with open("./files_with_errors.txt", "w") as f:
        f.write(str(files_with_errors))


def experiment():
    print(listdir("\\\\132.68.109.77/Public/Yevgeni/captures/scan-sets/2022-04-26/unmethylated"))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError("Not enough arguments for script")
    input = pathlib.Path(sys.argv[1])
    output = pathlib.Path(sys.argv[2])
    if not (input.is_dir() and output.is_dir()):
        raise ValueError("input or output aren't valid directories")
    run_radon_segment_for_files(dir=input.absolute(), output_prefix=output.absolute())
