from radon import segment_lines_with_radon
from os import listdir


def sort_func(file_name: str):
    name = file_name.split(sep=".")[0]
    return int(name)


def run_radon_segment_for_files(dir, output_prefix):
    files = listdir(dir)
    files.sort(key=sort_func)
    for file_name in files:
        segment_lines_with_radon(f"{dir}/{file_name}", output_prefix)


def experiment():
    print(listdir("\\\\132.68.109.77/Public/Yevgeni/captures/scan-sets/2022-04-26/unmethylated"))


if __name__ == "__main__":
    run_radon_segment_for_files("\\\\132.68.109.77/Public/Yevgeni/segmenter-team/2022-04-26/unmethylated",
                                "\\\\132.68.109.77/Public/Yevgeni/segmenter-team/2022-04-26-segments")
