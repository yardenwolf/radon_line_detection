from radon import segment_lines_with_radon
from os import listdir


def run_radon_segment_for_files(dir):
    for file_name in listdir(dir):
        segment_lines_with_radon(file_name, output_prefix="Q/Yevgeni/segmenter-team/20220426T125332Z")


if __name__ == "__main__":
    run_radon_segment_for_files("Q/Yevgeni/captures/scan-sets/2022-04-26")
