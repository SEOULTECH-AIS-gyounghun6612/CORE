"""
프로젝트 진행에 따른 실행 내역 및 결과와 같은 주요 내용을 생성, 출력, 기록하기 위한 모듈

--------------------------------------------------------------------
"""


@staticmethod
def Progress_bar(
    iteration: int,
    total: int,
    prefix: str = '',
    suffix: str = '',
    decimals: int = 1,
    length: int = 100,
    fill: str = '█'
):
    """
    Call in a loop to create terminal progress bar

    Parameters
    --------------------
    iteration
        current iteration
    total
        total iterations (Int)
    prefix
        prefix string (Str)
    suffix
        suffix string (Str)
    decimals
        positive number of decimals in percent complete (Int)
    length
        character length of bar (Int)
    fill
        bar fill character (Str)
    """
    percent = (
        "{0:." + str(decimals) + "f}"
    ).format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end="\r")
    # Print New Line on Complete
    if iteration == total:
        print()
