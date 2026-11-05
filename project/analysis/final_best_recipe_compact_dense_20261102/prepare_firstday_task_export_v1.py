"""Create a separately frozen exporter for the first-day task specialist."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE/'export_firstday_task_v1.py'


def main():
    assert not TARGET.exists(), 'Preserve every exporter version'
    source = (HERE/'export_cold_task_v1.py').read_text(encoding='utf8')
    source = source.replace('cold_task', 'firstday_task')
    compile(source, str(TARGET), 'exec')
    TARGET.write_text(source, encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
