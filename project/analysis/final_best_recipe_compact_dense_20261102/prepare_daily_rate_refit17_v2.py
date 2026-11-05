"""Preserve v1 and create a separately frozen refit for the verified binomial winner."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / 'export_daily_rate_refit17_v2.py'


def main():
    assert not TARGET.exists(), 'Preserve every exporter version'
    source = (HERE / 'export_daily_rate_refit17_v1.py').read_text(encoding='utf8')
    changes = [
        ("RUN=HERE/'daily_rate_refit17_run_v1'", "RUN=HERE/'daily_rate_refit17_run_v2'"),
        ('output/best_recipe_daily_rate_refit17_20191117_20261102',
         'output/best_recipe_daily_rate_refit17_v2_20191117_20261102'),
        ("paths=[Path(__file__),HERE/'prepare_daily_rate_refit17_v1.py'",
         "paths=[Path(__file__),HERE/'prepare_daily_rate_refit17_v2.py',HERE/'export_daily_rate_refit17_v1.py',HERE/'prepare_daily_rate_refit17_v1.py'"),
        ('delivery_daily_rate_refit17_v1.json', 'delivery_daily_rate_refit17_v2.json'),
    ]
    for old, new in changes:
        assert old in source, old
        source = source.replace(old, new)
    compile(source, str(TARGET), 'exec')
    TARGET.write_text(source, encoding='utf8')
    print(TARGET)


if __name__ == '__main__':
    main()
