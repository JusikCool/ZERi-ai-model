import yaml
from pathlib import Path
from model.m3_full_model.dataset import load_data
from validation.backtest.backtest import ukraine_inflation_backtest

CONFIG_PATH = Path("configs/config.yaml")

with open(CONFIG_PATH, encoding="utf-8") as f:
    config = yaml.safe_load(f)

df = load_data()
report = ukraine_inflation_backtest(df, config)

print("\n" + "=" * 60)
print("[우크라이나 전쟁 + 인플레이션 구간 백테스트] 2022-02-01 ~ 2022-07-01")
print("=" * 60)
print(report.to_string(index=False))
print(f"\nViolation Rate Pass: {report['vr_pass'].sum()}/{len(report)}")
print(f"Kupiec Pass:         {report['kupiec_pass'].sum()}/{len(report)}")

out = Path("model/saved/ukraine_inflation_backtest_results.csv")
report.to_csv(out, index=False)
print(f"\n결과 저장: {out}")
