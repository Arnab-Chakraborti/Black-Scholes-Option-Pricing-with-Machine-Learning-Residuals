from src.market_data import build_synthetic_dataset
from src.arbitrage import run_all_arbitrage_checks


df = build_synthetic_dataset()

results = run_all_arbitrage_checks(df)

for name, result in results.items():

    print("\n" + "=" * 50)
    print(name)

    if result.empty:
        print("No violations detected.")
        continue

    violation_columns = [
        col for col in result.columns
        if "violation" in col
    ]

    if violation_columns:

        for col in violation_columns:
            violations = result[col].sum()
            total = len(result)

            print(
                f"{col}: "
                f"{violations}/{total} violations"
            )

    else:
        print(f"Observations: {len(result)}")
