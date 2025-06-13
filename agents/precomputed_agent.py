import pandas as pd
import os

def precompute_advanced_kpi_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Store Name', 'KPI Name', 'Date'])

    # === ✅ Only Step 1: Daily Plan, Actual, and Achievement % ===
    for kpi in df['KPI Name'].unique():
        kpi_df = df[df['KPI Name'] == kpi].copy()
        kpi_df['Daily Plan'] = kpi_df.groupby(['Store Name', 'KPI Name'])['Plan'].diff().fillna(kpi_df['Plan'])
        kpi_df['Daily Actual'] = kpi_df.groupby(['Store Name', 'KPI Name'])['Actual'].diff().fillna(kpi_df['Actual'])
        kpi_df['Daily Achievement %'] = (kpi_df['Daily Actual'] / kpi_df['Daily Plan']) * 100
        kpi_df['Daily Achievement %'] = kpi_df['Daily Achievement %'].round(2)
        df.loc[kpi_df.index, ['Daily Plan', 'Daily Actual', 'Daily Achievement %']] = kpi_df[
            ['Daily Plan', 'Daily Actual', 'Daily Achievement %']
        ]

    return df

# === Run it ===
if __name__ == "__main__":
    input_file = "data/Gurugram Cluster.xlsx"
    output_file = "data/kpi_precomputed.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        exit()

    print("🔄 Reading raw KPI file...")
    raw_df = pd.read_excel(input_file)

    print("🧠 Precomputing KPI metrics (only till Daily Achievement %)...")
    computed_df = precompute_advanced_kpi_metrics(raw_df)

    print(f"💾 Saving precomputed data to {output_file}...")
    computed_df.to_excel(output_file, index=False)

    print("✅ Done! Preview:")
    print(computed_df.head(3))
