from pathlib import Path
import pandas as pd

def convert_uci_arff_to_csv():
    raw_dir = Path("ml/data/raw")
    # Locate the full chronic kidney disease arff file
    candidates = list(raw_dir.glob("**/*full*.arff"))
    if not candidates:
        candidates = list(raw_dir.glob("**/*.arff"))

    if not candidates:
        print("No .arff file found under ml/data/raw. Extract 'disease_full.rar' first.")
        return

    source_file = candidates[0]
    print(f"Parsing: {source_file}")

    columns = []
    data_rows = []
    is_data_section = False

    with open(source_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("%"):
                continue

            if line_str.lower().startswith("@attribute"):
                parts = line_str.split()
                # Attribute name is the second token
                attr_name = parts[1].strip("'\"")
                columns.append(attr_name)

            elif line_str.lower().startswith("@data"):
                is_data_section = True
                continue

            elif is_data_section:
                # Strip all leading/trailing whitespace and tabs from each cell
                cleaned_cells = [cell.strip().replace("\t", "") for cell in line_str.split(",")]
                if len(cleaned_cells) == len(columns):
                    data_rows.append(cleaned_cells)

    df = pd.DataFrame(data_rows, columns=columns)
    
    # Replace '?' with standard empty NaN
    df = df.replace("?", pd.NA)

    out_csv = raw_dir / "uci_ckd.csv"
    df.to_csv(out_csv, index=False)
    print(f"Successfully converted {len(df)} rows to: {out_csv}")

if __name__ == "__main__":
    convert_uci_arff_to_csv()