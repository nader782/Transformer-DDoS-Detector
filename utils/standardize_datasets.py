from typing import Dict, List, Optional, Tuple, Set, Union
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import json
import os


class DataStandardizer:
    def __init__(self, config_path: str, input_dir: str, output_dir: str):
        """
        Initialize the standardizer with paths and configuration.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Load JSON Configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.target_naming = self.config['cicflowmeter_library']
        self.target_columns = self.target_naming['columns_names']
        self.target_ts_pattern = self.target_naming['timestamp_pattern']
        self.features_to_drop = set(self.config.get(
            'cicflowmeter_library_duplicated_features', []))

        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _normalize_string(self, s: str) -> str:
        """
        Normalizes a column name for comparison (removes non-alphanumeric, lowercases, handles abbreviations).
        Used to match 'Flow Duration' with 'flow_duration'.
        """
        s = s.lower().strip()
        replacements = {
            ' ': '_', '/': '_', '.': '_',
            'source': 'src', 'destination': 'dst',
            'packet': 'pkt', 'length': 'len',
            'total': 'tot', 'forward': 'fwd',
            'backward': 'bwd', 'bytes': 'byts',
            'average': 'avg', 'variance': 'var',
            'std': 'std', 'duration': 'duration',
            'packets': 'pkts', 'count': 'cnt',
        }

        # Standardize separators
        s = s.replace(' ', '_').replace('.', '_').replace('/', '_')

        # Apply strict synonyms to parts of the string
        parts = s.split('_')
        normalized_parts = [replacements.get(p, p) for p in parts]
        # Return compressed string for comparison
        return "".join(normalized_parts)

    def _create_mapping_dictionary(self, current_columns: List[str], dataset_type: str) -> Dict[str, str]:
        """
        Creates a dictionary to map Current Column Name -> Target Column Name
        """
        phrases = {'Avg Fwd Segment Size': 'fwd_seg_size_avg',
                   'Avg Bwd Segment Size': 'bwd_seg_size_avg',
                   'Fwd Avg Bytes/Bulk': 'fwd_byts_b_avg',
                   'Bwd Avg Bytes/Bulk': 'bwd_byts_b_avg',
                   'Fwd Avg Packets/Bulk': 'fwd_pkts_b_avg',
                   'Bwd Avg Packets/Bulk': 'bwd_pkts_b_avg',
                   'Fwd Avg Bulk Rate': 'fwd_blk_rate_avg',
                   'Bwd Avg Bulk Rate': 'bwd_blk_rate_avg',
                   'Init_Win_bytes_backward': 'init_bwd_win_byts',
                   'Init_Win_bytes_forward': 'init_fwd_win_byts',
                   'Max Packet Length': 'pkt_len_max',
                   'Min Packet Length': 'pkt_len_min',
                   'Total Length of Fwd Packets': 'totlen_fwd_pkts',
                   'Total Length of Bwd Packets': 'totlen_bwd_pkts',
                   'act_data_pkt_fwd': 'fwd_act_data_pkts',
                   'min_seg_size_forward': 'fwd_seg_size_min',
                   'Average Packet Size': 'pkt_size_avg', }

        rephrased = {}
        for phrase, replacement in phrases.items():
            rephrased[self._normalize_string(phrase)] = replacement

        mapping = {}

        # Create a lookup map for the target columns
        # Key: Normalized String, Value: Actual Target Column Name
        target_map = {self._normalize_string(
            col): col for col in self.target_columns}

        target_map.update(rephrased)

        # Explicit overrides for tricky cases found in CIC datasets
        manual_overrides = {
            "flow_id": "flow_id",  # Sometimes missing in target, keep if needed or drop later
            "label": "label",
            "timestamp": "timestamp"
        }

        for col in current_columns:
            normalized_col = self._normalize_string(col)

            # 1. Try exact normalized match
            if normalized_col in target_map:
                mapping[col] = target_map[normalized_col]
            # 2. Handle known common variations manually if normalized logic fails
            elif "timestamp" in normalized_col:
                mapping[col] = "timestamp"
            elif normalized_col in manual_overrides:
                mapping[col] = manual_overrides[normalized_col]

        return mapping

    def _identify_dataset_type(self, columns: List[str]) -> str:
        """
        Identifies if the dataset is likely DDoS2019 or IDS2018 based on column headers.
        """
        cols_set = set(columns)

        # Get raw definitions from config
        ddos_cols = set(self.config['cic_ddos2019_naming']['columns_names'])
        ids_cols = set(self.config['cic_ids2018_naming']['columns_names'])

        # Calculate overlap
        ddos_overlap = len(cols_set.intersection(ddos_cols))
        ids_overlap = len(cols_set.intersection(ids_cols))

        if ddos_overlap > ids_overlap:
            return 'cic_ddos2019_naming'
        else:
            return 'cic_ids2018_naming'

    def process_file(
        self,
        filename: str,
        drop_label: Optional[Union[str, List[str]]] = None,
        keep_label: Optional[str] = None,
        max_rows: Optional[int] = None,
        random_state: int = 42,
    ) -> dict:
        """
        Main logic to process a single CSV file.

        Parameters
        ----------
        filename : str
            The CSV file name (relative to input_dir).
        drop_label : str or list of str, optional
            If provided, remove all rows where the Label column matches any of
            these values. Example: drop_label=['Benign', 'WebDDoS'].
        keep_label : str, optional
            If provided, keep only rows where the Label column equals this value.
            Example: keep_label='Benign' keeps only Benign rows.
        max_rows : int, optional
            If set and the cleaned dataset has more rows than this, randomly sample
            down to max_rows rows.
        random_state : int
            Random seed for reproducible sampling.

        Returns
        -------
        dict
            Summary of processing results (column counts, row counts, etc.).
        """
        filepath = os.path.join(self.input_dir, filename)
        print(f"Processing: {filename}...")

        result = {"File": filename}

        try:
            # Read CSV (Use low_memory=False for large files or specify dtypes if known)
            df = pd.read_csv(filepath, skipinitialspace=True, low_memory=False)
            original_columns = df.columns.tolist()
            rows_original = len(df)

            # 1. Identify Dataset
            dataset_key = self._identify_dataset_type(original_columns)
            source_config = self.config[dataset_key]
            source_ts_pattern = source_config['timestamp_pattern']

            # 2. Rename Columns
            mapping = self._create_mapping_dictionary(
                original_columns, dataset_key)
            df.rename(columns=mapping, inplace=True)

            # 3. Drop Duplicates (Defined in JSON)
            # Only drop if they exist in the dataframe
            cols_to_drop = [
                c for c in self.features_to_drop if c in df.columns]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, inplace=True)

            # 4. Standardize Timestamp
            if 'timestamp' in df.columns:
                try:
                    # Convert based on source pattern
                    df['timestamp'] = pd.to_datetime(
                        df['timestamp'], format="mixed", errors='coerce')
                    # Format to target pattern
                    df['timestamp'] = df['timestamp'].dt.strftime(
                        self.target_ts_pattern)
                except Exception as e:
                    print(
                        f"  Warning: Timestamp conversion failed for {filename}: {e}")

            # 5. Handle Missing/Extra Columns & Reordering
            present_target_cols = [
                c for c in self.target_columns if c in df.columns]

            # Log missing columns (cols in Target but not in File)
            missing_cols = list(set(self.target_columns) - set(df.columns))

            # Log dropped columns (cols in File but not in Target)
            dropped_cols = list(set(original_columns) - set(mapping.keys()))
            dropped_cols.extend(
                [c for c in df.columns if c not in present_target_cols])
            dropped_cols.extend(cols_to_drop)

            # 6. Reorder to target schema
            df_final = df[present_target_cols].copy()

            # ------------------------------------------------------------------
            # 7. Label-based row filtering
            # ------------------------------------------------------------------
            rows_after_rename = len(df_final)
            label_col = None

            # Find the actual label column name (case-insensitive search)
            for col in df_final.columns:
                if col.lower() == 'label':
                    label_col = col
                    break

            if label_col is not None:
                if drop_label is not None:
                    # Normalise to list
                    drop_labels = [drop_label] if isinstance(
                        drop_label, str) else list(drop_label)
                    drop_labels_lower = [lbl.strip().lower()
                                         for lbl in drop_labels]
                    mask = ~df_final[label_col].str.strip(
                    ).str.lower().isin(drop_labels_lower)
                    df_final = df_final[mask]
                    print(
                        f"  Dropped label(s) {drop_labels}: {rows_after_rename - len(df_final):,} rows removed")

                elif keep_label is not None:
                    mask = df_final[label_col].str.strip(
                    ).str.lower() == keep_label.strip().lower()
                    df_final = df_final[mask]
                    print(
                        f"  Kept only label='{keep_label}': {rows_after_rename - len(df_final):,} rows removed")
            else:
                print(
                    f"  Warning: No Label column found in {filename}, skipping label filter.")

            rows_after_label_filter = len(df_final)

            # ------------------------------------------------------------------
            # 8. Remove NaN and Infinite values
            # ------------------------------------------------------------------
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_final.dropna(inplace=True)
            rows_after_nan = len(df_final)
            print(
                f"  Removed NaN/Inf: {rows_after_label_filter - rows_after_nan:,} rows removed")

            # ------------------------------------------------------------------
            # 9. Remove duplicate rows
            # ------------------------------------------------------------------
            df_final.drop_duplicates(inplace=True)
            rows_after_dedup = len(df_final)
            print(
                f"  Removed duplicates: {rows_after_nan - rows_after_dedup:,} rows removed")

            # ------------------------------------------------------------------
            # 10. Chronological shrinking to max_rows
            # ------------------------------------------------------------------
            rows_after_shrink = rows_after_dedup
            if max_rows is not None and len(df_final) > max_rows:
                if 'timestamp' in df_final.columns:
                    df_final = df_final.sort_values('timestamp').head(max_rows)
                else:
                    df_final = df_final.head(max_rows)
                df_final.reset_index(drop=True, inplace=True)
                rows_after_shrink = len(df_final)
                print(f"  Sorted chronologically and trimmed to {max_rows:,} rows")

            # ------------------------------------------------------------------
            # 11. Save as PKL (same base filename, .pkl extension)
            # ------------------------------------------------------------------
            base_name = os.path.splitext(filename)[0]
            output_filename = base_name + '.pkl'
            output_path = os.path.join(self.output_dir, output_filename)
            df_final.to_pickle(output_path)
            print(
                f"  Saved to: {output_path}  ({len(df_final):,} rows × {len(df_final.columns)} cols)")

            # ------------------------------------------------------------------
            # 12. Build result summary
            # ------------------------------------------------------------------
            result.update({
                "Detected Type": dataset_key,
                "Original Rows": rows_original,
                "Rows After Label Filter": rows_after_label_filter,
                "Rows After NaN/Inf Removal": rows_after_nan,
                "Rows After Dedup": rows_after_dedup,
                "Final Rows": rows_after_shrink,
                "Original Column Count": len(original_columns),
                "Final Column Count": len(df_final.columns),
                "Missing Columns (Target cols not found)": sorted(missing_cols),
                "Dropped Columns (Extras/Duplicates)": sorted(set(dropped_cols)),
                "Final Columns": df_final.columns.tolist(),
                "Output File": output_path,
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            result["Error"] = str(e)

        return result

    def run(
        self,
        drop_label: Optional[Union[str, List[str]]] = None,
        keep_label: Optional[str] = None,
        max_rows: Optional[int] = None,
        random_state: int = 42,
    ) -> List[dict]:
        """
        Process all CSV files in input_dir.

        Parameters
        ----------
        drop_label : str or list of str, optional
            Remove all rows with any of these label values from every file.
        keep_label : str, optional
            Keep only rows with this label value in every file.
        max_rows : int, optional
            Maximum number of rows to keep per file after cleaning (random sample).
        random_state : int
            Random seed for reproducible sampling.

        Returns
        -------
        list of dict
            One summary dict per processed file.
        """
        csv_files = glob.glob(os.path.join(self.input_dir, "*.csv"))
        if not csv_files:
            print("No CSV files found in input directory.")
            return []

        print(f"Found {len(csv_files)} CSV file(s) in '{self.input_dir}'.")
        results = []
        for csv_file in csv_files:
            result = self.process_file(
                os.path.basename(csv_file),
                drop_label=drop_label,
                keep_label=keep_label,
                max_rows=max_rows,
                random_state=random_state,
            )
            results.append(result)
            print()

        return results


# ==========================================
# Execution Block (standalone usage)
# ==========================================
if __name__ == "__main__":
    CONFIG_FILE = "./utils/all_column_names.json"
    INPUT_DIRECTORY = "./data/CIC-DDoS2019"
    OUTPUT_DIRECTORY = "./data/CIC-DDoS2019"

    processor = DataStandardizer(
        CONFIG_FILE, INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    results = processor.run(drop_label='Benign',
                            max_rows=150_000, random_state=42)
