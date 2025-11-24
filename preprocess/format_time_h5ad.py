import pandas as pd
import numpy as np
import re
import argparse
import scanpy as sc
import anndata as ad
import sys
from format_time import parse_single_timepoint, has_multiple_references,has_multiple_units, get_common_unit

# Species gestation periods dictionary (unit: days)
GESTATION_PERIODS = {
    'Homo sapiens': 266,        # Human
    'Mus musculus': 20,          # Mouse
    'Danio rerio': 3,            # Zebrafish
    'Macaca fascicularis': 165,  # Cynomolgus monkey
    'Macaca mulatta': 165,       # Rhesus monkey
    'Oryctolagus cuniculus': 31, # Rabbit
    'Sus scrofa': 114,           # Pig
    'Callithrix jacchus': 140,   # Marmoset
    'Caenorhabditis elegans': 0  # C. elegans (no timepoint conversion)
}

def process_anndata_timepoints(adata, time_col='Timepoint', new_col='time'):
    """
    Process Timepoint information in AnnData object
    """
    # Extract Species and Stage from uns
    species = adata.uns.get('Species', 'Unknown')
    stage_info = adata.uns.get('Stage', 'Unknown')
    
    # Get unique timepoints from obs
    unique_timepoints = sorted(adata.obs[time_col].unique())
    timepoints_str = ",".join([str(tp) for tp in unique_timepoints])
    
    print(f"Processing timepoints for species: {species}")
    print(f"Unique timepoints found: {timepoints_str}")
    
    # Process timepoints
    if pd.isna(timepoints_str) or timepoints_str in ['Unknown', 'nan', '']:
        # No timepoints to process
        adata.obs[new_col] = np.nan
        adata.uns['Timepoint'] = ""
        adata.uns['timepoints'] = ""
        adata.uns['time_unit'] = "unknown"
        adata.uns['conversion_rule'] = "No timepoints to process"
        adata.uns['special_notes'] = "No timepoints to process"
        adata.uns['needs_conversion'] = "No"
        adata.uns['needs_reference_unification'] = "No"
        return adata
    
    adata.uns['Timepoint'] = timepoints_str

    # Split timepoints
    timepoints = timepoints_str.split(',')
    
    days_values = []
    original_units = []
    original_references = []
    conversion_rules = []
    special_notes = []
    
    # Parse each timepoint
    for tp in timepoints:
        tp = tp.strip()
        if tp:  # Non-empty timepoint
            days_value, rule, note, unit, reference = parse_single_timepoint(tp, species, stage_info)
            days_values.append(days_value)
            original_units.append(unit)
            original_references.append(reference)
            conversion_rules.append(rule)
            special_notes.append(note)
    
    # Check if time reference unification is needed
    needs_reference_unification = has_multiple_references(original_references)
    
    # Check if unit conversion is needed
    needs_conversion = has_multiple_units(original_units)
    
    # For C. elegans, no conversion
    if species == 'Caenorhabditis elegans':
        # C. elegans uses parsed values directly
        final_values = []
        for days_val in days_values:
            if np.isnan(days_val):
                final_values.append("")
            else:
                # C. elegans values might be large, keep as is
                final_values.append(f"{days_val:.2f}")
        
        # Create mapping from original timepoint to converted value
        timepoint_mapping = {}
        for i, tp in enumerate(timepoints):
            if i < len(final_values) and final_values[i]:
                timepoint_mapping[tp.strip()] = float(final_values[i])
            else:
                timepoint_mapping[tp.strip()] = np.nan
        
        # Apply mapping to obs
        adata.obs[new_col] = adata.obs[time_col].map(timepoint_mapping)
        
        # Store sorted time in uns
        adata.uns['timepoints'] = ",".join([v for v in final_values if v])
        adata.uns['time_unit'] = "unknown"
        adata.uns['conversion_rule'] = " | ".join(conversion_rules)
        adata.uns['special_notes'] = "C. elegans data, no conversion | " + " | ".join(special_notes)
        adata.uns['needs_conversion'] = "No"
        adata.uns['needs_reference_unification'] = "No"
        return adata
    
    # Get species gestation period
    gestation = GESTATION_PERIODS.get(species, 0)
    
    # If time reference unification is needed, convert all timepoints to fertilization reference
    unified_days_values = []
    if needs_reference_unification:
        for i, (days_val, reference) in enumerate(zip(days_values, original_references)):
            if reference == "birth" and not np.isnan(days_val):
                # Convert from birth reference to fertilization reference
                unified_days_values.append(days_val + gestation)
                conversion_rules[i] += f" (from birth to fertilization+{gestation} days)"
                special_notes[i] += f" | Birth reference converted to fertilization reference"
            else:
                unified_days_values.append(days_val)
    else:
        unified_days_values = days_values
    
    # Process timepoints based on whether conversion is needed
    if unified_days_values and not all(np.isnan(unified_days_values)):
        if needs_conversion:
            # Multiple units, perform unified conversion to days
            best_unit = "days"
            divisor = 1.0
            
            # Convert to final values
            final_values = []
            for days_val in unified_days_values:
                if np.isnan(days_val):
                    final_values.append("")
                else:
                    final_val = days_val / divisor
                    # Ensure at least two significant digits for non-zero values
                    if final_val == 0:
                        formatted_val = "0.00"
                    elif abs(final_val) < 0.01:
                        formatted_val = f"{final_val:.4f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 1:
                        formatted_val = f"{final_val:.3f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 10:
                        formatted_val = f"{final_val:.2f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 100:
                        formatted_val = f"{final_val:.1f}".rstrip('0').rstrip('.')
                    else:
                        formatted_val = f"{final_val:.0f}"
                    final_values.append(formatted_val)
            
            # Create mapping from original timepoint to converted value
            timepoint_mapping = {}
            for i, tp in enumerate(timepoints):
                if i < len(final_values) and final_values[i]:
                    timepoint_mapping[tp.strip()] = float(final_values[i])
                else:
                    timepoint_mapping[tp.strip()] = np.nan
            
            # Apply mapping to obs
            adata.obs[new_col] = adata.obs[time_col].map(timepoint_mapping)
            
            # Store sorted time in uns
            adata.uns['timepoints'] = ",".join([v for v in final_values if v])
            adata.uns['time_unit'] = best_unit
            
        else:
            # Single unit, keep original unit
            common_unit = get_common_unit(original_units)
            
            # Select appropriate divisor based on common unit
            unit_divisors = {
                "embryonic_days": 1.0,
                "gestational_weeks": 7.0,
                "postnatal_days": 1.0,
                "postnatal_days_range": 1.0,
                "hours": 1/24,
                "hours_after_fertilization": 1/24,
                "months": 30.0,
                "years": 365.0,
                "age_range": 365.0,
                "inequality": 1.0,
                "hpf": 1/24,
                "gestational_days": 1.0,
                "estimated_days": 1.0,
                "weeks": 7.0
            }
            
            divisor = unit_divisors.get(common_unit, 1.0)
            
            # Convert to final values
            final_values = []
            for days_val in unified_days_values:
                if np.isnan(days_val):
                    final_values.append("")
                else:
                    final_val = days_val / divisor
                    # Ensure at least two significant digits for non-zero values
                    if final_val == 0:
                        formatted_val = "0.00"
                    elif abs(final_val) < 0.01:
                        formatted_val = f"{final_val:.4f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 1:
                        formatted_val = f"{final_val:.3f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 10:
                        formatted_val = f"{final_val:.2f}".rstrip('0').rstrip('.')
                    elif abs(final_val) < 100:
                        formatted_val = f"{final_val:.1f}".rstrip('0').rstrip('.')
                    else:
                        formatted_val = f"{final_val:.0f}"
                    final_values.append(formatted_val)
            
            # Create mapping from original timepoint to converted value
            timepoint_mapping = {}
            for i, tp in enumerate(timepoints):
                if i < len(final_values) and final_values[i]:
                    timepoint_mapping[tp.strip()] = float(final_values[i])
                else:
                    timepoint_mapping[tp.strip()] = np.nan
            
            # Apply mapping to obs
            adata.obs[new_col] = adata.obs[time_col].map(timepoint_mapping)
            
            # Store sorted time in uns
            adata.uns['timepoints'] = ",".join([v for v in final_values if v])
            adata.uns['time_unit'] = common_unit
    else:
        adata.obs[new_col] = np.nan
        adata.uns['timepoints'] = ""
        adata.uns['time_unit'] = "unknown"
    
    # Record conversion rules and special notes
    if conversion_rules:
        adata.uns['conversion_rule'] = " | ".join(conversion_rules)
    if special_notes:
        adata.uns['special_notes'] = " | ".join(special_notes)
    
    # Store conversion flags
    adata.uns['needs_conversion'] = "Yes" if needs_conversion else "No"
    adata.uns['needs_reference_unification'] = "Yes" if needs_reference_unification else "No"
    
    return adata

# Main function for command line usage
def main():
    parser = argparse.ArgumentParser(description='Process timepoints in h5ad file')
    parser.add_argument('input', help='Input h5ad file path')
    parser.add_argument('output', help='Output h5ad file path')
    
    args = parser.parse_args()
    
    # Read h5ad file
    try:
        print(f"Reading h5ad file: {args.input}")
        adata = sc.read_h5ad(args.input)
    except Exception as e:
        print(f"Error reading h5ad file: {e}")
        sys.exit(1)
    
    time_col = "Timepoint"
    new_col = 'time'

    # Check if required fields exist
    if time_col not in adata.obs.columns:
        print("Error: {time_col} column not found in adata.obs")
        sys.exit(1)
    
    if 'Species' not in adata.uns:
        print("Warning: 'Species' not found in adata.uns, using 'Unknown'")
        adata.uns['Species'] = 'Unknown'
    
    if 'Stage' not in adata.uns:
        print("Warning: 'Stage' not found in adata.uns, using 'Unknown'")
        adata.uns['Stage'] = 'Unknown'
    
    # Process timepoints
    adata_processed = process_anndata_timepoints(adata)
    
    # Save processed h5ad file
    try:
        print(f"Saving processed h5ad file: {args.output}")
        adata_processed.write_h5ad(args.output)
    except Exception as e:
        print(f"Error saving h5ad file: {e}")
        sys.exit(1)
    
    # Display some statistics
    print(f"\nProcessing statistics:")
    print(f"Total cells: {adata_processed.n_obs}")
    print(f"Unique timepoints: {len(adata_processed.obs[time_col].unique())}")
    print(f"Time unit: {adata_processed.uns.get('time_unit', 'unknown')}")
    print(f"Needs conversion: {adata_processed.uns.get('needs_conversion', 'unknown')}")
    print(f"Needs reference unification: {adata_processed.uns.get('needs_reference_unification', 'unknown')}")
    
    # Show sorted time values
    timepoints = adata_processed.uns.get('timepoints', '')
    if timepoints:
        print(f"Sorted time values: {timepoints}")

if __name__ == "__main__":
    main()
