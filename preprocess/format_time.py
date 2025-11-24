import pandas as pd
import numpy as np
import re

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

def parse_single_timepoint(tp_str, species, stage_info):
    """
    Parse single timepoint, return value in days and conversion information
    """
    if pd.isna(tp_str) or tp_str in ['Unknown', 'nan', '']:
        return np.nan, "Cannot parse", "Empty or unknown", "unknown", "unknown"
    
    tp_str = str(tp_str).strip()
    
    # Special string handling
    special_cases = {
        'blastula stage': (5, "Estimated as 5 days based on blastula stage", "Blastula stage", "estimated_days", "fertilization"),
        'neonetal': (0, "Estimated as 0 days based on neonatal stage", "Neonatal stage (possible spelling error)", "estimated_days", "birth"),
        'neonetel': (0, "Estimated as 0 days based on neonatal stage", "Neonatal stage (possible spelling error)", "estimated_days", "birth"),
        'unknown': (np.nan, "Cannot parse", "Unknown timepoint", "unknown", "unknown")
    }
    
    for key, (value, rule, note, unit, reference) in special_cases.items():
        if key in tp_str.lower():
            return value, rule, note, unit, reference
    
    # C. elegans special handling - no timepoint conversion
    if species == 'Caenorhabditis elegans':
        # Handle range 100-130 -> 115
        if re.match(r'\d+-\d+', tp_str):
            nums = [float(x) for x in tp_str.split('-')]
            avg = sum(nums) / len(nums)
            return avg, f"Range {tp_str} averaged", "C. elegans range value", "unknown", "unknown"
        
        # Handle inequalities >650 -> 715, <100 -> 90
        elif re.match(r'[<>]\d+', tp_str):
            num = float(re.findall(r'(\d+\.?\d*)', tp_str)[0])
            if '>' in tp_str:
                result = num + 0.1 * num
                return result, f">{num} converted to {result}", "C. elegans inequality processing", "unknown", "unknown"
            elif '<' in tp_str:
                result = num - 0.1 * num
                return result, f"<{num} converted to {result}", "C. elegans inequality processing", "unknown", "unknown"
        
        # Other C. elegans formats
        else:
            try:
                num = float(tp_str)
                return num, f"C. elegans value {num}", "C. elegans value", "unknown", "unknown"
            except:
                return np.nan, f"Cannot parse C. elegans format: {tp_str}", "Unparseable C. elegans format", "unknown", "unknown"
    
    # Get species gestation period
    gestation = GESTATION_PERIODS.get(species, 0)
    
    # Embryonic days E (from fertilization)
    if re.match(r'E\d+', tp_str):
        embryo_match = re.findall(r'E(\d+\.?\d*)', tp_str)
        if embryo_match:
            days = float(embryo_match[0])
            return days, f"Embryonic day E{embryo_match[0]}", "Embryonic development days", "embryonic_days", "fertilization"
    
    # Gestational weeks GW (from fertilization)
    elif 'GW' in tp_str:
        # Handle range GW12-13
        range_match = re.findall(r'GW(\d+\.?\d*)-(\d+\.?\d*)', tp_str)
        if range_match:
            min_week, max_week = map(float, range_match[0])
            avg_week = (min_week + max_week) / 2
            days = avg_week * 7
            return days, f"Gestational week range {min_week}-{max_week} averaged", f"Average {avg_week:.1f} weeks", "gestational_weeks", "fertilization"
        
        # Handle single GW13
        week_match = re.findall(r'GW(\d+\.?\d*)', tp_str)
        if week_match:
            weeks = float(week_match[0])
            days = weeks * 7
            return days, f"Gestational {weeks} weeks", "Gestational weeks", "gestational_weeks", "fertilization"
    
    # Weeks (e.g., 9w) (from fertilization)
    elif re.match(r'\d+\s*w$', tp_str) or re.match(r'\d+w', tp_str):
        week_match = re.findall(r'(\d+\.?\d*)\s*w', tp_str)
        if week_match:
            weeks = float(week_match[0])
            days = weeks * 7
            return days, f"{weeks} weeks converted to days", f"{weeks} weeks", "weeks", "fertilization"
    
    # Postnatal days P (from birth)
    elif 'P' in tp_str:
        # Handle P range P0-P28
        p_range_match = re.findall(r'P(\d+\.?\d*)-P?(\d+\.?\d*)', tp_str)
        if p_range_match:
            min_p, max_p = map(float, p_range_match[0])
            avg_p = (min_p + max_p) / 2
            # Use postnatal days directly, without adding gestation
            days = avg_p
            return days, f"Postnatal range {min_p}-{max_p} days averaged", f"Average {avg_p} days", "postnatal_days_range", "birth"
        
        # Handle single P value
        postnatal_match = re.findall(r'P(\d+\.?\d*)', tp_str)
        if postnatal_match:
            postnatal_days = float(postnatal_match[0])
            # Use postnatal days directly, without adding gestation
            days = postnatal_days
            return days, f"Postnatal {postnatal_days} days", "Postnatal days", "postnatal_days", "birth"
    
    # Hours after fertilization (from fertilization)
    elif 'h' in tp_str.lower() and 'after' in tp_str.lower():
        hour_match = re.findall(r'(\d+\.?\d*)\s*h', tp_str.lower())
        if hour_match:
            hours = float(hour_match[0])
            days = hours / 24
            # Hours after fertilization directly as days
            return days, f"{hours} hours converted to days", f"{hours} hours (after fertilization)", "hours_after_fertilization", "fertilization"
    
    # Other hour timepoints (from fertilization)
    elif 'h' in tp_str.lower() and 'h' == tp_str[-1].lower():
        hour_match = re.findall(r'[-+]?\d+\.?\d*\s*h', tp_str.lower())
        if hour_match:
            # Extract number, note possible negative sign
            hours_str = hour_match[0].replace('h', '').strip()
            hours = float(hours_str)
            days = hours / 24
            return days, f"{hours} hours converted to days", f"{hours} hours", "hours", "fertilization"
    
    # Age in months (m for months) (from birth)
    elif 'month' in tp_str.lower() or (re.match(r'^\d+\.?\d*m$', tp_str.lower()) and len(tp_str) <= 10):
        month_match = re.findall(r'(\d+\.?\d*)-?(\d+\.?\d*)?\s*month', tp_str.lower())
        if not month_match:
            month_match = re.findall(r'(\d+\.?\d*)m', tp_str)
        
        if month_match:
            if isinstance(month_match[0], tuple) and month_match[0][1]:
                min_month, max_month = map(float, month_match[0])
                avg_month = (min_month + max_month) / 2
                days = avg_month * 30
                return days, f"{min_month}-{max_month} months averaged", f"Average {avg_month:.1f} months", "months", "birth"
            else:
                months = float(month_match[0][0] if isinstance(month_match[0], tuple) else month_match[0])
                days = months * 30
                return days, f"{months} months converted to days", f"{months} months", "months", "birth"
    
    # Handle inequalities with months (e.g., >40m)
    elif re.match(r'[<>]\d+\.?\d*m', tp_str, re.IGNORECASE):
        inequality_match = re.findall(r'([<>])(\d+\.?\d*)\s*m', tp_str, re.IGNORECASE)
        if inequality_match:
            operator, num_str = inequality_match[0]
            number = float(num_str)
            # Apply inequality adjustment (10%) to months directly
            if operator == '>':
                adjusted_months = number + 0.1 * number
                days = adjusted_months * 30
                return days, f">{number}m converted to {adjusted_months:.1f} months", "Inequality with months", "months", "birth"
            elif operator == '<':
                adjusted_months = number - 0.1 * number
                days = adjusted_months * 30
                return days, f"<{number}m converted to {adjusted_months:.1f} months", "Inequality with months", "months", "birth"
    
    # Age in years (including ranges) (from birth)
    elif 'y' in tp_str or (re.match(r'^\d+\.?\d*$', tp_str) and float(tp_str) > 1):
        # Handle range 60y-69y or 60-69y
        year_range_match = re.findall(r'(\d+\.?\d*)\s*y?\s*-\s*(\d+\.?\d*)\s*y', tp_str)
        if year_range_match:
            min_year, max_year = map(float, year_range_match[0])
            avg_year = (min_year + max_year) / 2
            days = avg_year * 365
            return days, f"Year range {min_year}-{max_year} averaged", f"Average {avg_year:.1f} years", "years", "birth"
        
        # Handle single 6y
        year_match = re.findall(r'(\d+\.?\d*)\s*y', tp_str)
        if year_match:
            years = float(year_match[0])
            days = years * 365
            return days, f"{years} years converted to days", f"{years} years", "years", "birth"
        
        # Pure numbers (assumed as years)
        elif re.match(r'^\d+\.?\d*$', tp_str):
            years = float(tp_str)
            days = years * 365
            return days, f"Pure number {years} parsed as years", f"{years} years", "years", "birth"
    
    # Age range 21-30 (without y) (from birth)
    elif re.match(r'\d+-\d+', tp_str) and 'h' not in tp_str and 'm' not in tp_str and 'y' not in tp_str:
        age_range = [float(x) for x in tp_str.split('-')]
        avg_age = sum(age_range) / len(age_range)
        days = avg_age * 365
        return days, f"Age range {age_range[0]}-{age_range[1]} averaged", f"Average {avg_age:.1f} years", "age_range", "birth"
    
    # Handle inequalities with years (e.g., >5y)
    elif re.match(r'[<>]\d+\.?\d*y', tp_str, re.IGNORECASE):
        inequality_match = re.findall(r'([<>])(\d+\.?\d*)\s*y', tp_str, re.IGNORECASE)
        if inequality_match:
            operator, num_str = inequality_match[0]
            number = float(num_str)
            # Apply inequality adjustment (10%) to years directly
            if operator == '>':
                adjusted_years = number + 0.1 * number
                days = adjusted_years * 365
                return days, f">{number}y converted to {adjusted_years:.1f} years", "Inequality with years", "years", "birth"
            elif operator == '<':
                adjusted_years = number - 0.1 * number
                days = adjusted_years * 365
                return days, f"<{number}y converted to {adjusted_years:.1f} years", "Inequality with years", "years", "birth"
    
    # Handle inequalities without units (e.g., >650, <100)
    elif re.match(r'[<>]\d+', tp_str):
        number = float(re.findall(r'(\d+\.?\d*)', tp_str)[0])
        if '>' in tp_str:
            days = number + 0.1 * number
            return days, f">{number} converted to {days:.1f}", "Inequality processing", "inequality", "unknown"
        elif '<' in tp_str:
            days = number - 0.1 * number
            return days, f"<{number} converted to {days:.1f}", "Inequality processing", "inequality", "unknown"
    
    # Zebrafish hours hpf (from fertilization)
    elif 'hpf' in tp_str:
        hours = float(re.findall(r'(\d+\.?\d*)hpf', tp_str)[0])
        days = hours / 24
        return days, f"{hours}hpf converted to days", "Zebrafish development hours", "hpf", "fertilization"
    
    # Rabbit gestational days GD (from fertilization)
    elif 'GD' in tp_str:
        days = float(re.findall(r'GD(\d+\.?\d*)', tp_str)[0])
        return days, f"Gestational {days} days", "Rabbit gestational days", "gestational_days", "fertilization"
    
    # Estimate based on Stage information
    stage_lower = str(stage_info).lower()
    if 'embryonic' in stage_lower:
        return 30, "Estimated as 30 days based on embryonic stage", "Embryonic stage estimation", "estimated_days", "fertilization"
    elif 'fetal' in stage_lower:
        return 150, "Estimated as 150 days based on fetal stage", "Fetal stage estimation", "estimated_days", "fertilization"
    elif 'newborn' in stage_lower or 'neonatal' in stage_lower:
        return 0, "Estimated as 0 days based on neonatal stage", "Neonatal stage estimation", "estimated_days", "birth"
    elif 'adult' in stage_lower:
        return 7300, "Estimated as 20 years based on adult stage", "Adult stage estimation", "estimated_days", "birth"
    elif 'paediatric' in stage_lower:
        return 1825, "Estimated as 5 years based on paediatric stage", "Paediatric stage estimation", "estimated_days", "birth"
    
    return np.nan, f"Cannot parse: {tp_str}", "Unparseable format", "unparseable", "unknown"

def has_multiple_references(references_list):
    """
    Check if multiple time references are present
    """
    # Filter out unknown and unparseable references
    valid_refs = [ref for ref in references_list if ref not in ["unknown", "unparseable"]]
    
    if len(valid_refs) <= 1:
        return False
    
    # Check if multiple different time references exist
    return len(set(valid_refs)) > 1

def has_multiple_units(units_list):
    """
    Check if multiple units are present
    """
    # Filter out unknown and unparseable units
    valid_units = [unit for unit in units_list if unit not in ["unknown", "unparseable"]]
    
    if len(valid_units) <= 1:
        return False
    
    # Check if multiple different units exist
    return len(set(valid_units)) > 1

def get_common_unit(units_list):
    """
    Get the most common unit from the list, ignoring unknown/unparseable
    """
    valid_units = [unit for unit in units_list if unit not in ["unknown", "unparseable"]]
    
    if not valid_units:
        return "unknown"
    
    # Return the most common unit
    return max(set(valid_units), key=valid_units.count)

def process_timepoint_dataframe(df):
    """
    Process Timepoint column for entire DataFrame
    """
    # Add new columns
    df['sorted_time'] = ""
    df['time_unit'] = ""
    df['conversion_rule'] = ""
    df['special_notes'] = ""
    df['needs_conversion'] = ""  # Mark if unit conversion is needed
    df['needs_reference_unification'] = ""  # Mark if time reference unification is needed
    
    for idx, row in df.iterrows():
        timepoint_str = row['Timepoint']
        species = row['Species']
        stage_info = row['Stage']
        
        # Handle multiple timepoints
        if pd.isna(timepoint_str):
            timepoints = [""]
        else:
            timepoints = str(timepoint_str).split(',')
        
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
        df.at[idx, 'needs_reference_unification'] = "Yes" if needs_reference_unification else "No"
        
        # Check if unit conversion is needed
        needs_conversion = has_multiple_units(original_units)
        df.at[idx, 'needs_conversion'] = "Yes" if needs_conversion else "No"
        
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
            
            df.at[idx, 'sorted_time'] = ",".join(final_values)
            df.at[idx, 'time_unit'] = "unknown"
            df.at[idx, 'conversion_rule'] = " | ".join(conversion_rules)
            df.at[idx, 'special_notes'] = "C. elegans data, no conversion | " + " | ".join(special_notes)
            continue
        
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
                
                # Join multiple values
                df.at[idx, 'sorted_time'] = ",".join(final_values)
                df.at[idx, 'time_unit'] = best_unit
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
                    "hours_after_fertilization": 1/24,  # Hours after fertilization need conversion
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
                
                # Join multiple values
                df.at[idx, 'sorted_time'] = ",".join(final_values)
                df.at[idx, 'time_unit'] = common_unit
        else:
            df.at[idx, 'sorted_time'] = ""
            df.at[idx, 'time_unit'] = "unknown"
        
        # Record conversion rules and special notes
        if conversion_rules:
            df.at[idx, 'conversion_rule'] = " | ".join(conversion_rules)
        if special_notes:
            df.at[idx, 'special_notes'] = " | ".join(special_notes)
    
    return df

# Read data and process
def main():
    # Read CSV file
    try:
        df = pd.read_csv('tedd_datasets_table.csv')
    except FileNotFoundError:
        print("Error: Cannot find 'tedd_datasets_table.csv' file")
        return
    
    # Process Timepoint column
    df_processed = process_timepoint_dataframe(df)
    
    # Save result as CSV
    df_processed.to_csv('tedd_datasets_table_processed.csv', index=False)
    print("Processing completed, result saved to tedd_datasets_table_processed.csv")
    
    # Display some statistics
    print(f"\nProcessing statistics:")
    print(f"Total rows: {len(df_processed)}")
    print(f"Rows needing time reference unification: {len(df_processed[df_processed['needs_reference_unification'] == 'Yes'])}")
    print(f"Rows needing unit conversion: {len(df_processed[df_processed['needs_conversion'] == 'Yes'])}")
    print(f"Unit distribution:")
    print(df_processed['time_unit'].value_counts())
    
if __name__ == "__main__":
    main()
