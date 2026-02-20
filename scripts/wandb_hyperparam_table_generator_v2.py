#!/usr/bin/env python3
"""
A minimal, self-contained script to generate a markdown table of
W&B run hyperparameters based on a YAML configuration file.

Includes:
1. Robust fetching of run parameters from config (top-level, nested 'erelela_override').
2. Third fallback to search recorded command-line arguments ('args' in summary).
3. Optional output to a markdown file.
"""
import argparse
import yaml
import wandb
import json
from pathlib import Path
from typing import Dict, Set, List, Any, Optional
import re # Added for command-line parsing
import ast # Added for argparse help extraction

class HyperparamTableGenerator:
    """
    Extracts, collates, and formats hyperparameter data from W&B runs 
    into a markdown table based on a configuration file.
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration and W&B API."""
        self.config = self._load_config(config_path)
        self.api = wandb.Api(timeout=600)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def _normalize_run_id(run_id: Optional[str]) -> str:
        """Return the bare run ID regardless of whether a full path was provided."""
        if not run_id:
            return ''
        return run_id.split('/')[-1]

    @staticmethod
    def _extract_argparse_help(script_path: str) -> Dict[str, str]:
        """
        Parse a Python script's AST to extract argparse help strings.
        Returns {arg_dest_name: help_string} without importing/executing the script.
        """
        with open(script_path, 'r') as f:
            source = f.read()

        tree = ast.parse(source)
        help_map: Dict[str, str] = {}

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Match calls like parser.add_argument(...) or any .add_argument(...)
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'add_argument'):
                continue

            # Extract the long option name from positional args (e.g. "--learning_rate")
            dest_name = None
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value.startswith('--'):
                        dest_name = arg.value.lstrip('-').replace('-', '_')

            # Check for explicit dest= keyword override
            for kw in node.keywords:
                if kw.arg == 'dest' and isinstance(kw.value, ast.Constant):
                    dest_name = kw.value.value

            if not dest_name:
                continue

            # Extract help= keyword value
            for kw in node.keywords:
                if kw.arg == 'help' and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    help_map[dest_name] = kw.value.value
                    break

        return help_map

    def _get_project_context(self, run_id: str, group_config: Dict[str, Any]) -> Optional[str]:
        """
        Determines the full W&B path (entity/project/run_id) or the closest context 
        for a given run ID based on configuration hierarchy.
        """
        # 1. Check if run_id is already a full path
        if len(run_id.split('/')) > 2:
            return run_id 

        # 2. Check for group-specific project context
        group_project = group_config.get('project')
        if group_project:
            if len(group_project.split('/')) == 1:
                for p in self.config.get('projects', []):
                    if p.split('/')[-1] == group_project:
                        group_project = p
                        break
            
            if len(group_project.split('/')) >= 2:
                return f"{group_project}/{run_id}"

        # 3. Fallback to root projects list 
        all_projects = self.config.get('projects', [])
        if all_projects:
            default_project = all_projects[0]
            return f"{default_project}/{run_id}"
            
        return None 

    def _collect_all_runs_mapped(self) -> Dict[str, wandb.apis.public.Run]:
        """
        Collect all required runs and return a map of {normalized_run_id: run_object}.
        """
        run_paths_to_fetch: Dict[str, str] = {} 

        for figure_config in self.config.get('figures', []):
            for group_config in figure_config.get('groups', []):
                run_ids = group_config.get('run_ids', [])
                if isinstance(run_ids, str):
                    run_ids = [run_ids]

                for run_id in run_ids:
                    if not run_id: continue
                    
                    full_path = self._get_project_context(run_id, group_config)
                    normalized_id = self._normalize_run_id(run_id)

                    if full_path and normalized_id not in run_paths_to_fetch:
                        run_paths_to_fetch[normalized_id] = full_path
                    elif not full_path:
                        print(f"Warning: Could not determine W&B path for run ID '{run_id}'. Skipping.")

        if not run_paths_to_fetch:
            return {}

        print(f"Fetching {len(run_paths_to_fetch)} unique run IDs from W&B...")
        fetched_runs = self._fetch_specific_runs(run_paths_to_fetch)

        return fetched_runs

    def _fetch_specific_runs(self, run_paths_map: Dict[str, str]) -> Dict[str, wandb.apis.public.Run]:
        """Fetch runs by their determined full paths."""
        runs_map = {}
        
        for normalized_id, run_path in run_paths_map.items():
            try:
                run = self.api.run(run_path)
                runs_map[normalized_id] = run
            except Exception as e:
                print(f"  Warning: Could not fetch run {run_path}. Error: {e}")
                
        return runs_map

    @staticmethod
    def _parse_cmd_args(args_string: str) -> Dict[str, Any]:
        """
        Parses a command-line arguments string into a dictionary, handling 
        standard key=value and complex click override syntax (e.g., --param=X=Y,A=B).
        """
        parsed_args: Dict[str, Any] = {}
        # Simple regex to find arguments: --key=value or --key value
        matches = re.findall(r'--(\w+)=?([^\s,]+(?:,[^\s,]+)*)?', args_string)
        
        for key, value in matches:
            # Clean up the key name for internal lookup
            wb_name = key.replace('-', '_')

            # Handle the erelela_override type, which contains key=value,key=value strings
            if 'override' in wb_name.lower() or wb_name in ['erelela_config']:
                override_args = {}
                # Split by comma for nested arguments (e.g., ELA_key=value,...)
                for item in value.split(','):
                    if '=' in item:
                        nested_key, nested_val = item.split('=', 1)
                        # We only care about the nested keys like ELA_... or RG_...
                        if nested_key.startswith(('ELA_', 'RG_', 'THER_', 'MiniWorld_')):
                            override_args[nested_key] = nested_val
                
                # If we found nested args, store them under the override key
                if override_args:
                    parsed_args[wb_name] = override_args
            
            # Handle all other arguments
            else:
                # Attempt to convert to basic types
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                try:
                    # Treat as a number if possible, but keep original if conversion fails
                    value = float(value) if '.' in value else int(value)
                except ValueError:
                    pass
                
                parsed_args[wb_name] = value

        return parsed_args

    def _get_config_value(self, run_config: Dict[str, Any], run_summary: Dict[str, Any], wb_name: str) -> Optional[Any]:
        """
        Extracts a config value using three fallbacks: 1) Top-level config, 
        2) Nested 'erelela_override', and 3) Command-line arguments.
        """
        value = None
        
        # 1. Try Top-level config
        if wb_name in run_config:
            value_obj = run_config[wb_name]
            if isinstance(value_obj, dict) and 'value' in value_obj:
                 value = value_obj['value']
            # Handles raw values or lists/dicts not wrapped in {'value': X}
            elif not isinstance(value_obj, list): 
                value = value_obj
        
        # 2. Fallback to Nested EReLELA Overrides
        is_erelela_key = wb_name.startswith(('ELA_', 'RG_', 'THER_', 'MiniWorld_'))
        if value is None and is_erelela_key:
            nested_config_key = 'erelela_override'
            if nested_config_key in run_config:
                nested_entry = run_config[nested_config_key]
                
                if isinstance(nested_entry, dict) and 'value' in nested_entry:
                    nested_value = nested_entry['value']
                else:
                    nested_value = nested_entry

                nested_dict = {}
                if isinstance(nested_value, dict):
                    nested_dict = nested_value
                elif isinstance(nested_value, str):
                    try:
                        nested_dict = json.loads(nested_value)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if wb_name in nested_dict:
                    value = nested_dict[wb_name]
        
        # 3. Fallback to Command-Line Arguments
        if value is None and 'args' in run_summary and run_summary['args']:
            # The 'args' summary key contains the raw command line string
            args_string = run_summary['args']
            
            # The key might be a direct command-line argument (e.g., --gamma=0.99)
            if wb_name in args_string:
                parsed_args = self._parse_cmd_args(args_string)
                
                if wb_name in parsed_args:
                    value = parsed_args[wb_name]
                
                # Check for nested EReLELA arguments inside the override
                elif is_erelela_key and 'erelela_override' in parsed_args and isinstance(parsed_args['erelela_override'], dict):
                    if wb_name in parsed_args['erelela_override']:
                        value = parsed_args['erelela_override'][wb_name]
            
        if value is None:
            return None
        
        # Final formatting cleanup
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (list, dict)):
            try:
                # Convert complex objects to a stable JSON string representation
                return json.dumps(value, sort_keys=True, indent=None, separators=(',', ':'))
            except TypeError:
                return str(value)
        
        return value


    def _extract_hyperparameter_values(
        self, 
        runs_map: Dict[str, wandb.apis.public.Run], 
        group_config: Dict[str, Any],
        common_hyperparams: Dict[str, str]
    ) -> Dict[str, List[Any]]:
        """
        Extracts unique values for all specified hyperparameters for a group's runs.
        """
        run_ids = group_config['run_ids']
        if isinstance(run_ids, str):
            run_ids = [run_ids]
        
        normalized_run_ids = {self._normalize_run_id(rid) for rid in run_ids}
        group_hyperparams = group_config.get('hyperparams', {})
        
        hyperparams_map = common_hyperparams.copy()
        hyperparams_map.update(group_hyperparams)
        
        extracted_values: Dict[str, Set[Any]] = {label: set() for label in hyperparams_map.values()}
        
        for run_id in normalized_run_ids:
            run = runs_map.get(run_id)
            if not run:
                continue
            
            run_config = run.config
            # W&B run object stores summary data (including 'args') in the run.summary attribute
            run_summary = run.summary
            
            for wb_name, output_label in hyperparams_map.items():
                value = self._get_config_value(run_config, run_summary, wb_name)
                
                if value is not None:
                    extracted_values[output_label].add(value)

        # Convert sets to sorted lists for final output
        final_values = {}
        for label, values in extracted_values.items():
            list_values = list(values)
            
            def try_numeric_sort(val):
                if isinstance(val, (int, float)):
                    return val
                try:
                    if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                        return str(val)
                    return float(val)
                except (ValueError, TypeError, AttributeError):
                    return str(val)
            
            list_values.sort(key=try_numeric_sort)
            final_values[label] = list_values

        return final_values


    def _discover_hyperparams_from_runs(
        self, runs_map: Dict[str, wandb.apis.public.Run]
    ) -> Dict[str, str]:
        """
        Auto-discover hyperparameters from W&B run configs when none are
        specified in the YAML configuration.  Collects the union of all
        top-level config keys (excluding W&B internal keys prefixed with '_')
        and returns a {key: key} identity mapping.
        """
        all_keys: Set[str] = set()
        for run in runs_map.values():
            for key in run.config.keys():
                if not key.startswith('_'):
                    # Escape pipe characters for markdown table safety
                    safe_key = key.replace('|', '\\|')
                    all_keys.add(safe_key)

        print(f"  Auto-discovered {len(all_keys)} hyperparameters from run configs.")
        return {k: k for k in sorted(all_keys)}

    def generate_hyperparameter_table(self, argparse_help: Optional[Dict[str, str]] = None) -> str:
        """
        Generates a markdown table of hyperparameters for all groups based on config definitions.
        """
        if not self.config.get('figures'):
            return "Configuration file has no 'figures' section defined to determine groups."

        print("\n--- Starting Hyperparameter Table Generation ---")
        
        runs_map = self._collect_all_runs_mapped()

        if not runs_map:
            return "No W&B runs found or fetched."

        common_hyperparams = self.config.get('hyperparams', {})
        all_labels: Set[str] = set(common_hyperparams.values())

        group_results: Dict[str, Dict[str, List[Any]]] = {}
        group_names: List[str] = []

        # Check if any group defines its own hyperparams
        any_group_hyperparams = False
        for figure_config in self.config.get('figures', []):
            for group_config in figure_config.get('groups', []):
                if group_config.get('hyperparams', {}):
                    any_group_hyperparams = True
                    break

        # Auto-discover if no hyperparams defined anywhere
        if not common_hyperparams and not any_group_hyperparams:
            print("  No hyperparams defined in YAML — auto-discovering from W&B run configs...")
            common_hyperparams = self._discover_hyperparams_from_runs(runs_map)
            all_labels = set(common_hyperparams.values())

        for figure_config in self.config.get('figures', []):
            for group_config in figure_config.get('groups', []):
                group_name = group_config['name']

                original_group_name = group_name
                suffix = 1
                while group_name in group_names:
                    group_name = f"{original_group_name} ({suffix})"
                    suffix += 1

                group_names.append(group_name)

                group_hyperparams = group_config.get('hyperparams', {})
                all_labels.update(group_hyperparams.values())

                extracted_values = self._extract_hyperparameter_values(
                    runs_map, group_config, common_hyperparams
                )

                group_results[group_name] = extracted_values

        if not all_labels:
            return "No hyperparameters found (neither in YAML config nor in W&B run configs)."

        sorted_labels = sorted(list(all_labels))

        # 4. Construct the markdown table
        has_descriptions = argparse_help is not None
        if has_descriptions:
            header = "| Hyperparameter | Description | " + " | ".join(group_names) + " |"
            separator = "|---|---|-" + "-|" * len(group_names)
        else:
            header = "| Hyperparameter | " + " | ".join(group_names) + " |"
            separator = "|---|-" + "-|" * len(group_names)

        table_rows: List[str] = [header, separator]

        for label in sorted_labels:
            row_content: List[str] = [label]

            if has_descriptions:
                desc = argparse_help.get(label, '')
                # Escape pipe chars in description for markdown safety
                desc = desc.replace('|', '\\|')
                row_content.append(desc)

            for group_name in group_names:
                values = group_results.get(group_name, {}).get(label, [])

                if values:
                    str_values = []
                    for v in values:
                        if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                            str_values.append(v)
                        elif isinstance(v, str) and (' ' in v or v.startswith('|')):
                            str_values.append(f'"{v}"')
                        else:
                            str_values.append(str(v))

                    cell_content = f"[{', '.join(str_values)}]"
                else:
                    cell_content = "[]"

                row_content.append(cell_content)

            table_rows.append("| " + " | ".join(row_content) + " |")

        return "\n".join(table_rows)

def main():
    """Main function to parse arguments and run the generator."""
    parser = argparse.ArgumentParser(description="W&B Hyperparameter Table Generator")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument('--output', '-o', type=str, required=False, help="Optional path to save the markdown table file")
    parser.add_argument('--script', '-s', type=str, help="Python script path to extract argparse help strings from")

    args = parser.parse_args()

    try:
        generator = HyperparamTableGenerator(args.config)

        argparse_help = None
        if args.script:
            print(f"Extracting argparse help strings from: {args.script}")
            argparse_help = HyperparamTableGenerator._extract_argparse_help(args.script)
            print(f"  Found {len(argparse_help)} argument descriptions.")

        table = generator.generate_hyperparameter_table(argparse_help=argparse_help)
        
        print("\n\n#####################################################")
        print("### Hyperparameter Table (Markdown Format) ###")
        print("#####################################################")
        print(table)
        print("#####################################################\n")
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(table)
            print(f"Successfully saved markdown table to: {output_path}")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Add a note on installation if not run inside a proper environment
    try:
        main()
    except NameError:
        print("\nNote: This script requires the 'wandb' and 'pyyaml' libraries.")
        print("Please install them using: `pip install wandb pyyaml`")
        print("Then run the script again.")

