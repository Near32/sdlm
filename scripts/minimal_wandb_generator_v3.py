#!/usr/bin/env python3
"""
Minimal W&B Figure Generator
Focuses on run ID specification and synchronization with local caching.
"""
from typing import Tuple, Dict, Set, Optional, Union
import argparse
import yaml
import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Any
import warnings
import pickle
import hashlib
import json
from tqdm import tqdm
import time
from colorspacious import cspace_convert
from matplotlib.colors import to_hex, to_rgb
warnings.filterwarnings('ignore')


class MinimalWandBGenerator:
    """Minimal W&B figure generator with run ID support, synchronization, and caching."""
    
    def __init__(self, config_path: str, cache_dir: str = None, use_cache: bool = True):
        """Initialize with configuration and caching options."""
        self.config = self._load_config(config_path)
        self.api = wandb.Api(timeout=600)
        #self.api = wandb.Api()
        self.use_cache = use_cache
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = self.config.get('cache_dir', 'wandb_cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cache enabled: {self.use_cache}")
        
        # Set style
        plt.style.use(self.config.get('plot_style', 'seaborn-v0_8'))
        if 'color_palette' in self.config:
            print(f"Setting Seaborn color palette: {self.config['color_palette']}")
            sns.set_palette(self.config['color_palette'])
   
        self._setup_font_sizes()

        # Set up default figure size
        self.default_figsize = self._setup_default_figsize()

        # Set up color palette
        self.color_palette = self._setup_color_palette()

        # Configure Kolmogorov-Smirnov test settings
        anchors = self.config.get('Kolmogorov-Smirnov-anchors', [])
        if isinstance(anchors, str):
            anchors = [anchors]
        self.ks_anchors = list(anchors)

        ks_kwargs = self.config.get('Kolmogorov-Smirnov-kwargs', {}) or {}
        if isinstance(ks_kwargs, dict):
            self.ks_kwargs = ks_kwargs
        else:
            print("Warning: Kolmogorov-Smirnov-kwargs must be a mapping; ignoring provided value.")
            self.ks_kwargs = {}

        # Track which runs we've already printed history keys for
        self._history_keys_printed: Set[str] = set()
        
    def _setup_font_sizes(self):
        """Set up font sizes from config."""
        font_config = self.config.get('font_sizes', {})
        
        if isinstance(font_config, (int, float)):
            # Single font size for everything
            plt.rcParams.update({
                'font.size': font_config,
                'axes.titlesize': font_config,
                'axes.labelsize': font_config,
                'xtick.labelsize': font_config,
                'ytick.labelsize': font_config,
                'legend.fontsize': font_config
            })
            print(f"Using font size: {font_config}")
            
        elif isinstance(font_config, dict):
            # Specific font sizes for different elements
            font_mapping = {
                'default': 'font.size',
                'title': 'axes.titlesize',
                'labels': 'axes.labelsize',
                'ticks': 'xtick.labelsize',
                'legend': 'legend.fontsize'
            }
            
            rcparams_update = {}
            for config_key, rcparam_key in font_mapping.items():
                if config_key in font_config:
                    rcparams_update[rcparam_key] = font_config[config_key]
                    if config_key == 'ticks':
                        # Set both x and y tick sizes
                        rcparams_update['ytick.labelsize'] = font_config[config_key]
            
            if rcparams_update:
                plt.rcParams.update(rcparams_update)
                print(f"Using custom font sizes: {font_config}")
        
        else:
            print("Using default font sizes")

    def _parse_figsize(
        self,
        figsize_config: Optional[Union[List, Tuple, Dict]],
        default: Optional[Tuple[float, float]] = None,
    ) -> Tuple[float, float]:
        """Parse figsize from various config formats.

        Supports multiple formats:
        - List/Tuple: [10, 6] or (10, 6)
        - Dict with width/height: {width: 10, height: 6}
        - Dict with width/aspect_ratio: {width: 10, aspect_ratio: 1.67}

        Args:
            figsize_config: The figsize configuration value.
            default: Default figsize to use if config is invalid/None.
                     If None, uses self.default_figsize if available, else (10, 6).

        Returns:
            Tuple of (width, height) for figure size.
        """
        # Determine the fallback default
        if default is None:
            default = getattr(self, 'default_figsize', (10, 6))

        if figsize_config is None:
            return default

        if isinstance(figsize_config, (list, tuple)) and len(figsize_config) == 2:
            return (float(figsize_config[0]), float(figsize_config[1]))

        if isinstance(figsize_config, dict):
            width = float(figsize_config.get('width', 10))
            if 'height' in figsize_config:
                height = float(figsize_config['height'])
            elif 'aspect_ratio' in figsize_config:
                height = width / float(figsize_config['aspect_ratio'])
            else:
                height = default[1] if default else 6
            return (width, height)

        return default

    def _setup_default_figsize(self) -> Tuple[float, float]:
        """Set up default figure size from config.

        Returns:
            Tuple of (width, height) for figure size.
        """
        figsize_config = self.config.get('figsize', None)
        # Use hardcoded default since self.default_figsize isn't set yet
        result = self._parse_figsize(figsize_config, default=(10, 6))

        if figsize_config is not None:
            print(f"Using global figure size: {result}")

        return result

    def _setup_color_palette(self):
        """Set up color palette from config."""
        color_config = self.config.get('color_palette', 'Set1')
        
        if isinstance(color_config, str):
            # Named palette (e.g., 'Set1', 'husl', 'viridis')
            try:
                if color_config in ['Set1', 'Set2', 'Set3', 'Paired', 'Dark2', 'Accent', 'Pastel1', 'Pastel2']:
                    # Qualitative palettes - get discrete colors
                    colors = sns.color_palette(color_config, n_colors=10)
                else:
                    # Other palettes
                    colors = sns.color_palette(color_config, n_colors=10)
                
                print(f"Using color palette: {color_config}")
                return colors
                
            except Exception as e:
                print(f"Warning: Could not load color palette '{color_config}': {e}")
                return sns.color_palette("Set1", n_colors=10)
                
        elif isinstance(color_config, list):
            # Custom color list
            print(f"Using custom colors: {color_config}")
            return color_config
            
        else:
            print("Warning: Invalid color palette config, using default 'Set1'")
            return sns.color_palette("Set1", n_colors=10)

    def _apply_lightness(self, color, lightness: float) -> str:
        """Adjust a color's lightness in CIELab space.

        Args:
            color: Hex string (e.g., "#ff7f0e") or RGB tuple (e.g., (1.0, 0.5, 0.06))
            lightness: Target L* value in CIELab (0-100)

        Returns:
            Hex color string with adjusted lightness
        """
        # Convert to RGB array [0, 1]
        if isinstance(color, str):
            rgb = np.array(to_rgb(color))
        else:
            rgb = np.array(color)

        # Convert to CIELab
        lab = cspace_convert(rgb, "sRGB1", "CIELab")

        # Set the lightness component
        lab[0] = lightness

        # Convert back to sRGB
        rgb_out = cspace_convert(lab, "CIELab", "sRGB1")
        rgb_out = np.clip(rgb_out, 0, 1)

        return to_hex(rgb_out)

    def _get_legend_export_config(self, figure_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get legend export configuration with defaults merged from global and figure-level."""
        global_config = self.config.get('legend_export', {})
        figure_config_override = figure_config.get('legend_export', {})

        defaults = {
            'enabled': False,
            'format': 'pdf',
            'fontsize': 12,
            'figsize': [8, 1.5],
            'ncol': 4,
            'loc': 'center',
            'frameon': False,
            'remove_from_main': False,
            'hide_xlabel': False,
            'hide_xticklabels': False,
            'hide_ylabel': False,
            'hide_yticklabels': False,
            'hide_title': False,
        }

        # Merge: defaults < global < figure-level
        result = {**defaults, **global_config, **figure_config_override}
        return result

    def _export_legend_to_file(
        self,
        handles: List,
        labels: List[str],
        output_path: Path,
        config: Dict[str, Any],
    ) -> None:
        """Export legend as a separate figure file.

        Args:
            handles: Legend handles from the main plot
            labels: Legend labels from the main plot
            output_path: Path to save the legend figure
            config: Legend export configuration dict
        """
        #figsize = config.get('figsize', [8, 1.5])
        #if isinstance(figsize, list):
        #    figsize = tuple(figsize)
        figsize = self._parse_figsize(config.get('figsize'))

        fontsize = config.get('fontsize', 12)
        ncol = config.get('ncol', 4)
        loc = config.get('loc', 'center')
        frameon = config.get('frameon', False)

        # Create a new figure for the legend
        fig = plt.figure(figsize=figsize)

        # Use ncol=0 as signal for auto (default matplotlib behavior)
        legend_kwargs = {
            'handles': handles,
            'labels': labels,
            'loc': loc,
            'frameon': frameon,
            'fontsize': fontsize,
        }
        if ncol > 0:
            legend_kwargs['ncol'] = ncol

        fig.legend(**legend_kwargs)

        # Remove axes
        fig.gca().set_axis_off()

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the figure
        fig.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved legend: {output_path}")

    def _apply_legend_export_modifications(
        self,
        fig: plt.Figure,
        config: Dict[str, Any],
    ) -> None:
        """Apply legend export modifications to a figure (remove legend, hide labels/title).

        Args:
            fig: The matplotlib figure to modify
            config: Legend export configuration dict
        """
        ax = fig.gca()

        # Remove legend from main figure if requested
        if config.get('remove_from_main', False):
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        # Hide axis labels if requested
        if config.get('hide_xlabel', False):
            ax.set_xlabel('')
        if config.get('hide_xticklabels', False):
            ax.tick_params(axis='x', labelbottom=False)

        if config.get('hide_ylabel', False):
            ax.set_ylabel('')
        if config.get('hide_yticklabels', False):
            ax.tick_params(axis='y', labelleft=False)

        # Hide title if requested
        if config.get('hide_title', False):
            ax.set_title('')

    def _apply_plot_label_visibility(self, fig: plt.Figure, plot_config: Dict[str, Any]) -> None:
        """Apply per-plot label/title visibility settings."""
        ax = fig.gca()
        if plot_config.get('hide_xlabel', False):
            ax.set_xlabel('')
        if plot_config.get('hide_xticklabels', False):
            ax.tick_params(axis='x', labelbottom=False)
        if plot_config.get('hide_ylabel', False):
            ax.set_ylabel('')
        if plot_config.get('hide_yticklabels', False):
            ax.tick_params(axis='y', labelleft=False)
        if plot_config.get('hide_title', False):
            ax.set_title('')

    def _collect_all_required_metrics(self) -> Tuple[Set[str], Set[str]]:
        """
        Collect all unique metrics and step columns needed across all figures and groups.
        Returns (all_metrics, all_step_columns)
        """
        all_metrics = set()
        #all_step_columns = set() 
        all_step_columns = {'_step'}

        for figure_config in self.config['figures']:
            # Add global figure metrics
            figure_metrics = set(figure_config['metrics'])
            all_metrics.update(figure_metrics)

            # Add step column
            all_step_columns.add(figure_config.get('step_column', '_step'))

            # Collect figure-level filter columns
            figure_filter = figure_config.get('data_filter', '')
            if figure_filter:
                filter_cols = self._extract_columns_from_filter(figure_filter)
                all_metrics.update(filter_cols)

            # Add group-specific metric aliases
            for group_config in figure_config['groups']:
                if 'metric_aliases' in group_config:
                    # Add the actual metric names used by this group
                    group_specific_metrics = set(group_config['metric_aliases'].values())
                    all_metrics.update(group_specific_metrics)
                else:
                    # No aliases, so group uses the same metrics as figure
                    all_metrics.update(figure_metrics)

                # Collect group-specific filter columns
                group_filter = group_config.get('data_filter', '')
                if group_filter:
                    filter_cols = self._extract_columns_from_filter(group_filter)
                    all_metrics.update(filter_cols)

                # Add recovery keys if epoch recovery is enabled
                recover_epoch_by_rank = group_config.get(
                    'recover_epoch_by_rank',
                    figure_config.get('recover_epoch_by_rank', False)
                )
                if recover_epoch_by_rank:
                    recovery_keys = group_config.get(
                        'recover_epoch_target_keys',
                        figure_config.get('recover_epoch_target_keys', None)
                    )
                    if recovery_keys is None:
                        recovery_keys = ['target_text', 'target_tokens_str', 'target_tokens']
                    elif isinstance(recovery_keys, str):
                        recovery_keys = [recovery_keys]
                    all_metrics.update([k for k in recovery_keys if k])

        return all_metrics, all_step_columns
    
    @staticmethod
    def _normalize_run_id(run_id: Optional[str]) -> str:
        """Return the bare run ID regardless of whether a full path was provided."""
        if not run_id:
            return ''
        # Handle substution from duplicated runs with different metric alias:
        eff_run_id = run_id
        #if ':' in run_id:
        #    # Extract the actual run ID after the colon
        #    eff_run_id = run_id.split(':')[-1]
        # Handle inputs like "entity/project/run_id" while leaving simple IDs untouched.
        return eff_run_id.split('/')[-1]
    
    def _apply_metric_aliases(self, data: pd.DataFrame, group_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply metric aliases to rename columns for a specific group."""
        if 'metric_aliases' not in group_config:
            return data
        
        aliases = group_config['metric_aliases']
        data_copy = data.copy()
        
        print(f"    Applying metric aliases for group '{group_config['name']}':")
        for global_metric, group_metric in aliases.items():
            if group_metric in data_copy.columns:
                if global_metric in data_copy.columns:
                    data_copy = data_copy.drop(columns=[global_metric])
                data_copy[global_metric] = data_copy[group_metric]
                print(f"      {group_metric} -> {global_metric}")
                # Optionally remove the original column to avoid confusion
                if global_metric != group_metric:
                    data_copy = data_copy.drop(columns=[group_metric])
            else:
                print(f"      Warning: Group metric '{group_metric}' not found in data for global metric '{global_metric}'")
        
        return data_copy

    def _parse_filter_string(self, filter_string: str) -> List[Tuple[str, str, Any]]:
        """Parse a filter string like 'col1==val1 & col2>val2' into conditions.

        Returns list of (column, operator, value) tuples.
        Supported operators: ==, !=, >, <, >=, <=
        """
        import re
        conditions = []
        # Split on '&' (with optional whitespace)
        parts = re.split(r'\s*&\s*', filter_string.strip())

        for part in parts:
            # Match: column_name operator value
            match = re.match(r'(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)', part.strip())
            if match:
                col, op, val = match.groups()
                # Try to convert value to appropriate type
                val = val.strip().strip('"').strip("'")
                try:
                    val = float(val)
                    if val.is_integer():
                        val = int(val)
                except ValueError:
                    pass  # Keep as string
                conditions.append((col, op, val))
            else:
                # Warn about unparseable parts and launch debugger
                if part.strip():
                    print(f"    WARNING: Could not parse filter condition: '{part.strip()}'")
                    print(f"    Expected format: 'column==value' or 'column>value' etc.")
                    print(f"    Supported operators: ==, !=, >, <, >=, <=")
                    print(f"    Launching debugger for investigation...")
                    import ipdb; ipdb.set_trace()

        return conditions

    def _apply_filter_conditions(self, data: pd.DataFrame, conditions: List[Tuple[str, str, Any]]) -> Tuple[pd.DataFrame, List[str]]:
        """Apply parsed filter conditions to a DataFrame.

        Returns:
            Tuple of (filtered_data, warnings_list)
        """
        filter_warnings = []

        if not conditions or data.empty:
            return data, filter_warnings

        mask = pd.Series([True] * len(data), index=data.index)

        for col, op, val in conditions:
            if col not in data.columns:
                msg = f"Filter column '{col}' not found in data, condition skipped"
                print(f"    WARNING: {msg}")
                print(f"    Available columns: {list(data.columns)}")
                print(f"    Launching debugger for investigation...")
                import ipdb; ipdb.set_trace()
                filter_warnings.append(msg)
                continue

            col_data = pd.to_numeric(data[col], errors='coerce')

            if op == '==':
                mask &= (col_data == val)
            elif op == '!=':
                mask &= (col_data != val)
            elif op == '>':
                mask &= (col_data > val)
            elif op == '<':
                mask &= (col_data < val)
            elif op == '>=':
                mask &= (col_data >= val)
            elif op == '<=':
                mask &= (col_data <= val)

        filtered = data[mask]
        rows_before = len(data)
        rows_after = len(filtered)

        # Check for problematic outcomes
        if rows_after == 0:
            msg = f"Filter resulted in 0 rows! Filter may be too restrictive or values don't exist"
            print(f"    ERROR: {msg}")
            print(f"    Filter conditions: {conditions}")
            print(f"    Launching debugger for investigation...")
            import ipdb; ipdb.set_trace()
            filter_warnings.append(msg)
        elif rows_after == rows_before:
            msg = f"Filter had no effect (all {rows_before} rows matched)"
            print(f"    INFO: {msg}")
            print(f"    Filter conditions: {conditions}")
            print(f"    Launching debugger for investigation...")
            import ipdb; ipdb.set_trace()
            filter_warnings.append(msg)
        else:
            print(f"    Filter applied: {rows_before} -> {rows_after} rows")

        return filtered, filter_warnings

    def _apply_loose_filter(
        self,
        data: pd.DataFrame,
        conditions: List[Tuple[str, str, Any]],
        step_column: str = '_step',
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply filter with per-step fallback to closest values.

        For each step value, if the exact filter match (e.g., epoch==100) has no data,
        find the closest filter value that does have data for that step.

        Args:
            data: DataFrame to filter
            conditions: List of (column, operator, value) tuples from _parse_filter_string
            step_column: Column name to use as the step/x-axis

        Returns:
            Tuple of (filtered_data, fallback_info)
            fallback_info contains which steps used fallback values
        """
        if not conditions or data.empty:
            return data, {'steps_with_fallback': {}}

        # 1. Identify equality conditions (can fallback) vs other conditions (strict)
        equality_conditions = [(col, val) for col, op, val in conditions if op == '==']
        other_conditions = [(col, op, val) for col, op, val in conditions if op != '==']

        # 2. Apply non-equality conditions first (these are strict, no fallback)
        if other_conditions:
            filtered, _ = self._apply_filter_conditions(data, other_conditions)
        else:
            filtered = data.copy()

        if filtered.empty:
            return filtered, {'steps_with_fallback': {}}

        # 3. For each step value that has data, apply loose filtering for equality conditions
        if step_column not in filtered.columns:
            print(f"    WARNING: Step column '{step_column}' not in data for loose filtering")
            # Fall back to strict filtering
            return self._apply_filter_conditions(data, conditions)

        all_steps = filtered[step_column].dropna().unique()
        result_rows = []
        fallback_info = {'steps_with_fallback': {}}

        for step in all_steps:
            step_data = filtered[filtered[step_column] == step]
            current_data = step_data

            step_fallbacks = {}
            for col, target_val in equality_conditions:
                if col not in current_data.columns:
                    continue

                col_data = pd.to_numeric(current_data[col], errors='coerce')

                # Try exact match first
                exact_match = current_data[col_data == target_val]

                if not exact_match.empty:
                    current_data = exact_match
                else:
                    # Find closest value that has data for this step
                    available_vals = col_data.dropna().unique()
                    if len(available_vals) > 0:
                        # Find closest value to target
                        closest_val = min(available_vals, key=lambda v: abs(v - target_val))
                        current_data = current_data[col_data == closest_val]
                        step_fallbacks[col] = {'requested': target_val, 'used': closest_val}

            if not current_data.empty:
                result_rows.append(current_data)
                if step_fallbacks:
                    fallback_info['steps_with_fallback'][step] = step_fallbacks

        if result_rows:
            result = pd.concat(result_rows, ignore_index=True)
        else:
            # Preserve schema to avoid downstream KeyErrors.
            result = data.iloc[0:0].copy()

        # Log summary of fallbacks
        n_fallback_steps = len(fallback_info['steps_with_fallback'])
        if n_fallback_steps > 0:
            print(f"    Loose filtering used fallback for {n_fallback_steps} steps")
            # Show details for first few steps
            shown = 0
            for step, fb_info in sorted(fallback_info['steps_with_fallback'].items()):
                if shown >= 5:
                    remaining = n_fallback_steps - shown
                    print(f"      ... and {remaining} more steps")
                    break
                parts = [f"{col} {info['requested']}->{info['used']}" for col, info in fb_info.items()]
                print(f"      step {step}: {', '.join(parts)}")
                shown += 1

        rows_before = len(data)
        rows_after = len(result)
        print(f"    Loose filter applied: {rows_before} -> {rows_after} rows")

        return result, fallback_info

    def _extract_columns_from_filter(self, filter_string: str) -> Set[str]:
        """Extract column names from a filter string."""
        import re
        columns = set()
        parts = re.split(r'\s*&\s*', filter_string.strip())
        for part in parts:
            match = re.match(r'(\w+)\s*(==|!=|>=|<=|>|<)', part.strip())
            if match:
                columns.add(match.group(1))
        return columns

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _generate_cache_key(self, run_id: str, metrics: List[str], step_columns: List[str]) -> str:
        """Generate a unique cache key for a run and its requested data."""
        # Create a deterministic hash based on run_id and requested columns
        data_spec = {
            'run_id': run_id,
            'metrics': sorted(metrics),
            'step_columns': sorted(step_columns)
        }
        data_string = json.dumps(data_spec, sort_keys=True)
        return hashlib.md5(data_string.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"run_data_{cache_key}.pkl"
    
    def _save_run_data_to_cache(self, run_id: str, metrics: List[str], 
                               step_columns: List[str], data: pd.DataFrame):
        """Save run data to cache."""
        if not self.use_cache:
            return
        
        cache_key = self._generate_cache_key(run_id, metrics, step_columns)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            cache_data = {
                'run_id': run_id,
                'metrics': metrics,
                'step_columns': step_columns,
                'data': data,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            print(f"    Cached data for run {run_id} ({len(data)} rows)")
        except Exception as e:
            print(f"    Warning: Could not save cache for run {run_id}: {e}")
    
    def _load_run_data_from_cache(self, run_id: str, metrics: List[str], 
                                 step_columns: List[str]) -> pd.DataFrame:
        """Load run data from cache if available."""
        if not self.use_cache:
            return pd.DataFrame()
        
        cache_key = self._generate_cache_key(run_id, metrics, step_columns)
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return pd.DataFrame()
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify the cached data matches our request
            if (cache_data['run_id'] == run_id and 
                set(cache_data['metrics']) >= set(metrics) and
                set(cache_data['step_columns']) >= set(step_columns)):
                
                # Extract only the columns we need
                needed_columns = ['run_id', 'group'] + metrics + step_columns
                available_columns = [col for col in needed_columns if col in cache_data['data'].columns]
                
                print(f"    Loaded from cache: run {run_id} ({len(cache_data['data'])} rows)")
                return cache_data['data'][available_columns].copy()
            
        except Exception as e:
            print(f"    Warning: Could not load cache for run {run_id}: {e}")
        
        return pd.DataFrame()
    
    def _clear_cache(self):
        """Clear all cached data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            print(f"Cleared cache directory: {self.cache_dir}")
    
    def _list_cache_contents(self):
        """List contents of cache directory."""
        cache_files = list(self.cache_dir.glob("run_data_*.pkl"))
        print(f"\nCache directory: {self.cache_dir}")
        print(f"Cached files: {len(cache_files)}")
        
        total_size = 0
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                size = cache_file.stat().st_size
                total_size += size
                
                print(f"  {cache_file.name}")
                print(f"    Run ID: {cache_data['run_id']}")
                print(f"    Metrics: {cache_data['metrics']}")
                print(f"    Rows: {len(cache_data['data'])}")
                print(f"    Size: {size / 1024:.1f} KB")
                print(f"    Cached: {cache_data['timestamp']}")
                print()
            except Exception as e:
                print(f"  {cache_file.name}: Error reading cache file: {e}")
        
    def _get_all_projects(self) -> List[str]:
        """Get list of all projects from config."""
        projects = self.config.get('projects', self.config.get('project'))
        if isinstance(projects, str):
            projects = [projects]
        return projects
        """Get list of all projects from config."""
        projects = self.config.get('projects', self.config.get('project'))
        if isinstance(projects, str):
            projects = [projects]
        return projects
    
    def _fetch_specific_runs(self, run_ids: List[str], projects: List[str] = []) -> List[wandb.apis.public.Run]:
        """Fetch specific runs by their IDs across all projects."""
        if len(projects):
            all_projects = projects
        else:
            all_projects = self._get_all_projects()
        found_runs = {}
        
        print(f"Fetching {len(run_ids)} specific run IDs...")
        
        '''
        for project in all_projects:
            try:
                project_runs = self.api.runs(project)
                for run in project_runs:
                    if run.id in run_ids:
                        found_runs.append(run)
                        print(f"  Found run {run.id} in project {project}")
                        run_ids.remove(run.id)
                        if not run_ids:
                            break
                if not run_ids:
                    break
            except Exception as e:
                print(f"Warning: Could not access project {project}: {e}")
                continue
        '''
        assert len(projects) >= 1
        for run_id in run_ids:
            eff_run_id = run_id
            if ':' in run_id:
                # If it contains a colon, it is a purposeful duplication -treat it as a path 
                eff_run_id = run_id.split(':')[-1]
            
            # Check if eff_run_id already contains a full path (e.g., entity/project/run_id)
            if len(eff_run_id.split('/')) > 2:
                # If it's a full path, use it directly
                run_path = eff_run_id
            else:
                # Otherwise, use the group's default project path
                run_path = f"{projects[0]}/{eff_run_id}"

            print(f"    Fetching run: {run_path}")
            run = self.api.run(run_path)
            found_runs[run_id] = run

        #if run_ids:
        #    print(f"Warning: Could not find runs: {run_ids}")
        
        return found_runs
    
    def _get_metrics_for_run(self, run_id: str) -> List[str]:
        """Determine which specific metrics to fetch for a given run based on its group aliases.

        Also includes filter columns from both figure-level and group-level data_filter.
        """
        for figure_config in self.config['figures']:
            for group_config in figure_config['groups']:
                run_ids = group_config['run_ids']
                if isinstance(run_ids, str):
                    run_ids = [run_ids]

                normalized_run_ids = {self._normalize_run_id(rid) for rid in run_ids}
                if self._normalize_run_id(run_id) in normalized_run_ids:
                    # Start with base metrics
                    if 'metric_aliases' in group_config:
                        # Use group-specific metric names (values in aliases)
                        metrics = set(group_config['metric_aliases'].values())
                    else:
                        # Use global metric names
                        metrics = set(figure_config['metrics'])

                    # Add filter columns from figure-level data_filter
                    figure_filter = figure_config.get('data_filter', '')
                    if figure_filter:
                        filter_cols = self._extract_columns_from_filter(figure_filter)
                        metrics.update(filter_cols)

                    # Add filter columns from group-level data_filter
                    group_filter = group_config.get('data_filter', '')
                    if group_filter:
                        filter_cols = self._extract_columns_from_filter(group_filter)
                        metrics.update(filter_cols)

                    # Add recovery keys if epoch recovery is enabled
                    recover_epoch_by_rank = group_config.get(
                        'recover_epoch_by_rank',
                        figure_config.get('recover_epoch_by_rank', False)
                    )
                    if recover_epoch_by_rank:
                        recovery_keys = group_config.get(
                            'recover_epoch_target_keys',
                            figure_config.get('recover_epoch_target_keys', None)
                        )
                        if recovery_keys is None:
                            recovery_keys = ['target_text', 'target_tokens_str', 'target_tokens']
                        elif isinstance(recovery_keys, str):
                            recovery_keys = [recovery_keys]
                        metrics.update([k for k in recovery_keys if k])

                    return list(metrics)

        # Fallback: return all global metrics if run not found in any group
        all_global_metrics = set()
        for figure_config in self.config['figures']:
            all_global_metrics.update(figure_config['metrics'])
            # Also include filter columns from figure-level filters
            figure_filter = figure_config.get('data_filter', '')
            if figure_filter:
                filter_cols = self._extract_columns_from_filter(figure_filter)
                all_global_metrics.update(filter_cols)
            # And from group-level filters
            for group_config in figure_config['groups']:
                group_filter = group_config.get('data_filter', '')
                if group_filter:
                    filter_cols = self._extract_columns_from_filter(group_filter)
                    all_global_metrics.update(filter_cols)
        return list(all_global_metrics)
    
    def _extract_run_data(
        self,
        run: wandb.apis.public.Run,
        metrics: List[str],
        step_columns: List[str],
        max_retry: int=2,
        eff_run_id: str = None,
    ) -> pd.DataFrame:
        """Extract metrics with fallback strategy for async logging."""
        # Get metrics specific to this run's group (includes relevant filter columns)
        run_specific_metrics = self._get_metrics_for_run(run.id)
        self._print_history_keys(run)

        # Try cache first
        cached_data = self._load_run_data_from_cache(run.id, run_specific_metrics, step_columns)
        #if 'dzo' in run.id or 'daa' in run.id:
        #    import ipdb; ipdb.set_trace()
        if not cached_data.empty:
            return cached_data

        print(f"    Extracting from W&B: run {run.id}")
        columns_to_extract = list(set(run_specific_metrics + step_columns))
        if '_step' not in columns_to_extract:
            columns_to_extract.append('_step')

        # Try unified extraction first (faster)
        #if "fw048qur" in run.id:
        #    import ipdb; ipdb.set_trace()
        df = self._extract_unified_with_fallback(run, columns_to_extract, max_retry)

        if df.empty:
            return pd.DataFrame()

        # Check if extracted data is missing expected columns
        extracted_cols = set(df.columns)
        expected_cols = set(columns_to_extract)
        missing_from_extraction = expected_cols - extracted_cols
        if missing_from_extraction:
            print(f"    WARNING: Extracted data missing columns for run {run.id}")
            print(f"    Expected columns: {sorted(expected_cols)}")
            print(f"    Extracted columns: {sorted(extracted_cols)}")
            print(f"    Missing columns: {sorted(missing_from_extraction)}")
            print(f"    The W&B run may not have logged these metrics.")
            print(f"    Launching debugger for investigation...")
            import ipdb; ipdb.set_trace()

        # Add metadata
        effective_run_id = eff_run_id if eff_run_id else run.id
        df['run_id'] = effective_run_id
        df['group'] = ''

        # Cache result
        self._save_run_data_to_cache(effective_run_id, metrics, step_columns, df)
        return df

    def _print_history_keys(self, run: wandb.apis.public.Run) -> None:
        """Print available history keys once per run when debug_history_keys is enabled."""
        if not self.config.get('debug_history_keys', False):
            return

        run_id = run.id
        if run_id in self._history_keys_printed:
            return

        keys: List[str] = []
        try:
            df = run.history(samples=1)
            if df is not None and not df.empty:
                keys = list(df.columns)
        except Exception as e:
            print(f"    Warning: run.history failed for {run_id}: {e}")

        if not keys:
            try:
                history = run.scan_history(page_size=1)
                first_row = next(iter(history), None)
                if isinstance(first_row, dict):
                    keys = list(first_row.keys())
            except Exception as e:
                print(f"    Warning: run.scan_history failed for {run_id}: {e}")

        if keys:
            keys_sorted = sorted(keys)
            print(f"    History keys for run {run_id} ({len(keys_sorted)}): {keys_sorted}")
        else:
            print(f"    Warning: Could not read history keys for run {run_id}")

        self._history_keys_printed.add(run_id)
    
    def _extract_unified_with_fallback(
        self, 
        run: wandb.apis.public.Run, 
        columns_to_extract: List[str], 
        max_retry: int
    ) -> pd.DataFrame:
        """Try unified extraction first, fall back to separate calls if data seems incomplete."""
        
        # First attempt: unified extraction (original approach)
        unified_df = self._fetch_unified_from_wandb(run, columns_to_extract, max_retry)
        
        if unified_df.empty:
            print(f"      Unified extraction failed, trying separate extraction")
            return self._extract_separate_with_batch_merge(run, columns_to_extract, max_retry)
        
        # Check data completeness heuristic
        expected_cols = set(columns_to_extract) - {'_step'}  # _step always included
        actual_cols = set(unified_df.columns) - {'_step'}
        
        missing_cols = expected_cols - actual_cols
        if missing_cols:
            print(f"      Missing columns in unified extraction: {missing_cols}")
            print(f"      Trying separate extraction for better coverage")
            return self._extract_separate_with_batch_merge(run, columns_to_extract, max_retry)
        
        # Quick data sparsity check
        non_null_ratios = {}
        for col in expected_cols:
            if col in unified_df.columns:
                non_null_ratio = unified_df[col].notna().mean()
                non_null_ratios[col] = non_null_ratio
        
        # If any metric has very low coverage, try separate extraction
        min_coverage = min(non_null_ratios.values()) if non_null_ratios else 1.0
        if min_coverage < 0.1:  # Less than 10% coverage
            print(f"      Low data coverage detected (min: {min_coverage:.1%}), trying separate extraction")
            separate_df = self._extract_separate_with_batch_merge(run, columns_to_extract, max_retry)
            
            # Compare coverage and use better result
            if not separate_df.empty:
                separate_coverage = min([separate_df[col].notna().mean() for col in expected_cols 
                                       if col in separate_df.columns], default=0)
                if separate_coverage > min_coverage * 1.5:  # 50% better coverage
                    print(f"      Using separate extraction (coverage: {separate_coverage:.1%} vs {min_coverage:.1%})")
                    return separate_df
        
        return unified_df
    
    def _fetch_unified_from_wandb(
        self,
        run: wandb.apis.public.Run,
        columns_to_extract: List[str],
        max_retry: int
    ) -> pd.DataFrame:
        """Original unified extraction method."""
        retry = 0

        # Get history sampling config
        full_history = self.config.get('full_history', False)
        history_samples = self.config.get('history_samples', 10000)

        while retry < max_retry:
            try:
                if full_history or history_samples is None:
                    # Fetch all data without sampling
                    history = run.scan_history(keys=columns_to_extract)
                    df = pd.DataFrame(history)
                else:
                    # Fetch sampled data (faster but truncates)
                    history = run.history(keys=columns_to_extract, samples=history_samples)
                    df = pd.DataFrame(history)

                if not df.empty:
                    print(f"      Unified extraction: {len(df)} rows, {len(df.columns)} columns")
                return df

            except Exception as e:
                print(f"      Error in unified extraction: {e}")
                time.sleep(30)
                retry += 1

        return pd.DataFrame()
    
    def _extract_separate_with_batch_merge(
        self, 
        run: wandb.apis.public.Run, 
        columns_to_extract: List[str], 
        max_retry: int
    ) -> pd.DataFrame:
        """Optimized separate extraction with efficient merging."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        # Skip _step as it's included automatically
        metrics_to_fetch = [col for col in columns_to_extract if col != '_step']
        
        if not metrics_to_fetch:
            return pd.DataFrame()
        
        # Parallel extraction with limited concurrency to avoid rate limits
        max_workers = min(3, len(metrics_to_fetch))  # Conservative concurrency
        column_data = {}
        lock = threading.Lock()
        
        def fetch_single_metric(metric):
            df = self._fetch_single_column_from_wandb(run, metric, max_retry)
            if not df.empty:
                with lock:
                    column_data[metric] = df
            return metric, not df.empty
        
        print(f"      Fetching {len(metrics_to_fetch)} metrics with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single_metric, metric): metric 
                      for metric in metrics_to_fetch}
            
            for future in as_completed(futures):
                metric, success = future.result()
                if not success:
                    print(f"      Failed to fetch {metric}")
        
        if not column_data:
            return pd.DataFrame()
        
        # Efficient merge using pandas concat + groupby
        return self._fast_merge_on_step(list(column_data.values()))
    
    def _fast_merge_on_step(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Fast merge without creating full step ranges."""
        if len(dataframes) == 1:
            return dataframes[0]
        
        # Concatenate all data
        all_data = pd.concat(dataframes, ignore_index=True)
        all_data['_step'] = pd.to_numeric(all_data['_step'], errors='coerce')
        all_data = all_data.dropna(subset=['_step'])
        
        if all_data.empty:
            return pd.DataFrame()
        
        # Group by step and combine (takes first non-null value for each metric)
        numeric_columns = all_data.select_dtypes(include=[np.number]).columns
        
        # Use groupby with first() to efficiently merge
        merged = all_data.groupby('_step').first().reset_index()
        
        # Only interpolate if we have sparse data (optional optimization)
        total_steps = merged['_step'].max() - merged['_step'].min() + 1
        if len(merged) < total_steps * 0.5:  # Less than 50% coverage
            print(f"      Applying minimal interpolation ({len(merged)}/{total_steps} steps)")
            merged = self._minimal_interpolation(merged, numeric_columns)
        
        print(f"      Fast merge result: {len(merged)} rows, {len(merged.columns)} columns")
        return merged
    
    def _minimal_interpolation(self, data: pd.DataFrame, numeric_columns) -> pd.DataFrame:
        """Apply interpolation only where beneficial."""
        #data = data.set_index('_step').sort_index()
        
        # Only fill small gaps (e.g., <= 10 steps)
        data[numeric_columns] = data[numeric_columns].interpolate(
            method='linear', 
            limit=10,  # Maximum gap size to fill
            limit_area='inside'
        )
        
        return data.reset_index()
    
    def _fetch_single_column_from_wandb(
        self,
        run: wandb.apis.public.Run,
        column: str,
        max_retry: int
    ) -> pd.DataFrame:
        """Fetch single column (used only in fallback mode)."""
        retry = 0

        # Get history sampling config
        full_history = self.config.get('full_history', False)
        history_samples = self.config.get('history_samples', 5000)  # Lower default for single column

        while retry < max_retry:
            try:
                if full_history or history_samples is None:
                    # Fetch all data without sampling
                    history = run.scan_history(keys=[column])
                    df = pd.DataFrame(history)
                else:
                    # Fetch sampled data (faster but truncates)
                    history = run.history(keys=[column], samples=history_samples)
                    df = pd.DataFrame(history)

                return df

            except Exception as e:
                print(f"      Error fetching {column}: {e}")
                time.sleep(10)  # Shorter wait
                retry += 1

        return pd.DataFrame()
  
    def _collect_run_requirements(self) -> Dict[str, Any]:
        """Collect all run IDs and their potential projects."""
        all_run_ids = list() #set()
        project_specific_runs = {}  # project -> set of run_ids
        
        for figure_config in self.config['figures']:
            for group_config in figure_config['groups']:
                run_ids = group_config['run_ids']
                if isinstance(run_ids, str):
                    run_ids = [run_ids]
                
                # Check if group specifies specific projects
                if 'project' in group_config:
                    projects = group_config['project']
                    if isinstance(projects, str):
                        projects = [projects]
                    
                    for project in projects:
                        if project not in project_specific_runs:
                            project_specific_runs[project] = set()
                        project_specific_runs[project].update(run_ids)
                else:
                    # No specific project - need to search all
                    #all_run_ids.update(run_ids)
                    all_run_ids.extend(run_ids)
        
        return {
            'all_projects_runs': list(all_run_ids),
            'project_specific_runs': {k: list(v) for k, v in project_specific_runs.items()}
        }

    def _export_filtered_data_for_verification(
        self,
        filtered_data: pd.DataFrame,
        group_name: str,
        figure_name: str,
        output_dir: Path,
        effective_filter: Optional[str],
        rows_before_filter: int,
        run_ids: Set[str],
        filter_warnings: List[str] = None,
    ) -> None:
        """Export filtered data to CSV and metadata to JSON for human verification.

        Creates files in output_dir/filter_diagnostics/:
        - {figure_name}_{group_name}_filtered_data.csv: The actual filtered DataFrame
        - {figure_name}_{group_name}_filter_info.json: Metadata about the filtering
        """
        diag_dir = output_dir / "filter_diagnostics" / figure_name.replace('/', '+')
        diag_dir.mkdir(parents=True, exist_ok=True)

        safe_group_name = group_name.replace('/', '+').replace(' ', '_')

        # Reorder columns: identity cols first, then filter cols, then the rest
        filter_cols = []
        if effective_filter:
            filter_cols = list(self._extract_columns_from_filter(effective_filter))

        # Build column order: run_id, _step, filter columns, then everything else
        identity_cols = ['run_id']
        step_cols = ['_step'] if '_step' in filtered_data.columns else []
        priority_cols = identity_cols + step_cols + [c for c in filter_cols if c in filtered_data.columns]
        other_cols = [c for c in filtered_data.columns if c not in priority_cols]
        ordered_cols = priority_cols + other_cols

        # Reorder the DataFrame for export
        if not ordered_cols:
            export_df = filtered_data.copy()
        else:
            export_df = filtered_data.copy()
            missing_cols = [c for c in ordered_cols if c not in export_df.columns]
            for col in missing_cols:
                export_df[col] = np.nan
            export_df = export_df[ordered_cols].copy()

        # Export the filtered DataFrame to CSV
        csv_filename = f"{safe_group_name}_filtered_data.csv"
        csv_path = diag_dir / csv_filename
        export_df.to_csv(csv_path, index=False)

        # Determine which filter columns are present/missing
        filter_cols_present = [c for c in filter_cols if c in filtered_data.columns]
        filter_cols_missing = [c for c in filter_cols if c not in filtered_data.columns]

        # Create filter metadata
        filter_info = {
            "figure_name": figure_name,
            "group_name": group_name,
            "run_ids": sorted(list(run_ids)),
            "filter_applied": effective_filter,
            "filter_columns_requested": filter_cols,
            "filter_columns_present": filter_cols_present,
            "filter_columns_missing": filter_cols_missing,
            "rows_after_run_id_filter": rows_before_filter,
            "rows_after_data_filter": len(filtered_data),
            "rows_removed_by_filter": rows_before_filter - len(filtered_data),
            "csv_column_order": list(ordered_cols),
            "warnings": filter_warnings or [],
            "has_errors": any("ERROR" in w or "not found" in w for w in (filter_warnings or [])),
        }

        # Add summary statistics for filter columns if filter was applied
        if effective_filter and filter_cols_present:
            filter_col_stats = {}
            for col in filter_cols_present:
                col_data = pd.to_numeric(filtered_data[col], errors='coerce')
                unique_vals = sorted([v for v in col_data.dropna().unique().tolist() if pd.notna(v)])
                filter_col_stats[col] = {
                    "unique_values_in_filtered_data": unique_vals,
                    "min": float(col_data.min()) if pd.notna(col_data.min()) else None,
                    "max": float(col_data.max()) if pd.notna(col_data.max()) else None,
                    "count": int(col_data.notna().sum()),
                }
            filter_info["filter_column_stats"] = filter_col_stats

        # Export filter metadata to JSON
        json_filename = f"{safe_group_name}_filter_info.json"
        json_path = diag_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(filter_info, f, indent=2)

        print(f"    Exported filter diagnostics: {csv_path.name}, {json_path.name}")

    def _group_runs_data(self, data: pd.DataFrame, group_config: Dict[str, Any],
                         figure_filter: str = None,
                         output_dir: Path = None,
                         figure_name: str = None,
                         figure_config: Dict[str, Any] = None) -> pd.DataFrame:
        """Filter data by run IDs, apply filters, and assign group."""
        group_name = group_config['name']
        run_ids = group_config['run_ids']

        if isinstance(run_ids, str):
            run_ids = [run_ids]

        # Filter by run IDs
        normalized_run_ids = {self._normalize_run_id(rid) for rid in run_ids}
        filtered_data = data[data['run_id'].isin(normalized_run_ids)].copy()
        rows_after_run_filter = len(filtered_data)

        # Apply data filter: group-level overrides figure-level
        effective_filter = group_config.get('data_filter', figure_filter)
        filter_warnings = []
        used_artifact_fallback = False

        if effective_filter:
            print(f"    Applying data filter for group '{group_name}': {effective_filter}")

            # Early check: are all filter columns present in the data?
            required_filter_cols = self._extract_columns_from_filter(effective_filter)
            available_cols = set(filtered_data.columns)
            missing_filter_cols = required_filter_cols - available_cols

            if missing_filter_cols:
                msg = f"Filter columns missing from data: {sorted(missing_filter_cols)}"
                print(f"    WARNING: {msg}")
                print(f"    Filter string: {effective_filter}")
                print(f"    Required filter columns: {sorted(required_filter_cols)}")
                print(f"    Available columns in data: {sorted(available_cols)}")
                print(f"    This may indicate the columns weren't fetched from W&B.")
                print(f"    Check _collect_all_required_metrics() and W&B run data.")
                filter_warnings.append(msg)

            # Optionally recover missing epoch values before applying drop_missing/filters.
            recover_epoch_by_rank = group_config.get(
                'recover_epoch_by_rank',
                figure_config.get('recover_epoch_by_rank', False) if figure_config else False
            )
            if recover_epoch_by_rank and 'epoch' in required_filter_cols and 'epoch' in filtered_data.columns:
                target_keys = group_config.get(
                    'recover_epoch_target_keys',
                    figure_config.get('recover_epoch_target_keys', None) if figure_config else None
                )
                step_col = figure_config.get('step_column', '_step') if figure_config else '_step'
                filtered_data = self._recover_epoch_by_rank(
                    filtered_data,
                    step_column=step_col,
                    target_keys=target_keys,
                    filter_warnings=filter_warnings,
                )

            # Optionally drop rows with missing filter variables before applying filters.
            drop_missing = group_config.get(
                'drop_missing',
                figure_config.get('drop_missing', False) if figure_config else False
            )
            if drop_missing and required_filter_cols:
                if missing_filter_cols:
                    msg = "drop_missing requested but filter columns are missing from data"
                    print(f"    WARNING: {msg}")
                    filter_warnings.append(msg)
                else:
                    rows_before_drop = len(filtered_data)
                    filtered_data = filtered_data.dropna(subset=list(required_filter_cols))
                    rows_after_drop = len(filtered_data)
                    dropped = rows_before_drop - rows_after_drop
                    if dropped > 0:
                        msg = f"drop_missing removed {dropped} rows with missing filter columns"
                        print(f"    INFO: {msg}")
                        filter_warnings.append(msg)

            conditions = self._parse_filter_string(effective_filter)

            # Determine if loose filtering is enabled (group-level overrides figure-level)
            loose_filtering = group_config.get(
                'loose_filtering',
                figure_config.get('loose_filtering', False) if figure_config else False
            )

            if loose_filtering:
                # Apply loose filtering PER-RUN to ensure each run gets fallback independently
                step_column = figure_config.get('step_column', '_step') if figure_config else '_step'
                print(f"    Loose filtering enabled (step_column: {step_column}), applying per-run")

                per_run_results = []
                all_fallback_info = {'steps_with_fallback': {}}

                for run_id in normalized_run_ids:
                    run_data = filtered_data[filtered_data['run_id'] == run_id]
                    if run_data.empty:
                        print(f"      Run {run_id}: no data after run filter")
                        continue

                    run_filtered, run_fallback = self._apply_loose_filter(
                        run_data, conditions, step_column
                    )

                    if not run_filtered.empty:
                        per_run_results.append(run_filtered)
                        print(f"      Run {run_id}: {len(run_filtered)} rows after loose filter")
                    else:
                        print(f"      Run {run_id}: no data after loose filter")

                    # Merge fallback info with run_id prefix for tracking
                    for step, fb in run_fallback.get('steps_with_fallback', {}).items():
                        key = f"{run_id}:{step}"
                        all_fallback_info['steps_with_fallback'][key] = fb

                if per_run_results:
                    filtered_data = pd.concat(per_run_results, ignore_index=True)
                else:
                    filtered_data = pd.DataFrame()

                if all_fallback_info['steps_with_fallback']:
                    n_fallbacks = len(all_fallback_info['steps_with_fallback'])
                    filter_warnings.append(
                        f"Loose filtering used fallback for {n_fallbacks} run:step combinations"
                    )
            else:
                # Use strict filtering (existing behavior)
                filtered_data, filter_warnings_from_apply = self._apply_filter_conditions(filtered_data, conditions)
                filter_warnings.extend(filter_warnings_from_apply)

            # If drop_missing leaves no data, try to recover from artifacts.
            if drop_missing and filtered_data.empty:
                print(f"    drop_missing resulted in 0 rows for group '{group_name}', trying artifact fallback")
                artifact_data = self._load_group_data_from_artifacts(
                    group_config=group_config,
                    figure_config=figure_config,
                )
                if not artifact_data.empty:
                    # Apply the same filter strictly to recovered data.
                    if effective_filter:
                        artifact_data, filter_warnings_from_apply = self._apply_filter_conditions(
                            artifact_data, conditions
                        )
                        filter_warnings.extend(filter_warnings_from_apply)
                    if not artifact_data.empty:
                        filtered_data = artifact_data
                        used_artifact_fallback = True
                        filter_warnings.append("Used artifact fallback due to empty data after drop_missing")

        # Export filtered data for human verification
        if output_dir is not None and figure_name is not None:
            self._export_filtered_data_for_verification(
                filtered_data,
                group_name,
                figure_name,
                output_dir,
                effective_filter,
                rows_after_run_filter,
                normalized_run_ids,
                filter_warnings=filter_warnings,
            )

        # Apply metric aliases: rename group-specific metrics to global names
        filtered_data = self._apply_metric_aliases(filtered_data, group_config)
        
        filtered_data['group'] = group_name
        
        # Clean up data types for numeric columns
        for col in filtered_data.columns:
            if col not in ['run_id', 'group']:  # Skip metadata columns
                # Convert string "NaN" to proper NaN and ensure numeric type
                filtered_data[col] = filtered_data[col].replace('NaN', np.nan)
                filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

        # Add temporal debugging for numeric metrics
        for col in filtered_data.columns:
            if col not in ['run_id', 'group', '_step', 'obs_count']:
             # Clean the column first
                cleaned_col = filtered_data[col].replace('NaN', np.nan)
                cleaned_col = pd.to_numeric(cleaned_col, errors='coerce')
                
                # Find non-zero, non-NaN values
                valid_mask = cleaned_col.notna() & (cleaned_col != 0)
                if valid_mask.any():
                    valid_data = filtered_data[valid_mask]
                    
                    print(f"\n  === Temporal Analysis for '{col}' in group '{group_config['name']}' ===")
                    print(f"  Total data points: {len(filtered_data)}")
                    print(f"  Valid non-zero points: {valid_mask.sum()}")
                    
                    # _step range for valid data
                    if '_step' in valid_data.columns:
                        step_values = pd.to_numeric(valid_data['_step'], errors='coerce').dropna()
                        if len(step_values) > 0:
                            print(f"  _step range for valid data: {step_values.min():.0f} to {step_values.max():.0f}")
                            print(f"  _step coverage: {len(step_values)} points")
                    
                    # obs_count range for valid data  
                    if 'obs_count' in valid_data.columns:
                        obs_values = pd.to_numeric(valid_data['obs_count'], errors='coerce').dropna()
                        if len(obs_values) > 0:
                            print(f"  obs_count range for valid data: {obs_values.min():.0f} to {obs_values.max():.0f}")
                            print(f"  obs_count coverage: {len(obs_values)} points")
                    
                    # Show the relationship between _step and obs_count for valid data
                    if '_step' in valid_data.columns and 'obs_count' in valid_data.columns:
                        step_clean = pd.to_numeric(valid_data['_step'], errors='coerce')
                        obs_clean = pd.to_numeric(valid_data['obs_count'], errors='coerce')
                        both_valid = step_clean.notna() & obs_clean.notna()
                        
                        if both_valid.any():
                            step_vals = step_clean[both_valid]
                            obs_vals = obs_clean[both_valid]
                            print(f"  _step vs obs_count correlation: {step_vals.corr(obs_vals):.3f}")
                            
                            # Show some sample points
                            sample_data = valid_data[both_valid][['_step', 'obs_count', col]].head(5)
                            print(f"  Sample valid points:")
                            for _, row in sample_data.iterrows():
                                print(f"    _step={row['_step']}, obs_count={row['obs_count']}, {col}={row[col]}")
                    
                    # Show overall dataset ranges for comparison
                    print(f"  --- Overall dataset ranges for comparison ---")
                    if '_step' in filtered_data.columns:
                        all_steps = pd.to_numeric(filtered_data['_step'], errors='coerce').dropna()
                        if len(all_steps) > 0:
                            print(f"  Full _step range: {all_steps.min():.0f} to {all_steps.max():.0f}")
                    
                    if 'obs_count' in filtered_data.columns:
                        all_obs = pd.to_numeric(filtered_data['obs_count'], errors='coerce').dropna()
                        if len(all_obs) > 0:
                            print(f"  Full obs_count range: {all_obs.min():.0f} to {all_obs.max():.0f}")
                    
                    print(f"  ========================================\n")
    
        return filtered_data

    def _recover_epoch_by_rank(
        self,
        data: pd.DataFrame,
        step_column: str = '_step',
        target_keys: Optional[List[str]] = None,
        filter_warnings: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Recover missing epoch values by ranking rows per run/target/k."""
        if data.empty or 'epoch' not in data.columns:
            return data

        if target_keys is None:
            target_keys = ['target_text', 'target_tokens_str', 'target_tokens']

        target_key = next((k for k in target_keys if k in data.columns), None)
        if target_key is None:
            msg = "recover_epoch_by_rank enabled but no target key found in data"
            print(f"    WARNING: {msg}")
            if filter_warnings is not None:
                filter_warnings.append(msg)
            return data

        step_col = step_column if step_column in data.columns else ('_step' if '_step' in data.columns else None)
        if step_col is None:
            msg = "recover_epoch_by_rank enabled but no step column available"
            print(f"    WARNING: {msg}")
            if filter_warnings is not None:
                filter_warnings.append(msg)
            return data

        df = data.copy()
        original_index = df.index

        def _normalize_target(val: Any) -> Optional[str]:
            if pd.isna(val):
                return None
            if isinstance(val, (list, dict)):
                try:
                    return json.dumps(val, ensure_ascii=True, sort_keys=True)
                except Exception:
                    return str(val)
            return str(val)

        df['__target_id__'] = df[target_key].apply(_normalize_target)
        if 'run_id' in df.columns:
            df = df.sort_values(['run_id', step_col])
            df['__target_id__'] = df.groupby('run_id')['__target_id__'].ffill().bfill()
        else:
            df = df.sort_values(step_col)
            df['__target_id__'] = df['__target_id__'].ffill().bfill()

        missing_mask = df['epoch'].isna()
        if not missing_mask.any():
            df = df.loc[original_index]
            df.drop(columns=['__target_id__'], inplace=True, errors='ignore')
            return df

        group_cols = []
        if 'run_id' in df.columns:
            group_cols.append('run_id')
        if '__target_id__' in df.columns:
            group_cols.append('__target_id__')
        if 'k' in df.columns:
            group_cols.append('k')

        if not group_cols:
            msg = "recover_epoch_by_rank enabled but no grouping columns available"
            print(f"    WARNING: {msg}")
            if filter_warnings is not None:
                filter_warnings.append(msg)
            df = df.loc[original_index]
            df.drop(columns=['__target_id__'], inplace=True, errors='ignore')
            return df

        df = df.sort_values(group_cols + [step_col])
        df['__epoch_rank__'] = df.groupby(group_cols).cumcount() + 1
        df.loc[missing_mask, 'epoch'] = df.loc[missing_mask, '__epoch_rank__']

        filled = int(missing_mask.sum())
        msg = f"recover_epoch_by_rank filled {filled} missing epoch values using {target_key}"
        print(f"    INFO: {msg}")
        if filter_warnings is not None:
            filter_warnings.append(msg)

        df.drop(columns=['__target_id__', '__epoch_rank__'], inplace=True, errors='ignore')
        df = df.loc[original_index]
        return df

    def _load_group_data_from_artifacts(
        self,
        group_config: Dict[str, Any],
        figure_config: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Attempt to recover metrics from local/W&B artifacts for the group's runs."""
        group_name = group_config['name']
        run_ids = group_config.get('run_ids', [])
        if isinstance(run_ids, str):
            run_ids = [run_ids]

        # Normalize run IDs to match artifact naming conventions.
        normalized_run_ids = [self._normalize_run_id(rid) for rid in run_ids]

        metrics = figure_config.get('metrics', []) if figure_config else []
        if not metrics:
            return pd.DataFrame()

        recovered_frames = []
        for run_id in normalized_run_ids:
            summary = self._load_summary_from_artifact(run_id)
            if not summary:
                continue
            run_frame = self._build_epoch_k_metrics_frame(summary, metrics)
            if run_frame.empty:
                continue
            run_frame['run_id'] = run_id
            run_frame['group'] = group_name
            recovered_frames.append(run_frame)

        if not recovered_frames:
            return pd.DataFrame()

        return pd.concat(recovered_frames, ignore_index=True)

    def _load_summary_from_artifact(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load summary dict from local artifacts or W&B if available."""
        # Attempt W&B artifact download as a fallback.
        try:
            api = wandb.Api()
        except Exception as e:
            print(f"    Warning: W&B API unavailable for artifact recovery: {e}")
            return None

        projects = self.config.get('projects', [])
        artifact_basenames = ["evaluation_summary", "batch_results"]
        cache_root = Path(self.config.get('cache_dir', 'wandb_cache')) / "artifacts"
        cache_root.mkdir(parents=True, exist_ok=True)

        for project in projects:
            for base in artifact_basenames:
                artifact_name = f"{base}-{run_id}"
                artifact_ref = f"{project}/{artifact_name}:latest"
                try:
                    artifact = api.artifact(artifact_ref, type="results")
                    download_dir = cache_root / artifact_name
                    artifact.download(root=str(download_dir))
                    candidate = download_dir / f"{base}.json"
                    if candidate.exists():
                        with open(candidate, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, dict) and isinstance(data.get('summary'), dict):
                            return data['summary']
                        if isinstance(data, dict):
                            return data
                except Exception:
                    continue

        return None

    def _build_epoch_k_metrics_frame(
        self,
        summary: Dict[str, Any],
        metrics: List[str],
    ) -> pd.DataFrame:
        """Build a DataFrame with epoch/k grids for requested metrics from summary dict."""
        rows_by_key: Dict[Tuple[float, float], Dict[str, Any]] = {}

        for metric in metrics:
            metric_map = None
            if metric == "avg_lcs_ratio":
                metric_map = summary.get("avg_lcs_ratio_by_epoch_by_k")
            elif metric.startswith("avg_"):
                base_metric = metric[len("avg_"):]
                prompt_map = summary.get("avg_prompt_metrics_by_epoch_by_k", {})
                semantic_map = summary.get("avg_semantic_metrics_by_epoch_by_k", {})
                metric_map = prompt_map.get(base_metric) or semantic_map.get(base_metric)

            if not metric_map:
                continue

            for epoch_key, k_dict in metric_map.items():
                try:
                    epoch_val = float(epoch_key)
                except (TypeError, ValueError):
                    continue
                if not isinstance(k_dict, dict):
                    continue
                for k_key, value in k_dict.items():
                    try:
                        k_val = float(k_key)
                    except (TypeError, ValueError):
                        continue
                    if value is None:
                        continue
                    key = (epoch_val, k_val)
                    row = rows_by_key.setdefault(key, {"epoch": epoch_val, "k": k_val})
                    row[metric] = value

        if not rows_by_key:
            return pd.DataFrame()

        return pd.DataFrame(list(rows_by_key.values()))
        
    def _synchronize_data(self, data: pd.DataFrame, sync_config: Dict[str, Any], 
                        step_column: str = '_step') -> pd.DataFrame:
        """Synchronize data to a common x-axis grid."""
        if not sync_config:
            return data
        
        sync_method = sync_config.get('method', 'interpolate')
        
        if sync_method == 'interpolate':
            return self._sync_by_interpolation(data, sync_config, step_column)
        elif sync_method in ('global_interpolate', 'common_grid'):
            return self._sync_by_common_grid_interpolation(data, sync_config, step_column)
        elif sync_method == 'bin':
            return self._sync_by_binning(data, sync_config, step_column)
        elif sync_method == 'resample':
            return self._sync_by_resampling(data, sync_config, step_column)
        else:
            print(f"Unknown synchronization method: {sync_method}")
            return data
    
    def _sync_by_interpolation(
        self, 
        data: pd.DataFrame, 
        sync_config: Dict[str, Any], 
        step_column: str,
    ) -> pd.DataFrame:
        """Interpolate all data to a common grid using _step as synchronization axis when available."""
        
        # Determine the common grid
        groups = data['group'].unique()
        
        # Only include numeric columns for interpolation
        potential_metrics = [col for col in data.columns if col not in ['group', 'run_id', step_column, '_step']]
        numeric_metrics = []
        
        for metric in potential_metrics:
            try:
                # Convert to numeric, coercing invalid values to NaN
                numeric_series = pd.to_numeric(data[metric], errors='coerce')
                
                # Only include if there's at least some numeric data
                if numeric_series.notna().any():
                    numeric_metrics.append(metric)
                else:
                    print(f"    Warning: No valid numeric data in column '{metric}', skipping")
            except Exception:
                print(f"    Warning: Could not process column '{metric}', skipping")
               
        print(f"    Interpolating {len(numeric_metrics)} numeric metrics: {numeric_metrics}")
        
        # Add the step_column to metrics if it's not _step (so we can interpolate it too)
        if step_column != '_step' and step_column not in numeric_metrics:
            try:
                pd.to_numeric(data[step_column], errors='raise')
                numeric_metrics.append(step_column)
            except (ValueError, TypeError):
                print(f"    Warning: Step column '{step_column}' is not numeric")
        
        interpolated_data = []
        
        for group in groups:
            group_data = data[data['group'] == group]
            
            for run_id in group_data['run_id'].unique():
                run_data = group_data[group_data['run_id'] == run_id].copy()
                
                # Use _step as the synchronization axis if available, otherwise fall back to step_column
                effective_step_col = '_step' if '_step' in run_data.columns else step_column
                
                try:
                    run_data[effective_step_col] = pd.to_numeric(run_data[effective_step_col], errors='coerce')
                    run_data = run_data.dropna(subset=[effective_step_col])
                    run_data = run_data.sort_values(effective_step_col)
                except (ValueError, TypeError):
                    print(f"    Warning: Non-numeric effective step column for run {run_id}, skipping interpolation")
                    continue
                
                if len(run_data) < 2:
                    print(f"    Warning: Not enough data points for run {run_id}, skipping interpolation")
                    continue
                
                # Determine the step range from the effective step column
                step_min = run_data[effective_step_col].min()
                step_max = run_data[effective_step_col].max()
                
                # Create interpolation grid
                if 'grid' in sync_config:
                    if isinstance(sync_config['grid'], list):
                        common_steps = np.array(sync_config['grid'])
                    else:
                        grid_config = sync_config['grid']
                        start = grid_config.get('start', step_min)
                        stop = grid_config.get('stop', step_max)
                        num_points = grid_config.get('num_points', 100)
                        common_steps = np.linspace(start, stop, num_points)
                else:
                    num_points = sync_config.get('num_points', 100)
                    common_steps = np.linspace(step_min, step_max, num_points)
                
                # Create interpolated run DataFrame
                interpolated_run = pd.DataFrame({effective_step_col: common_steps})
                interpolated_run['__sync_step__'] = common_steps
                interpolated_run['group'] = group
                interpolated_run['run_id'] = run_id
                
                # If we're using _step as effective step but need step_column in output, 
                # we need to handle the step_column mapping
                if effective_step_col == '_step' and step_column != '_step':
                    # The step_column will be interpolated as one of the metrics
                    pass
                elif effective_step_col == step_column:
                    # Direct mapping - the interpolated steps are the step_column values
                    interpolated_run[step_column] = common_steps
                
                # Interpolate each numeric metric using the effective step column
                for metric in numeric_metrics:
                    if metric in run_data.columns:
                        if True: #try:
                            run_data[metric] = pd.to_numeric(run_data[metric], errors='coerce')
                            if metric == effective_step_col:
                                # For the step column itself, just copy the interpolated steps
                                interpolated_run[metric] = common_steps
                            else:
                                valid_data = run_data[[effective_step_col, metric]].dropna()
                            
                                if len(valid_data) >= 1:
                                    interpolated_run[metric] = self._smart_interpolate_with_fills(
                                        common_steps,
                                        valid_data[effective_step_col].astype(float).values,
                                        valid_data[metric].astype(float).values,
                                    )
                                else:
                                    interpolated_run[metric] = np.nan
                        else: #except (ValueError, TypeError) as e:
                            print(f"    Warning: Could not interpolate metric '{metric}' for run {run_id}: {e}")
                            interpolated_run[metric] = np.nan
                    else:
                        interpolated_run[metric] = np.nan
                
                interpolated_data.append(interpolated_run)
        
        return pd.concat(interpolated_data, ignore_index=True) if interpolated_data else data
    
    def _sync_by_common_grid_interpolation(
        self,
        data: pd.DataFrame,
        sync_config: Dict[str, Any],
        step_column: str,
    ) -> pd.DataFrame:
        """Interpolate every run onto a shared grid so downstream aggregation sees aligned rows."""
        if data.empty:
            return data

        working = data.copy()

        # Prefer the configured step column for alignment; fall back to _step if needed
        sync_axis_candidates = []
        if step_column in working.columns:
            sync_axis_candidates.append(step_column)
        if '_step' in working.columns and '_step' != step_column:
            sync_axis_candidates.append('_step')

        if not sync_axis_candidates:
            print("    Warning: No suitable step column found for common grid interpolation")
            return data

        sync_axis_col = None
        for candidate in sync_axis_candidates:
            try:
                working[candidate] = pd.to_numeric(working[candidate], errors='coerce')
                sync_axis_col = candidate
                break
            except (ValueError, TypeError):
                print(f"    Warning: Step column '{candidate}' is not numeric; trying fallback")

        if sync_axis_col is None:
            print("    Warning: Could not coerce any step column to numeric; skipping common grid interpolation")
            return data

        working = working.dropna(subset=[sync_axis_col])
        if working.empty:
            print("    Warning: No valid step values after cleaning; skipping common grid interpolation")
            return working

        # Determine the shared grid of step values
        if 'grid' in sync_config:
            grid_config = sync_config['grid']
            if isinstance(grid_config, list):
                common_steps = np.array(grid_config, dtype=float)
            else:
                start = grid_config.get('start', working[sync_axis_col].min())
                stop = grid_config.get('stop', working[sync_axis_col].max())
                num_points = grid_config.get('num_points', sync_config.get('num_points', 100))
                common_steps = np.linspace(start, stop, num_points)
        else:
            global_min = working[sync_axis_col].min()
            global_max = working[sync_axis_col].max()
            if 'num_points' in sync_config:
                num_points = sync_config.get('num_points', 100)
                common_steps = np.linspace(global_min, global_max, num_points)
            else:
                common_steps = np.sort(working[sync_axis_col].astype(float).unique())

        if common_steps.size == 0:
            print("    Warning: Common grid is empty; skipping common grid interpolation")
            return working

        # Decide which columns can be interpolated
        reserved_cols = {'group', 'run_id', '__sync_step__'}
        potential_metrics = [col for col in working.columns if col not in reserved_cols]
        if sync_axis_col in potential_metrics:
            potential_metrics.remove(sync_axis_col)

        numeric_metrics = []
        for metric in potential_metrics:
            try:
                numeric_series = pd.to_numeric(working[metric], errors='coerce')
            except Exception:
                print(f"    Warning: Could not process column '{metric}', skipping")
                continue

            if numeric_series.notna().any():
                numeric_metrics.append(metric)
            else:
                print(f"    Warning: No valid numeric data in column '{metric}', skipping")

        if step_column != sync_axis_col and step_column not in numeric_metrics and step_column in working.columns:
            try:
                pd.to_numeric(working[step_column], errors='raise')
                numeric_metrics.append(step_column)
            except (ValueError, TypeError):
                print(f"    Warning: Step column '{step_column}' is not numeric; will not be interpolated")

        interpolated_rows: List[pd.DataFrame] = []

        for group in working['group'].unique():
            group_data = working[working['group'] == group]

            for run_id in group_data['run_id'].unique():
                run_data = group_data[group_data['run_id'] == run_id].copy()
                run_data = run_data.sort_values(sync_axis_col)
                run_data = run_data.drop_duplicates(subset=sync_axis_col, keep='last')

                if run_data.empty:
                    continue

                run_common = pd.DataFrame({
                    '__sync_step__': common_steps,
                    'group': group,
                    'run_id': run_id,
                })

                run_common[sync_axis_col] = common_steps
                if step_column == sync_axis_col:
                    run_common[step_column] = common_steps

                for metric in numeric_metrics:
                    if metric not in run_data.columns:
                        run_common[metric] = np.nan
                        continue

                    metric_series = pd.to_numeric(run_data[metric], errors='coerce')
                    valid_mask = metric_series.notna()
                    if not valid_mask.any():
                        run_common[metric] = np.nan
                        continue

                    valid_steps = run_data.loc[valid_mask, sync_axis_col].astype(float).values
                    valid_values = metric_series.loc[valid_mask].astype(float).values

                    run_common[metric] = self._linear_interpolate_on_grid(common_steps, valid_steps, valid_values)

                interpolated_rows.append(run_common)

        if not interpolated_rows:
            print("    Warning: Common grid interpolation produced no data; returning original data")
            return working

        return pd.concat(interpolated_rows, ignore_index=True)
    
    def _smart_interpolate_with_fills(self, target_steps: np.ndarray, 
                                     valid_steps: np.ndarray, 
                                     valid_values: np.ndarray) -> np.ndarray:
        """
        Interpolate with smart filling:
        - 0 fill before the first valid point
        - Linear interpolation between valid points  
        - Forward fill (last value) after the last valid point
        """
        if len(valid_steps) == 0:
            return np.full(len(target_steps), np.nan)
        
        if len(valid_steps) == 1:
            # Single point: 0 before, forward fill after
            single_step = valid_steps[0]
            single_value = valid_values[0]
            result = np.where(target_steps < single_step, 0.0, single_value)
            return result
        
        # Multiple points: use different strategies for different regions
        min_valid_step = valid_steps.min()#[0]
        max_valid_step = valid_steps.max()#[0]
        #last_valid_value = valid_values[np.argmax(valid_steps)]  # Value at the highest step
        #max_step_idx = int(np.argmax(valid_steps))
        #if hasattr(valid_values, 'iloc'):
        #    last_valid_value = valid_values.iloc[max_step_idx]
        #last_valid_value = valid_values.iloc[-1][0]
        last_valid_value = valid_values[-1]

        result = np.full(len(target_steps), np.nan)
        
        for i, step in enumerate(target_steps):
            if step < min_valid_step:
                # Before first valid point: fill with 0
                result[i] = 0.0
            elif step > max_valid_step:
                # After last valid point: forward fill with last value
                result[i] = last_valid_value
            else:
                # Between valid points: linear interpolation
                result[i] = np.interp(step, valid_steps, valid_values)
        
        return result
    
    def _linear_interpolate_on_grid(
        self,
        target_steps: np.ndarray,
        valid_steps: np.ndarray,
        valid_values: np.ndarray,
    ) -> np.ndarray:
        """Standard linear interpolation constrained to the convex hull of the data."""
        if len(valid_steps) == 0:
            return np.full(len(target_steps), np.nan)

        valid_df = pd.DataFrame({
            'step': valid_steps,
            'value': valid_values,
        }).dropna()

        if valid_df.empty:
            return np.full(len(target_steps), np.nan)

        valid_df = valid_df.drop_duplicates(subset='step', keep='last').sort_values('step')

        unique_steps = valid_df['step'].to_numpy(dtype=float)
        unique_values = valid_df['value'].to_numpy(dtype=float)

        if unique_steps.size == 1:
            result = np.full(len(target_steps), np.nan)
            result[np.isclose(target_steps, unique_steps[0])] = unique_values[0]
            return result

        interpolated = np.interp(target_steps, unique_steps, unique_values)
        outside_mask = (target_steps < unique_steps[0]) | (target_steps > unique_steps[-1])
        interpolated[outside_mask] = np.nan
        return interpolated
    
    def depr_sync_by_interpolation(self, data: pd.DataFrame, sync_config: Dict[str, Any], 
        step_column: str) -> pd.DataFrame:
        """Interpolate all data to a common grid."""
        # Determine the common grid
        if 'grid' in sync_config:
            if isinstance(sync_config['grid'], list):
                common_steps = np.array(sync_config['grid'])
            else:
                grid_config = sync_config['grid']
                start = grid_config.get('start', data[step_column].min())
                stop = grid_config.get('stop', data[step_column].max())
                num_points = grid_config.get('num_points', 100)
                common_steps = np.linspace(start, stop, num_points)
        else:
            step_min = data[step_column].min()
            step_max = data[step_column].max()
            num_points = sync_config.get('num_points', 100)
            common_steps = np.linspace(step_min, step_max, num_points)
        
        # Interpolate each run's metrics to the common grid
        interpolated_data = []
        groups = data['group'].unique()
        
        # Only include numeric columns for interpolation
        potential_metrics = [col for col in data.columns if col not in ['group', 'run_id', step_column]]
        numeric_metrics = []
        
        for metric in potential_metrics:
            # Check if the column is numeric
            try:
                #pd.to_numeric(data[metric], errors='raise')
                data[metric] = data[metric].fillna(0)
                numeric_metrics.append(metric)
            except (ValueError, TypeError):
                print(f"    Warning: Skipping non-numeric column '{metric}' for interpolation")
                continue
        
        print(f"    Interpolating {len(numeric_metrics)} numeric metrics: {numeric_metrics}")
        
        for group in groups:
            group_data = data[data['group'] == group]
            
            for run_id in group_data['run_id'].unique():
                run_data = group_data[group_data['run_id'] == run_id].copy()
                
                # Ensure step column is numeric
                try:
                    run_data[step_column] = pd.to_numeric(run_data[step_column], errors='coerce')
                    run_data = run_data.dropna(subset=[step_column])
                    run_data = run_data.sort_values(step_column)
                except (ValueError, TypeError):
                    print(f"    Warning: Non-numeric step column for run {run_id}, skipping interpolation")
                    continue
                
                if len(run_data) < 2:
                    print(f"    Warning: Not enough data points for run {run_id}, skipping interpolation")
                    continue
                
                interpolated_run = pd.DataFrame({step_column: common_steps})
                interpolated_run['group'] = group
                interpolated_run['run_id'] = run_id
                
                # Interpolate each numeric metric
                for metric in numeric_metrics:
                    if metric in run_data.columns:
                        # Convert to numeric and remove NaN values
                        try:
                            run_data[metric] = pd.to_numeric(run_data[metric], errors='coerce')
                            valid_data = run_data[[step_column, metric]].dropna()
                            
                            if len(valid_data) > 1:
                                interpolated_run[metric] = np.interp(
                                    common_steps, 
                                    valid_data[step_column].astype(float), 
                                    valid_data[metric].astype(float)
                                )
                            else:
                                interpolated_run[metric] = np.nan
                        except (ValueError, TypeError) as e:
                            print(f"    Warning: Could not interpolate metric '{metric}' for run {run_id}: {e}")
                            interpolated_run[metric] = np.nan
                    else:
                        interpolated_run[metric] = np.nan
                
                interpolated_data.append(interpolated_run)
        
        return pd.concat(interpolated_data, ignore_index=True) if interpolated_data else data
    
    def _sync_by_binning(self, data: pd.DataFrame, sync_config: Dict[str, Any], 
                        step_column: str) -> pd.DataFrame:
        """Bin data into uniform intervals."""
        # Ensure step column is numeric
        try:
            data = data.copy()
            data[step_column] = pd.to_numeric(data[step_column], errors='coerce')
            data = data.dropna(subset=[step_column])
        except (ValueError, TypeError):
            print(f"    Warning: Non-numeric step column, cannot bin data")
            return data
        
        if data.empty:
            return data
        
        num_bins = sync_config.get('num_bins', 50)
        step_min = data[step_column].min()
        step_max = data[step_column].max()
        
        bins = np.linspace(step_min, step_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        data['bin'] = pd.cut(data[step_column], bins=bins, labels=False, include_lowest=True)
        
        group_cols = ['run_id', 'group', 'bin']
        potential_metrics = [col for col in data.columns if col not in group_cols + [step_column]]
        
        # Only include numeric columns for aggregation
        numeric_metrics = []
        for metric in potential_metrics:
            try:
                data[metric] = pd.to_numeric(data[metric], errors='coerce')
                numeric_metrics.append(metric)
            except (ValueError, TypeError):
                print(f"    Warning: Skipping non-numeric column '{metric}' for binning")
        
        print(f"    Binning {len(numeric_metrics)} numeric metrics: {numeric_metrics}")
        
        if not numeric_metrics:
            print("    Warning: No numeric metrics found for binning")
            return data.drop('bin', axis=1)
        
        binned_data = data.groupby(group_cols)[numeric_metrics].mean().reset_index()
        binned_data[step_column] = binned_data['bin'].map(lambda x: bin_centers[int(x)] if pd.notna(x) else np.nan)
        binned_data = binned_data.drop('bin', axis=1)
        
        return binned_data.dropna(subset=[step_column])
    
    def _sync_by_resampling(self, data: pd.DataFrame, sync_config: Dict[str, Any], 
                          step_column: str) -> pd.DataFrame:
        """Resample data to specific intervals."""
        # Ensure step column is numeric
        try:
            data = data.copy()
            data[step_column] = pd.to_numeric(data[step_column], errors='coerce')
            data = data.dropna(subset=[step_column])
        except (ValueError, TypeError):
            print(f"    Warning: Non-numeric step column, cannot resample data")
            return data
        
        if data.empty:
            return data
        
        interval = sync_config.get('interval', 1000)
        data[f'{step_column}_resampled'] = (data[step_column] // interval) * interval
        
        group_cols = ['run_id', 'group', f'{step_column}_resampled']
        potential_metrics = [col for col in data.columns if col not in group_cols + [step_column]]
        
        # Only include numeric columns for aggregation
        numeric_metrics = []
        for metric in potential_metrics:
            try:
                data[metric] = pd.to_numeric(data[metric], errors='coerce')
                numeric_metrics.append(metric)
            except (ValueError, TypeError):
                print(f"    Warning: Skipping non-numeric column '{metric}' for resampling")
        
        print(f"    Resampling {len(numeric_metrics)} numeric metrics: {numeric_metrics}")
        
        if not numeric_metrics:
            print("    Warning: No numeric metrics found for resampling")
            return data.drop(f'{step_column}_resampled', axis=1)
        
        resampled_data = data.groupby(group_cols)[numeric_metrics].mean().reset_index()
        resampled_data[step_column] = resampled_data[f'{step_column}_resampled']
        resampled_data = resampled_data.drop(f'{step_column}_resampled', axis=1)
        
        return resampled_data
    
    def _apply_smoothing(self, data: pd.DataFrame, smoothing_config: Dict[str, Any], 
                        metric: str, step_column: str = '_step') -> pd.DataFrame:
        """Apply smoothing to metric data."""
        if not smoothing_config:
            return data
        
        smoothing_method = smoothing_config.get('method', 'running_average')
        window_size = smoothing_config.get('window_size', 32)
        
        print(f"    Applying smoothing: {smoothing_method} with window {window_size}")
        
        smoothed_data = []
        
        for group in data['group'].unique():
            for run_id in data[data['group'] == group]['run_id'].unique():
                run_data = data[(data['group'] == group) & (data['run_id'] == run_id)].copy()
                run_data = run_data.sort_values(step_column)
                
                if len(run_data) < window_size:
                    print(f"      Warning: Run {run_id} has {len(run_data)} points, less than window size {window_size}")
                    smoothed_data.append(run_data)
                    continue
                
                if smoothing_method == 'running_average':
                    run_data[metric] = run_data[metric].rolling(
                        window=window_size, 
                        center=smoothing_config.get('center', False),
                        min_periods=smoothing_config.get('min_periods', 1)
                    ).mean()
                
                elif smoothing_method == 'exponential':
                    alpha = smoothing_config.get('alpha', 0.1)
                    run_data[metric] = run_data[metric].ewm(
                        alpha=alpha,
                        adjust=smoothing_config.get('adjust', True)
                    ).mean()
                
                elif smoothing_method == 'savgol':
                    try:
                        from scipy.signal import savgol_filter
                        polyorder = smoothing_config.get('polyorder', 3)
                        if window_size > len(run_data):
                            window_size = len(run_data) if len(run_data) % 2 == 1 else len(run_data) - 1
                        if window_size <= polyorder:
                            window_size = polyorder + 1 if (polyorder + 1) % 2 == 1 else polyorder + 2
                        
                        run_data[metric] = savgol_filter(
                            run_data[metric].values, 
                            window_size, 
                            polyorder
                        )
                    except ImportError:
                        print(f"      Warning: scipy not available for savgol filter, using running average")
                        run_data[metric] = run_data[metric].rolling(window=window_size).mean()
                
                # Remove NaN values created by smoothing
                run_data = run_data.dropna(subset=[metric])
                smoothed_data.append(run_data)

        return pd.concat(smoothed_data, ignore_index=True) if smoothed_data else data

    def _calculate_statistics(
        self,
        data: pd.DataFrame,
        metric: str,
        step_column: str = '_step',
        group_axis: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calculate mean, median, and standard error for grouped data."""
    
        # Clean the metric column first
        print(f"    Original data: {len(data)} rows")
        print(f"    Data types in {metric}: {data[metric].apply(type).value_counts()}")
    
        # Convert to numeric, dropping non-numeric values
        data = data.copy()
        data[metric] = pd.to_numeric(data[metric], errors='coerce')
        data = data.dropna(subset=[metric])
    
        print(f"    After cleaning: {len(data)} rows")
    
        if data.empty:
            print(f"    Warning: No numeric data found for {metric}")
            return pd.DataFrame()
    
        axis_column = group_axis or step_column
        grouping_keys = [axis_column, 'group']
    
        stats = (
            data.groupby(grouping_keys)[metric]
            .agg(['mean', 'median', 'std', 'count'])
            .reset_index()
        )
        stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
        stats['stderr'] = stats['stderr'].fillna(0)
        stats['median'] = stats['median'].fillna(stats['mean'])
    
        # Preserve the display step column even if we grouped on another axis
        if step_column not in stats.columns and step_column in data.columns:
            step_values = (
                data.groupby(grouping_keys)[step_column]
                .mean()
                .reset_index()
            )
            stats = stats.merge(step_values, on=grouping_keys, how='left')
    
        return stats

    def _export_plot_data_to_json(
        self,
        stats_data: pd.DataFrame,
        metric: str,
        step_column: str,
        output_path: Path,
        group_order: List[str] = None,
        group_axis: Optional[str] = None,
    ) -> None:
        """Export the plotted statistics data to JSON for each group.

        Creates a JSON file with the following structure:
        {
            "metric": "metric_name",
            "step_column": "step_column_name",
            "groups": {
                "Group A": {
                    "x_values": [...],
                    "mean": [...],
                    "median": [...],
                    "std": [...],
                    "stderr": [...],
                    "count": [...]
                },
                ...
            }
        }
        """
        if stats_data.empty:
            return

        # Determine x-axis column
        x_column = group_axis if group_axis and group_axis in stats_data.columns else step_column

        # Build the export structure
        export_data = {
            "metric": metric,
            "step_column": step_column,
            "x_axis_column": x_column,
            "groups": {}
        }

        # Get groups in specified order
        if group_order is not None:
            available_groups = set(stats_data['group'].unique())
            groups = [g for g in group_order if g in available_groups]
        else:
            groups = stats_data['group'].unique().tolist()

        for group in groups:
            group_data = stats_data[stats_data['group'] == group].sort_values(x_column)

            # Convert to native Python types for JSON serialization
            group_export = {
                "x_values": group_data[x_column].tolist(),
                "mean": [float(v) if pd.notna(v) else None for v in group_data['mean']],
                "median": [float(v) if pd.notna(v) else None for v in group_data['median']],
                "std": [float(v) if pd.notna(v) else None for v in group_data['std']],
                "stderr": [float(v) if pd.notna(v) else None for v in group_data['stderr']],
                "count": [int(v) if pd.notna(v) else None for v in group_data['count']],
            }

            # Include the display step column if different from x_axis
            if step_column in group_data.columns and step_column != x_column:
                group_export["step_values"] = group_data[step_column].tolist()

            export_data["groups"][group] = group_export

        # Write JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"    Saved plot data JSON: {output_path}")

    def _create_line_plot(
        self,
        stats_data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        step_column: str = '_step',
        group_order: list[str]=None,
        group_axis: Optional[str]=None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Tuple[plt.Figure, List, List[str]]:
        """Create line plot with mean ± standard error.

        Returns:
            Tuple of (figure, legend_handles, legend_labels)
        """
        fig, ax = plt.subplots(figsize=self._parse_figsize(plot_config.get('figsize')))

        # Use specified group order, or fall back to data order
        if group_order is not None:
            # Only include groups that actually exist in the data
            available_groups = set(stats_data['group'].unique())
            groups = [group for group in group_order if group in available_groups]
            
            # Add any groups from data that weren't in the config (shouldn't happen, but be safe)
            for group in available_groups:
                if group not in groups:
                    groups.append(group)
                    
            print(f"    Plotting groups in config order: {groups}")
        else:
            groups = stats_data['group'].unique()
            print(f"    Plotting groups in data order: {groups}")
        
        # Use configured color palette with preserved order
        for i, group in enumerate(groups):
            group_data = stats_data[stats_data['group'] == group].sort_values(step_column)
            
            if group_data.empty:
                continue
                
            if step_column in group_data.columns:
                x = group_data[step_column]
            elif group_axis and group_axis in group_data.columns:
                x = group_data[group_axis]
            else:
                raise KeyError(f"Step column '{step_column}' (or fallback '{group_axis}') not found in statistics data.")
            y = group_data['mean']
            yerr = group_data['stderr']
            
            # Use custom color if specified, otherwise fall back to palette
            if group_colors and group in group_colors:
                color = group_colors[group]
            else:
                color_idx = i % len(self.color_palette)
                color = self.color_palette[color_idx]
            
            ax.plot(x, y, label=group, linewidth=2, alpha=0.8, color=color)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, color=color)

        ax.set_xlabel(plot_config.get('xlabel', step_column))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f'{metric} over {step_column}'))

        # Capture handles/labels before creating legend
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(loc=plot_config.get('legend_loc', 'best'))
        ax.grid(True, alpha=0.3)

        if 'xlim' in plot_config:
            ax.set_xlim(plot_config['xlim'])
        if 'ylim' in plot_config:
            ax.set_ylim(plot_config['ylim'])
        if plot_config.get('xscale') == 'log':
            ax.set_xscale('log')
        if plot_config.get('yscale') == 'log':
            ax.set_yscale('log')

        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig, handles, labels

    def _create_count_plot(
        self,
        stats_data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        step_column: str,
        group_order: list[str] = None,
        group_axis: Optional[str] = None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Tuple[plt.Figure, List, List[str]]:
        """Plot the number of contributing runs per datapoint for each group.

        Returns:
            Tuple of (figure, legend_handles, legend_labels)
        """
        fig, ax = plt.subplots(figsize=self._parse_figsize(plot_config.get('figsize')))

        if group_order is not None:
            available_groups = set(stats_data['group'].unique())
            groups = [group for group in group_order if group in available_groups]
            for group in available_groups:
                if group not in groups:
                    groups.append(group)
        else:
            groups = stats_data['group'].unique()
        
        line_styles = [
            '-',
            '--',
            ':',
            '-.',
            (0, (5, 2, 1, 2)),
            (0, (3, 1, 1, 1)),
        ]
        markers = ['o', 's', '^', 'D', 'v', 'P']
        
        value_offset = 0.01
        
        for i, group in enumerate(groups):
            group_data = stats_data[stats_data['group'] == group].sort_values(step_column)
            if group_data.empty:
                continue

            # Use custom color if specified, otherwise fall back to palette
            if group_colors and group in group_colors:
                color = group_colors[group]
            else:
                color_idx = i % len(self.color_palette)
                color = self.color_palette[color_idx]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            adjusted_counts = group_data['count'] + i * value_offset
            
            if step_column in group_data.columns:
                x_values = group_data[step_column]
            elif group_axis and group_axis in group_data.columns:
                x_values = group_data[group_axis]
            else:
                raise KeyError(f"Step column '{step_column}' (or fallback '{group_axis}') not found in statistics data.")
            
            ax.plot(
                x_values,
                adjusted_counts,
                label=group,
                color=color,
                linewidth=2,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                markevery=max(len(group_data) // 20, 1),
            )
        
        ax.set_xlabel(plot_config.get('xlabel', step_column))
        ax.set_ylabel("# runs contributing (offset for visibility)")
        base_title = plot_config.get('title', f"{metric} over {step_column}")
        ax.set_title(f"{base_title} — sample counts")

        # Capture handles/labels before creating legend
        handles, labels = ax.get_legend_handles_labels()

        ax.legend(loc=plot_config.get('legend_loc', 'best'))
        ax.grid(True, alpha=0.3)

        if 'xlim' in plot_config:
            ax.set_xlim(plot_config['xlim'])
        if plot_config.get('xscale') == 'log':
            ax.set_xscale('log')

        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig, handles, labels

    def _create_pre_sync_group_plot(
        self,
        stats_data: pd.DataFrame,
        group_name: str,
        metric: str,
        step_column: str,
        plot_config: Dict[str, Any],
        color,
        output_path: Path,
    ):
        """Create a diagnostic plot for a single group's raw (pre-sync) data."""
        if stats_data.empty:
            return

        plot_cfg = dict(plot_config) if plot_config else {}
        fig, ax = plt.subplots(figsize=self._parse_figsize(plot_cfg.get('figsize')))

        group_stats = stats_data.sort_values(step_column)
        x = group_stats[step_column]
        mean = group_stats['mean']
        median = group_stats.get('median')
        stderr = group_stats['stderr']
        
        ax.plot(x, mean, color=color, linewidth=2, label='Mean', alpha=0.8)
        ax.fill_between(x, mean - stderr, mean + stderr, color=color, alpha=0.5, label='Mean ± SE')
        
        if median is not None:
            ax.plot(
                x,
                median,
                color=color,
                linewidth=2,
                linestyle='--',
                alpha=0.8,
                label='Median',
            )
        
        ax.set_xlabel(plot_cfg.get('xlabel', step_column))
        ax.set_ylabel(plot_cfg.get('ylabel', metric))
        
        base_title = plot_cfg.get('title', f"{metric} over {step_column}")
        ax.set_title(f"{base_title} — {group_name} (pre-sync)")
        
        if 'xlim' in plot_cfg:
            ax.set_xlim(plot_cfg['xlim'])
        if 'ylim' in plot_cfg:
            ax.set_ylim(plot_cfg['ylim'])
        if plot_cfg.get('xscale') == 'log':
            ax.set_xscale('log')
        if plot_cfg.get('yscale') == 'log':
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        ax.legend(loc=plot_cfg.get('legend_loc', 'best'))
        
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved pre-sync plot: {output_path}")

    def _create_group_bar_plot(
        self,
        data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        group_order: Optional[List[str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Optional[plt.Figure]:
        """Create bar plot of mean ± standard error across groups without synchronization."""
        required_cols = {'group', metric}
        if data.empty or not required_cols.issubset(data.columns):
            print(f"    Skipping bar plot for '{metric}': missing columns {required_cols - set(data.columns)}")
            return None

        working = data[['group', metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working = working.dropna(subset=[metric])
        if working.empty:
            print(f"    Skipping bar plot for '{metric}': no numeric values")
            return None

        stats = (
            working.groupby('group')[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        stats['stderr'] = stats['std'] / np.sqrt(stats['count'].replace(0, np.nan))
        stats['stderr'] = stats['stderr'].fillna(0)

        if group_order:
            ordered = [g for g in group_order if g in stats['group'].values]
            ordered += [g for g in stats['group'].values if g not in ordered]
        else:
            ordered = list(stats['group'].values)

        if not ordered:
            print(f"    Skipping bar plot for '{metric}': no groups with data")
            return None

        stats = stats.set_index('group').loc[ordered].reset_index()

        fig, ax = plt.subplots(figsize=self._parse_figsize(plot_config.get('figsize')))
        positions = np.arange(len(ordered))
        # Use custom colors if specified, otherwise fall back to palette
        if group_colors is None:
            group_colors = {}
        colors = [
            group_colors.get(g, self.color_palette[i % len(self.color_palette)])
            for i, g in enumerate(ordered)
        ]

        ax.bar(
            positions,
            stats['mean'],
            yerr=stats['stderr'],
            color=colors,
            capsize=plot_config.get('capsize', 4),
            alpha=plot_config.get('alpha', 0.9),
            edgecolor=plot_config.get('edgecolor', 'black'),
            linewidth=plot_config.get('linewidth', 1),
        )

        if plot_config.get('log_scale', False):
            ax.set_yscale('log')

        if 'ylim' in plot_config:
            ax.set_ylim(plot_config['ylim'])

        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, rotation=plot_config.get('xtick_rotation', 30), ha='right')
        ax.set_xlabel(plot_config.get('xlabel', ''))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f"{metric} — mean ± SE"))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig

    def _create_conditioned_grouped_bar_plot(
        self,
        data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        condition_column: str,
        condition_values: Optional[List[Any]] = None,
        group_order: Optional[List[str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Optional[plt.Figure]:
        """Create grouped bar plot with bars clustered by condition values.

        X-axis shows condition values (e.g., epochs), with clustered bars for each group.
        Y-axis shows mean values with stderr error bars.
        """
        required_cols = {'group', metric, condition_column}
        if data.empty or not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            print(f"    Skipping conditioned bar plot for '{metric}': missing columns {missing}")
            return None

        working = data[['group', condition_column, metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working[condition_column] = pd.to_numeric(working[condition_column], errors='coerce')
        working = working.dropna(subset=[metric, condition_column])

        if working.empty:
            print(f"    Skipping conditioned bar plot for '{metric}': no numeric values")
            return None

        # Filter to specified condition values if provided
        if condition_values is not None:
            working = working[working[condition_column].isin(condition_values)]
            if working.empty:
                print(f"    Skipping conditioned bar plot: no data for specified condition values")
                return None
            conditions = [c for c in condition_values if c in working[condition_column].values]
        else:
            conditions = sorted(working[condition_column].unique())

        # Determine group order
        if group_order:
            groups = [g for g in group_order if g in working['group'].unique()]
            groups += [g for g in working['group'].unique() if g not in groups]
        else:
            groups = sorted(working['group'].unique())

        if not groups or not conditions:
            print(f"    Skipping conditioned bar plot: no groups or conditions")
            return None

        # Calculate statistics for each (condition, group) pair
        stats = (
            working.groupby([condition_column, 'group'])[metric]
            .agg(['mean', 'std', 'count'])
            .reset_index()
        )
        stats['stderr'] = stats['std'] / np.sqrt(stats['count'].replace(0, np.nan))
        stats['stderr'] = stats['stderr'].fillna(0)

        # Prepare figure
        figsize = self._parse_figsize(plot_config.get('figsize'))
        fig, ax = plt.subplots(figsize=figsize)

        n_groups = len(groups)
        n_conditions = len(conditions)

        # Calculate bar width
        bar_width = plot_config.get('bar_width')
        if bar_width is None:
            bar_width = 0.8 / n_groups

        x_positions = np.arange(n_conditions)

        # Set up colors
        if group_colors is None:
            group_colors = {}

        # Collect text annotation data for later (after axes are configured)
        text_annotations = []

        # Plot bars for each group
        for i, group in enumerate(groups):
            offset = (i - (n_groups - 1) / 2) * bar_width
            group_x = x_positions + offset

            # Get data for this group across all conditions
            means = []
            stderrs = []
            for cond in conditions:
                row = stats[(stats[condition_column] == cond) & (stats['group'] == group)]
                if len(row) > 0:
                    means.append(row['mean'].values[0])
                    stderrs.append(row['stderr'].values[0])
                else:
                    means.append(np.nan)
                    stderrs.append(0)

            color = group_colors.get(group, self.color_palette[i % len(self.color_palette)])

            ax.bar(
                group_x,
                means,
                width=bar_width,
                yerr=stderrs,
                label=group,
                color=color,
                capsize=plot_config.get('capsize', 4),
                alpha=plot_config.get('alpha', 0.9),
                edgecolor=plot_config.get('edgecolor', 'black'),
                linewidth=plot_config.get('linewidth', 1),
            )

            # Collect text annotation positions for show_values
            if plot_config.get('show_values', False):
                for x, mean, stderr in zip(group_x, means, stderrs):
                    if not np.isnan(mean):
                        text_annotations.append((x, mean, stderr))

        # Configure axes
        if plot_config.get('log_scale', False):
            ax.set_yscale('log')

        if 'ylim' in plot_config:
            ax.set_ylim(plot_config['ylim'])

        ax.set_xticks(x_positions)
        condition_labels = [str(c) for c in conditions]
        ax.set_xticklabels(condition_labels, rotation=plot_config.get('xtick_rotation', 0))
        ax.set_xlabel(plot_config.get('xlabel', condition_column))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f"{metric} by {condition_column}"))
        ax.legend(loc=plot_config.get('legend_loc', 'best'))
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels after axes are configured (so we can compute proper padding)
        if text_annotations:
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            padding = 0.02 * y_range if y_range > 0 else 0.01
            for x, mean, stderr in text_annotations:
                y_pos = mean + stderr + padding
                ax.text(
                    x, y_pos, f'{mean:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90
                )

        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig

    def _create_group_violin_plot(
        self,
        data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        group_order: Optional[List[str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Optional[plt.Figure]:
        """Create violin plot showing distribution of metric values across groups."""
        required_cols = {'group', metric}
        if data.empty or not required_cols.issubset(data.columns):
            print(f"    Skipping violin plot for '{metric}': missing columns {required_cols - set(data.columns)}")
            return None

        working = data[['group', metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working = working.dropna(subset=[metric])
        if working.empty:
            print(f"    Skipping violin plot for '{metric}': no numeric values")
            return None

        log_scale = plot_config.get('log_scale', False)
        if log_scale:
            n_before = len(working)
            working = working[working[metric] > 0]
            n_dropped = n_before - len(working)
            if n_dropped > 0:
                print(f"    Warning: dropped {n_dropped} non-positive values for log-scale violin plot")
            if working.empty:
                print(f"    Skipping violin plot for '{metric}': no positive values for log scale")
                return None

        if group_order:
            ordered = [g for g in group_order if g in working['group'].unique()]
            ordered += [g for g in working['group'].unique() if g not in ordered]
        else:
            ordered = list(working['group'].unique())

        if not ordered:
            print(f"    Skipping violin plot for '{metric}': no groups with data")
            return None

        # Collect raw values per group
        group_values = []
        valid_groups = []
        for g in ordered:
            vals = working[working['group'] == g][metric].values
            if len(vals) > 0:
                group_values.append(vals)
                valid_groups.append(g)

        if not valid_groups:
            print(f"    Skipping violin plot for '{metric}': no groups with data")
            return None

        fig, ax = plt.subplots(figsize=self._parse_figsize(plot_config.get('figsize')))

        if log_scale:
            ax.set_yscale('log')

        positions = np.arange(len(valid_groups))

        if group_colors is None:
            group_colors = {}
        colors = [
            group_colors.get(g, self.color_palette[i % len(self.color_palette)])
            for i, g in enumerate(valid_groups)
        ]

        show_means = plot_config.get('show_means', True)
        show_medians = plot_config.get('show_medians', True)
        show_extrema = plot_config.get('show_extrema', True)
        show_points = plot_config.get('show_points', False)
        point_size = plot_config.get('point_size', 3)
        cut = plot_config.get('cut', 0)
        bw_method = plot_config.get('bw_method', 'scott')
        alpha = plot_config.get('alpha', 0.7)
        edgecolor = plot_config.get('edgecolor', 'black')
        linewidth = plot_config.get('linewidth', 1)

        # Build violinplot kwargs
        vp_kwargs: Dict[str, Any] = {
            'positions': positions,
            'showmeans': show_means,
            'showmedians': show_medians,
            'showextrema': show_extrema,
            'bw_method': bw_method,
        }

        vp = ax.violinplot(group_values, **vp_kwargs)

        # Style each violin body individually
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(colors[i])
            body.set_alpha(alpha)
            body.set_edgecolor(edgecolor)
            body.set_linewidth(linewidth)

            # Clip to data range if cut=0
            if cut == 0 and i < len(group_values):
                v_min = group_values[i].min()
                v_max = group_values[i].max()
                for path in body.get_paths():
                    path.vertices[:, 1] = np.clip(path.vertices[:, 1], v_min, v_max)

        # Style the statistical lines
        for partname in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
            if partname in vp:
                vp[partname].set_edgecolor('black')
                vp[partname].set_linewidth(1)

        # Overlay individual data points if requested
        if show_points:
            for i, (pos, vals) in enumerate(zip(positions, group_values)):
                jitter = np.random.default_rng(42).uniform(-0.05, 0.05, size=len(vals))
                ax.scatter(
                    pos + jitter, vals,
                    s=point_size, color=colors[i], alpha=0.5, zorder=3, edgecolors='none',
                )

        if 'ylim' in plot_config:
            ax.set_ylim(plot_config['ylim'])

        ax.set_xticks(positions)
        ax.set_xticklabels(valid_groups, rotation=plot_config.get('xtick_rotation', 30), ha='right')
        ax.set_xlabel(plot_config.get('xlabel', ''))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f"{metric} — distribution"))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig

    def _create_conditioned_grouped_violin_plot(
        self,
        data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        condition_column: str,
        condition_values: Optional[List[Any]] = None,
        group_order: Optional[List[str]] = None,
        group_colors: Optional[Dict[str, str]] = None,
    ) -> Optional[plt.Figure]:
        """Create grouped violin plot with violins clustered by condition values.

        X-axis shows condition values (e.g., epochs), with clustered violins for each group.
        Y-axis shows the full distribution of metric values.
        """
        required_cols = {'group', metric, condition_column}
        if data.empty or not required_cols.issubset(data.columns):
            missing = required_cols - set(data.columns)
            print(f"    Skipping conditioned violin plot for '{metric}': missing columns {missing}")
            return None

        working = data[['group', condition_column, metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working[condition_column] = pd.to_numeric(working[condition_column], errors='coerce')
        working = working.dropna(subset=[metric, condition_column])

        if working.empty:
            print(f"    Skipping conditioned violin plot for '{metric}': no numeric values")
            return None

        log_scale = plot_config.get('log_scale', False)
        if log_scale:
            n_before = len(working)
            working = working[working[metric] > 0]
            n_dropped = n_before - len(working)
            if n_dropped > 0:
                print(f"    Warning: dropped {n_dropped} non-positive values for log-scale violin plot")
            if working.empty:
                print(f"    Skipping conditioned violin plot for '{metric}': no positive values for log scale")
                return None

        # Filter to specified condition values if provided
        if condition_values is not None:
            working = working[working[condition_column].isin(condition_values)]
            if working.empty:
                print(f"    Skipping conditioned violin plot: no data for specified condition values")
                return None
            conditions = [c for c in condition_values if c in working[condition_column].values]
        else:
            conditions = sorted(working[condition_column].unique())

        # Determine group order
        if group_order:
            groups = [g for g in group_order if g in working['group'].unique()]
            groups += [g for g in working['group'].unique() if g not in groups]
        else:
            groups = sorted(working['group'].unique())

        if not groups or not conditions:
            print(f"    Skipping conditioned violin plot: no groups or conditions")
            return None

        # Prepare figure
        figsize = self._parse_figsize(plot_config.get('figsize'))
        fig, ax = plt.subplots(figsize=figsize)

        if log_scale:
            ax.set_yscale('log')

        n_groups = len(groups)
        n_conditions = len(conditions)

        # Calculate violin width
        violin_width = plot_config.get('violin_width')
        if violin_width is None:
            violin_width = 0.8 / n_groups

        x_positions = np.arange(n_conditions)

        # Set up colors
        if group_colors is None:
            group_colors = {}

        alpha = plot_config.get('alpha', 0.7)
        edgecolor = plot_config.get('edgecolor', 'black')
        linewidth = plot_config.get('linewidth', 1)
        show_means = plot_config.get('show_means', True)
        show_medians = plot_config.get('show_medians', True)
        show_extrema = plot_config.get('show_extrema', True)
        show_points = plot_config.get('show_points', False)
        point_size = plot_config.get('point_size', 3)
        cut = plot_config.get('cut', 0)
        bw_method = plot_config.get('bw_method', 'scott')

        # Collect text annotation data
        text_annotations = []
        legend_patches = []

        # Plot violins for each group
        for i, group in enumerate(groups):
            offset = (i - (n_groups - 1) / 2) * violin_width
            color = group_colors.get(group, self.color_palette[i % len(self.color_palette)])

            group_data_per_cond = []
            group_positions = []
            for j, cond in enumerate(conditions):
                vals = working[(working[condition_column] == cond) & (working['group'] == group)][metric].values
                if len(vals) >= 2:  # Need at least 2 points for KDE
                    group_data_per_cond.append(vals)
                    group_positions.append(x_positions[j] + offset)

            if not group_data_per_cond:
                continue

            vp = ax.violinplot(
                group_data_per_cond,
                positions=group_positions,
                widths=violin_width * 0.9,
                showmeans=show_means,
                showmedians=show_medians,
                showextrema=show_extrema,
                bw_method=bw_method,
            )

            # Style each violin body and clip to data range if cut=0
            for body_idx, body in enumerate(vp['bodies']):
                body.set_facecolor(color)
                body.set_alpha(alpha)
                body.set_edgecolor(edgecolor)
                body.set_linewidth(linewidth)

                if cut == 0 and body_idx < len(group_data_per_cond):
                    v_min = group_data_per_cond[body_idx].min()
                    v_max = group_data_per_cond[body_idx].max()
                    for path in body.get_paths():
                        path.vertices[:, 1] = np.clip(path.vertices[:, 1], v_min, v_max)

            # Style statistical lines
            for partname in ('cmeans', 'cmedians', 'cmins', 'cmaxes', 'cbars'):
                if partname in vp:
                    vp[partname].set_edgecolor('black')
                    vp[partname].set_linewidth(1)

            # Overlay individual data points if requested
            if show_points:
                rng = np.random.default_rng(42 + i)
                for pos, vals in zip(group_positions, group_data_per_cond):
                    jitter = rng.uniform(-violin_width * 0.15, violin_width * 0.15, size=len(vals))
                    ax.scatter(
                        pos + jitter, vals,
                        s=point_size, color=color, alpha=0.5, zorder=3, edgecolors='none',
                    )

            # Collect show_values annotations
            if plot_config.get('show_values', False):
                for pos, vals in zip(group_positions, group_data_per_cond):
                    mean_val = np.mean(vals)
                    text_annotations.append((pos, mean_val))

            # Create legend patch for this group
            import matplotlib.patches as mpatches
            legend_patches.append(mpatches.Patch(facecolor=color, edgecolor=edgecolor, alpha=alpha, label=group))

        if not legend_patches:
            print(f"    Skipping conditioned violin plot for '{metric}': no plottable data")
            plt.close(fig)
            return None

        if 'ylim' in plot_config:
            ax.set_ylim(plot_config['ylim'])

        ax.set_xticks(x_positions)
        condition_labels = [str(c) for c in conditions]
        ax.set_xticklabels(condition_labels, rotation=plot_config.get('xtick_rotation', 0))
        ax.set_xlabel(plot_config.get('xlabel', condition_column))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f"{metric} by {condition_column}"))
        ax.legend(handles=legend_patches, loc=plot_config.get('legend_loc', 'best'))
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels after axes are configured
        if text_annotations:
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            padding = 0.02 * y_range if y_range > 0 else 0.01
            for x, mean_val in text_annotations:
                y_pos = mean_val + padding
                ax.text(
                    x, y_pos, f'{mean_val:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=90,
                )

        plt.tight_layout()
        self._apply_plot_label_visibility(fig, plot_config)
        return fig

    def _compute_group_final_values(
        self,
        data: pd.DataFrame,
        metric: str,
        step_column: str = '_step',
        plot_config: Dict[str, Any] = None,
        group_order: Optional[List[str]] = None,
    ) -> tuple[dict[str, list[float]], float]:
        """Compute per-run final values for each group."""
        if data.empty:
            return {}, float('nan')

        if step_column not in data.columns:
            print(f"    Warning: Step column '{step_column}' missing; cannot compute final values.")
            return {}, float('nan')

        working = data[['group', 'run_id', step_column, metric]].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working = working.dropna(subset=[metric, step_column])

        if working.empty:
            print(f"    Warning: No valid numeric data for metric '{metric}' to compute final values.")
            return {}, float('nan')

        if plot_config and 'xlim' in plot_config and len(plot_config['xlim']) > 1:
            target_x = plot_config['xlim'][1]
            print(f"    Using target x from config xlim: {target_x}")
        else:
            target_x = working[step_column].max()
            print(f"    Using max x from processed data: {target_x}")

        if group_order:
            groups = [g for g in group_order if g in working['group'].unique()]
            for g in working['group'].unique():
                if g not in groups:
                    groups.append(g)
        else:
            groups = list(working['group'].unique())

        final_values: dict[str, list[float]] = {}
        for group in groups:
            group_data = working[working['group'] == group]
            if group_data.empty:
                continue

            per_run_values: list[float] = []
            for run_id in group_data['run_id'].unique():
                run_data = group_data[group_data['run_id'] == run_id]
                if run_data.empty:
                    continue

                idx_closest = (run_data[step_column] - target_x).abs().idxmin()
                final_value = run_data.loc[idx_closest, metric]
                if pd.notna(final_value):
                    per_run_values.append(float(final_value))

            if per_run_values:
                final_values[group] = per_run_values

        return final_values, float(target_x)

    def _compute_group_auc_values(
        self,
        data: pd.DataFrame,
        metric: str,
        step_column: str = '_step',
        group_order: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """Compute area under the curve (AUC) for each run within every group."""
        if data.empty:
            return {}

        axis_candidates = []
        if step_column in data.columns:
            axis_candidates.append(step_column)
        if '__sync_step__' in data.columns and '__sync_step__' not in axis_candidates:
            axis_candidates.append('__sync_step__')
        if '_step' in data.columns and '_step' not in axis_candidates:
            axis_candidates.append('_step')

        axis_column = None
        for candidate in axis_candidates:
            numeric = pd.to_numeric(data[candidate], errors='coerce')
            if numeric.notna().any():
                axis_column = candidate
                break

        if axis_column is None:
            print("    Warning: No numeric axis available for AUC computation.")
            return {}

        working_columns = ['group', 'run_id', metric, axis_column]
        working = data[working_columns].copy()
        working[metric] = pd.to_numeric(working[metric], errors='coerce')
        working[axis_column] = pd.to_numeric(working[axis_column], errors='coerce')
        working = working.dropna(subset=[metric, axis_column])

        if working.empty:
            print("    Warning: No data available after cleaning for AUC computation.")
            return {}

        if group_order:
            groups = [g for g in group_order if g in working['group'].unique()]
            for g in working['group'].unique():
                if g not in groups:
                    groups.append(g)
        else:
            groups = list(working['group'].unique())

        auc_values: Dict[str, List[float]] = {}
        for group in groups:
            group_data = working[working['group'] == group]
            if group_data.empty:
                continue

            per_run_auc: List[float] = []
            for run_id in group_data['run_id'].unique():
                run_data = group_data[group_data['run_id'] == run_id].copy()
                if run_data.empty:
                    continue

                run_data = run_data.sort_values(axis_column)
                run_data = run_data.drop_duplicates(subset=[axis_column])
                if len(run_data) < 2:
                    continue

                auc = np.trapz(run_data[metric].values, run_data[axis_column].values)
                per_run_auc.append(float(auc))

            if per_run_auc:
                auc_values[group] = per_run_auc

        return auc_values

    def _extract_final_values_from_processed_data(
        self,
        data: pd.DataFrame,
        metric: str,
        step_column: str = '_step',
        plot_config: Dict[str, Any] = None,
        group_order: Optional[List[str]] = None,
        precomputed: Optional[tuple[dict[str, list[float]], float]] = None,
    ) -> str:
        """Extract final values directly from processed data (post-sync, post-smooth)."""
        if precomputed is not None:
            final_values, target_x = precomputed
        else:
            final_values, target_x = self._compute_group_final_values(
                data,
                metric,
                step_column,
                plot_config,
                group_order=group_order,
            )

        if not final_values:
            return ""

        final_results = []
        for group, values in final_values.items():
            if not values:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
            stderr_val = std_val / np.sqrt(len(values)) if values else 0.0

            final_results.append({
                'Group': group,
                'N Runs': len(values),
                'Final Step': target_x,
                'Mean': f"{mean_val:.4f}",
                'Std Error': f"{stderr_val:.4f}",
                'Mean ± SE': f"{mean_val:.4f} ± {stderr_val:.4f}"
            })

        if not final_results:
            return ""

        df_results = pd.DataFrame(final_results)
        return df_results.to_markdown(index=False)

    def _escape_latex(self, text: str) -> str:
        """Escape characters with special meaning in LaTeX."""
        replacements = {
            '\\': r'\textbackslash{}',
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        escaped = str(text)
        for key, value in replacements.items():
            escaped = escaped.replace(key, value)
        return escaped

    def _generate_ks_latex_table(
        self,
        data: Optional[pd.DataFrame],
        metric: str,
        figure_name: str,
        plot_config: Dict[str, Any],
        step_column: str,
        group_order: Optional[List[str]] = None,
        samples: Optional[Dict[str, List[float]]] = None,
        descriptor: Optional[str] = None,
        label_suffix: str = "_ks",
    ) -> str:
        """Generate LaTeX table with Kolmogorov-Smirnov test results."""
        if not self.ks_anchors:
            return ""

        try:
            from scipy.stats import ks_2samp
        except ImportError:
            print("Warning: scipy.stats.ks_2samp not available; skipping Kolmogorov-Smirnov tests.")
            return ""

        if samples is None:
            if data is None:
                print("    Warning: No data provided to compute KS samples.")
                return ""

            computed_samples, target_x = self._compute_group_final_values(
                data,
                metric,
                step_column,
                plot_config,
                group_order=group_order,
            )
            sample_map = computed_samples
            if descriptor is None:
                if np.isfinite(target_x):
                    descriptor = f"step {target_x:.0f}"
                else:
                    descriptor = "final step"
        else:
            sample_map = samples

        if not sample_map:
            print("    Warning: No samples available for KS tests.")
            return ""

        anchors = list(self.ks_anchors)
        anchor_presence = {anchor: anchor in sample_map and bool(sample_map[anchor]) for anchor in anchors}
        for anchor, present in anchor_presence.items():
            if not present:
                print(f"    Warning: Anchor '{anchor}' has no data for KS test.")

        if group_order:
            groups = [g for g in group_order if g in sample_map]
            for g in sample_map:
                if g not in groups:
                    groups.append(g)
        else:
            groups = list(sample_map.keys())

        table_rows: list[str] = []
        for group in groups:
            row_cells = []
            group_sample = sample_map.get(group, [])
            for anchor in anchors:
                anchor_sample = sample_map.get(anchor, [])

                if group == anchor:
                    row_cells.append('--')
                    continue

                if not group_sample or not anchor_sample:
                    row_cells.append('N/A')
                    continue

                try:
                    result = ks_2samp(
                        group_sample,
                        anchor_sample,
                        **self.ks_kwargs,
                    )
                    cell = f"$D={result.statistic:.4f},\\,p={result.pvalue:.3e}$"
                except Exception as exc:
                    print(f"    Warning: KS test failed for '{group}' vs '{anchor}': {exc}")
                    cell = 'error'

                row_cells.append(cell)

            escaped_group = self._escape_latex(group)
            table_rows.append(f"{escaped_group} & " + " & ".join(row_cells) + r" \\")

        if not table_rows:
            return ""

        escaped_anchors = [self._escape_latex(anchor) for anchor in anchors]
        column_spec = 'l' + 'c' * len(anchors)
        header = "Group & " + " & ".join(escaped_anchors) + r" \\"

        descriptor = descriptor or "distribution"
        escaped_descriptor = self._escape_latex(descriptor)
        caption = (
            f"Kolmogorov--Smirnov tests for {self._escape_latex(figure_name)} "
            f"({self._escape_latex(metric)}) — {escaped_descriptor}"
        )

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\begin{{tabular}}{{{column_spec}}}",
            r"\hline",
            header,
            r"\hline",
            *table_rows,
            r"\hline",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
        ]

        safe_label = f"{figure_name}_{metric}{label_suffix}".replace(' ', '_')
        safe_label = ''.join(ch for ch in safe_label if ch.isalnum() or ch in {'_', '-'})
        lines.append(rf"\label{{tab:{safe_label}}}")
        lines.append(r"\end{table}")

        return "\n".join(lines) + "\n"

    def _generate_final_results_table(self, stats_data: pd.DataFrame, metric: str, 
                                    step_column: str = '_step') -> str:
        """Generate markdown table with final results (mean ± stderr) for each group."""
        if stats_data.empty:
            return ""
        
        # Get the final (maximum step) results for each group
        final_results = []
        for group in stats_data['group'].unique():
            group_data = stats_data[stats_data['group'] == group]
            if group_data.empty:
                continue
            
            # Get the row with maximum step value for this group
            final_row = group_data.loc[group_data[step_column].idxmax()]
            final_results.append({
                'Group': group,
                'Final Step': int(final_row[step_column]),
                'Mean': f"{final_row['mean']:.4f}",
                'Std Error': f"{final_row['stderr']:.4f}",
                'Mean ± SE': f"{final_row['mean']:.4f} ± {final_row['stderr']:.4f}"
            })
        
        # Convert to markdown table
        if not final_results:
            return ""
        
        df_results = pd.DataFrame(final_results)
        return df_results.to_markdown(index=False)
    
    def generate_figures(self):
        """Generate all figures based on configuration."""
        output_dir = Path(self.config.get('output_dir', 'figures'))
        output_dir.mkdir(exist_ok=True)
        
        # Collect all unique metrics and step columns needed
        '''
        all_metrics = set()
        all_step_columns = set()
        
        for figure_config in self.config['figures']:
            all_metrics.update(figure_config['metrics'])
            all_step_columns.add(figure_config.get('step_column', '_step'))
        '''
        all_metrics, all_step_columns = self._collect_all_required_metrics()

        print(f"Required metrics: {list(all_metrics)}")
        print(f"Required step columns: {list(all_step_columns)}")
        
        # Collect run requirements with project optimization
        run_requirements = self._collect_run_requirements()
        
        # Fetch runs efficiently
        all_runs = {} #mapping from eff_id to runs
        
        # Fetch runs that could be in any project
        if run_requirements['all_projects_runs']:
            print(f"\nFetching {len(run_requirements['all_projects_runs'])} runs from all projects...")
            all_projects_runs = self._fetch_specific_runs(
                run_requirements['all_projects_runs'], 
                self._get_all_projects()
            )
            for eff_run_id, run in all_projects_runs.items():
                all_runs[eff_run_id] = run
        
        # Fetch runs from specific projects
        for project, run_ids in tqdm(run_requirements['project_specific_runs'].items()):
            print(f"\nFetching {len(run_ids)} runs specifically from project {project}...")
            project_runs = self._fetch_specific_runs(run_ids, [project])
            for eff_run_id, run in project_runs.items():
                all_runs[eff_run_id] = run
        
        if not all_runs:
            print("No runs found.")
            return
        
        print(f"\nTotal runs found: {len(all_runs)}")
        
        # Extract data from all runs
        all_data = []
        for eff_run_id, run in tqdm(all_runs.items()):
            run_data = self._extract_run_data(run, list(all_metrics), list(all_step_columns), eff_run_id=eff_run_id)
            if run_data.empty:
                print(f"Run {eff_run_id} is empty. Skipping.")
            else:
                all_data.append(run_data)
        
        if not all_data:
            print("No data extracted from runs.")
            return
        
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Extracted data: {len(combined_data)} rows from {len(all_runs)} runs")

        # Generate figures
        for figure_config in self.config['figures']:
            print(f"\nGenerating figure: {figure_config['name']}")
            
            # Prepare data for this figure
            figure_data = []
            group_order = []
            group_colors = {}  # Maps group name -> custom color (if specified)
            pre_sync_dir = output_dir / "pre_sync_diagnostics" / figure_config['name']
            step_column = figure_config.get('step_column', '_step')

            # Get figure-level default filter
            figure_filter = figure_config.get('data_filter', None)

            for group_config in figure_config['groups']:
                group_data = self._group_runs_data(
                    combined_data,
                    group_config,
                    figure_filter=figure_filter,
                    output_dir=output_dir,
                    figure_name=figure_config['name'],
                    figure_config=figure_config,
                )
                if not group_data.empty:
                    print(f"Group '{group_config['name']}' raw data stats:")
                    for metric in figure_config['metrics']:
                        print(f"  Metric '{metric}' range: {group_data[metric].min()} to {group_data[metric].max()}")
                        print(f"  Non-zero values: {(group_data[metric] != 0).sum()}/{len(group_data)}")
                        print(f"  Sample values: {group_data[metric].dropna().head().tolist()}")
                    unique_runs = len(group_data['run_id'].unique())
                    print(f"  Group '{group_config['name']}': {unique_runs} runs")
                    
                    # Show which projects the runs came from
                    if 'project' in group_config:
                        projects = group_config['project']
                        if isinstance(projects, str):
                            projects = [projects]
                        print(f"    Specified projects: {projects}")
                    
                    figure_data.append(group_data)
                    group_order.append(group_config['name'])

                    # Use custom color if specified, otherwise use palette
                    if 'color' in group_config:
                        group_color = group_config['color']
                    else:
                        color_idx = (len(group_order) - 1) % len(self.color_palette)
                        group_color = self.color_palette[color_idx]

                    # Apply lightness adjustment if specified
                    if 'lightness' in group_config:
                        group_color = self._apply_lightness(group_color, group_config['lightness'])
                        print(f"    Applied lightness={group_config['lightness']} to '{group_config['name']}'")

                    group_colors[group_config['name']] = group_color

                    for metric in figure_config['metrics']:
                        if metric not in group_data.columns:
                            continue
                        raw_stats = self._calculate_statistics(group_data, metric, step_column)
                        raw_stats = raw_stats[raw_stats['group'] == group_config['name']]
                        if raw_stats.empty:
                            continue
                        
                        diagnostic_plot_config = figure_config.get('lineplot_config', {})
                        diag_filename = (
                            f"{group_config['name']}_{metric}_pre_sync."
                            f"{self.config.get('output_format', 'png')}"
                        )
                        diag_path = pre_sync_dir / diag_filename.replace('/','+')
                        self._create_pre_sync_group_plot(
                            raw_stats,
                            group_config['name'],
                            metric,
                            step_column,
                            diagnostic_plot_config,
                            group_color,
                            diag_path,
                        )
                else:
                    print(f"  Group '{group_config['name']}': No data found")
            
            if not figure_data:
                print(f"No data for figure {figure_config['name']}")
                continue
            
            figure_combined = pd.concat(figure_data, ignore_index=True)
            raw_figure_combined = figure_combined.copy()

            barplot_cfg = figure_config.get('barplot_config')
            barplot_enabled = False
            barplot_settings: Dict[str, Any] = {}
            if isinstance(barplot_cfg, bool):
                barplot_enabled = barplot_cfg
            elif isinstance(barplot_cfg, dict):
                barplot_enabled = barplot_cfg.get('enabled', True)
                barplot_settings = {k: v for k, v in barplot_cfg.items() if k != 'enabled'}

            # Parse violinplot_config
            violin_cfg = figure_config.get('violinplot_config')
            violin_enabled = False
            violin_settings: Dict[str, Any] = {}
            if isinstance(violin_cfg, bool):
                violin_enabled = violin_cfg
            elif isinstance(violin_cfg, dict):
                violin_enabled = violin_cfg.get('enabled', True)
                violin_settings = {k: v for k, v in violin_cfg.items() if k != 'enabled'}

            # Apply synchronization if specified
            sync_config = figure_config.get('synchronization', {})
            if sync_config:
                print(f"  Applying synchronization: {sync_config.get('method', 'interpolate')}")
                figure_combined = self._synchronize_data(
                    figure_combined, 
                    sync_config, 
                    figure_config.get('step_column', '_step')
                )
            
            # Generate plots for each metric
            for metric in figure_config['metrics']:
                print(f"  Plotting metric: {metric}")
                
                if metric not in figure_combined.columns:
                    print(f"    Metric {metric} not found in data")
                    continue

                raw_plot_config = figure_config.get('lineplot_config', {}) or {}
                plot_config = dict(raw_plot_config)
                
                # Apply smoothing if specified
                smoothing_config = figure_config.get('smoothing', {})
                if smoothing_config:
                    print(f"  Applying smoothing to {metric}")
                    figure_combined = self._apply_smoothing(
                        figure_combined,
                        smoothing_config,
                        metric,
                        figure_config.get('step_column', '_step')
                    )

                final_values, target_x = self._compute_group_final_values(
                    figure_combined,
                    metric,
                    figure_config.get('step_column', '_step'),
                    plot_config,
                    group_order=group_order,
                )

                if final_values:
                    if np.isfinite(target_x):
                        descriptor = f"step {target_x:.0f}"
                    else:
                        descriptor = "final step"

                    ks_table = self._generate_ks_latex_table(
                        None,
                        metric,
                        figure_config['name'],
                        plot_config,
                        figure_config.get('step_column', '_step'),
                        group_order=group_order,
                        samples=final_values,
                        descriptor=descriptor,
                        label_suffix="_ks",
                    )
                    if ks_table:
                        ks_filename = f"{figure_config['name']}_{metric}_ks.tex"
                        ks_path = output_dir / ks_filename.replace('/','+')
                        with open(ks_path, 'w') as f:
                            f.write(ks_table)
                        print(f"    Saved KS table: {ks_path}")

                auc_samples = self._compute_group_auc_values(
                    figure_combined,
                    metric,
                    figure_config.get('step_column', '_step'),
                    group_order=group_order,
                )
                if auc_samples:
                    auc_ks_table = self._generate_ks_latex_table(
                        None,
                        metric,
                        figure_config['name'],
                        plot_config,
                        figure_config.get('step_column', '_step'),
                        group_order=group_order,
                        samples=auc_samples,
                        descriptor="Area under curve (AUC)",
                        label_suffix="_ks_auc",
                    )
                    if auc_ks_table:
                        auc_filename = f"{figure_config['name']}_{metric}_ks_auc.tex"
                        auc_path = output_dir / auc_filename.replace('/','+')
                        with open(auc_path, 'w') as f:
                            f.write(auc_ks_table)
                        print(f"    Saved KS AUC table: {auc_path}")

                aggregate_on_sync_axis = bool(sync_config.get('aggregate_on_sync_axis', False))
                group_axis_column = '__sync_step__' if aggregate_on_sync_axis and '__sync_step__' in figure_combined.columns else None
                if aggregate_on_sync_axis and group_axis_column is None:
                    print("    Warning: aggregate_on_sync_axis requested but sync axis column not found; using step_column instead.")

                stats = self._calculate_statistics(
                    figure_combined, 
                    metric, 
                    figure_config.get('step_column', '_step'),
                    group_axis=group_axis_column,
                )
                
                if stats.empty:
                    print(f"    No statistics calculated for {metric}")
                    continue
                
                plot_config['ylabel'] = plot_config.get('ylabel', metric)
                plot_config['title'] = plot_config.get('title', f"{figure_config['name']}: {metric}")

                # Get legend export configuration
                legend_export_config = self._get_legend_export_config(figure_config)

                fig, handles, labels = self._create_line_plot(
                    stats, metric, plot_config,
                    figure_config.get('step_column', '_step'),
                    group_order=group_order,
                    group_axis=group_axis_column,
                    group_colors=group_colors,
                )

                count_fig, count_handles, count_labels = self._create_count_plot(
                    stats,
                    metric,
                    plot_config,
                    figure_config.get('step_column', '_step'),
                    group_order=group_order,
                    group_axis=group_axis_column,
                    group_colors=group_colors,
                )

                if barplot_enabled:
                    bar_fig = self._create_group_bar_plot(
                        raw_figure_combined,
                        metric,
                        barplot_settings,
                        group_order=group_order,
                        group_colors=group_colors,
                    )
                else:
                    bar_fig = None

                # Handle legend export for line plot
                if legend_export_config.get('enabled', False) and handles and labels:
                    legend_format = legend_export_config.get('format', 'pdf')
                    legend_filename = f"{figure_config['name']}_{metric}_legend.{legend_format}"
                    legend_path = output_dir / legend_filename.replace('/', '+')
                    self._export_legend_to_file(handles, labels, legend_path, legend_export_config)

                    # Apply modifications to main figure (remove legend, hide labels/title)
                    self._apply_legend_export_modifications(fig, legend_export_config)

                filename = f"{figure_config['name']}_{metric}.{self.config.get('output_format', 'png')}"
                filepath = output_dir / filename.replace('/','+')
                fig.savefig(filepath, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                plt.close(fig)

                # Apply legend export modifications to count plot if enabled
                if legend_export_config.get('enabled', False):
                    self._apply_legend_export_modifications(count_fig, legend_export_config)

                count_filename = f"{figure_config['name']}_{metric}_counts.{self.config.get('output_format', 'png')}"
                count_path = output_dir / count_filename.replace('/','+')
                count_fig.savefig(count_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                plt.close(count_fig)

                if bar_fig is not None:
                    # Apply legend export modifications if enabled
                    if legend_export_config.get('enabled', False):
                        self._apply_legend_export_modifications(bar_fig, legend_export_config)

                    bar_filename = f"{figure_config['name']}_{metric}_barplot.{self.config.get('output_format', 'png')}"
                    bar_path = output_dir / bar_filename.replace('/','+')
                    bar_fig.savefig(bar_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                    plt.close(bar_fig)
                    print(f"    Saved bar plot: {bar_path}")

                # Handle conditioned grouped bar plot
                cond_barplot_cfg = figure_config.get('conditioned_barplot_config')
                if cond_barplot_cfg and cond_barplot_cfg.get('enabled', True):
                    condition_column = cond_barplot_cfg.get('condition_column')
                    if condition_column:
                        cond_barplot_settings = {
                            k: v for k, v in cond_barplot_cfg.items()
                            if k not in ('enabled', 'condition_column', 'condition_values')
                        }
                        condition_values = cond_barplot_cfg.get('condition_values')

                        cond_bar_fig = self._create_conditioned_grouped_bar_plot(
                            raw_figure_combined,
                            metric,
                            cond_barplot_settings,
                            condition_column=condition_column,
                            condition_values=condition_values,
                            group_order=group_order,
                            group_colors=group_colors,
                        )

                        if cond_bar_fig is not None:
                            # Apply legend export modifications if enabled
                            if legend_export_config.get('enabled', False):
                                self._apply_legend_export_modifications(cond_bar_fig, legend_export_config)

                            cond_bar_filename = f"{figure_config['name']}_{metric}_conditioned_barplot.{self.config.get('output_format', 'png')}"
                            cond_bar_path = output_dir / cond_bar_filename.replace('/', '+')
                            cond_bar_fig.savefig(cond_bar_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                            plt.close(cond_bar_fig)
                            print(f"    Saved conditioned bar plot: {cond_bar_path}")
                    else:
                        print("    Warning: conditioned_barplot_config enabled but 'condition_column' not specified")

                # Handle simple violin plot
                if violin_enabled:
                    violin_fig = self._create_group_violin_plot(
                        raw_figure_combined,
                        metric,
                        violin_settings,
                        group_order=group_order,
                        group_colors=group_colors,
                    )
                else:
                    violin_fig = None

                if violin_fig is not None:
                    # Apply legend export modifications if enabled
                    if legend_export_config.get('enabled', False):
                        self._apply_legend_export_modifications(violin_fig, legend_export_config)

                    violin_filename = f"{figure_config['name']}_{metric}_violinplot.{self.config.get('output_format', 'png')}"
                    violin_path = output_dir / violin_filename.replace('/', '+')
                    violin_fig.savefig(violin_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                    plt.close(violin_fig)
                    print(f"    Saved violin plot: {violin_path}")

                # Handle conditioned grouped violin plot
                cond_violin_cfg = figure_config.get('conditioned_violinplot_config')
                if cond_violin_cfg and cond_violin_cfg.get('enabled', True):
                    condition_column = cond_violin_cfg.get('condition_column')
                    if condition_column:
                        cond_violin_settings = {
                            k: v for k, v in cond_violin_cfg.items()
                            if k not in ('enabled', 'condition_column', 'condition_values')
                        }
                        condition_values = cond_violin_cfg.get('condition_values')

                        cond_violin_fig = self._create_conditioned_grouped_violin_plot(
                            raw_figure_combined,
                            metric,
                            cond_violin_settings,
                            condition_column=condition_column,
                            condition_values=condition_values,
                            group_order=group_order,
                            group_colors=group_colors,
                        )

                        if cond_violin_fig is not None:
                            # Apply legend export modifications if enabled
                            if legend_export_config.get('enabled', False):
                                self._apply_legend_export_modifications(cond_violin_fig, legend_export_config)

                            cond_violin_filename = f"{figure_config['name']}_{metric}_conditioned_violinplot.{self.config.get('output_format', 'png')}"
                            cond_violin_path = output_dir / cond_violin_filename.replace('/', '+')
                            cond_violin_fig.savefig(cond_violin_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                            plt.close(cond_violin_fig)
                            print(f"    Saved conditioned violin plot: {cond_violin_path}")
                    else:
                        print("    Warning: conditioned_violinplot_config enabled but 'condition_column' not specified")

                # Export plot data to JSON
                json_filename = f"{figure_config['name']}_{metric}_plot_data.json"
                json_path = output_dir / json_filename.replace('/', '+')
                self._export_plot_data_to_json(
                    stats,
                    metric,
                    figure_config.get('step_column', '_step'),
                    json_path,
                    group_order=group_order,
                    group_axis=group_axis_column,
                )

                # Generate and save markdown table
                # Extract final values from processed data (includes sync + smoothing effects)
                markdown_table = self._extract_final_values_from_processed_data(
                    figure_combined, 
                    metric, 
                    figure_config.get('step_column', '_step'),
                    plot_config,
                    group_order=group_order,
                    precomputed=(final_values, target_x) if final_values else None,
                )
                
                # Alternative: use statistics-based table (commented out)
                # markdown_table = self._generate_final_results_table(
                #    stats, metric, figure_config.get('step_column', '_step')
                #)
                
                if markdown_table:
                    table_filename = f"{figure_config['name']}_{metric}_results.md"
                    table_filepath = output_dir / table_filename.replace('/','+')
                    with open(table_filepath, 'w') as f:
                        f.write(f"# Final Results: {figure_config['name']} - {metric}\n\n")
                        f.write(markdown_table)
                    print(f"    Saved table: {table_filepath}")
                
                print(f"    Saved: {filepath}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Minimal W&B Figure Generator with Caching")
    parser.add_argument('--config', '-c', type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument('--no-cache', action='store_true', help="Disable caching (always fetch from W&B)")
    parser.add_argument('--clear-cache', action='store_true', help="Clear all cached data")
    parser.add_argument('--list-cache', action='store_true', help="List cached data")
    parser.add_argument('--cache-dir', type=str, help="Custom cache directory")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MinimalWandBGenerator(
        args.config, 
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache
    )
    
    if args.clear_cache:
        generator._clear_cache()
        print("Cache cleared.")
        return
    
    if args.list_cache:
        generator._list_cache_contents()
        return
    
    generator.generate_figures()


if __name__ == "__main__":
    main()
