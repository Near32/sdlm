#!/usr/bin/env python3
"""
Minimal W&B Figure Generator
Focuses on run ID specification and synchronization with local caching.
"""
from typing import Tuple, Dict, Set, Optional
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

    def _collect_all_required_metrics(self) -> Tuple[Set[str], Set[str]]:
        """
        Collect all unique metrics and step columns needed across all figures and groups.
        Returns (all_metrics, all_step_columns)
        """
        all_metrics = set()
        all_step_columns = set()
        
        for figure_config in self.config['figures']:
            # Add global figure metrics
            figure_metrics = set(figure_config['metrics'])
            all_metrics.update(figure_metrics)
            
            # Add step column
            all_step_columns.add(figure_config.get('step_column', '_step'))
            
            # Add group-specific metric aliases
            for group_config in figure_config['groups']:
                if 'metric_aliases' in group_config:
                    # Add the actual metric names used by this group
                    group_specific_metrics = set(group_config['metric_aliases'].values())
                    all_metrics.update(group_specific_metrics)
                else:
                    # No aliases, so group uses the same metrics as figure
                    all_metrics.update(figure_metrics)
        
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
        """Determine which specific metrics to fetch for a given run based on its group aliases."""
        for figure_config in self.config['figures']:
            for group_config in figure_config['groups']:
                run_ids = group_config['run_ids']
                if isinstance(run_ids, str):
                    run_ids = [run_ids]
                
                normalized_run_ids = {self._normalize_run_id(rid) for rid in run_ids}
                if self._normalize_run_id(run_id) in normalized_run_ids:
                    if 'metric_aliases' in group_config:
                        # Use group-specific metric names (values in aliases)
                        return list(group_config['metric_aliases'].values())
                    else:
                        # Use global metric names
                        return figure_config['metrics']
        
        # Fallback: return all global metrics if run not found in any group
        all_global_metrics = set()
        for figure_config in self.config['figures']:
            all_global_metrics.update(figure_config['metrics'])
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
        run_specific_metrics = self._get_metrics_for_run(run.id)
        
        # Try cache first
        cached_data = self._load_run_data_from_cache(run.id, run_specific_metrics, step_columns)
        #if 'dzo' in run.id or 'daa' in run.id:
        #    import ipdb; ipdb.set_trace()
        if not cached_data.empty:
            return cached_data
    
        print(f"    Extracting from W&B: run {run.id}")
        columns_to_extract = list(set(run_specific_metrics + step_columns))
        
        # Try unified extraction first (faster)
        df = self._extract_unified_with_fallback(run, columns_to_extract, max_retry)
        
        if df.empty:
            return pd.DataFrame()
        
        # Add metadata
        effective_run_id = eff_run_id if eff_run_id else run.id
        df['run_id'] = effective_run_id
        df['group'] = ''
        
        # Cache result
        self._save_run_data_to_cache(effective_run_id, metrics, step_columns, df)
        return df
    
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
        while retry < max_retry:
            try:
                if True: #retry > 0:
                    history = run.history(keys=columns_to_extract, samples=10000)
                    df = pd.DataFrame(history) #if history else pd.DataFrame()
                else:
                    history = run.scan_history(keys=columns_to_extract)
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
        while retry < max_retry:
            try:
                if True: #retry > 0:
                    history = run.history(keys=[column], samples=5000)  # Reduced samples for speed
                    df = pd.DataFrame(history) #if history else pd.DataFrame()
                else:
                    history = run.scan_history(keys=[column])
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
    
    def _group_runs_data(self, data: pd.DataFrame, group_config: Dict[str, Any]) -> pd.DataFrame:
        """Filter data by run IDs and assign group."""
        group_name = group_config['name']
        run_ids = group_config['run_ids']
        
        if isinstance(run_ids, str):
            run_ids = [run_ids]
        
        # Filter by run IDs
        normalized_run_ids = {self._normalize_run_id(rid) for rid in run_ids}
        filtered_data = data[data['run_id'].isin(normalized_run_ids)].copy()
        
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

    def _create_line_plot(
        self, 
        stats_data: pd.DataFrame, 
        metric: str, 
        plot_config: Dict[str, Any], 
        step_column: str = '_step',
        group_order: list[str]=None,
        group_axis: Optional[str]=None,
    ) -> plt.Figure:
        """Create line plot with mean ± standard error."""
        fig, ax = plt.subplots(figsize=plot_config.get('figsize', (10, 6)))
        
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
            
            # Get color from configured palette - index matches config order
            color_idx = i % len(self.color_palette)
            color = self.color_palette[color_idx]
            
            ax.plot(x, y, label=group, linewidth=2, alpha=0.8, color=color)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.5, color=color)

        ax.set_xlabel(plot_config.get('xlabel', step_column))
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f'{metric} over {step_column}'))
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
        return fig
    
    def _create_count_plot(
        self,
        stats_data: pd.DataFrame,
        metric: str,
        plot_config: Dict[str, Any],
        step_column: str,
        group_order: list[str] = None,
        group_axis: Optional[str] = None,
    ) -> plt.Figure:
        """Plot the number of contributing runs per datapoint for each group."""
        fig, ax = plt.subplots(figsize=plot_config.get('figsize', (10, 6)))
        
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
        ax.legend(loc=plot_config.get('legend_loc', 'best'))
        ax.grid(True, alpha=0.3)
        
        if 'xlim' in plot_config:
            ax.set_xlim(plot_config['xlim'])
        if plot_config.get('xscale') == 'log':
            ax.set_xscale('log')
        
        plt.tight_layout()
        return fig
    
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
        fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', (10, 6)))
        
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

        fig, ax = plt.subplots(figsize=plot_config.get('figsize', (8, 5)))
        positions = np.arange(len(ordered))
        colors = [self.color_palette[i % len(self.color_palette)] for i in range(len(ordered))]

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

        ax.set_xticks(positions)
        ax.set_xticklabels(ordered, rotation=plot_config.get('xtick_rotation', 30), ha='right')
        ax.set_ylabel(plot_config.get('ylabel', metric))
        ax.set_title(plot_config.get('title', f"{metric} — mean ± SE"))
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
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
            pre_sync_dir = output_dir / "pre_sync_diagnostics" / figure_config['name']
            step_column = figure_config.get('step_column', '_step')
            for group_config in figure_config['groups']:
                group_data = self._group_runs_data(combined_data, group_config)
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
                    
                    color_idx = (len(group_order) - 1) % len(self.color_palette)
                    group_color = self.color_palette[color_idx]


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

                raw_plot_config = figure_config.get('plot_config', {}) or {}
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
                
                fig = self._create_line_plot(
                    stats, metric, plot_config, 
                    figure_config.get('step_column', '_step'),
                    group_order=group_order,
                    group_axis=group_axis_column,
                )
                
                count_fig = self._create_count_plot(
                    stats,
                    metric,
                    plot_config,
                    figure_config.get('step_column', '_step'),
                    group_order=group_order,
                    group_axis=group_axis_column,
                )

                if barplot_enabled:
                    bar_fig = self._create_group_bar_plot(
                        raw_figure_combined,
                        metric,
                        barplot_settings,
                        group_order=group_order,
                    )
                else:
                    bar_fig = None
                
                filename = f"{figure_config['name']}_{metric}.{self.config.get('output_format', 'png')}"
                filepath = output_dir / filename.replace('/','+')
                fig.savefig(filepath, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                plt.close(fig)
                
                count_filename = f"{figure_config['name']}_{metric}_counts.{self.config.get('output_format', 'png')}"
                count_path = output_dir / count_filename.replace('/','+')
                count_fig.savefig(count_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                plt.close(count_fig)

                if bar_fig is not None:
                    bar_filename = f"{figure_config['name']}_{metric}_barplot.{self.config.get('output_format', 'png')}"
                    bar_path = output_dir / bar_filename.replace('/','+')
                    bar_fig.savefig(bar_path, dpi=self.config.get('dpi', 300), bbox_inches='tight')
                    plt.close(bar_fig)
                    print(f"    Saved bar plot: {bar_path}")
                
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
