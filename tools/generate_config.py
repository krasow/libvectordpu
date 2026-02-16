#!/usr/bin/env python3
"""
Generate config.h from build.config
Converts key=value pairs into C preprocessor macros
"""

import os
import sys
from pathlib import Path


def parse_build_config(config_path):
    """Parse build.config and return a dictionary of config values."""
    config = {}
    try:
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    except FileNotFoundError:
        print(f"Error: {config_path} not found", file=sys.stderr)
        sys.exit(1)
    return config


def generate_config_h(config, output_path):
    """Generate config.h from configuration dictionary."""
    header_guard = "CONFIG_H"
    
    lines = [
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "/* Auto-generated from build.config */",
        "",
    ]
    
    for key, value in config.items():
        # Convert to macro format (uppercase, spaces to underscores)
        macro_name = key.upper().replace(' ', '_').replace('-', '_')
        
        # Determine if value is numeric or string
        if value.isdigit():
            # Numeric value
            lines.append(f"#ifndef {macro_name}")
            lines.append(f"#define {macro_name} {value}")
            lines.append(f"#endif")
        elif value.lower() in ('true', 'false', 'yes', 'no', 'on', 'off'):
            # Boolean value - convert to 0 or 1
            bool_value = 1 if value.lower() in ('true', 'yes', 'on', '1') else 0
            lines.append(f"#ifndef {macro_name}")
            lines.append(f"#define {macro_name} {bool_value}")
            lines.append(f"#endif")
        else:
            # String value
            lines.append(f"#ifndef {macro_name}")
            lines.append(f"#define {macro_name} \"{value}\"")
            lines.append(f"#endif")
    
    lines.extend([
        "",
        f"#endif /* {header_guard} */",
    ])
    
    try:
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Generated {output_path}")
    except Exception as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    # Get paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    build_config = repo_root / "build.config"
    output_config_h = repo_root / "common" / "config.h"
    
    # Parse and generate
    config = parse_build_config(str(build_config))
    generate_config_h(config, str(output_config_h))


if __name__ == "__main__":
    main()
