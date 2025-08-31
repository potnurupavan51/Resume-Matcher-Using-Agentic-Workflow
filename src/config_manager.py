"""
Configuration Manager for Resume Matcher
Handles dynamic updates to config.py file
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

def update_config_top_n(new_top_n: int, config_file_path: str = "config.py") -> bool:
    """
    Updates the top_n value in config.py file dynamically.
    
    Args:
        new_top_n (int): The new top_n value to set
        config_file_path (str): Path to the config.py file
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Validate input
        if not isinstance(new_top_n, int) or new_top_n <= 0:
            logger.error(f"Invalid top_n value: {new_top_n}. Must be a positive integer.")
            return False
            
        # Check if config file exists
        if not os.path.exists(config_file_path):
            logger.error(f"Config file not found: {config_file_path}")
            return False
            
        # Read the current config file
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_content = file.read()
            
        # Update top_n value using regex
        # Pattern to match: top_n=<any_number> #this is a final selection value
        top_n_pattern = r'(top_n\s*=\s*)\d+(\s*#.*final selection value.*)'
        
        if re.search(top_n_pattern, config_content):
            # Replace the top_n value
            updated_content = re.sub(
                top_n_pattern, 
                rf'\g<1>{new_top_n}\g<2>', 
                config_content
            )
            
            # Update top_k_retrieved_count as well (it depends on top_n)
            # Pattern to match: top_k_retrieved_count = top_n*4
            top_k_pattern = r'(top_k_retrieved_count\s*=\s*top_n\s*\*\s*)\d+(\s*#.*rag.*value.*)'
            updated_content = re.sub(
                top_k_pattern,
                rf'\g<1>4\g<2>',
                updated_content
            )
            
            # Write the updated content back to the file
            with open(config_file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
                
            logger.info(f"✅ Successfully updated top_n to {new_top_n} in {config_file_path}")
            logger.info(f"✅ top_k_retrieved_count automatically updated to {new_top_n * 4}")
            return True
            
        else:
            logger.error("Could not find top_n pattern in config file")
            return False
            
    except Exception as e:
        logger.error(f"Error updating config file: {str(e)}")
        return False

def get_current_top_n(config_file_path: str = "config.py") -> int:
    """
    Retrieves the current top_n value from config.py
    
    Args:
        config_file_path (str): Path to the config.py file
        
    Returns:
        int: Current top_n value, or 5 as default if not found
    """
    try:
        if not os.path.exists(config_file_path):
            logger.warning(f"Config file not found: {config_file_path}, using default top_n=5")
            return 5
            
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config_content = file.read()
            
        # Extract top_n value using regex
        top_n_match = re.search(r'top_n\s*=\s*(\d+)', config_content)
        
        if top_n_match:
            return int(top_n_match.group(1))
        else:
            logger.warning("Could not find top_n in config file, using default value 5")
            return 5
            
    except Exception as e:
        logger.error(f"Error reading config file: {str(e)}, using default top_n=5")
        return 5

def validate_top_n_input(user_input: str) -> tuple[bool, int]:
    """
    Validates user input for top_n value
    
    Args:
        user_input (str): User input string
        
    Returns:
        tuple[bool, int]: (is_valid, parsed_value)
    """
    try:
        if not user_input.strip():
            return False, 0
            
        value = int(user_input.strip())
        
        if value <= 0:
            return False, 0
            
        if value > 50:  # Reasonable upper limit
            return False, 0
            
        return True, value
        
    except ValueError:
        return False, 0
