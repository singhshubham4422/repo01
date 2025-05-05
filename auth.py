import os
import json
import hashlib
from typing import Dict, Tuple, Optional

# User data file path
USERS_FILE = "users.json"

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_default_user():
    """Create a default user if no users exist."""
    if not os.path.exists(USERS_FILE) or os.path.getsize(USERS_FILE) == 0:
        default_users = {
            "user": {
                "password": hash_password("password"),
                "created_at": "0"
            }
        }
        with open(USERS_FILE, 'w') as f:
            json.dump(default_users, f)

def load_users() -> Dict:
    """Load users from the JSON file."""
    # Create default user if needed
    create_default_user()
    
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_users(users: Dict) -> None:
    """Save users to the JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def register_user(username: str, password: str) -> Tuple[bool, str]:
    """
    Register a new user.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        Tuple of (success, message)
    """
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists"
    
    # Add the new user
    users[username] = {
        "password": hash_password(password),
        "created_at": str(os.path.getmtime(USERS_FILE) if os.path.exists(USERS_FILE) else 0)
    }
    
    # Save users
    save_users(users)
    
    return True, "User registered successfully"

def authenticate(username: str, password: str) -> Tuple[bool, str]:
    """
    Authenticate a user.
    
    Args:
        username: Username
        password: Password
        
    Returns:
        Tuple of (success, message)
    """
    users = load_users()
    
    # Check if username exists
    if username not in users:
        return False, "Invalid username or password"
    
    # Check if password is correct
    if users[username]["password"] != hash_password(password):
        return False, "Invalid username or password"
    
    return True, "Authentication successful"

def get_user_data(username: str) -> Optional[Dict]:
    """
    Get user data.
    
    Args:
        username: Username
        
    Returns:
        User data or None if user does not exist
    """
    users = load_users()
    
    if username not in users:
        return None
    
    return users[username]