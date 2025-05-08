#!/usr/bin/env python3
"""
Utility script to change a user's password in the API backend.
"""

import argparse
import os
import sys

# Add the parent directory to the path so we can import from the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the API package
from models import User
from auth import get_password_hash
from db import get_db


def change_password(identifier, new_password):
    """
    Change a user's password in the database.

    Args:
        identifier (str): User identifier (phone number or ID)
        new_password (str): New password to set

    Returns:
        User: The updated user object or None if user not found
    """
    try:
        # Get database session
        db = next(get_db())

        # Determine if identifier is a phone number or ID
        user = None
        if identifier.startswith("+") or identifier.isdigit() and len(identifier) >= 10:
            # Format phone number if it doesn't start with +
            if not identifier.startswith("+"):
                identifier = f"+{identifier}"
            user = db.query(User).filter(User.phone == identifier).first()
        else:
            try:
                user_id = int(identifier)
                user = db.query(User).filter(User.id == user_id).first()
            except ValueError:
                print(
                    f"Invalid identifier: {identifier}. Must be a phone number or user ID."
                )
                return None

        if not user:
            print(f"User not found with identifier: {identifier}")
            return None

        # Update password
        password_hash = get_password_hash(new_password)
        user.password_hash = password_hash

        db.commit()

        print("Password changed successfully for user:")
        print(f"ID: {user.id}")
        print(f"Phone: {user.phone}")
        print(f"Name: {user.first_name} {user.last_name}")

        return user

    except Exception as e:
        print(f"Error changing password: {str(e)}")
        if "db" in locals():
            db.rollback()
        return None
    finally:
        if "db" in locals():
            db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Change a user's password in the API backend"
    )
    parser.add_argument(
        "--user", required=True, help="User identifier (phone number or ID)"
    )
    parser.add_argument("--password", required=True, help="New password to set")

    args = parser.parse_args()

    user = change_password(args.user, args.password)

    if user:
        print("\nPassword changed successfully. You can now log in to the UI with:")
        print(f"Phone: {user.phone}")
        print(f"Password: {args.password}")
    else:
        print("Failed to change password.")
        sys.exit(1)


if __name__ == "__main__":
    main()
