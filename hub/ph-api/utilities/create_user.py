#!/usr/bin/env python3
"""
Utility script to create a user in the API backend.
This can be used to create a user for login to the UI.
"""

import argparse
import os
import sys
from datetime import datetime

# Add the parent directory to the path so we can import from the api package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from the API package
from models import User
from auth import get_password_hash
from db import get_db


def create_user(
    phone, password, first_name="", last_name="", email="", email_verified=False
):
    """
    Create a new user in the database with the given credentials.

    Args:
        phone (str): Phone number in E.164 format (e.g., +15551234567)
        password (str): Plain text password
        first_name (str, optional): User's first name
        last_name (str, optional): User's last name
        email (str, optional): User's email
        email_verified (bool, optional): Whether the email is verified

    Returns:
        User: The created user object
    """
    try:
        # Get database session using the existing function
        db = next(get_db())

        # Check if user with this phone already exists
        existing_user = db.query(User).filter(User.phone == phone).first()
        if existing_user:
            print(f"User with phone {phone} already exists (ID: {existing_user.id}).")
            return existing_user

        # Create new user
        password_hash = get_password_hash(password)

        user = User(
            phone=phone,
            phone_verified=True,  # Set to True so user can login immediately
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            email=email,
            email_verified=email_verified,
            picture="",
            created_at=datetime.utcnow(),
            last_logged_in=datetime.utcnow(),
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        print(f"User created successfully with ID: {user.id}")
        print(f"Phone: {user.phone}")
        print(f"Name: {user.first_name} {user.last_name}")

        return user

    except Exception as e:
        print(f"Error creating user: {str(e)}")
        if "db" in locals():
            db.rollback()
        return None
    finally:
        if "db" in locals():
            db.close()


def main():
    parser = argparse.ArgumentParser(description="Create a user in the API backend")
    parser.add_argument(
        "--phone",
        required=True,
        help="Phone number in E.164 format (e.g., +15551234567)",
    )
    parser.add_argument("--password", required=True, help="Password for the user")
    parser.add_argument("--first-name", default="", help="First name")
    parser.add_argument("--last-name", default="", help="Last name")
    parser.add_argument("--email", default="", help="Email address")
    parser.add_argument(
        "--email-verified", action="store_true", help="Mark email as verified"
    )

    args = parser.parse_args()

    # Format phone number if it doesn't start with +
    phone = args.phone
    if not phone.startswith("+"):
        phone = f"+{phone}"

    user = create_user(
        phone=phone,
        password=args.password,
        first_name=args.first_name,
        last_name=args.last_name,
        email=args.email,
        email_verified=args.email_verified,
    )

    if user:
        print("\nUser created successfully. You can now log in to the UI with:")
        print(f"Phone: {phone}")
        print(f"Password: {args.password}")
    else:
        print("Failed to create user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
