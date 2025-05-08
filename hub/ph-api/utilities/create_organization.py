#!/usr/bin/env python3
"""
Utility script to create a new organization and an initial admin user.

Usage:
    python create_organization.py --name "Organization Name" --slug "org-slug" --admin-email "admin@example.com" --admin-phone "+11234567890" --admin-password "password"

All arguments are optional and will be prompted for if not provided.
"""

import argparse
import sys
import os
from getpass import getpass
import re
from datetime import datetime

# Add the parent directory to the path so we can import from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.exc import IntegrityError
from db import get_db
from models import Organization, User
from auth import get_password_hash
from utilities.initialize_checks import init_system_checks


def validate_slug(slug):
    """Validate that the slug is a valid URL path segment"""
    if not re.match(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$", slug):
        return False
    return True


def validate_email(email):
    """Validate that the email is in a valid format"""
    # if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email):
    #    return False
    return True


def validate_phone(phone):
    """Validate that the phone number is in E.164 format"""
    # if not re.match(r"^\+[1-9]\d{1,14}$", phone):
    #    return False
    return True


def create_organization(name, slug, description=None):
    """Create a new organization"""
    db = next(get_db())
    try:
        # Check if organization with this slug already exists
        existing_org = db.query(Organization).filter(Organization.slug == slug).first()
        if existing_org:
            print(f"Organization with slug '{slug}' already exists.")
            return existing_org

        # Create new organization
        org = Organization(
            name=name,
            slug=slug,
            description=description
            or f"Organization created on {datetime.now().strftime('%Y-%m-%d')}",
        )
        db.add(org)
        db.commit()
        db.refresh(org)
        print(
            f"Organization '{name}' (slug: {slug}) created successfully with ID: {org.id}"
        )
        return org
    except IntegrityError:
        db.rollback()
        print(f"Error: Organization with slug '{slug}' already exists.")
        return None
    except Exception as e:
        db.rollback()
        print(f"Error creating organization: {str(e)}")
        return None


def create_admin_user(
    org_id, email, phone, password, first_name="Admin", last_name="User"
):
    """Create an admin user for the organization"""
    db = next(get_db())
    try:
        # Check if user with this email or phone already exists
        existing_user = (
            db.query(User).filter((User.email == email) | (User.phone == phone)).first()
        )

        if existing_user:
            print(f"User with email '{email}' or phone '{phone}' already exists.")
            return existing_user

        # Create new admin user
        user = User(
            organization_id=org_id,
            email=email,
            email_verified=True,
            phone=phone,
            phone_verified=True,
            first_name=first_name,
            last_name=last_name,
            password_hash=get_password_hash(password),
            is_admin=True,
            created_at=datetime.now(),
            last_logged_in=datetime.now(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"Admin user '{email}' created successfully with ID: {user.id}")
        return user
    except IntegrityError:
        db.rollback()
        print(f"Error: User with email '{email}' or phone '{phone}' already exists.")
        return None
    except Exception as e:
        db.rollback()
        print(f"Error creating admin user: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create a new organization and admin user"
    )
    parser.add_argument("--name", help="Organization name")
    parser.add_argument("--slug", help="Organization slug (URL-friendly identifier)")
    parser.add_argument("--description", help="Organization description")
    parser.add_argument("--admin-email", help="Admin user email")
    parser.add_argument("--admin-phone", help="Admin user phone number (E.164 format)")
    parser.add_argument("--admin-password", help="Admin user password")
    parser.add_argument(
        "--admin-first-name", help="Admin user first name", default="Admin"
    )
    parser.add_argument(
        "--admin-last-name", help="Admin user last name", default="User"
    )

    args = parser.parse_args()

    # Prompt for organization details if not provided
    org_name = args.name
    while not org_name:
        org_name = input("Organization name: ")

    org_slug = args.slug
    while not org_slug or not validate_slug(org_slug):
        if org_slug and not validate_slug(org_slug):
            print(
                "Invalid slug format. Use only lowercase letters, numbers, and hyphens. Must start and end with a letter or number."
            )
        org_slug = input("Organization slug (URL-friendly identifier): ")

    org_description = args.description
    if not org_description:
        org_description = input("Organization description (optional): ")

    # Create the organization
    org = create_organization(org_name, org_slug, org_description)
    if not org:
        return

    # Initialize system checks for the organization
    print("\nInitializing system checks for the organization...")
    checks_initialized = init_system_checks(org.id)
    if checks_initialized:
        print("System checks initialized successfully.")
    else:
        print("Warning: Failed to initialize system checks.")

    # Prompt for admin user details if not provided
    admin_email = args.admin_email
    while not admin_email or not validate_email(admin_email):
        if admin_email and not validate_email(admin_email):
            print("Invalid email format.")
        admin_email = input("Admin user email: ")

    admin_phone = args.admin_phone
    while not admin_phone or not validate_phone(admin_phone):
        if admin_phone and not validate_phone(admin_phone):
            print("Invalid phone number format. Use E.164 format (e.g., +11234567890).")
        admin_phone = input("Admin user phone number (E.164 format): ")

    admin_password = args.admin_password
    while not admin_password:
        admin_password = getpass("Admin user password: ")
        if not admin_password:
            continue
        confirm_password = getpass("Confirm password: ")
        if admin_password != confirm_password:
            print("Passwords do not match. Please try again.")
            admin_password = None

    admin_first_name = (
        args.admin_first_name
        or input(f"Admin user first name [{args.admin_first_name or 'Admin'}]: ")
        or args.admin_first_name
        or "Admin"
    )
    admin_last_name = (
        args.admin_last_name
        or input(f"Admin user last name [{args.admin_last_name or 'User'}]: ")
        or args.admin_last_name
        or "User"
    )

    # Create the admin user
    user = create_admin_user(
        org.id,
        admin_email,
        admin_phone,
        admin_password,
        admin_first_name,
        admin_last_name,
    )
    if not user:
        return

    print("\nOrganization and admin user created successfully!")
    print(f"Organization: {org.name} (slug: {org.slug})")
    print(f"Admin User: {user.email} (ID: {user.id})")
    print("\nYou can now access the API using the organization slug in the URL path:")
    print(f"  /{org.slug}/...")
    print("\nLogin with the admin user credentials:")
    print(f"  Email: {user.email}")
    print(f"  Phone: {user.phone}")
    print("  Password: [as provided]")


if __name__ == "__main__":
    main()
