"""
app/hr_loader.py
Handles loading, filtering, and access-role classification of employee data
from the HR CSV file.
"""

import csv
from app.config import settings


def load_employees(csv_path: str) -> list[dict]:
    """
    Read the HR CSV, classify each employee, and return a list of dicts.

    Access Role logic:
      - NOT (Full-Time AND Active)  → "Restricted"
      - VP designation_level        → "C-Level"
      - Mapped department           → Finance / Engineering / Marketing / HR
      - Full-Time+Active, unmapped  → skipped (no valid access role)
    """
    employees = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            is_active_fulltime = (
                row["employment_type"] == "Full-Time" and
                row["employment_status"] == "Active"
            )

            if not is_active_fulltime:
                access_role = "Restricted"
            elif row["designation_level"] == "VP":
                access_role = "C-Level"
            else:
                access_role = settings.dept_bucket.get(row["department"])
                if access_role is None:
                    continue  # Full-Time + Active but unmapped department — skip

            employees.append({
                "employee_id":        row["employee_id"],
                "name":               row["full_name"],
                "gender":             row["gender"],
                "role":               row["role"],
                "department":         row["department"],
                "access_role":        access_role,
                "designation_level":  row["designation_level"],
                "email":              row["email"],
                "phone":              row["phone"],
                "location":           row["location"],
                "date_of_birth":      row["date_of_birth"],
                "date_of_joining":    row["date_of_joining"],
                "employment_type":    row["employment_type"],
                "employment_status":  row["employment_status"],
                "manager_id":         row["manager_id"],
                "salary":             row["salary"],
                "leave_balance":      row["leave_balance"],
                "leaves_taken":       row["leaves_taken"],
                "attendance_pct":     row["attendance_pct"],
                "performance_rating": row["performance_rating"],
                "last_review_date":   row["last_review_date"],
                "exit_date":          row["exit_date"],
            })

    return employees


def save_employees(employees: list[dict], output_path: str = settings.employees_output_path) -> None:
    """Write the processed employee list to a CSV file."""
    if not employees:
        print("No employees to save.")
        return

    fieldnames = list(employees[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(employees)

    print(f"Saved {len(employees)} employees -> {output_path}")
