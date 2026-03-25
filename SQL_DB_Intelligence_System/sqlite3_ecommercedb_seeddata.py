import sqlite3

conn = sqlite3.connect("C:\sqlite3\ecommerce.db")
cursor = conn.cursor()

# Insert sample data
cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", [
    (1, "Satyendra", "Satyendra@test.com", "India"),
    (2, "Smruti", "Smruti@test.com", "India"),
    (3, "Sidhant", "Sidhant@test.com", "India"),
    (4, "Swati", "Swati@test.com", "India"),
    (5, "Divit", "Divit@test.com", "India")
])

cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", [
    (1, 4, "2026-03-22", 360.0),
    (2, 8, "2026-03-22", 900.0),
    (3, 12, "2026-03-21", 1000.0),
    (4, 20, "2026-03-21", 2500.0),
    (5, 20, "2026-03-21", 2500.0),
])

cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [
    (1, "Laptop", "Electronics", 1000),
    (2, "Phone", "Electronics", 500),
    (3, "Desktop", "Electronics", 1000),
    (4, "Monitor", "Electronics", 500),
    (5, "Tab", "Electronics", 10),
])

cursor.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?)", [
    (1, 1, 1, 1),
    (2, 2, 2, 2),
    (3, 3, 3, 3),
    (4, 4, 4, 4),
    (5, 5, 5, 5),
])

conn.commit()
conn.close()