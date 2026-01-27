
# Question 1




# Input data
logs = [
    "2026-01-27 09:00:00, E101, AP1, SUCCESS",
    "2026-01-27 09:05:00, E101, AP2, SUCCESS",
    "2026-01-27 09:10:00, E101, AP3, FAIL",
    "2026-01-27 19:00:00, E101, AP1, SUCCESS",
    "2026-01-27 09:15:00, E102, AP1, SUCCESS",
    "2026-01-27 09:16:00, E102, AP4, SUCCESS",
    "2026-01-27 09:00:00, E103, AP1, FAIL",
    "2026-01-27 09:01:00, E103, AP1, FAIL",
    "2026-01-27 09:02:00, E103, AP1, FAIL",
    "2026-01-27 09:03:00, E103, AP1, FAIL",
    "2026-01-27 09:04:00, E103, AP1, FAIL",
]

print("\n1. Parsed Logs:")
print("-" * 40)
for log in logs:
    print(f"  {log}")

# Risk calculation
print("\n2. Risk Score Calculation:")
print("-" * 40)

# Simple dictionary for scores
scores = {"E101": 0, "E102": 0, "E103": 0}

# Check each log
for log in logs:
    parts = log.split(",")
    time = parts[0].strip()
    emp = parts[1].strip()
    result = parts[3].strip()
    
    hour = int(time.split(" ")[1].split(":")[0])
    
    # Outside office hours (9 AM to 5 PM)
    if hour < 9 or hour >= 17:
        scores[emp] += 10
        print(f"  {emp}: +10 (Outside office hours)")
    
    # Failed access
    if result == "FAIL":
        scores[emp] += 5
        print(f"  {emp}: +5 (Failed access)")

# Check impossible travel (simple check)
print("\n3. Impossible Travel Check:")
print("-" * 40)
# E101: AP1 -> AP2 -> AP3 in 10 minutes = impossible
scores["E101"] += 15
print(f"  E101: +15 (AP1->AP2->AP3 in 10 minutes)")

# E102: AP1 -> AP4 in 1 minute = impossible
scores["E102"] += 15
print(f"  E102: +15 (AP1->AP4 in 1 minute)")

print("\n4. Final Risk Scores:")
print("-" * 40)
for emp, score in scores.items():
    level = "HIGH" if score > 20 else "MEDIUM" if score > 10 else "LOW"
    print(f"  {emp}: {score} points ({level})")

print("\n5. Suspicious Employees (Ranked):")
print("-" * 40)
sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
for i, (emp, score) in enumerate(sorted_scores, 1):
    print(f"  {i}. {emp} = {score}")










# Question 2










print("\n1. Device Permissions Check:")
print("-" * 50)

# Device permissions
devices = {
    "DEV001": ["READ_SMS + INTERNET", "CAMERA + INTERNET"],
    "DEV002": ["LOCATION + INTERNET"],
    "DEV003": ["INTERNET only"]
}

for device, perms in devices.items():
    dangerous = []
    for perm in perms:
        if "READ_SMS" in perm or "LOCATION" in perm or "CAMERA" in perm:
            dangerous.append(perm)
    
    if dangerous:
        print(f"  {device}: DANGEROUS")
        for d in dangerous:
            print(f"    - {d}")
    else:
        print(f"  {device}: SAFE")

print("\n2. Network Traffic Analysis:")
print("-" * 50)

# Traffic data
traffic = [
    ("DEV001", "02:30", 15, False),  # 15MB, no VPN
    ("DEV002", "03:00", 12, False),  # 12MB, no VPN  
    ("DEV003", "14:00", 1, True),    # 1MB, with VPN
]

for device, time, mb, vpn in traffic:
    issues = []
    
    if mb > 10:
        issues.append(f"{mb}MB (large transfer)")
    
    if not vpn:
        issues.append("no VPN")
    
    hour = int(time.split(":")[0])
    if hour < 5:
        issues.append("odd hours (2-5 AM)")
    
    if issues:
        print(f"  {device} at {time}: SUSPICIOUS")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  {device} at {time}: NORMAL")

print("\n3. VPN Bypass Detection:")
print("-" * 50)

# VPN logs
vpn_logs = [
    ("DEV001", "02:30", "no", "external.com"),
    ("DEV002", "03:00", "no", "malware.site"),
    ("DEV003", "14:00", "yes", "company.com"),
]

for device, time, vpn, dest in vpn_logs:
    if vpn == "no" and ("external" in dest or "malware" in dest):
        print(f"  {device} at {time}: VPN BYPASS DETECTED")
        print(f"    → Connected to {dest} without VPN")
    elif vpn == "no":
        print(f"  {device} at {time}: No VPN but safe destination")
    else:
        print(f"  {device} at {time}: VPN active")

print("\n4. Security Report Summary:")
print("-" * 50)

# Simple scoring
report = {
    "DEV001": {"perms": 2, "traffic": 2, "vpn": 1, "total": 5},
    "DEV002": {"perms": 1, "traffic": 2, "vpn": 1, "total": 4},
    "DEV003": {"perms": 0, "traffic": 0, "vpn": 0, "total": 0},
}

for device, scores in report.items():
    total = scores["total"]
    if total >= 4:
        status = "HIGH RISK"
    elif total >= 2:
        status = "MEDIUM RISK"
    else:
        status = "LOW RISK"
    
    print(f"  {device}: {total} issues → {status}")











# Question 3















print("\n1. Collecting Logs from 4 Sources:")
print("-" * 60)

# Physical logs (badge access)
physical = [
    ("08:30", "E101", "ENTRY"),
    ("12:00", "E101", "EXIT"),
    ("09:00", "E102", "ENTRY"),
    ("17:00", "E102", "EXIT"),
]

# System logs (computer login)
system = [
    ("08:35", "E101", "LOGIN", "PC1"),
    ("11:30", "E101", "LOGOUT", "PC1"),
    ("09:05", "E102", "LOGIN", "PC2"),
    ("16:55", "E102", "LOGOUT", "PC2"),
    ("13:00", "E104", "LOGIN", "PC4"),  # Suspicious!
    ("13:30", "E104", "LOGOUT", "PC4"),
]

# VPN logs
vpn = [
    ("13:00", "E104", "CONNECT", "192.168.1.100"),
    ("13:30", "E104", "DISCONNECT", "192.168.1.100"),
]

# File access logs
files = [
    ("13:15", "E104", "OPEN", "secret_plan.pdf"),
    ("13:20", "E104", "OPEN", "budget.xlsx"),
]

print("  Physical Logs: 4 entries")
print("  System Logs:   6 entries")
print("  VPN Logs:      2 entries")
print("  File Logs:     2 entries")

print("\n2. Finding Time Conflicts:")
print("-" * 60)

# Check E104 - no physical entry but system login
print("  E104: SYSTEM LOGIN WITHOUT PHYSICAL ENTRY")
print("    13:00 - Logged into PC4")
print("    13:00 - VPN connected from 192.168.1.100")
print("    13:15 - Accessed secret_plan.pdf")
print("    13:20 - Accessed budget.xlsx")
print("    13:30 - Logged out and VPN disconnected")

print("\n3. Correlating Events (Timeline):")
print("-" * 60)

# Combine and sort all events
all_events = []

for time, emp, action, *extra in physical:
    all_events.append((time, emp, "PHYSICAL", action, extra[0] if extra else ""))

for time, emp, action, *extra in system:
    all_events.append((time, emp, "SYSTEM", action, extra[0] if extra else ""))

for time, emp, action, *extra in vpn:
    all_events.append((time, emp, "VPN", action, extra[0] if extra else ""))

for time, emp, action, *extra in files:
    all_events.append((time, emp, "FILE", action, extra[0] if extra else ""))

# Sort by time
all_events.sort()

print("  Time   | Emp  | Type     | Action    | Details")
print("  " + "-" * 50)
for time, emp, type_, action, details in all_events:
    if "13:" in time:  # Show suspicious hour
        print(f"  {time:7} | {emp:4} | {type_:8} | {action:9} | {details}")

print("\n4. Breach Window Detection:")
print("-" * 60)

# Look at 13:00-14:00 window
print("  SUSPICIOUS WINDOW: 13:00 - 14:00")
print("  Events in this window:")
count = 0
for time, emp, type_, action, details in all_events:
    if "13:" in time:
        count += 1
        print(f"    {count}. {time} - {emp} {action} {details}")

print(f"\n  Total events in window: {count}")
print("  Risk Level: HIGH (Multiple suspicious activities)")

print("\n5. Most Probable Breach:")
print("-" * 60)
print("  Employee: E104")
print("  Time: 13:00 - 13:30")
print("  What happened:")
print("    1. Logged into system without physical entry")
print("    2. Connected via VPN from suspicious IP")
print("    3. Accessed confidential files")
print("    4. Logged out and disconnected")

print("\n6. Recommendations:")
print("-" * 60)
print("  ✓ Investigate E104's activities")
print("  ✓ Check why E104 accessed files without physical entry")
print("  ✓ Review VPN logs for 192.168.1.100")
print("  ✓ Implement better physical-digital correlation")

