# # time = int(input("Enter time (0-23): "))
# # day_type = input("Enter day type (weekday/weekend): ").lower()
# # balance = float(input("Enter card balance: "))

# # decision = ""
# # path = ""

# # if balance < 10:
# #     decision = "No Discount"
# #     path = "Low balance denial: Balance is below 10."
# # elif day_type == "weekend":
# #     if (12 <= time <= 14) or (18 <= time <= 20):
# #         decision = "Partial Discount"
# #         path = "Weekend rule: Peak hours (12-14 or 18-20) override full discount."
# #     else:
# #         decision = "Full Discount"
# #         path = "Weekend rule: Standard weekend hours allow full discount."
# # else:
# #     if 7 <= time <= 10:
# #         if balance > 50:
# #             decision = "Full Discount"
# #             path = "Weekday morning: Balance > 50 allows full discount."
# #         else:
# #             decision = "No Discount"
# #             path = "Weekday morning: Balance <= 50 denies discount."
# #     elif 12 <= time <= 14 or 18 <= time <= 20:
# #         decision = "Partial Discount"
# #         path = "Weekday peak: Lunch or dinner rush receives partial discount."
# #     else:
# #         decision = "No Discount"
# #         path = "Weekday standard: No active discount window."

# # print(f"Decision: {decision}")
# # print(f"Logical Path: {path}")






# violations = 0
# threshold = 3
# log = []

# while violations < threshold:
#     entry = input("Enter violations at checkpoint (comma separated) or 'exit': ")
#     if entry.lower() == 'exit':
#         break
    
#     current_checks = entry.split(',')
#     for check in current_checks:
#         check = check.strip().lower()
        
#         if violations >= threshold:
#             log.append(f"Skipped {check}: Fine threshold already reached.")
#             continue
            
#         if check in ["speed", "signal", "zone"]:
#             violations += 1
#             log.append(f"Check: {check} - Violation recorded.")
#         else:
#             log.append(f"Check: {check} - No violation.")

# final_status = "Critical - Fine Threshold Reached" if violations >= threshold else "Active - Below Threshold"

# print("\n--- Processing Log ---")
# for item in log:
#     print(item)
# print(f"\nFinal Status: {final_status}")
# print(f"Total Violations: {violations}")













# flats_data = {
#     "Flat A": [15.5, -2.0, 18.2, 500.0, 16.1],
#     "Flat B": [10.1, 12.5, 11.2, 13.0, 10.5],
#     "Flat C": [5.0, 25.0, 4.5, 28.0, 5.2]
# }

# summary = {}
# most_inconsistent_flat = ""
# max_variance = -1

# for flat, readings in flats_data.items():
#     valid_readings = []
#     for r in readings:
#         if 0 <= r <= 100:
#             valid_readings.append(r)
    
#     if not valid_readings:
#         continue
        
#     avg_usage = sum(valid_readings) / len(valid_readings)
#     peak_usage = max(valid_readings)
    
#     diffs = [abs(valid_readings[i] - valid_readings[i-1]) for i in range(1, len(valid_readings))]
#     variance = sum(diffs) / len(diffs) if diffs else 0
    
#     summary[flat] = {"Average": avg_usage, "Peak": peak_usage}
    
#     if variance > max_variance:
#         max_variance = variance
#         most_inconsistent_flat = flat

# print("--- Building Summary Report ---")
# for flat, stats in summary.items():
#     print(f"{flat} -> Average: {stats['Average']:.2f} units, Peak: {stats['Peak']:.2f} units")

# print(f"\nFlat with most inconsistent pattern: {most_inconsistent_flat}")