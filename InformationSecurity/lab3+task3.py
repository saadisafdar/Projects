# email = {
#     "from": "support@email.com",
    
#     "to": "stu@edu.pk",
#     "subject": "URGENT: Meeting Reminder",
#     "body": "Don't forget about the meeting tomorrow at 10 AM.",
# }
# print("Emails Recieved:")
# for key, value in email.items():
#     print(f"{key}: {value}")
# warnings = []
# if "urgent" in email["subject"].lower():
#     warnings.append("The email subject contains the word 'URGENT'.")
# if "http://" in email["body"] or "https://" in email["body"]:
#     warnings.append("The email contains a non HTTPS link.")
# if email["from"].split("@")[-1].count("-") > 0 or len(email["from"].split("@")[-1].split(".")) < 2:
#     warnings.append("The sender's email domain looks suspicious.")
# if warnings:
#     print("\nWarnings:")
#     for w in warnings:
#         print("- ", w)
# else:
#     print("\n - No obivous threats detected in the email. - ", "Still be cautious!")


# import time
# from queue import Queue

# SERVICE_CAPACITY = 5
# requests = Queue()

# for i in range(8):
#     requests.put(f"Requests_{i}")

# processed = 0
# print("Server starting capacity:" , SERVICE_CAPACITY)
# while not requests.empty():
#     if processed >= SERVICE_CAPACITY:
#         print("Server Overloaded: response time increasing...")
#         time.sleep(1)
#     req = requests.get()
#     print("Processing", req)
#     processed += 1
#     time.sleep(0.1)


# users = {"student" : "1234pass"}

# def authenticate(username, password):
#     return users.get(username) == password

# wordlist = ["1234", "pass1234", "password", "1234pass"]

# for guess in wordlist:
#     print("Trying password:", guess)
#     if authenticate("student", guess):
#         print("Password found: ", guess)
#         break
# else:
#     print("Password not found in the tested wordlist")


# def sender(msg):
#     print("[Sender] sending:", msg)
#     return msg
# def network_path(message):
#     print("[Hacker] intercepted:", message)
#     return message
# def receiver(msg):
#     print("[Receiver] recieiving:", msg)
    
# if __name__ == "__main__":
#     s = sender("Sensitive: password123")
#     r = network_path(s)
#     receiver(r)


# def analyze_emails():
#     suspicious_words = ["urgent", "verify", "click here", "account"]
#     risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
#     for i in range(1, 4):
#         subject = input(f"Enter subject of Email {i}: ").lower()
#         body = input(f"Enter body of Email {i}: ").lower()
#         email_text = subject + " " + body
#         flagged = [word for word in suspicious_words if word in email_text]
#         red_flags = len(flagged)
#         if red_flags == 0:
#             risk = "Low Risk"
#         elif red_flags == 1:
#             risk = "Medium Risk"
#         else:
#             risk = "High Risk"
#         risk_counts[risk] += 1
#         print(f"\nEmail {i}:")
#         print(f"Suspicious words found: {flagged if flagged else 'None'}")
#         print(f"Total red flags: {red_flags}")
#         print(f"Risk level: {risk}\n")
#     print("Overall Summary:")
#     print("Total emails analyzed: 3")
#     print(f"Low Risk emails: {risk_counts['Low Risk']}")
#     print(f"Medium Risk emails: {risk_counts['Medium Risk']}")
#     print(f"High Risk emails: {risk_counts['High Risk']}")
# analyze_emails()


# def sender(msg):
#     checksum = sum(ord(c) for c in msg)
#     packet = f"{msg}|{checksum}"
#     print("Sender sends:", packet)
#     return packet

# def network(packet, tamper=False):
#     if tamper:
#         tampered_msg = packet.split('|')[0].replace("abc123", "ab123")  # Example tampering
#         checksum = packet.split('|')[1]  # Keep old checksum without recalculating
#         tampered_packet = f"{tampered_msg}|{checksum}"
#         print("Attacker sends on:", tampered_packet)
#         return tampered_packet
#     else:
#         print("Network sends:", packet)
#         return packet

# def receiver(packet):
#     msg, checksum_str = packet.split('|')
#     checksum = sum(ord(c) for c in msg)
#     print("Receiver gets packet:", packet)
#     if str(checksum) == checksum_str:
#         print("OK")
#     else:
#         print("Tampered!")

# # Demonstration without tampering
# pkt = sender("abc123")
# pkt = network(pkt, tamper=False)
# receiver(pkt)

# print()  # Separator

# # Demonstration with tampering
# pkt = sender("abc123")
# pkt = network(pkt, tamper=True)
# receiver(pkt)




def process_requests(requests, capacity, per_source_limit):
    processed_count = 0
    blocked_sources = set()
    source_count = {}
    for req_id, source in requests:
        if source in blocked_sources:
            print(f"Blocked {req_id} from {source}")
            continue
        if processed_count >= capacity:
            print("Server overloaded!")
            break
        count = source_count.get(source, 0)
        if count >= per_source_limit:
            blocked_sources.add(source)
            print(f"Blocked {req_id} from {source}")
            continue
        print(f"Processing {req_id} from {source}")
        processed_count += 1
        source_count[source] = count + 1
    print("\nSummary:")
    print(f"Total processed requests: {processed_count}")
    print(f"Blocked sources: {', '.join(blocked_sources) if blocked_sources else 'None'}")
requests = [
    (1, "10.0.0.1"),
    (2, "10.0.0.2"),
    (3, "10.0.0.1"),
    (4, "10.0.0.1"),
    (5, "10.0.0.3"),
    (6, "10.0.0.2"),
]
capacity = 5
per_source_limit = 2
process_requests(requests, capacity, per_source_limit)
