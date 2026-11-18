#!/usr/bin/env python3
"""
Generates a SYNTHETIC demo dataset (demo_emails_SYNTHETIC.csv) so you can test that the
pipeline runs end to end. !! It is NOT real data !!
Results on this file say nothing about real-world performance and must NOT be
reported as your project results. Use a real labelled dataset for submission.
"""
import csv
import random

random.seed(7)
BRANDS = ["PayPal", "Amazon", "Netflix", "Microsoft", "your bank", "Apple ID", "DHL"]
NAMES = ["Priya", "Arun", "Meena", "Karthik", "Sneha", "Ravi", "Divya", "Vijay"]
TOPICS = ["quarterly report", "team lunch", "project timeline", "invoice 4821",
          "client meeting", "training session", "holiday schedule", "code review"]

def rnd_ip():
    return ".".join(str(random.randint(11, 220)) for _ in range(4))

def phishing():
    b = random.choice(BRANDS)
    link = random.choice([
        f"http://{rnd_ip()}/login/verify",
        f"http://secure-{b.split()[0].lower()}-update.xyz/confirm?id={random.randint(1000,9999)}",
        "http://bit.ly/3xYz" + str(random.randint(10, 99)),
        f"http://{b.split()[0].lower()}.account-verify.top/signin",
    ])
    body = random.choice([
        f"URGENT: Your {b} account has been suspended! Verify your password immediately: {link}",
        f"Security alert - unusual activity detected on your {b} account. Click here to confirm your login within 24 hours: {link}",
        f"Dear customer, final notice. Update your payment details or your {b} account will expire. {link}",
        f"Congratulations WINNER! Claim your prize now, limited time offer! {link}",
    ])
    if random.random() < 0.08:        # hard case: phishing with NO link
        body = f"Dear {b} customer, reply with your password and account number to avoid suspension."
    return body

def safe():
    n, t = random.choice(NAMES), random.choice(TOPICS)
    body = random.choice([
        f"Hi {n}, please find attached the {t}. Let me know if you have questions. Thanks.",
        f"Hello team, reminder about the {t} tomorrow at 10am in the main conference room.",
        f"Hi {n}, thanks for your update on the {t}. I will review it this afternoon.",
        f"Dear {n}, your order has shipped. Track it at https://www.example-store.com/orders/{random.randint(1000,9999)}",
    ])
    if random.random() < 0.12:        # hard case: legitimate mail with urgency words + link
        body = (f"Reminder: please confirm your attendance for the {t} by Friday. "
                f"Update your availability here: https://intranet.company.com/calendar")
    return body

rows = [(phishing(), "Phishing Email") for _ in range(300)] + \
       [(safe(), "Safe Email") for _ in range(300)]
random.shuffle(rows)
with open("demo_emails_SYNTHETIC.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Email Text", "Email Type"])
    w.writerows(rows)
print("Wrote demo_emails_SYNTHETIC.csv with", len(rows), "SYNTHETIC emails")
