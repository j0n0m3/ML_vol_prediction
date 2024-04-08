import smtplib
from config import EMAIL_ADDRESS, EMAIL_PASSWORD

def generate_message(message, subject):

    EMAIL = EMAIL_ADDRESS
    PASSWORD = EMAIL_PASSWORD

    recipient = EMAIL
    auth = (EMAIL, PASSWORD)
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(auth[0], auth[1])
    
        message = f"Subject: {subject}\n\n{body}"
        body = message
        subject = subject
        server.sendmail(from_addr = auth[0], to_addrs = recipient, msg = message)
        
        print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
    finally:
        server.quit()
    
send_message(message = "Test message", subject = "Trade Subject")