from datetime import datetime

def generate_current_time():
    now = datetime.now() # current date and time
    return now.strftime("%Y%m%d_%H%M")
    