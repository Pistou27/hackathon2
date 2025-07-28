import schedule, time
from datetime import datetime
from pipeline import run

TOPICS = ["Nouveautés Python 3.13", "Pourquoi dormir augmente la créativité"]

def job():
    topic = TOPICS[datetime.now().day % len(TOPICS)]
    run(topic, out_dir=f"auto/{datetime.now():%Y%m%d}")

schedule.every().day.at("08:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)