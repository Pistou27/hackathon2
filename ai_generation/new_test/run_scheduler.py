import schedule, time
from pipeline import run

def job():
    topic = "Les avancées de l'IA en 2025"
    run(topic, out_dir="outputs", temperature=0.4, with_image=True)

schedule.every().hour.at(":00").do(job)

print("⏱️ Scheduler actif (toutes les heures)...")
while True:
    schedule.run_pending()
    time.sleep(1)