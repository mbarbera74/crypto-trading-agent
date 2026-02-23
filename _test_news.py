"""Test: invia news importanti via WhatsApp + Telegram"""
from engine.accumulation_monitor import AccumulationMonitor

m = AccumulationMonitor()
news = m._fetch_important_news()
print(f"Found {len(news)} important news")
for n in news:
    print(f"  - {n['title'][:80]}")

msg = m._format_news_message(news)
if msg:
    print()
    r1 = m.notifier.send(msg)
    print(f"WhatsApp: {r1}")
    m.telegram.send_sync(msg)
    print("Telegram: sent")
else:
    print("No important news found")
