"""Fix dashboard - button-triggered loading with session_state persistence"""

f = open('dashboard/app.py', 'r', encoding='utf-8')
lines = f.readlines()
f.close()

# FUTURES: button-triggered
for i in range(len(lines)):
    if '@st.cache_data' in lines[i] and i+1 < len(lines) and '_load_futures' in lines[i+1]:
        lines[i]   = '    if run_futures:\n'
        lines[i+1] = '        from analysis.futures_monitor import FuturesMonitor\n'
        lines[i+2] = '        with st.spinner("Recupero dati futures..."):\n'
        lines[i+3] = '            st.session_state["futures_data"] = FuturesMonitor().analyze()\n'
        lines[i+4] = '\n'
        lines[i+5] = '    fa = st.session_state.get("futures_data")\n'
        lines[i+6] = '    if fa:\n'
        print(f"FUTURES fixed at line {i+1}")
        break

# MONITOR: button-triggered
for i in range(len(lines)):
    if '@st.cache_data' in lines[i] and i+1 < len(lines) and '_load_monitor' in lines[i+1]:
        lines[i]   = '    if run_monitor:\n'
        lines[i+1] = '        with st.spinner("Analisi segnali accumulo..."):\n'
        lines[i+2] = '            analyzer = MarketAnalyzer()\n'
        lines[i+3] = '            st.session_state["monitor_report"] = analyzer.full_analysis(period="1y")\n'
        lines[i+4] = '\n'
        lines[i+5] = '    report = st.session_state.get("monitor_report")\n'
        lines[i+6] = '    if report:\n'
        print(f"MONITOR fixed at line {i+1}")
        break

# NEWS: button-triggered
for i in range(len(lines)):
    if '@st.cache_data' in lines[i] and i+1 < len(lines) and '_load_calendar' in lines[i+1]:
        lines[i]   = '    if run_calendar:\n'
        lines[i+1] = '        from analysis.news_calendar import NewsCalendarProvider\n'
        lines[i+2] = '        with st.spinner("Recupero earnings e news..."):\n'
        lines[i+3] = '            st.session_state["news_calendar"] = NewsCalendarProvider().get_full_calendar()\n'
        lines[i+4] = '\n'
        lines[i+5] = '    calendar = st.session_state.get("news_calendar")\n'
        lines[i+6] = '    if calendar:\n'
        print(f"NEWS fixed at line {i+1}")
        break

f = open('dashboard/app.py', 'w', encoding='utf-8')
f.writelines(lines)
f.close()
print("Done!")
