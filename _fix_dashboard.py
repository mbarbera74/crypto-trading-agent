"""Fix dashboard loading - replace session_state with @st.cache_data"""
import sys

f = open('dashboard/app.py', 'r', encoding='utf-8')
lines = f.readlines()
f.close()

# Fix button variable names back to original
for i in range(len(lines)):
    lines[i] = lines[i].replace('run_futures_refresh', 'run_futures')
    lines[i] = lines[i].replace('run_monitor_refresh', 'run_monitor')
    lines[i] = lines[i].replace('run_calendar_refresh', 'run_calendar')

# FUTURES fix
for i in range(len(lines)):
    if '# Auto-load al primo accesso' in lines[i] and i+8 < len(lines) and 'futures_data' in lines[i+1]:
        lines[i]   = '    @st.cache_data(ttl=300, show_spinner="Recupero dati futures...")\n'
        lines[i+1] = '    def _load_futures():\n'
        lines[i+2] = '        from analysis.futures_monitor import FuturesMonitor\n'
        lines[i+3] = '        return FuturesMonitor().analyze()\n'
        lines[i+4] = '\n'
        lines[i+5] = '    if run_futures:\n'
        lines[i+6] = '        _load_futures.clear()\n'
        lines[i+7] = '\n'
        lines[i+8] = '    fa = _load_futures()\n'
        lines[i+9] = '    if fa:\n'
        print(f"Fixed FUTURES at line {i+1}")
        break

# MONITOR fix
for i in range(len(lines)):
    if '# Auto-load al primo accesso' in lines[i] and i+8 < len(lines) and 'monitor_report' in lines[i+1]:
        lines[i]   = '    @st.cache_data(ttl=600, show_spinner="Analisi segnali accumulo...")\n'
        lines[i+1] = '    def _load_monitor():\n'
        lines[i+2] = '        analyzer = MarketAnalyzer()\n'
        lines[i+3] = '        return analyzer.full_analysis(period="1y")\n'
        lines[i+4] = '\n'
        lines[i+5] = '    if run_monitor:\n'
        lines[i+6] = '        _load_monitor.clear()\n'
        lines[i+7] = '\n'
        lines[i+8] = '    report = _load_monitor()\n'
        lines[i+9] = '    if report:\n'
        print(f"Fixed MONITOR at line {i+1}")
        break

# NEWS fix
for i in range(len(lines)):
    if '# Auto-load al primo accesso' in lines[i] and i+9 < len(lines) and 'news_calendar' in lines[i+1]:
        lines[i]   = '    from analysis.news_calendar import NewsCalendarProvider\n'
        lines[i+1] = '\n'
        lines[i+2] = '    @st.cache_data(ttl=600, show_spinner="Recupero earnings e news...")\n'
        lines[i+3] = '    def _load_calendar():\n'
        lines[i+4] = '        return NewsCalendarProvider().get_full_calendar()\n'
        lines[i+5] = '\n'
        lines[i+6] = '    if run_calendar:\n'
        lines[i+7] = '        _load_calendar.clear()\n'
        lines[i+8] = '\n'
        lines[i+9] = '    calendar = _load_calendar()\n'
        lines[i+10] = '    if calendar:\n'
        print(f"Fixed NEWS at line {i+1}")
        break

f = open('dashboard/app.py', 'w', encoding='utf-8')
f.writelines(lines)
f.close()
print("All done!")
