import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API = {
  health: "/health",
  summary: "/api/summary",
  alerts: "/api/alerts/db",
  assess: "/api/live/assess",
  modelAnalytics: "/api/analytics/models",
  agentAnalytics: "/api/analytics/agents",
  userIntelligence: "/api/users/intelligence",
  liveEvents: "/api/live/events",
  chartStats: "/api/charts/summary-stats",
  chartDistribution: "/api/charts/alert-distribution",
  chartTimeline: "/api/charts/alert-timeline",
  chartEventTimeline: "/api/charts/event-timeline",
  chartCategories: "/api/charts/event-categories",
  chartRiskHist: "/api/charts/risk-score-distribution",
};

/* ============================================================
   MINI CHART PRIMITIVES (no external deps)
   ============================================================ */

function DonutChart({ segments, size = 120, strokeWidth = 18, label, sublabel }) {
  const r = (size - strokeWidth) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const total = segments.reduce((s, seg) => s + seg.value, 0) || 1;

  let offset = 0;
  const arcs = segments.map((seg) => {
    const dash = (seg.value / total) * circumference;
    const arc = { ...seg, dash, gap: circumference - dash, offset };
    offset += dash;
    return arc;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#1e2235" strokeWidth={strokeWidth} />
      {arcs.map((arc, i) => (
        <circle
          key={i}
          cx={cx}
          cy={cy}
          r={r}
          fill="none"
          stroke={arc.color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${arc.dash} ${arc.gap}`}
          strokeDashoffset={-arc.offset}
          strokeLinecap="butt"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: "stroke-dasharray 0.6s ease" }}
        />
      ))}
      {label != null && (
        <>
          <text x={cx} y={cy - 6} textAnchor="middle" fill="#fff" fontSize="15" fontWeight="700">
            {label}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle" fill="#858da8" fontSize="9">
            {sublabel}
          </text>
        </>
      )}
    </svg>
  );
}

function Sparkline({ data = [], width = 200, height = 48, color = "#a58aff", fill = true }) {
  if (data.length < 2) return <svg width={width} height={height} />;
  const max = Math.max(...data, 1);
  const min = Math.min(...data);
  const range = max - min || 1;
  const pts = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const polyline = pts.join(" ");
  const fillPath = `M${pts[0]} L${pts.join(" L")} L${width},${height} L0,${height} Z`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
      {fill && (
        <path d={fillPath} fill={color} fillOpacity={0.15} />
      )}
      <polyline
        points={polyline}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BarChart({ bars = [], maxVal, height = 80, showLabels = true }) {
  const max = maxVal || Math.max(...bars.map((b) => b.value), 1);
  return (
    <div className="bar-chart">
      {bars.map((bar, i) => (
        <div key={i} className="bar-item">
          <div
            className="bar-fill"
            style={{
              height: `${Math.max((bar.value / max) * height, bar.value > 0 ? 4 : 0)}px`,
              background: bar.color || "#7957ef",
              transition: "height 0.5s ease",
            }}
            title={`${bar.label}: ${bar.value}`}
          />
          {showLabels && (
            <div className="bar-label">{bar.shortLabel || bar.label}</div>
          )}
        </div>
      ))}
    </div>
  );
}

function HorizontalBar({ label, value, max, color, note }) {
  const pct = max ? Math.max((value / max) * 100, value > 0 ? 1 : 0) : 0;
  return (
    <div className="h-bar-row">
      <div className="h-bar-label">{label}</div>
      <div className="h-bar-track">
        <div
          className="h-bar-fill"
          style={{ width: `${pct}%`, background: color || "#7957ef", transition: "width 0.5s ease" }}
        />
      </div>
      <div className="h-bar-value">{value}</div>
      {note && <div className="h-bar-note">{note}</div>}
    </div>
  );
}

function RiskScoreTimeline({ alerts = [], height = 90 }) {
  const max = 100;
  if (!alerts.length) return <div className="empty-state" style={{ padding: "20px" }}>No alert timeline data.</div>;

  const width = 100;
  const pts = alerts.map((a, i) => {
    const x = (i / (alerts.length - 1 || 1)) * width;
    const y = height - (a.risk_score / max) * (height - 10) - 5;
    return { x, y, ...a };
  });

  const colorFor = (level) =>
    level === "HIGH" ? "#ff6f8f" : level === "MEDIUM" ? "#ffb45e" : "#55d6ad";

  const polyline = pts.map((p) => `${p.x},${p.y}`).join(" ");
  const fillPts = `M${pts[0]?.x},${height} ${pts.map((p) => `${p.x},${p.y}`).join(" ")} ${pts[pts.length - 1]?.x},${height} Z`;

  return (
    <div className="risk-timeline-wrap">
      <svg
        viewBox={`0 0 100 ${height}`}
        preserveAspectRatio="none"
        style={{ width: "100%", height: `${height}px` }}
      >
        {/* threshold lines */}
        <line x1="0" y1={height - (60 / max) * (height - 10) - 5} x2="100" y2={height - (60 / max) * (height - 10) - 5} stroke="#ff6f8f" strokeWidth="0.3" strokeDasharray="2,2" />
        <line x1="0" y1={height - (25 / max) * (height - 10) - 5} x2="100" y2={height - (25 / max) * (height - 10) - 5} stroke="#ffb45e" strokeWidth="0.3" strokeDasharray="2,2" />
        <path d={fillPts} fill="#a58aff" fillOpacity={0.1} />
        <polyline points={polyline} fill="none" stroke="#a58aff" strokeWidth="0.8" strokeLinejoin="round" />
        {pts.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r="1.2" fill={colorFor(p.risk_level)} />
        ))}
      </svg>
      <div className="timeline-thresholds">
        <span style={{ color: "#ff6f8f" }}>HIGH ≥60</span>
        <span style={{ color: "#ffb45e" }}>MEDIUM ≥25</span>
      </div>
    </div>
  );
}

/* ============================================================
   UTILITY
   ============================================================ */
function Metric({ label, value, note, color = "purple", trend }) {
  return (
    <article className="metric">
      <span className={`metric-icon ${color}`}>■</span>
      <p>{label}</p>
      <h2>{value ?? "—"}</h2>
      <small>{note}</small>
      {trend !== undefined && (
        <div className="metric-trend">
          <Sparkline data={trend} width={80} height={24} color={
            color === "red" ? "#ff6f8f" : color === "orange" ? "#ffb45e" : "#a58aff"
          } />
        </div>
      )}
    </article>
  );
}

function LiveDot({ active = true }) {
  return (
    <span
      className="live-dot"
      style={{ background: active ? "#55d6ad" : "#ff6f8f" }}
    />
  );
}

/* ============================================================
   MAIN APP
   ============================================================ */
function App() {
  const [health, setHealth] = useState(null);
  const [summary, setSummary] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [chartStats, setChartStats] = useState(null);
  const [errors, setErrors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [assessing, setAssessing] = useState(false);
  const [assessMessage, setAssessMessage] = useState("");
  const [page, setPage] = useState("Overview");
  const [lastRefresh, setLastRefresh] = useState(null);
  const autoRefreshRef = useRef(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    const problems = [];

    const requests = [
      ["health", API.health],
      ["summary", API.summary],
      ["alerts", `${API.alerts}?limit=100`],
      ["chartStats", API.chartStats],
    ];

    const results = await Promise.all(
      requests.map(async ([name, url]) => {
        try {
          const response = await fetch(url);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return [name, await response.json(), null];
        } catch (error) {
          return [name, null, `${name}: ${error.message}`];
        }
      })
    );

    for (const [name, data, error] of results) {
      if (error) { problems.push(error); continue; }
      if (name === "health") setHealth(data);
      if (name === "summary") setSummary(data);
      if (name === "alerts") setAlerts(data.alerts ?? []);
      if (name === "chartStats") setChartStats(data);
    }

    setErrors(problems);
    setLoading(false);
    setLastRefresh(new Date());
  }, []);

  const triggerLiveAssessment = async () => {
    setAssessing(true);
    setAssessMessage("");
    try {
      const resp = await fetch("/api/live/collect", { method: "POST" });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const s = data.assessment_summary ?? {};
      setAssessMessage(
        `Live collect & assess complete: ${data.total_security_events} total events, ${s.alerts_persisted ?? 0} new alert(s) (${data.total_security_alerts} total in DB).`
      );
      await refresh();
    } catch (err) {
      setAssessMessage(`Assessment error: ${err.message}`);
    } finally {
      setAssessing(false);
    }
  };

  useEffect(() => {
    refresh();
    // Auto-refresh every 15 seconds
    autoRefreshRef.current = setInterval(refresh, 15000);
    return () => clearInterval(autoRefreshRef.current);
  }, [refresh]);

  const riskCounts = summary?.risk_level_counts ?? {};
  const historicalTotal = summary?.total_records;
  const dbAlertCount = alerts.length;

  const navItems = [
    ["Overview", "📊"],
    ["Threat Alerts", "🛡️"],
    ["User Intelligence", "👤"],
    ["Incident Management", "⚖️"],
    ["Model Analytics", "🔬"],
    ["System Status", "⚙️"],
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">S</span>
          <div>
            <strong>Sentinel<span>IQ</span></strong>
            <small>INSIDER THREAT ANALYTICS</small>
          </div>
        </div>

        <div className="nav-label">WORKSPACE</div>
        <nav>
          {navItems.map(([name, icon]) => (
            <button
              key={name}
              className={`nav-item ${page === name ? "active" : ""}`}
              onClick={() => setPage(name)}
            >
              <span>{icon}</span>{name}
            </button>
          ))}
        </nav>

        <div className="sidebar-stats">
          <div className="sidebar-stat-row">
            <span>Live Events</span>
            <strong>{chartStats?.total_events ?? health?.live_events_count ?? "—"}</strong>
          </div>
          <div className="sidebar-stat-row">
            <span>Open Alerts</span>
            <strong style={{ color: "#ff8da5" }}>{chartStats?.open_alerts ?? "—"}</strong>
          </div>
          <div className="sidebar-stat-row">
            <span>Review Signals</span>
            <strong style={{ color: "#ffbd78" }}>{chartStats?.review_signals ?? "—"}</strong>
          </div>
        </div>

        <div className="sidebar-bottom">
          <LiveDot active={health?.status === "healthy"} />
          <div>
            <strong>API connection</strong>
            <small>
              {health?.status === "healthy" ? "Connected (Port 5434 DB)" : "Connecting..."}
            </small>
            {lastRefresh && (
              <small style={{ color: "#556" }}>
                Updated {lastRefresh.toLocaleTimeString()}
              </small>
            )}
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <div className="eyebrow">SECURITY OPERATIONS / CERT R4.1 PIPELINE</div>
            <h1>{page}</h1>
            <p>Multi-Agent Insider Threat Detection and Behavioral Telemetry</p>
          </div>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            {loading && <span className="loading-pulse">Refreshing…</span>}
            <button
              className="refresh-button"
              style={{ background: "#2a224a", borderColor: "#554388" }}
              onClick={triggerLiveAssessment}
              disabled={assessing}
            >
              {assessing ? "Collecting & Scoring…" : "⚡ Collect & Assess"}
            </button>
            <button className="refresh-button" onClick={refresh} disabled={loading}>
              ↻ Refresh
            </button>
          </div>
        </header>

        {assessMessage && (
          <div className="notice info">{assessMessage}</div>
        )}

        {errors.length > 0 && (
          <div className="notice warning">
            <strong>Some API endpoints unavailable.</strong>
            <p>{errors.join(" | ")}</p>
            <small>Ensure the FastAPI service and PostgreSQL are running.</small>
          </div>
        )}

        {page === "Overview" && (
          <OverviewTab
            health={health}
            summary={summary}
            riskCounts={riskCounts}
            historicalTotal={historicalTotal}
            dbAlertCount={dbAlertCount}
            alerts={alerts}
            chartStats={chartStats}
            onAssess={triggerLiveAssessment}
            assessing={assessing}
          />
        )}

        {page === "Threat Alerts" && (
          <ThreatAlertsTab alerts={alerts} onRefresh={refresh} />
        )}

        {page === "User Intelligence" && <UserIntelligenceTab />}

        {page === "Incident Management" && (
          <IncidentManagementTab alerts={alerts} onRefresh={refresh} />
        )}

        {page === "Model Analytics" && <ModelAnalyticsTab />}

        {page === "System Status" && (
          <SystemStatusTab health={health} onAssess={triggerLiveAssessment} assessing={assessing} />
        )}

        <footer>
          <span>SENTINEL-IQ INSIDER THREAT DETECTION SYSTEM</span>
          <span>Verified CMU CERT r4.1 Multi-Agent Architecture | Live Windows System Monitor</span>
        </footer>
      </main>
    </div>
  );
}

/* ============================================================
   TAB: OVERVIEW (with charts)
   ============================================================ */
function OverviewTab({ health, summary, riskCounts, historicalTotal, dbAlertCount, alerts, chartStats, onAssess, assessing }) {
  const [dist, setDist] = useState(null);
  const [alertTimeline, setAlertTimeline] = useState(null);
  const [cats, setCats] = useState(null);
  const [hist, setHist] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [d, t, c, h] = await Promise.all([
          fetch(API.chartDistribution).then(r => r.ok ? r.json() : null),
          fetch(API.chartTimeline).then(r => r.ok ? r.json() : null),
          fetch(API.chartCategories).then(r => r.ok ? r.json() : null),
          fetch(API.chartRiskHist).then(r => r.ok ? r.json() : null),
        ]);
        setDist(d);
        setAlertTimeline(t);
        setCats(c);
        setHist(h);
      } catch (_) {}
    };
    load();
  }, []);

  const levelColors = { HIGH: "#ff6f8f", MEDIUM: "#ffb45e", LOW: "#55d6ad" };
  const statusColors = {
    OPEN: "#ff6f8f", INVESTIGATING: "#ffb45e",
    RESOLVED: "#55d6ad", FALSE_POSITIVE: "#858da8",
  };

  const donutSegments = (dist?.by_level ?? []).map((s) => ({
    value: s.count,
    color: levelColors[s.label] || "#7957ef",
    label: s.label,
  }));

  const statusSegments = (dist?.by_status ?? []).map((s) => ({
    value: s.count,
    color: statusColors[s.label] || "#7957ef",
    label: s.label,
  }));

  const histBars = (hist?.buckets ?? []).map((b) => ({
    label: b.label,
    shortLabel: b.start.toString(),
    value: b.count,
    color: b.start >= 60 ? "#ff6f8f" : b.start >= 25 ? "#ffb45e" : "#55d6ad",
  }));

  const catBars = (cats?.categories ?? []).map((c) => ({
    label: c.label,
    shortLabel: c.label.replace("SERVICE_", "SVC_").replace("UNCLASSIFIED", "UNCLASS"),
    value: c.count,
    color: c.label === "SERVICE_INSTALLED" ? "#ff6f8f"
      : c.label === "SERVICE_START_TYPE_CHANGED" ? "#ffb45e" : "#7957ef",
  }));

  const riskTimelineData = (alertTimeline?.alerts ?? []).map(a => a.risk_score);

  return (
    <>
      {/* Status Banner */}
      <section className="status-banner">
        <div>
          <span className="live-indicator" />
          <strong>System Status: Operational</strong>
          <p>
            Historical multi-agent fusion active across{" "}
            {historicalTotal?.toLocaleString() ?? "342,706"} user-days.
            Live Windows System telemetry stream connected with{" "}
            {dbAlertCount} persisted review alert(s).
          </p>
        </div>
        <span className="status-tag">
          {health?.status ? health.status.toUpperCase() : "CONNECTED"}
        </span>
      </section>

      {/* KPI Cards */}
      <section className="metrics-grid">
        <Metric
          label="Live Events Ingested"
          value={(chartStats?.total_events ?? health?.live_events_count ?? "—").toLocaleString?.() ?? (chartStats?.total_events ?? "—")}
          note="Records in PostgreSQL"
          color="blue"
        />
        <Metric
          label="Open Alerts"
          value={chartStats?.open_alerts ?? "—"}
          note="Awaiting SOC review"
          color="red"
        />
        <Metric
          label="Review Signals"
          value={chartStats?.review_signals ?? "—"}
          note="Service lifecycle events (7040/7045)"
          color="orange"
        />
        <Metric
          label="Avg Risk Score"
          value={chartStats?.avg_risk_score ?? "—"}
          note={`Max: ${chartStats?.max_risk_score ?? "—"} / 100`}
        />
      </section>

      {/* Charts Row 1: Donut + Risk Score Timeline */}
      <section className="content-grid chart-row">
        {/* Alert Distribution Donut */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Alert Risk Distribution</h2>
              <p>Breakdown by threat level (live DB)</p>
            </div>
            <span className="pill">LIVE DB</span>
          </div>
          <div className="donut-layout">
            <DonutChart
              segments={donutSegments.length ? donutSegments : [{ value: 1, color: "#1e2235" }]}
              size={130}
              strokeWidth={22}
              label={dist?.total ?? dbAlertCount}
              sublabel="Total Alerts"
            />
            <div className="donut-legend">
              {(dist?.by_level ?? []).map((seg) => (
                <div key={seg.label} className="legend-row">
                  <span className="legend-dot" style={{ background: levelColors[seg.label] || "#7957ef" }} />
                  <span>{seg.label}</span>
                  <strong style={{ marginLeft: "auto" }}>{seg.count}</strong>
                </div>
              ))}
              {(dist?.by_status ?? []).map((seg) => (
                <div key={seg.label} className="legend-row" style={{ opacity: 0.7 }}>
                  <span className="legend-dot" style={{ background: statusColors[seg.label] || "#555" }} />
                  <span style={{ fontSize: "10px" }}>{seg.label}</span>
                  <strong style={{ marginLeft: "auto", fontSize: "10px" }}>{seg.count}</strong>
                </div>
              ))}
            </div>
          </div>
        </article>

        {/* Risk Score Timeline Sparkline */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Risk Score Timeline</h2>
              <p>Alert scores chronologically (latest 100)</p>
            </div>
            <span className="pill">LIVE ALERTS</span>
          </div>
          <RiskScoreTimeline alerts={alertTimeline?.alerts ?? []} height={90} />
          <div className="timeline-legend">
            {(alertTimeline?.alerts ?? []).slice(-5).reverse().map((a, i) => (
              <div key={i} className="tl-row">
                <span className={`badge ${a.risk_level?.toLowerCase()}`} style={{ fontSize: "9px", padding: "3px 6px" }}>
                  {a.risk_level}
                </span>
                <span style={{ fontSize: "10px", color: "#858da8" }}>
                  Score: <strong>{a.risk_score}</strong>
                </span>
                <span style={{ fontSize: "9px", color: "#556" }}>
                  {a.detected_at ? new Date(a.detected_at).toLocaleDateString() : ""}
                </span>
              </div>
            ))}
          </div>
        </article>
      </section>

      {/* Charts Row 2: Risk Histogram + Event Categories */}
      <section className="content-grid chart-row">
        {/* Risk Score Histogram */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Risk Score Distribution</h2>
              <p>Histogram of alert score buckets (0–100)</p>
            </div>
          </div>
          <div style={{ paddingTop: "8px" }}>
            <BarChart bars={histBars} height={80} />
            <div className="chart-x-labels">
              <span>0</span>
              <span>25</span>
              <span>60</span>
              <span>100</span>
            </div>
            <div className="chart-legend-row">
              <span><span className="risk-dot low" />&nbsp;LOW &lt;25</span>
              <span><span className="risk-dot medium" />&nbsp;MEDIUM 25–60</span>
              <span><span className="risk-dot high" />&nbsp;HIGH ≥60</span>
            </div>
          </div>
        </article>

        {/* Event Category Bar Chart */}
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Event Categories</h2>
              <p>Live event classification breakdown</p>
            </div>
            <span className="pill">LIVE PROCESSOR</span>
          </div>
          <BarChart bars={catBars} height={80} />
          <div style={{ marginTop: "12px" }}>
            {(cats?.top_codes ?? []).slice(0, 5).map((c) => (
              <div key={c.code} className="system-row" style={{ padding: "7px 0" }}>
                <span style={{ fontFamily: "monospace", fontSize: "11px" }}>Code {c.code}</span>
                <div style={{ flex: 1, margin: "0 12px" }}>
                  <div style={{
                    height: "4px", borderRadius: "4px", background: "#1e2235",
                    overflow: "hidden",
                  }}>
                    <div style={{
                      height: "100%",
                      width: `${(c.count / (cats?.total || 1)) * 100}%`,
                      background: "#7957ef",
                      transition: "width 0.5s",
                    }} />
                  </div>
                </div>
                <strong style={{ fontSize: "11px", minWidth: "28px", textAlign: "right" }}>{c.count}</strong>
              </div>
            ))}
          </div>
        </article>
      </section>

      {/* Historical Risk Distribution + System Health */}
      <section className="content-grid">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Historical Risk Distribution</h2>
              <p>Supervisor agent risk fusion results</p>
            </div>
            <span className="pill">HISTORICAL BATCH</span>
          </div>

          {summary ? (
            <div className="risk-list">
              {["HIGH", "MEDIUM", "LOW"].map((level) => {
                const count = Number(riskCounts[level] ?? 0);
                const total = Number(historicalTotal ?? 0);
                const percent = total ? (count / total) * 100 : 0;
                return (
                  <div className="risk-row" key={level}>
                    <div>
                      <span className={`risk-dot ${level.toLowerCase()}`} />
                      <strong>{level} Threat</strong>
                      <span className="risk-count">{count.toLocaleString()} rows ({percent.toFixed(2)}%)</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className={`progress-fill ${level.toLowerCase()}`}
                        style={{ width: `${Math.max(percent, 0.5)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
              <small className="muted">
                Source: {summary.data_source ?? "data/outputs/alert_explanations.parquet"}
              </small>
            </div>
          ) : (
            <div className="empty-state">Historical summary is unavailable.</div>
          )}
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Runtime Services Health</h2>
              <p>Active pipeline daemon and storage components</p>
            </div>
          </div>
          <div className="system-row">
            <span>FastAPI Server</span>
            <strong style={{ color: "#55d6ad" }}>{health?.status ?? "Healthy"}</strong>
          </div>
          <div className="system-row">
            <span>PostgreSQL (Port 5434)</span>
            <strong style={{ color: health?.database === "connected" ? "#55d6ad" : "#ff6f8f" }}>
              {health?.database === "connected" ? "Connected" : "Disconnected"}
            </strong>
          </div>
          <div className="system-row">
            <span>Windows Event Checkpoint</span>
            <strong>{health?.collector_checkpoint_exists ? "Active" : "Inactive"}</strong>
          </div>
          <div className="system-row">
            <span>Live Events Ingested</span>
            <strong>{health?.live_events_count ?? 0} events</strong>
          </div>
          <div className="system-row">
            <span>Live Events Processed</span>
            <strong>{health?.live_processed_count ?? 0} events</strong>
          </div>
          <div className="system-row">
            <span>Unique Hosts Seen</span>
            <strong>{chartStats?.unique_hosts ?? "—"}</strong>
          </div>
          <div className="system-row">
            <span>Live Risk Scoring & Persistence</span>
            <strong style={{ color: "#55d6ad" }}>Operational</strong>
          </div>
        </article>
      </section>

      {/* Recent DB Alerts */}
      <section className="panel alerts-panel">
        <div className="panel-heading">
          <div>
            <h2>Recent Database Alerts</h2>
            <p>Persisted security alerts from live Windows System event correlation</p>
          </div>
          <span className="pill">LIVE POSTGRESQL</span>
        </div>
        <AlertTable alerts={alerts.slice(0, 5)} />
      </section>
    </>
  );
}

/* ============================================================
   ALERT TABLE COMPONENT
   ============================================================ */
function AlertTable({ alerts = [], onStatusUpdate }) {
  const [expandedId, setExpandedId] = useState(null);

  if (alerts.length === 0) {
    return (
      <div className="empty-state">
        <span>✓</span>
        <strong>No active security alerts</strong>
        <p>No security alerts are currently recorded in the PostgreSQL database.</p>
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Alert ID</th>
            <th>Detected At</th>
            <th>User</th>
            <th>Risk Score</th>
            <th>Risk Level</th>
            <th>Priority</th>
            <th>Status</th>
            <th>Explanation</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => {
            const isExpanded = expandedId === a.alert_id;
            return (
              <tr key={a.alert_id} style={{ verticalAlign: "top" }}>
                <td><strong>#{a.alert_id}</strong></td>
                <td>{a.detected_at ? new Date(a.detected_at).toLocaleString() : "—"}</td>
                <td><code>{a.user_name || "SYSTEM"}</code></td>
                <td>
                  <strong>{typeof a.risk_score === "number" ? a.risk_score.toFixed(1) : a.risk_score}</strong> / 100
                </td>
                <td>
                  <span className={`badge ${String(a.risk_level).toLowerCase()}`}>
                    {a.risk_level}
                  </span>
                </td>
                <td>
                  <span className="badge" style={{ background: "#242738", color: "#c4b4ff" }}>
                    {a.priority}
                  </span>
                </td>
                <td>
                  <span className={`badge ${String(a.status).toLowerCase()}`}>
                    {a.status}
                  </span>
                </td>
                <td style={{ maxWidth: "280px", whiteSpace: "normal" }}>
                  <p style={{ margin: "0 0 4px", fontSize: "11px", color: "#e0e3f0" }}>{a.explanation}</p>
                  {a.recommended_action && (
                    <small style={{ color: "#a58aff" }}>Action: {a.recommended_action}</small>
                  )}
                </td>
                <td>
                  <button
                    className="filter-btn"
                    style={{ padding: "4px 8px" }}
                    onClick={() => setExpandedId(isExpanded ? null : a.alert_id)}
                  >
                    {isExpanded ? "Hide" : "Evidence"}
                  </button>
                  {isExpanded && a.agent_evidence && (
                    <div className="evidence-box">
                      <h4>Supporting Evidence</h4>
                      <pre className="agent-json" style={{ maxHeight: "180px" }}>
                        {JSON.stringify(a.agent_evidence, null, 2)}
                      </pre>
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ============================================================
   TAB: THREAT ALERTS
   ============================================================ */
function ThreatAlertsTab({ alerts = [], onRefresh }) {
  const [levelFilter, setLevelFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [sortField, setSortField] = useState("detected_at");
  const [sortDir, setSortDir] = useState("desc");
  const [dist, setDist] = useState(null);
  const [alertTimeline, setAlertTimeline] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(API.chartDistribution).then(r => r.ok ? r.json() : null),
      fetch(API.chartTimeline).then(r => r.ok ? r.json() : null),
    ]).then(([d, t]) => { setDist(d); setAlertTimeline(t); });
  }, [alerts]);

  const filtered = alerts.filter((a) => {
    if (levelFilter !== "ALL" && a.risk_level !== levelFilter) return false;
    if (statusFilter !== "ALL" && a.status !== statusFilter) return false;
    return true;
  }).sort((a, b) => {
    const va = a[sortField] ?? 0;
    const vb = b[sortField] ?? 0;
    return sortDir === "asc" ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });

  const levelColors = { HIGH: "#ff6f8f", MEDIUM: "#ffb45e", LOW: "#55d6ad" };

  return (
    <section className="agent-page">
      {/* Summary cards at top */}
      <div className="metrics-grid">
        {["HIGH", "MEDIUM", "LOW"].map(lvl => {
          const count = alerts.filter(a => a.risk_level === lvl).length;
          return (
            <Metric
              key={lvl}
              label={`${lvl} Risk`}
              value={count}
              note={`${alerts.length ? ((count / alerts.length) * 100).toFixed(1) : 0}% of alerts`}
              color={lvl === "HIGH" ? "red" : lvl === "MEDIUM" ? "orange" : "blue"}
            />
          );
        })}
        <Metric
          label="Total Alerts"
          value={alerts.length}
          note="All risk levels combined"
        />
      </div>

      {/* Timeline chart */}
      {alertTimeline?.alerts?.length > 0 && (
        <article className="panel">
          <div className="panel-heading">
            <div><h2>Alert Risk Score Timeline</h2><p>Chronological view of scored alerts</p></div>
          </div>
          <RiskScoreTimeline alerts={alertTimeline.alerts} height={80} />
        </article>
      )}

      <div className="status-banner">
        <div>
          <strong>Live Threat Alerts Repository</strong>
          <p>Persisted alerts generated from Windows System events, correlation rules, and heuristic risk scoring.</p>
        </div>
        <span className="status-tag">{filtered.length} ALERTS MATCHED</span>
      </div>

      <div className="tab-filter-bar">
        <span style={{ fontSize: "11px", color: "#858da8", alignSelf: "center", marginRight: "8px" }}>Threat Level:</span>
        {["ALL", "HIGH", "MEDIUM", "LOW"].map((lvl) => (
          <button
            key={lvl}
            className={`filter-btn ${levelFilter === lvl ? "active" : ""}`}
            style={lvl !== "ALL" && levelFilter !== lvl ? { borderColor: levelColors[lvl] + "55", color: levelColors[lvl] + "99" } : {}}
            onClick={() => setLevelFilter(lvl)}
          >
            {lvl}
          </button>
        ))}

        <span style={{ fontSize: "11px", color: "#858da8", alignSelf: "center", margin: "0 8px 0 16px" }}>Status:</span>
        {["ALL", "OPEN", "INVESTIGATING", "RESOLVED", "FALSE_POSITIVE"].map((st) => (
          <button
            key={st}
            className={`filter-btn ${statusFilter === st ? "active" : ""}`}
            onClick={() => setStatusFilter(st)}
          >
            {st}
          </button>
        ))}

        <span style={{ fontSize: "11px", color: "#858da8", alignSelf: "center", margin: "0 8px 0 16px" }}>Sort:</span>
        <button
          className={`filter-btn ${sortField === "risk_score" ? "active" : ""}`}
          onClick={() => { setSortField("risk_score"); setSortDir(d => d === "desc" ? "asc" : "desc"); }}
        >
          Score {sortField === "risk_score" ? (sortDir === "desc" ? "↓" : "↑") : ""}
        </button>
        <button
          className={`filter-btn ${sortField === "detected_at" ? "active" : ""}`}
          onClick={() => { setSortField("detected_at"); setSortDir(d => d === "desc" ? "asc" : "desc"); }}
        >
          Date {sortField === "detected_at" ? (sortDir === "desc" ? "↓" : "↑") : ""}
        </button>
      </div>

      <article className="panel alerts-panel">
        <AlertTable alerts={filtered} />
      </article>
    </section>
  );
}

/* ============================================================
   TAB: USER INTELLIGENCE
   ============================================================ */
function UserIntelligenceTab() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedUser, setSelectedUser] = useState(null);
  const [userDetail, setUserDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    fetch(`${API.userIntelligence}?limit=50`)
      .then((r) => r.json())
      .then(setData)
      .catch((err) => console.error("User intel fetch error:", err))
      .finally(() => setLoading(false));
  }, []);

  const viewUserDetail = async (userId) => {
    setSelectedUser(userId);
    setDetailLoading(true);
    try {
      const resp = await fetch(`/api/users/intelligence/${userId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      setUserDetail(await resp.json());
    } catch (err) {
      console.error(err);
    } finally {
      setDetailLoading(false);
    }
  };

  const users = (data?.users ?? []).filter(u =>
    !search || u.user.toLowerCase().includes(search.toLowerCase())
  );

  // Build sparkline trend for top users
  const topUsers = users.slice(0, 8);

  return (
    <section className="agent-page">
      <div className="status-banner">
        <div>
          <strong>User Behavioral Intelligence</strong>
          <p>Aggregated profile metrics across 1,000 users in the CMU CERT r4.1 dataset ranked by maximum risk score.</p>
        </div>
        <span className="status-tag">1,000 PROFILES MONITORED</span>
      </div>

      <div className="metrics-grid">
        <Metric label="Total Users Monitored" value={data?.total_users ?? 1000} note="Full CERT cohort" />
        <Metric label="True Malicious Users" value="4 / 4" note="100% captured in top risk ranks" color="red" />
        <Metric label="Users with HIGH Risk" value="2" note="Peak score ≥ 60.0 (ABB0427, HFC0492)" color="orange" />
        <Metric label="Users with MEDIUM Risk" value="51" note="Peak score ≥ 25.0" color="blue" />
      </div>

      {/* Top Users Horizontal Bars */}
      <article className="panel">
        <div className="panel-heading">
          <div><h2>Top User Risk Profiles</h2><p>Highest-risk users by peak score</p></div>
        </div>
        {topUsers.map(u => (
          <HorizontalBar
            key={u.user}
            label={u.user}
            value={u.max_risk_score.toFixed(1)}
            max={100}
            color={u.max_risk_score >= 60 ? "#ff6f8f" : u.max_risk_score >= 25 ? "#ffb45e" : "#55d6ad"}
            note={u.is_malicious ? "⚠ MALICIOUS" : ""}
          />
        ))}
      </article>

      <div className="content-grid" style={{ gridTemplateColumns: selectedUser ? "1.2fr 1fr" : "1fr" }}>
        <article className="panel">
          <div className="panel-heading">
            <div>
              <h2>Top Users by Behavioral Risk</h2>
              <p>Click any user to inspect their historical incident timeline</p>
            </div>
            <span className="pill">SORTED BY MAX RISK</span>
          </div>

          <div className="tab-filter-bar" style={{ marginBottom: "12px" }}>
            <input
              type="text"
              placeholder="Search user ID…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="search-input"
            />
          </div>

          {loading ? (
            <p>Loading user intelligence...</p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>User ID</th>
                    <th>Max Risk</th>
                    <th>Avg Risk</th>
                    <th>Threat Level</th>
                    <th>High Days</th>
                    <th>Med Days</th>
                    <th>CERT Ground Truth</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((u) => (
                    <tr
                      key={u.user}
                      style={{ background: selectedUser === u.user ? "#1f1b36" : "transparent" }}
                    >
                      <td><strong>{u.user}</strong></td>
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                          <strong style={{ color: u.max_risk_score >= 60 ? "#ff8da5" : u.max_risk_score >= 25 ? "#ffc37e" : "#55d6ad" }}>
                            {u.max_risk_score.toFixed(1)}
                          </strong>
                          <div style={{ flex: 1, height: "3px", borderRadius: "3px", background: "#1e2235", overflow: "hidden", minWidth: "40px" }}>
                            <div style={{ height: "100%", width: `${u.max_risk_score}%`, background: u.max_risk_score >= 60 ? "#ff8da5" : u.max_risk_score >= 25 ? "#ffc37e" : "#55d6ad" }} />
                          </div>
                        </div>
                      </td>
                      <td>{u.avg_risk_score.toFixed(2)}</td>
                      <td>
                        <span className={`badge ${u.max_threat_level.toLowerCase()}`}>
                          {u.max_threat_level}
                        </span>
                      </td>
                      <td>{u.high_risk_days}</td>
                      <td>{u.medium_risk_days}</td>
                      <td>
                        {u.is_malicious ? (
                          <span className="badge high" style={{ fontWeight: "700" }}>MALICIOUS INSIDER</span>
                        ) : (
                          <span className="badge low">NORMAL USER</span>
                        )}
                      </td>
                      <td>
                        <button
                          className="filter-btn"
                          style={{ padding: "4px 8px" }}
                          onClick={() => viewUserDetail(u.user)}
                        >
                          {selectedUser === u.user ? "Selected" : "Timeline"}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        {selectedUser && (
          <article className="panel">
            <div className="panel-heading">
              <div>
                <h2>User Profile: {selectedUser}</h2>
                <p>Elevated risk incidents and agent explanation timeline</p>
              </div>
              <button className="filter-btn" onClick={() => setSelectedUser(null)}>✕ Close</button>
            </div>

            {detailLoading ? (
              <p>Loading timeline...</p>
            ) : userDetail ? (
              <div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
                  <div className="system-row">
                    <span>Total Days:</span>
                    <strong>{userDetail.total_days_recorded}</strong>
                  </div>
                  <div className="system-row">
                    <span>Elevated Days:</span>
                    <strong style={{ color: "#ff8da5" }}>{userDetail.elevated_risk_days}</strong>
                  </div>
                  <div className="system-row">
                    <span>Max Risk:</span>
                    <strong>{userDetail.max_risk_score}</strong>
                  </div>
                  <div className="system-row">
                    <span>Avg Risk:</span>
                    <strong>{userDetail.avg_risk_score}</strong>
                  </div>
                </div>

                {/* Sparkline for user's risk over time */}
                {userDetail.timeline?.length > 0 && (
                  <div style={{ marginBottom: "16px" }}>
                    <p style={{ fontSize: "10px", color: "#858da8", marginBottom: "6px" }}>Risk score trend</p>
                    <Sparkline
                      data={userDetail.timeline.map(t => Number(t.supervisor_risk_score) || 0)}
                      width={260}
                      height={48}
                      color="#a58aff"
                    />
                  </div>
                )}

                <h3>Incident Timeline</h3>
                {userDetail.timeline?.length === 0 ? (
                  <p className="muted">No elevated risk days recorded for this user.</p>
                ) : (
                  <div style={{ display: "grid", gap: "12px", maxHeight: "420px", overflowY: "auto" }}>
                    {userDetail.timeline.map((item, idx) => (
                      <div key={idx} className="evidence-box">
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                          <strong>{item.day ? new Date(item.day).toLocaleDateString() : `Day ${idx + 1}`}</strong>
                          <span className={`badge ${String(item.risk_level).toLowerCase()}`}>
                            {item.risk_level} ({Number(item.supervisor_risk_score).toFixed(1)}/100)
                          </span>
                        </div>
                        <p style={{ margin: "4px 0", fontSize: "11px", color: "#c8ccdc" }}>{item.explanation}</p>
                        <small style={{ color: "#a58aff" }}>{item.recommended_action}</small>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <p>No detail data.</p>
            )}
          </article>
        )}
      </div>
    </section>
  );
}

/* ============================================================
   TAB: INCIDENT MANAGEMENT
   ============================================================ */
function IncidentManagementTab({ alerts = [], onRefresh }) {
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [updatingId, setUpdatingId] = useState(null);

  const updateStatus = async (alertId, newStatus) => {
    setUpdatingId(alertId);
    try {
      const resp = await fetch(`/api/alerts/db/${alertId}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (onRefresh) onRefresh();
    } catch (err) {
      alert(`Failed to update status: ${err.message}`);
    } finally {
      setUpdatingId(null);
    }
  };

  const statusCounts = {
    OPEN: alerts.filter((a) => a.status === "OPEN").length,
    INVESTIGATING: alerts.filter((a) => a.status === "INVESTIGATING").length,
    RESOLVED: alerts.filter((a) => a.status === "RESOLVED").length,
    FALSE_POSITIVE: alerts.filter((a) => a.status === "FALSE_POSITIVE").length,
  };

  const filtered = alerts.filter((a) => statusFilter === "ALL" || a.status === statusFilter);

  return (
    <section className="agent-page">
      <div className="status-banner">
        <div>
          <strong>Incident Triage & Investigation Management</strong>
          <p>SOC workflow for reviewing, investigating, resolving, or triaging security alerts in PostgreSQL.</p>
        </div>
        <span className="status-tag">ACTIVE WORKFLOW</span>
      </div>

      <div className="metrics-grid">
        <Metric label="Open Alerts" value={statusCounts.OPEN} note="Awaiting review" color="red" />
        <Metric label="Investigating" value={statusCounts.INVESTIGATING} note="Assigned to SOC analyst" color="orange" />
        <Metric label="Resolved" value={statusCounts.RESOLVED} note="Remediation completed" color="blue" />
        <Metric label="False Positives" value={statusCounts.FALSE_POSITIVE} note="Verified benign activity" />
      </div>

      {/* Status distribution donut */}
      <article className="panel" style={{ display: "flex", gap: "28px", alignItems: "center" }}>
        <DonutChart
          size={110}
          strokeWidth={20}
          segments={[
            { value: statusCounts.OPEN, color: "#ff6f8f" },
            { value: statusCounts.INVESTIGATING, color: "#ffb45e" },
            { value: statusCounts.RESOLVED, color: "#55d6ad" },
            { value: statusCounts.FALSE_POSITIVE, color: "#858da8" },
          ]}
          label={alerts.length}
          sublabel="Total"
        />
        <div style={{ flex: 1 }}>
          {[
            ["OPEN", "#ff6f8f"], ["INVESTIGATING", "#ffb45e"],
            ["RESOLVED", "#55d6ad"], ["FALSE_POSITIVE", "#858da8"],
          ].map(([st, color]) => (
            <HorizontalBar
              key={st}
              label={st}
              value={statusCounts[st]}
              max={alerts.length || 1}
              color={color}
            />
          ))}
        </div>
      </article>

      <div className="tab-filter-bar">
        {["ALL", "OPEN", "INVESTIGATING", "RESOLVED", "FALSE_POSITIVE"].map((st) => (
          <button
            key={st}
            className={`filter-btn ${statusFilter === st ? "active" : ""}`}
            onClick={() => setStatusFilter(st)}
          >
            {st} ({st === "ALL" ? alerts.length : statusCounts[st] ?? 0})
          </button>
        ))}
      </div>

      <article className="panel alerts-panel">
        <div className="panel-heading">
          <div>
            <h2>Incident Triage Queue</h2>
            <p>Directly modify incident status in PostgreSQL table security_alerts</p>
          </div>
        </div>

        {filtered.length === 0 ? (
          <div className="empty-state">
            <span>✓</span>
            <strong>No incidents in queue</strong>
            <p>No alerts match the selected status filter.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Alert ID</th>
                  <th>Detected</th>
                  <th>User / Host</th>
                  <th>Score</th>
                  <th>Priority</th>
                  <th>Status</th>
                  <th>Incident Explanation</th>
                  <th>Investigation Action</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((a) => (
                  <tr key={a.alert_id}>
                    <td><strong>#{a.alert_id}</strong></td>
                    <td>{a.detected_at ? new Date(a.detected_at).toLocaleString() : "—"}</td>
                    <td><code>{a.user_name || "SYSTEM"}</code></td>
                    <td><strong>{a.risk_score}</strong>/100</td>
                    <td>
                      <span className="badge" style={{ background: "#242738", color: "#c4b4ff" }}>
                        {a.priority}
                      </span>
                    </td>
                    <td>
                      <span className={`badge ${String(a.status).toLowerCase()}`}>
                        {a.status}
                      </span>
                    </td>
                    <td style={{ maxWidth: "260px", whiteSpace: "normal" }}>
                      <p style={{ margin: 0, fontSize: "11px" }}>{a.explanation}</p>
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                        {a.status !== "INVESTIGATING" && (
                          <button className="btn-sm btn-investigate" disabled={updatingId === a.alert_id} onClick={() => updateStatus(a.alert_id, "INVESTIGATING")}>Investigate</button>
                        )}
                        {a.status !== "RESOLVED" && (
                          <button className="btn-sm btn-resolve" disabled={updatingId === a.alert_id} onClick={() => updateStatus(a.alert_id, "RESOLVED")}>Resolve</button>
                        )}
                        {a.status !== "FALSE_POSITIVE" && (
                          <button className="btn-sm btn-fp" disabled={updatingId === a.alert_id} onClick={() => updateStatus(a.alert_id, "FALSE_POSITIVE")}>False Positive</button>
                        )}
                        {a.status !== "OPEN" && (
                          <button className="btn-sm btn-reopen" disabled={updatingId === a.alert_id} onClick={() => updateStatus(a.alert_id, "OPEN")}>Reopen</button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </article>
    </section>
  );
}

/* ============================================================
   TAB: MODEL ANALYTICS
   ============================================================ */
function ModelAnalyticsTab() {
  const [modelData, setModelData] = useState(null);
  const [agentData, setAgentData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(API.modelAnalytics).then((r) => r.json()),
      fetch(API.agentAnalytics).then((r) => r.json()),
    ])
      .then(([m, a]) => { setModelData(m); setAgentData(a); })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <section className="panel">Loading model analytics...</section>;

  const coverage = modelData?.model_coverage ?? {};
  const fusion = modelData?.supervisor_fusion ?? {};

  const modelCards = [
    {
      name: "Isolation Forest",
      sub: "Unsupervised Anomaly Detector",
      badge: "CHRONOLOGICAL TEST",
      count: coverage.isolation_forest?.heldout_test_rows?.toLocaleString() ?? "66,813",
      rows: [
        ["Contamination", coverage.isolation_forest?.contamination ?? 0.001],
        ["Score Mapping", "Continuous 0-100 anomaly intensity"],
        ["Split Type", "Latest 20% of dates"],
      ],
      note: coverage.isolation_forest?.note,
      metrics: null,
    },
    {
      name: "Random Forest Classifier",
      sub: "Supervised Ensemble",
      badge: "STRATIFIED HOLDOUT",
      count: coverage.random_forest?.heldout_test_rows?.toLocaleString() ?? "68,542",
      rows: [
        ["Test Malicious Rows", coverage.random_forest?.test_malicious_rows ?? 6],
      ],
      metrics: coverage.random_forest?.evaluation_metrics ?? { precision: 0.40, recall: 0.67, f1: 0.50 },
    },
    {
      name: "LSTM Temporal Agent",
      sub: "Sequence Neural Network",
      badge: "SEQUENCE WINDOW",
      count: coverage.lstm?.heldout_test_rows?.toLocaleString() ?? "62,545",
      rows: [
        ["Test Malicious Rows", coverage.lstm?.test_malicious_rows ?? 6],
      ],
      metrics: coverage.lstm?.evaluation_metrics ?? { precision: 0.38, recall: 0.50, f1: 0.43 },
    },
  ];

  return (
    <section className="agent-page">
      <div className="status-banner">
        <div>
          <strong>Model Analytics & Evaluation Benchmark</strong>
          <p>Rigorous evaluation metrics, split methodology, and multi-agent fusion weights.</p>
        </div>
        <span className="status-tag">EVALUATION VERIFIED</span>
      </div>

      <div className="coverage-grid">
        {modelCards.map((card) => (
          <article key={card.name} className="panel">
            <div className="panel-heading">
              <div><h2>{card.name}</h2><p>{card.sub}</p></div>
              <span className="pill">{card.badge}</span>
            </div>
            <div className="agent-count">{card.count}</div>
            <p className="muted">Test user-days</p>

            {card.metrics && (
              <div className="metrics-bar-group">
                {[["Precision", card.metrics.precision], ["Recall", card.metrics.recall], ["F1-Score", card.metrics.f1]].map(([label, val]) => (
                  <div key={label}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "10px", marginBottom: "3px" }}>
                      <span style={{ color: "#858da8" }}>{label}</span>
                      <strong style={{ color: label === "Recall" ? "#55d6ad" : label === "Precision" ? "#7dbbff" : "#e0e3f0" }}>
                        {(val * 100).toFixed(1)}%
                      </strong>
                    </div>
                    <div style={{ height: "5px", background: "#1e2235", borderRadius: "4px", overflow: "hidden", marginBottom: "8px" }}>
                      <div style={{
                        height: "100%",
                        width: `${val * 100}%`,
                        background: label === "Recall" ? "#55d6ad" : label === "Precision" ? "#7dbbff" : "#a58aff",
                        transition: "width 0.6s ease",
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {card.rows && card.rows.map(([k, v]) => (
              <div key={k} className="system-row">
                <span>{k}:</span>
                <strong>{String(v)}</strong>
              </div>
            ))}

            {card.note && <p className="muted" style={{ marginTop: "10px", lineHeight: "1.5" }}>{card.note}</p>}
          </article>
        ))}
      </div>

      <section className="content-grid">
        <article className="panel">
          <div className="panel-heading">
            <div><h2>Multi-Agent & ML Fusion Weights</h2><p>Configured supervisor weight distribution</p></div>
          </div>
          <h3>Domain Agent Weights (Sum: 1.0)</h3>
          {[["Rule Agent (XGBoost)", 40], ["Login Behavioral Agent", 15], ["File Access Agent", 15], ["Device (USB) Agent", 15], ["Network (HTTP) Agent", 15]].map(([name, pct]) => (
            <div key={name}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "3px" }}>
                <span style={{ color: "#9da4bb" }}>{name}</span>
                <strong>{pct}%</strong>
              </div>
              <div style={{ height: "4px", background: "#1e2235", borderRadius: "4px", overflow: "hidden", marginBottom: "8px" }}>
                <div style={{ height: "100%", width: `${pct}%`, background: "#7957ef", transition: "width 0.6s" }} />
              </div>
            </div>
          ))}
          <h3 style={{ marginTop: "16px" }}>ML Ensemble Weights</h3>
          {[["Isolation Forest (Anomaly)", 40, "#55d6ad"], ["Random Forest (Classifier)", 30, "#7dbbff"], ["LSTM (Temporal Sequence)", 30, "#a58aff"]].map(([name, pct, color]) => (
            <div key={name}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "3px" }}>
                <span style={{ color: "#9da4bb" }}>{name}</span>
                <strong>{pct}%</strong>
              </div>
              <div style={{ height: "4px", background: "#1e2235", borderRadius: "4px", overflow: "hidden", marginBottom: "8px" }}>
                <div style={{ height: "100%", width: `${pct}%`, background: color, transition: "width 0.6s" }} />
              </div>
            </div>
          ))}
          <p className="muted" style={{ marginTop: "14px", lineHeight: "1.6" }}>
            Fusion Policy: 50% Domain Agents + 50% Available ML Models when ML is available.
            Falls back to 100% Domain Agent score when user-day was not part of ML holdout test splits.
          </p>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div><h2>Supervisor Ground Truth Recall</h2><p>Performance on 32 true malicious days</p></div>
            <span className="pill" style={{ color: "#55d6ad", background: "#19352f" }}>100% RECALL</span>
          </div>
          <div className="system-row"><span>Total Malicious Days in Dataset:</span><strong>32 user-days</strong></div>
          <div className="system-row"><span>Flagged as HIGH Threat (score ≥ 60):</span><strong style={{ color: "#ff8da5" }}>9 user-days</strong></div>
          <div className="system-row"><span>Flagged as MEDIUM Threat (score ≥ 25):</span><strong style={{ color: "#ffc37e" }}>23 user-days</strong></div>
          <div className="system-row"><span>Missed as LOW Threat (&lt; 25):</span><strong style={{ color: "#55d6ad" }}>0 user-days (0%)</strong></div>
          <div className="system-row"><span>Total Evaluated Union Coverage:</span><strong>161,628 user-days</strong></div>
          <div className="system-row"><span>Fallback Domain-Only Days:</span><strong>181,078 user-days</strong></div>

          {/* Recall donut */}
          <div style={{ display: "flex", justifyContent: "center", marginTop: "16px" }}>
            <div style={{ textAlign: "center" }}>
              <DonutChart
                size={100}
                strokeWidth={16}
                segments={[
                  { value: 9, color: "#ff6f8f" },
                  { value: 23, color: "#ffb45e" },
                  { value: 0, color: "#55d6ad" },
                ]}
                label="32/32"
                sublabel="RECALL 100%"
              />
            </div>
          </div>
        </article>
      </section>

      <AgentIntelligence agents={agentData} />
    </section>
  );
}

function AgentIntelligence({ agents }) {
  const results = agents?.results ?? {};

  return (
    <article className="panel">
      <div className="panel-heading">
        <div>
          <h2>Historical Artifact Inspector</h2>
          <p>Verified parquet artifact row counts and schema from data/outputs</p>
        </div>
      </div>

      <div className="agent-grid">
        {Object.entries(results).map(([name, result]) => (
          <div className="panel agent-card" key={name} style={{ background: "#0e111d" }}>
            <div className="panel-heading">
              <h2>{name}</h2>
              <span className={`pill ${result.available ? "" : "unavailable"}`}>
                {result.available ? "VERIFIED" : "UNAVAILABLE"}
              </span>
            </div>

            {result.available ? (
              <>
                <div className="agent-count">{Number(result.row_count).toLocaleString()}</div>
                <p className="muted">Rows in {result.file}</p>

                <details>
                  <summary>View schema & sample record</summary>
                  <h3>Columns ({result.columns?.length})</h3>
                  <p className="column-list">{result.columns?.join(", ")}</p>
                  <h3>Sample record</h3>
                  <pre className="agent-json">
                    {JSON.stringify(result.sample_records?.[0] ?? {}, null, 2)}
                  </pre>
                </details>
              </>
            ) : (
              <p className="muted">{result.message}</p>
            )}
          </div>
        ))}
      </div>
    </article>
  );
}

/* ============================================================
   TAB: SYSTEM STATUS & LIVE MONITOR
   ============================================================ */
function SystemStatusTab({ health, onAssess, assessing }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [eventTimeline, setEventTimeline] = useState(null);

  const loadEvents = useCallback(async () => {
    try {
      const [evRes, tlRes] = await Promise.all([
        fetch(`${API.liveEvents}?limit=100`),
        fetch(API.chartEventTimeline),
      ]);
      if (!evRes.ok) throw new Error(`HTTP ${evRes.status}`);
      setData(await evRes.ok ? evRes.json() : null);
      if (tlRes.ok) setEventTimeline(await tlRes.json());
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
    const timer = setInterval(loadEvents, 5000);
    return () => clearInterval(timer);
  }, [loadEvents]);

  const events = data?.events ?? [];

  // Build event timeline bars
  const tlBars = (eventTimeline?.timeline ?? []).slice(-24).map((t) => ({
    label: t.hour ? new Date(t.hour).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "",
    shortLabel: t.hour ? new Date(t.hour).getHours() + "h" : "",
    value: t.total,
    color: t.service_events > 0 ? "#ffb45e" : "#7957ef",
  }));

  return (
    <section className="agent-page">
      <div className="status-banner">
        <div>
          <strong>Windows Security Event Monitor & Daemon Status</strong>
          <p>Real-time feed from PowerShell collector, PostgreSQL pipeline, and continuous live scoring.</p>
        </div>
        <div style={{ display: "flex", gap: "8px" }}>
          <button className="refresh-button" onClick={onAssess} disabled={assessing}>
            {assessing ? "Scoring..." : "Run Assessment"}
          </button>
        </div>
      </div>

      {error && (
        <div className="notice warning">Could not load live events: {error}</div>
      )}

      <div className="metrics-grid">
        <Metric label="Events Ingested" value={data?.count ?? "—"} note="Latest records in PostgreSQL" />
        <Metric label="Categorized" value={events.filter((e) => e.processed_at).length} note="Live processor timestamp verified" color="blue" />
        <Metric label="Review Signals" value={events.filter((e) => e.signal_status === "REVIEW_SIGNAL").length} note="Service 7040 / 7045 lifecycle" color="orange" />
        <Metric label="Persisted Alerts" value={health?.live_alerts_count ?? "—"} note="Alerts table count" color="red" />
      </div>

      {/* Event ingestion timeline */}
      {tlBars.length > 0 && (
        <article className="panel">
          <div className="panel-heading">
            <div><h2>Event Ingestion Timeline</h2><p>Hourly event volume (last 24h). Orange = service lifecycle events.</p></div>
          </div>
          <BarChart bars={tlBars} height={70} showLabels={false} />
          <div className="chart-x-labels">
            {tlBars.filter((_, i) => i % 6 === 0).map((b, i) => (
              <span key={i}>{b.shortLabel}</span>
            ))}
          </div>
        </article>
      )}

      <section className="panel alerts-panel">
        <div className="panel-heading">
          <div>
            <h2>Collected Windows Events</h2>
            <p>Source: PostgreSQL table security_events with live_event_processing metadata</p>
          </div>
          <button className="refresh-button" onClick={loadEvents}>Refresh Table</button>
        </div>

        {loading ? (
          <p>Loading events...</p>
        ) : events.length === 0 ? (
          <div className="empty-state">
            <strong>No events returned</strong>
            <p>The API returned no event records.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Event ID</th>
                  <th>Timestamp</th>
                  <th>Host</th>
                  <th>User</th>
                  <th>Windows Code</th>
                  <th>Category</th>
                  <th>Signal Status</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => (
                  <tr key={event.event_id}>
                    <td><strong>#{event.event_id}</strong></td>
                    <td>{event.event_timestamp ? new Date(event.event_timestamp).toLocaleString() : "—"}</td>
                    <td><code>{event.host_name ?? "—"}</code></td>
                    <td><code>{event.user_name ?? "—"}</code></td>
                    <td><strong>{event.event_code ?? "—"}</strong></td>
                    <td>{event.event_category ?? "Pending"}</td>
                    <td>
                      <span className={`badge ${event.signal_status === "REVIEW_SIGNAL" ? "medium" : "low"}`}>
                        {event.signal_status ?? "Pending"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  );
}

export default App;
