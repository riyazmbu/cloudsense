
from statistics import mean, pstdev
from typing import List

def z_score(value: float, values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    sd = pstdev(values)
    if sd == 0:
        return 0.0
    return (value - mean(values)) / sd

def detect_cost_anomalies(records):
    # Calculate dataset statistics once. The previous implementation recomputed
    # mean/std-dev for every resource, which becomes unnecessarily expensive on
    # large demo uploads.
    costs = [float(r.monthly_cost or 0) for r in records]
    results = []
    avg = mean(costs) if costs else 0.0
    sd = pstdev(costs) if len(costs) >= 2 else 0.0

    for r in records:
        cost = float(r.monthly_cost or 0)
        z = (cost - avg) / sd if sd else 0.0

        # Simple explainable anomaly rule for the MVP.
        anomalous = abs(z) >= 1.5 if len(costs) >= 4 else False

        results.append({
            "resource_id": r.resource_id,
            "service": r.service,
            "environment": r.environment,
            "monthly_cost": round(cost, 2),
            "z_score": round(z, 2),
            "anomaly": anomalous,
            "severity": (
                "high" if abs(z) >= 2.5
                else "medium" if abs(z) >= 1.5
                else "normal"
            ),
        })

    return results

def service_cost_anomalies(records):
    service_values = {}
    for r in records:
        key = r.service or "Unknown"
        service_values.setdefault(key, []).append(float(r.monthly_cost or 0))

    output = []
    for service, values in service_values.items():
        avg = mean(values) if values else 0
        output.append({
            "service": service,
            "resource_count": len(values),
            "total_monthly_cost": round(sum(values), 2),
            "average_resource_cost": round(avg, 2),
            "max_resource_cost": round(max(values), 2) if values else 0,
        })

    return sorted(output, key=lambda x: x["total_monthly_cost"], reverse=True)

def forecast_monthly_cost(records, months=3):
    current = sum(float(r.monthly_cost or 0) for r in records)

    # MVP baseline forecast. With historical billing periods, this can become
    # a real time-series model (moving average/linear regression/Prophet).
    forecast = []
    for i in range(1, months + 1):
        forecast.append({
            "month_offset": i,
            "forecast_monthly_cost": round(current, 2),
            "method": "baseline_current_run_rate",
        })

    return {
        "current_monthly_cost": round(current, 2),
        "forecast": forecast,
        "note": "Forecast uses current run-rate because the MVP dataset has one billing period."
    }
