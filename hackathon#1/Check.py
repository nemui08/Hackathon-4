import pandas as pd
from collections import Counter

# ==========================
# CONFIG
# ==========================

LOG_FILE = "cart_web.log"   # เปลี่ยนชื่อไฟล์ตามจริง
CHUNK_SIZE = 500000

cols = [
    "timestamp",
    "ip",
    "method",
    "endpoint",
    "status",
    "response_time"
]

# ==========================
# STORAGE
# ==========================

requests_per_min = Counter()
response_sum = Counter()
response_count = Counter()

status_200 = Counter()
status_404 = Counter()
status_500 = Counter()

print("Starting analysis...")

# ==========================
# READ LOG
# ==========================

for chunk in pd.read_csv(
        LOG_FILE,
        sep="|",
        header=None,
        names=cols,
        usecols=[0,1,2,3,4,5],   # ข้ามคอลัมน์ username
        chunksize=CHUNK_SIZE):

    # Clean text
    for col in ["ip", "method", "endpoint"]:
        chunk[col] = chunk[col].astype(str).str.strip()

    chunk["status"] = pd.to_numeric(
        chunk["status"],
        errors="coerce"
    )

    chunk["response_time"] = pd.to_numeric(
        chunk["response_time"],
        errors="coerce"
    )

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"].astype(str).str.strip(),
        errors="coerce"
    )

    chunk["minute"] = chunk["timestamp"].dt.floor("min")

    # ------------------
    # Requests
    # ------------------

    req_counts = chunk.groupby("minute").size()

    for minute, count in req_counts.items():
        requests_per_min[minute] += count

    # ------------------
    # Response Time
    # ------------------

    stats = chunk.groupby("minute")["response_time"].agg(
        ["sum", "count"]
    )

    for minute, row in stats.iterrows():
        response_sum[minute] += row["sum"]
        response_count[minute] += row["count"]

    # ------------------
    # Status Codes
    # ------------------

    s200 = chunk[chunk["status"] == 200].groupby("minute").size()
    s404 = chunk[chunk["status"] == 404].groupby("minute").size()
    s500 = chunk[chunk["status"] >= 500].groupby("minute").size()

    for minute, count in s200.items():
        status_200[minute] += count

    for minute, count in s404.items():
        status_404[minute] += count

    for minute, count in s500.items():
        status_500[minute] += count

print("Finished reading log.")

# ==========================
# BUILD RESULT
# ==========================

all_minutes = sorted(requests_per_min.keys())

result = pd.DataFrame({
    "minute": all_minutes
})

result["requests"] = result["minute"].map(
    requests_per_min
)

result["avg_response_time"] = result["minute"].apply(
    lambda x:
    response_sum[x] / response_count[x]
    if response_count[x] > 0 else 0
)

result["status_200"] = result["minute"].map(
    status_200
).fillna(0)

result["status_404"] = result["minute"].map(
    status_404
).fillna(0)

result["status_500"] = result["minute"].map(
    status_500
).fillna(0)

# ==========================
# ERROR RATE
# ==========================

result["error_rate"] = (
    result["status_500"] /
    result["requests"]
) * 100

# ==========================
# SAVE
# ==========================

result.to_csv(
    "timeline_analysis.csv",
    index=False
)

print("Saved timeline_analysis.csv")

# ==========================
# SUMMARY
# ==========================

print("\n====================")
print("TOP REQUEST PEAKS")
print("====================")

print(
    result.nlargest(
        20,
        "requests"
    )[[
        "minute",
        "requests"
    ]]
    .reset_index(drop=True)
)

print("\n====================")
print("TOP LATENCY PEAKS")
print("====================")

print(
    result.nlargest(
        20,
        "avg_response_time"
    )[[
        "minute",
        "avg_response_time"
    ]]
    .reset_index(drop=True)
)

print("\n====================")
print("TOP ERROR PEAKS")
print("====================")

print(
    result.nlargest(
        20,
        "status_500"
    )[[
        "minute",
        "status_500"
    ]]
    .reset_index(drop=True)
)

print("\n====================")
print("TOP ERROR RATE")
print("====================")

print(
    result.nlargest(
        20,
        "error_rate"
    )[[
        "minute",
        "requests",
        "status_500",
        "error_rate"
    ]]
    .reset_index(drop=True)
)

print("\n====================")
print("RESPONSE TIME STATS")
print("====================")

print(
    result["avg_response_time"].describe()
)

import matplotlib.pyplot as plt

# Requests
plt.figure(figsize=(15,5))
plt.plot(result["minute"], result["requests"])
plt.title("Requests Per Minute")
plt.savefig("requests_per_minute.png")
plt.close()

# Latency
plt.figure(figsize=(15,5))
plt.plot(result["minute"], result["avg_response_time"])
plt.title("Average Response Time")
plt.savefig("latency.png")
plt.close()

# Errors
plt.figure(figsize=(15,5))
plt.plot(result["minute"], result["status_500"])
plt.title("500 Errors")
plt.savefig("errors.png")
plt.close()

slow = result.nlargest(
    10,
    "avg_response_time"
)

print(slow)

errors = result.nlargest(
    10,
    "status_500"
)

print(errors)

print(result["avg_response_time"].describe())

outage_start = "2024-06-16 10:26:00"
outage_end   = "2024-06-16 10:45:00"

for chunk in pd.read_csv(
        LOG_FILE,
        sep="|",
        header=None,
        names=cols,
        usecols=[0,1,2,3,4,5],
        chunksize=500000):

    chunk["timestamp"] = pd.to_datetime(
        chunk["timestamp"].astype(str).str.strip(),
        errors="coerce"
    )

    outage = chunk[
        (chunk["timestamp"] >= outage_start)
        &
        (chunk["timestamp"] <= outage_end)
    ]

    if len(outage) > 0:

        print("\nTOP IP")
        print(outage["ip"].value_counts().head(20))

        print("\nTOP ENDPOINT")
        print(outage["endpoint"].value_counts().head(20))

        print("\nMETHOD")
        print(outage["method"].value_counts())

        break

TOP_IP = Counter()
TOP_ENDPOINT = Counter()

for chunk in pd.read_csv(
        LOG_FILE,
        sep="|",
        header=None,
        names=cols,
        usecols=[0,1,2,3,4,5],
        chunksize=500000):

    TOP_IP.update(chunk["ip"].astype(str).str.strip())

    TOP_ENDPOINT.update(
        chunk["endpoint"].astype(str).str.strip()
    )

print("\nTOP 20 IP")
print(TOP_IP.most_common(20))

print("\nTOP 20 ENDPOINT")
print(TOP_ENDPOINT.most_common(20))