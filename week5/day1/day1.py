import pandas as pd

packets = [
    # normal web browsing: one TCP conversation to port 443
    {
        "time": 0.00,
        "src": "10.0.0.5",
        "dst": "142.250.1.1",
        "sport": 51000,
        "dport": 443,
        "proto": "TCP",
        "size": 60,
    },
    {
        "time": 0.05,
        "src": "10.0.0.5",
        "dst": "142.250.1.1",
        "sport": 51000,
        "dport": 443,
        "proto": "TCP",
        "size": 1500,
    },
    {
        "time": 0.10,
        "src": "10.0.0.5",
        "dst": "142.250.1.1",
        "sport": 51000,
        "dport": 443,
        "proto": "TCP",
        "size": 1500,
    },
    {
        "time": 0.20,
        "src": "10.0.0.5",
        "dst": "142.250.1.1",
        "sport": 51000,
        "dport": 443,
        "proto": "TCP",
        "size": 52,
    },
    # DNS lookup: UDP to port 53
    {
        "time": 0.30,
        "src": "10.0.0.5",
        "dst": "8.8.8.8",
        "sport": 51001,
        "dport": 53,
        "proto": "UDP",
        "size": 70,
    },
    {
        "time": 0.35,
        "src": "10.0.0.5",
        "dst": "8.8.8.8",
        "sport": 51001,
        "dport": 53,
        "proto": "UDP",
        "size": 120,
    },
    # suspicious: one sender knocking on many doors with tiny packets
    {
        "time": 1.00,
        "src": "10.0.0.99",
        "dst": "10.0.0.5",
        "sport": 40000,
        "dport": 21,
        "proto": "TCP",
        "size": 44,
    },
    {
        "time": 1.01,
        "src": "10.0.0.99",
        "dst": "10.0.0.5",
        "sport": 40000,
        "dport": 22,
        "proto": "TCP",
        "size": 44,
    },
    {
        "time": 1.02,
        "src": "10.0.0.99",
        "dst": "10.0.0.5",
        "sport": 40000,
        "dport": 23,
        "proto": "TCP",
        "size": 44,
    },
    {
        "time": 1.03,
        "src": "10.0.0.99",
        "dst": "10.0.0.5",
        "sport": 40000,
        "dport": 80,
        "proto": "TCP",
        "size": 44,
    },
]

df = pd.DataFrame(packets)
print("PACKETS:")
print(df)

flows = df.groupby(["src", "dst", "sport", "dport", "proto"]).agg(
    n_packets=("size", "count"),
    total_bytes=("size", "sum"),
    duration=("time", lambda t: t.max() - t.min()),
)
print("\nFLOWS:")
print(flows)
