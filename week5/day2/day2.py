from scapy.all import IP, TCP, UDP, wrpcap, rdpcap
import pandas as pd

# 1) Build packets: normal web, DNS, and a port scan
pkts = []
pkts.append(
    IP(src="10.0.0.5", dst="142.250.1.1") / TCP(sport=51000, dport=443) / ("x" * 1400)
)
pkts.append(
    IP(src="10.0.0.5", dst="142.250.1.1") / TCP(sport=51000, dport=443) / ("x" * 1400)
)
pkts.append(IP(src="10.0.0.5", dst="8.8.8.8") / UDP(sport=51001, dport=53) / ("d" * 30))
for port in [21, 22, 23, 80, 443, 3306]:
    pkts.append(
        IP(src="10.0.0.99", dst="10.0.0.5") / TCP(sport=40000, dport=port, flags="S")
    )

# 2) Save as a real PCAP file
wrpcap("week5/day2/sample.pcap", pkts)
print("Wrote sample.pcap with", len(pkts), "packets")

# 3) Read it back, like you would any capture
captured = rdpcap("week5/day2/sample.pcap")
print("Read back:", len(captured), "packets")

# 4) Peek at one packet in full
print("\nFIRST PACKET:")
captured[0].show()

# 5) Turn every packet into a row
rows = []
for p in captured:
    if IP in p:
        proto = "TCP" if TCP in p else "UDP" if UDP in p else "OTHER"
        layer = p[TCP] if TCP in p else p[UDP]
        rows.append(
            {
                "src": p[IP].src,
                "dst": p[IP].dst,
                "sport": layer.sport,
                "dport": layer.dport,
                "proto": proto,
                "size": len(p),
            }
        )

df = pd.DataFrame(rows)
print("\nPACKET TABLE:")
print(df)

# 6) Same idea as Day 1: one sender, how many different ports?
print("\nDISTINCT PORTS PER SOURCE:")
print(df.groupby("src")["dport"].nunique())
