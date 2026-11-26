# week7/day6/mitre_map.py
# Label -> ATT&CK tactic/technique + a first step for the analyst.
# The mapping is a judgement call. Verify IDs at attack.mitre.org.


def _row(tactic, tid, tname, nxt):
    return {"tactic": tactic, "id": tid, "name": tname, "next": nxt}


BRUTE = _row(
    "Credential Access",
    "T1110",
    "Brute Force",
    "Check the source for repeated failed logins; block it and reset the targeted account",
)
FLOOD = _row(
    "Impact",
    "T1499",
    "Endpoint Denial of Service",
    "Check the target service load; rate-limit or block the source",
)

MITRE = {
    "BENIGN": None,
    "FTP Patator": BRUTE,
    "SSH Patator": BRUTE,
    "Web Attack Brute Force": BRUTE,
    "PortScan": _row(
        "Discovery",
        "T1046",
        "Network Service Discovery",
        "Find the scanning source; check which ports it hit and whether any later connected",
    ),
    "Bot": _row(
        "Command and Control",
        "T1071",
        "Application Layer Protocol",
        "Find the infected host, review its outbound connections, isolate it",
    ),
    "DDoS": _row(
        "Impact",
        "T1498",
        "Network Denial of Service",
        "Confirm traffic volume, contact upstream provider, enable filtering",
    ),
    "DoS Hulk": FLOOD,
    "DoS GoldenEye": FLOOD,
    "DoS slowloris": FLOOD,
    "DoS Slowhttptest": FLOOD,
    "Web Attack XSS": _row(
        "Initial Access",
        "T1190",
        "Exploit Public-Facing Application",
        "Review web server logs for the request; patch or filter the input",
    ),
    "Web Attack Sql Injection": _row(
        "Initial Access",
        "T1190",
        "Exploit Public-Facing Application",
        "Review web server and database logs; check for data access",
    ),
    "Heartbleed": _row(
        "Initial Access",
        "T1190",
        "Exploit Public-Facing Application",
        "Check OpenSSL version on the target; patch and rotate certificates",
    ),
    "Infiltration": _row(
        "Command and Control",
        "T1105",
        "Ingress Tool Transfer",
        "Look for files the host downloaded; isolate and inspect it",
    ),
}


def enrich(label):
    """Return the ATT&CK info for a label, or None for BENIGN/unknown."""
    return MITRE.get(label)
