import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e.get("type") == "pending_pushed":
        for it in e["items"]:
            print(f"  PUSH  d{e['day']} -> {e['agent']}: {it[:95]}")
    if e.get("type") == "action" and e.get("tool") in (
            "challenge_agent", "accept_battle", "decline_battle"):
        print(f"  ACT   d{e['day']} {e['agent']} {e['tool']} {str(e.get('result'))[:70]}")
