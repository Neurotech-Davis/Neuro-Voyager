import pyxdf
import numpy as np

streams, header = pyxdf.load_xdf("data/Trial2.xdf")
print("Number of streams:", len(streams))
for i, s in enumerate(streams):
    print(f"\nStream {i}:", s["info"]["name"][0], "Type:", s["info"]["type"][0])
    ts = s["time_series"]
    print("time_series type:", type(ts))
    if isinstance(ts, list):
        print("Length of list:", len(ts))
        if len(ts) > 0:
            print("First element type:", type(ts[0]))
            print("First element len:", len(ts[0]) if isinstance(ts[0], (list, np.ndarray)) else 'N/A')
    elif isinstance(ts, np.ndarray):
        print("time_series shape:", ts.shape)
        
    print("nominal_srate:", s["info"]["nominal_srate"][0])
    times = s.get("time_stamps", [])
    if isinstance(times, np.ndarray):
        print("time_stamps shape:", times.shape)
    else:
        print("time_stamps length:", len(times))
