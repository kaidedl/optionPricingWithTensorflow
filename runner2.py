import BlackScholes
import Heston
from datetime import datetime
import sys

numRuns=int(sys.argv[1])
device_name=sys.argv[2]
model=sys.argv[3]

print(model)
res=""
for n in [ 1, 10000, 25000, 50000, 100000 ]:
	if device_name=="np":
		engine=BlackScholes.BlackScholesNp(n) if model=="BS" else Heston.HestonNp(n)
	elif device_name=="cpu":
		engine=BlackScholes.BlackScholesTf(n, "/cpu:0") if model=="BS" else Heston.HestonTf(n,False,"/cpu:0")
	else:
		engine=BlackScholes.BlackScholesTf(n, "/gpu:0") if model=="BS" else Heston.HestonTf(n,False,"/gpu:0")

	t0=datetime.now()
	for _ in range(numRuns):
		engine.price()

	t1=datetime.now()
	res += "\n"+device_name+","+str(n)+": "+str((t1-t0)/numRuns)

print(res)