import chess
import time

times = []
for i in range(1,3010, 100):
    print("initializing {} many chessboards".format(i))
    start = time.time()
    for j in range(1,i):
        chess.Board().is_game_over(claim_draw=False)
    times.append(time.time() - start)

import matplotlib.pyplot as plt
print(times)
plt.plot(times)
plt.show()

