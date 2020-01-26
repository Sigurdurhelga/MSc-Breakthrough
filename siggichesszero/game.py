from mcts import MCTS
from playchess import PlayChess
import timeit
from multiprocessing import Pool

def play_game(white_think, black_think):
  print("playgame started with {} {}".format(white_think, black_think))
  whiteplayer = MCTS()
  blackplayer = MCTS(negamaxing=True)
  board = PlayChess()
  while True:
    for _ in range(white_think):
        whiteplayer.do_rollout(board)
    board = whiteplayer.choose_child(board)
    # print("===================")
    # board.print_board()
    # print("===================")
    if board.is_terminal():
      break

    for _ in range(black_think):
        blackplayer.do_rollout(board)
    board = blackplayer.choose_child(board)
    print("===================")
    board.print_board()
    print("===================")
    if board.terminal:
        break
  return board.board.result(claim_draw=True)

test_times = 5
with open("results.out",'w') as f:
  for j in range(24,25):
    for k in range(1,10):
      f.write(f"Testing with white think {j} and black think {k}\n")
      time = 0
      print(f"Testing with white think {j} and black think {k}")
      with Pool(test_times) as p:
        print(p.starmap(play_game, [(j,k), (j,k), (j,k), (j,k), (j,k)]))
      f.write(f"game length average {time}\n")






