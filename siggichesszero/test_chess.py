import chess
import queue
import copy

a = chess.Board("3k4/8/8/8/8/8/8/K7 b - - 0 1")

print(a.is_check())
print(a.is_checkmate())
print(a.is_game_over(claim_draw=True))
print(a.is_variant_draw())
print(a.is_variant_win())
print(a.is_variant_loss())
print(a.result())
exit()

frontier = queue.Queue()
frontier.put((a,0))
while not frontier.empty():
  board, depth = frontier.get()
  if depth > 2:
    continue
  if board.is_game_over():
    print("board is game over ")
    print(board)
    break
  print("examining board")
  print(board)
  print("=======================")
  for m in board.generate_legal_moves():
    board.push(m)
    frontier.put((copy.deepcopy(board),depth+1))
    board.pop()

"""
for m in a.generate_legal_moves():
  a.push(m)
  print(a)
  a.pop()
  print("================")
  """