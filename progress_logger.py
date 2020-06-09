import time


class ProgressLogger(object):
  """Meant to be used for total_amount >> 100"""
  def __init__(self, name, total_amount):
    self.name = name
    self.total_amount = total_amount
    self.cur_amount = 0
    self.last_time = time.time()
  
  def percentage(self, amount):
    return amount * 100 // self.total_amount
  
  def update_log(self):
    next_amount = self.cur_amount + 1
    cur_percentage = self.percentage(self.cur_amount)
    next_percentage = self.percentage(next_amount)
    if next_percentage > cur_percentage:
      next_time = time.time()
      print(
        '%s %02d%% %.1fs ' %
            (self.name, next_percentage, next_time - self.last_time),
        end='\r' if next_percentage < 100 else '\n'
      )
      self.last_time = next_time
    self.cur_amount = next_amount