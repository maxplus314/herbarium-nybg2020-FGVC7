from tensorflow import keras


class SkippableSeq(keras.utils.Sequence):
  def __init__(self, seq):
    super(SkippableSeq, self).__init__()
    self.start = 0
    self.seq = seq

  def __iter__(self):
    return self

  def __next__(self):
    res = self.seq[self.start]
    self.start = (self.start + 1) % len(self)
    return res

  def __getitem__(self, i):
    if isinstance(i, slice):
      assert i.step == None == i.stop and self.start == 0, \
          'only one suffix slicing allowed'
      oth = copy.copy(self)
      oth.start = i.start
      return oth
    else:
      return self.seq[(self.start + i) % len(self)]

  def __len__(self):
    return len(self.seq)


class PostprocessSeq(SkippableSeq):
  def __init__(self, postprocess, seq):
    super(PostprocessSeq, self).__init__(seq)
    self.postprocess = postprocess

  def __next__(self):
    return self.postprocess(super(PostprocessSeq, self).__next__())

  def __getitem__(self, i):
    return self.postprocess(super(PostprocessSeq, self).__getitem__(i))


def make_enqueuer_generator(sequence, workers):
  data_enqueuer = keras.utils.OrderedEnqueuer(sequence)
  data_enqueuer.start(workers=workers, max_queue_size=workers + 1)
  return data_enqueuer.get()
