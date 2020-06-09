import pathlib
import multiprocessing

import numpy as np
import PIL

import progress_logger


def prepare_images_child(df, dir_in, dir_pre, resolution, proc_id):
  logger = progress_logger.ProgressLogger(
      'prepare images', len(df['file_name']))
  dir_in, dir_pre = map(pathlib.Path, (dir_in, dir_pre))
  for i, filename in enumerate(df['file_name']):
    if proc_id == 0:
      logger.update_log()

    filename_in = dir_in / filename
    filename_out = dir_pre / filename
    if filename_out.exists():
      continue
    filename_out.parents[0].mkdir(parents=True, exist_ok=True)
    
    im = PIL.Image.open(filename_in)
    if im.width > im.height:
      im = im.rotate(270, expand=True)
    im = im.resize(resolution[::-1], resample=PIL.Image.LANCZOS)
    im.save(filename_out)


def prepare_images(df, dir_in, dir_pre, resolution, workers):
  with multiprocessing.Pool(workers) as p:
    dfs = [(df.loc[idx], dir_in, dir_pre, resolution, proc_id)
           for proc_id, idx in enumerate(np.array_split(df.index, workers))]
    p.starmap(prepare_images_child, dfs)
