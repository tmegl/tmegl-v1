import logging,os

def setLogger(id: str|None=None, log_file: str | None = None, console: bool = True, rewrite: bool = False,mode="w") -> logging.Logger:
    logger = logging.getLogger(id)
    logger.setLevel(logging.DEBUG)
    mode = mode.lower()
    logging.shutdown()

    if log_file is not None:
        log_file = log_file[:-4] if log_file.endswith(".log") else log_file
        if rewrite == False and os.path.exists(log_file + ".log"):
            for i in range(10000):
                if os.path.exists(log_file + f"_{i}.log") == False:
                    log_file = log_file + f"_{i}.log"
                    break
            else:
                raise Exception("too much tries to find a log file")
        else:
            log_file = log_file + ".log"
        os.makedirs(os.path.dirname(log_file),exist_ok=True) if os.path.dirname(log_file)!="" else None
        fh = logging.FileHandler(log_file, mode=mode, encoding="utf8")
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger